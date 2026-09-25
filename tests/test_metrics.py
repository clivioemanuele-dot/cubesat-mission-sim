"""Test delle metriche di missione su una serie costruita a mano e su una corsa vera."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cubesat_sim.analysis.metrics import compute_metrics
from cubesat_sim.core.config import MissionConfig, load_config
from cubesat_sim.core.results import SimulationResults
from cubesat_sim.environment.orbit import CircularOrbit
from cubesat_sim.simulation.runner import run_simulation

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"
CONFIG = load_config(BASELINE)
PERIOD_S = CircularOrbit.from_config(
    CONFIG.orbit, CONFIG.simulation.start_epoch
).period_s
DURATION_S = 12_000.0  # due orbite complete (2 x 5677 s) e un pezzo della terza
STEP_S = 10.0


def config_data(duration_s: float, **sections: dict[str, Any]) -> dict[str, Any]:
    """Scenario di riferimento con durata e valori cambiati, come dizionario."""
    data = CONFIG.model_dump(mode="json")
    data["simulation"]["duration_h"] = duration_s / 3600.0
    for section, values in sections.items():
        data[section].update(values)
    return data


def synthetic_results() -> SimulationResults:
    """Serie inventata con risultati noti.

    Modi: DETUMBLE fino a 1000 s, SUN_POINTING fino a 5000 s, NADIR fino a
    5300 s, poi DOWNLINK fino a 5600 s e di nuovo SUN_POINTING. Potenze
    costanti: 10 W prodotti, 2 W consumati. Due episodi di saturazione.
    """
    time_s = np.arange(0.0, DURATION_S, STEP_S)
    mode = np.full(time_s.size, "SUN_POINTING", dtype=object)
    mode[time_s < 1000.0] = "DETUMBLE"
    mode[(time_s >= 5000.0) & (time_s < 5300.0)] = "NADIR"
    mode[(time_s >= 5300.0) & (time_s < 5600.0)] = "DOWNLINK"
    error_deg = np.full(time_s.size, 0.5)
    error_deg[mode == "DETUMBLE"] = math.nan
    error_deg[time_s == 1000.0] = 30.0  # inizio della manovra verso il Sole
    error_deg[mode == "NADIR"] = 2.0
    error_deg[mode == "DOWNLINK"] = 5.0
    visible = (time_s >= 5300.0) & (time_s < 5700.0)  # 400 s, 300 in DOWNLINK
    antenna_deg = np.where(visible, 5.0, math.nan)
    saturated = ((time_s >= 2000.0) & (time_s < 2050.0)) | (
        (time_s >= 8000.0) & (time_s < 8100.0)
    )
    saturation_s = np.concatenate(([0.0], np.cumsum(saturated[:-1] * STEP_S)))
    series = pd.DataFrame(
        {
            "time_s": time_s,
            "mode": mode.astype(str),
            "station_visible": visible,
            "pointing_error_rad": np.radians(error_deg),
            "antenna_error_rad": np.radians(antenna_deg),
            "wheel_momentum_x_nms": np.linspace(0.0, 0.004, time_s.size),
            "wheel_momentum_y_nms": np.zeros(time_s.size),
            "wheel_momentum_z_nms": np.full(time_s.size, -0.001),
            "state_of_charge": 0.7 - 1e-5 * time_s,
            "energy_generated_j": 10.0 * time_s,
            "energy_consumed_j": 2.0 * time_s,
            "wheel_saturation_time_s": saturation_s,
            "data_acquired_bit": np.full(time_s.size, 5e8),
            "data_downlinked_bit": np.full(time_s.size, 3e8),
            "data_stored_bit": np.full(time_s.size, 2e8),
            "data_lost_bit": np.zeros(time_s.size),
        }
    )
    transitions = [
        {"time_s": 1000.0, "previous": "DETUMBLE", "current": "SUN_POINTING"},
        {"time_s": 5000.0, "previous": "SUN_POINTING", "current": "NADIR"},
        {"time_s": 5300.0, "previous": "NADIR", "current": "DOWNLINK"},
        {"time_s": 5600.0, "previous": "DOWNLINK", "current": "SUN_POINTING"},
    ]
    metadata = {"config": config_data(DURATION_S), "transitions": transitions}
    return SimulationResults(timeseries=series, metadata=metadata)


def test_detumble_and_time_in_modes() -> None:
    metrics = compute_metrics(synthetic_results())
    assert metrics.detumble_time_h == pytest.approx(1000.0 / 3600.0)
    hours = metrics.mode_hours
    assert hours["DETUMBLE"] == pytest.approx(1000.0 / 3600.0)
    assert hours["NADIR"] == pytest.approx(300.0 / 3600.0)
    assert hours["SAFE"] == 0.0
    assert sum(hours.values()) == pytest.approx(DURATION_S / 3600.0)


@pytest.mark.parametrize(
    ("transitions", "expected_s"),
    [
        # Batteria scarica durante il detumble: SAFE, poi di nuovo DETUMBLE.
        (
            [
                (600.0, "DETUMBLE", "SAFE"),
                (1300.0, "SAFE", "DETUMBLE"),
                (2000.0, "DETUMBLE", "SUN_POINTING"),
            ],
            2000.0,
        ),
        # Rotazione smorzata mentre è in SAFE: conta l'uscita dal SAFE.
        ([(600.0, "DETUMBLE", "SAFE"), (3000.0, "SAFE", "SUN_POINTING")], 3000.0),
        # Mai arrivato a un modo di puntamento.
        ([(600.0, "DETUMBLE", "SAFE")], None),
    ],
)
def test_detumble_ends_in_a_pointing_mode(
    transitions: list[tuple[float, str, str]], expected_s: float | None
) -> None:
    results = synthetic_results()
    results.metadata["transitions"] = [
        {"time_s": time_s, "previous": previous, "current": current}
        for time_s, previous, current in transitions
    ]
    detumble_h = compute_metrics(results).detumble_time_h
    if expected_s is None:
        assert detumble_h is None
    else:
        assert detumble_h == pytest.approx(expected_s / 3600.0)


def test_pointing_statistics_per_mode() -> None:
    stats = {item.mode: item for item in compute_metrics(synthetic_results()).pointing}
    assert list(stats) == ["DOWNLINK", "NADIR", "SUN_POINTING"]  # ordine di priorità
    assert stats["NADIR"].mean_deg == pytest.approx(2.0)
    sun = stats["SUN_POINTING"]
    assert sun.max_deg == pytest.approx(30.0)
    assert sun.p95_deg == pytest.approx(0.5)  # la manovra è un solo campione
    assert sun.mean_deg > 0.5


def test_wheels_and_link() -> None:
    metrics = compute_metrics(synthetic_results())
    assert metrics.wheel_saturation_events == 2
    assert metrics.wheel_saturation_s == pytest.approx(150.0)
    assert metrics.max_wheel_momentum_fraction == pytest.approx(0.4)
    assert metrics.station_visible_h == pytest.approx(400.0 / 3600.0)
    assert metrics.link_availability == pytest.approx(300.0 / 400.0)


def test_energy_per_complete_orbit() -> None:
    metrics = compute_metrics(synthetic_results())
    assert len(metrics.orbits) == 2
    first = metrics.orbits[0]
    assert first.generated_wh == pytest.approx(10.0 * PERIOD_S / 3600.0)
    assert first.net_wh == pytest.approx(8.0 * PERIOD_S / 3600.0)
    last_inside_s = STEP_S * math.floor(PERIOD_S / STEP_S)
    assert first.min_state_of_charge == pytest.approx(0.7 - 1e-5 * last_inside_s)
    assert metrics.min_state_of_charge == pytest.approx(0.7 - 1e-5 * 11_990.0)


def test_data_totals() -> None:
    metrics = compute_metrics(synthetic_results())
    assert metrics.data_acquired_gbit == pytest.approx(0.5)
    assert metrics.data_downlinked_gbit == pytest.approx(0.3)
    assert metrics.data_on_board_gbit == pytest.approx(0.2)
    assert metrics.data_lost_gbit == 0.0


def test_metrics_of_a_real_short_run() -> None:
    # Rilascio quasi fermo: esce dal DETUMBLE dopo il tempo di conferma.
    data = config_data(600.0, initial_state={"max_rate_deg_s": 0.1})
    metrics = compute_metrics(run_simulation(MissionConfig.model_validate(data)))
    assert metrics.detumble_time_h == pytest.approx(10.0 / 3600.0)
    assert sum(metrics.mode_hours.values()) == pytest.approx(600.0 / 3600.0)
    assert metrics.orbits == ()
