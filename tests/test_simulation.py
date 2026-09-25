"""Test del ciclo di simulazione su scenari brevi dello scenario di riferimento."""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cubesat_sim.core.config import MissionConfig, load_config
from cubesat_sim.core.results import SimulationResults
from cubesat_sim.simulation.runner import run_simulation

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"
CONFIG = load_config(BASELINE)
SHORT_S = 120.0  # 2 minuti dal rilascio
CALM_S = 600.0  # 10 minuti: detumble immediato, manovra e puntamento al Sole
WHEEL_COLUMNS = ["wheel_momentum_x_nms", "wheel_momentum_y_nms", "wheel_momentum_z_nms"]


def scenario(duration_s: float, **changes: dict[str, Any]) -> MissionConfig:
    """Scenario di riferimento accorciato, con alcuni valori cambiati per sezione."""
    data = CONFIG.model_dump()
    data["simulation"]["duration_h"] = duration_s / 3600.0
    for section, values in changes.items():
        data[section].update(values)
    return MissionConfig.model_validate(data)


@pytest.fixture(scope="module")
def tumbling() -> SimulationResults:
    """Primi 2 minuti dopo il rilascio a 10 gradi/s."""
    return run_simulation(scenario(SHORT_S))


@pytest.fixture(scope="module")
def calm() -> SimulationResults:
    """Rilascio quasi fermo (0,1 gradi/s) sul lato illuminato dell'orbita."""
    return run_simulation(
        scenario(
            CALM_S,
            initial_state={"max_rate_deg_s": 0.1},
            orbit={"initial_argument_of_latitude_deg": 180.0},
        )
    )


def test_same_seed_gives_identical_results(tumbling: SimulationResults) -> None:
    again = run_simulation(scenario(SHORT_S))
    pd.testing.assert_frame_equal(
        again.timeseries, tumbling.timeseries, check_exact=True
    )
    assert again.metadata == tumbling.metadata


def test_different_seed_gives_different_results(tumbling: SimulationResults) -> None:
    other = run_simulation(scenario(SHORT_S, simulation={"seed": 7}))
    assert not other.timeseries.equals(tumbling.timeseries)


def test_output_grid_and_metadata(tumbling: SimulationResults) -> None:
    series = tumbling.timeseries
    assert series["time_s"].tolist() == [10.0 * k for k in range(12)]
    assert set(series["mode"]) == {"DETUMBLE"}
    metadata = tumbling.metadata
    assert metadata["seed"] == 42
    assert metadata["transitions"] == []
    initial_rate = np.linalg.norm(metadata["initial_state"]["angular_velocity_rad_s"])
    assert initial_rate == pytest.approx(math.radians(10.0))
    assert MissionConfig.model_validate(metadata["config"]) == scenario(SHORT_S)


def test_results_survive_save_and_load(
    tumbling: SimulationResults, tmp_path: Path
) -> None:
    tumbling.save(tmp_path)
    loaded = SimulationResults.load(tmp_path)
    pd.testing.assert_frame_equal(loaded.timeseries, tumbling.timeseries, rtol=1e-12)
    assert loaded.metadata["initial_state"] == tumbling.metadata["initial_state"]


def test_energy_and_data_balances_close(calm: SimulationResults) -> None:
    # Bilancio della batteria e conservazione dei dati sull'intera corsa.
    last = calm.timeseries.iloc[-1]
    capacity_j = CONFIG.battery.capacity_j
    initial_j = CONFIG.battery.initial_soc * capacity_j
    stored_change_j = last["state_of_charge"] * capacity_j - initial_j
    balance_j = last["energy_generated_j"] - last["energy_consumed_j"]
    accounted_j = (
        stored_change_j
        + last["energy_loss_j"]
        + last["energy_wasted_j"]
        - last["energy_unmet_j"]
    )
    assert balance_j == pytest.approx(accounted_j, rel=1e-9)
    assert last["energy_generated_j"] > 0.0
    stored_bit = last["data_acquired_bit"] - last["data_downlinked_bit"]
    assert last["data_stored_bit"] == pytest.approx(stored_bit)


def test_calm_release_points_panels_to_sun(calm: SimulationResults) -> None:
    # Detumble immediato, poi una sola transizione: la manovra verso il Sole,
    # limitata a 1 grado/s, non riporta il satellite in DETUMBLE.
    transitions = calm.metadata["transitions"]
    assert len(transitions) == 1
    assert transitions[0]["previous"] == "DETUMBLE"
    assert transitions[0]["current"] == "SUN_POINTING"
    assert transitions[0]["time_s"] == CONFIG.modes.confirmation_time_s
    series = calm.timeseries
    rates_deg_s = np.degrees(series["angular_rate_rad_s"].to_numpy())
    assert rates_deg_s.max() < CONFIG.modes.detumble_enter_rate_deg_s
    # Alla fine i pannelli guardano il Sole: errore piccolo, potenza di picco.
    last = series.iloc[-1]
    assert math.degrees(last["pointing_error_rad"]) < 1.0
    assert last["solar_power_w"] > 19.0


def test_safe_with_fast_rotation_uses_bdot() -> None:
    # Batteria sotto la soglia al rilascio: SAFE ha la priorità sul DETUMBLE.
    # Il satellite ruota ancora a 10 gradi/s, quindi il controllo resta il
    # B-dot: le ruote non vengono usate e i carichi del carico utile sono spenti.
    result = run_simulation(scenario(SHORT_S, battery={"initial_soc_pct": 25.0}))
    transitions = result.metadata["transitions"]
    assert len(transitions) == 1
    assert transitions[0]["current"] == "SAFE"
    assert transitions[0]["cause"].startswith("batteria scarica")
    series = result.timeseries
    assert (series[WHEEL_COLUMNS] == 0.0).all().all()
    safe = series[series["mode"] == "SAFE"]
    platform_w = CONFIG.loads.obc_w + CONFIG.loads.adcs_base_w
    torquers_w = 3 * CONFIG.loads.magnetorquer_max_power_w
    assert safe["consumed_power_w"].max() <= platform_w + torquers_w
