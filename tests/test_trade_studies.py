"""Test degli studi di compromesso."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cubesat_sim.analysis.trade_studies import (
    battery_sizing_trade,
    pointing_energy_trade,
)
from cubesat_sim.core.config import load_config
from cubesat_sim.core.results import SimulationResults
from cubesat_sim.core.vectors import angle_between
from cubesat_sim.environment.earth import julian_date
from cubesat_sim.environment.orbit import CircularOrbit
from cubesat_sim.environment.sun import is_in_eclipse, sun_position_eci

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"
CONFIG = load_config(BASELINE)


# --- Studio 1: puntare la Terra o il Sole ---


def test_sun_pointing_gives_peak_power_in_sunlight() -> None:
    # Pannelli +X verso il Sole: 20,7 W a 1 UA, circa 20,56 W all'equinozio
    # (distanza dal Sole 1,0034 UA). Frazione al Sole: 1 - 34,8 / 94,6.
    trade = pointing_energy_trade(CONFIG)
    assert trade.period_min == pytest.approx(94.62, abs=0.01)
    assert trade.sunlit_fraction == pytest.approx(1 - 34.8 / 94.6, abs=0.005)
    assert trade.sun_pointing_power_w == pytest.approx(20.56, abs=0.05)
    expected_wh = (
        trade.sun_pointing_power_w * trade.sunlit_fraction * trade.period_min / 60
    )
    assert trade.sun_pointing_wh == pytest.approx(expected_wh, rel=1e-9)


def test_nadir_power_follows_sun_nadir_geometry() -> None:
    # In NADIR i pannelli +X sono perpendicolari al nadir e girati verso il
    # Sole: lo vedono con coseno sin(theta), dove theta è l'angolo tra Sole e
    # nadir. Verifica indipendente dal calcolo degli assetti. La tolleranza
    # copre la distanza dal Sole, che lungo l'orbita cambia di +-7000 km
    # (+-0,01 % sul flusso).
    trade = pointing_energy_trade(CONFIG)
    epoch = CONFIG.simulation.start_epoch
    orbit = CircularOrbit.from_config(CONFIG.orbit, epoch)
    sines: list[float] = []
    for elapsed_s in np.arange(0.0, orbit.period_s, CONFIG.simulation.output_step_s):
        position, _ = orbit.state_eci(float(elapsed_s))
        sun = sun_position_eci(julian_date(epoch, float(elapsed_s)))
        if not is_in_eclipse(position, sun):
            sines.append(np.sin(angle_between(sun - position, -position)))
    expected_w = trade.sun_pointing_power_w * float(np.mean(sines))
    assert trade.nadir_pointing_power_w == pytest.approx(expected_w, rel=2e-4)


def test_pointing_at_earth_costs_energy() -> None:
    trade = pointing_energy_trade(CONFIG)
    assert trade.nadir_pointing_wh < trade.sun_pointing_wh
    assert trade.loss_wh == pytest.approx(
        trade.sun_pointing_wh - trade.nadir_pointing_wh
    )
    assert 0.0 < trade.loss_fraction < 0.5


# --- Studio 2: dimensionamento della batteria ---

# Profilo inventato con risultato calcolabile a mano: orbite di 6000 s, le
# prime 2400 s in eclissi (0 W prodotti), poi 3600 s al Sole (10 W); consumo
# costante di 2 W. In ogni eclissi il bus chiede 2 W x 2400 s = 4800 J, e la
# batteria cede 4800 / 0,95 J = 1,4035 Wh.
NIGHT_DRAW_WH = 2.0 * 2400.0 / 0.95 / 3600.0


def eclipse_profile(modes: list[str] | None = None) -> SimulationResults:
    """Risultati inventati con il profilo di eclissi descritto sopra."""
    time_s = np.arange(0.0, 86_400.0, 10.0)
    generated_w = np.where(time_s % 6000.0 >= 2400.0, 10.0, 0.0)
    generated_j = np.concatenate(([0.0], np.cumsum(generated_w[:-1] * 10.0)))
    series = pd.DataFrame(
        {
            "time_s": time_s,
            "mode": modes or ["SUN_POINTING"] * time_s.size,
            "energy_generated_j": generated_j,
            "energy_consumed_j": 2.0 * time_s,
        }
    )
    metadata = {"config": CONFIG.model_dump(mode="json"), "transitions": []}
    return SimulationResults(timeseries=series, metadata=metadata)


def test_min_capacity_matches_hand_calculation() -> None:
    # La prima eclissi parte dal 70 %: serve 0,7 C - 1,4035 >= 0,3 C, cioè
    # C >= 3,51 Wh. Le eclissi successive partono da batteria piena e chiedono
    # meno (C >= 2,0 Wh): vince la prima.
    trade = battery_sizing_trade(eclipse_profile())
    assert trade.min_capacity_wh == pytest.approx(NIGHT_DRAW_WH / 0.4, abs=0.011)
    assert trade.safe_threshold == pytest.approx(0.3)


def test_drawdown_and_depth_of_discharge() -> None:
    trade = battery_sizing_trade(eclipse_profile())
    assert trade.max_drawdown_wh == pytest.approx(NIGHT_DRAW_WH)
    assert trade.max_depth_of_discharge == pytest.approx(NIGHT_DRAW_WH / 40.0)


def test_run_with_safe_is_rejected() -> None:
    modes = ["SUN_POINTING"] * 8640
    modes[100] = "SAFE"
    with pytest.raises(ValueError, match="SAFE"):
        battery_sizing_trade(eclipse_profile(modes))
