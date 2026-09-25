"""Test dei consumi elettrici."""

from pathlib import Path

import numpy as np
import pytest

from cubesat_sim.core.config import load_config
from cubesat_sim.resources.loads import PowerLoads

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"
CONFIG = load_config(BASELINE)
LOADS = PowerLoads(CONFIG.loads, CONFIG.actuators)
ZERO = np.zeros(3)


def test_base_consumption_matches_assumptions() -> None:
    # docs/assumptions.md: computer di bordo + ADCS = 2,0 W, senza attuatori
    # né carico utile.
    power = LOADS.consumption(ZERO, ZERO, transmitting=False, imaging=False)
    assert power.total_w == pytest.approx(2.0)
    assert power.actuators_w == 0.0


@pytest.mark.parametrize(
    ("wheel_torque", "expected_w"),
    [
        ([1e-3, 1e-3, 1e-3], 1.5),  # tre ruote alla coppia massima
        ([0.5e-3, 0.0, 0.0], 0.25),  # una ruota a metà coppia
        ([0.0, -0.5e-3, 0.0], 0.25),  # il verso della coppia non conta
    ],
)
def test_wheel_power_is_proportional_to_torque(
    wheel_torque: list[float], expected_w: float
) -> None:
    power = LOADS.consumption(
        np.array(wheel_torque), ZERO, transmitting=False, imaging=False
    )
    assert power.actuators_w == pytest.approx(expected_w)


def test_magnetorquer_power_is_proportional_to_dipole() -> None:
    # Un asse al massimo e uno a metà: 1,5 volte il consumo di un magnetorquer.
    dipole = np.array([0.2, -0.1, 0.0])
    power = LOADS.consumption(ZERO, dipole, transmitting=False, imaging=False)
    assert power.actuators_w == pytest.approx(1.5 * 0.25)


@pytest.mark.parametrize(
    ("transmitting", "imaging", "payload_w"),
    [(True, False, 4.0), (False, True, 3.0), (True, True, 7.0)],
)
def test_payload_loads(transmitting: bool, imaging: bool, payload_w: float) -> None:
    power = LOADS.consumption(ZERO, ZERO, transmitting=transmitting, imaging=imaging)
    assert power.transmitter_w + power.camera_w == pytest.approx(payload_w)
    assert power.total_w == pytest.approx(2.0 + payload_w)


def test_worst_case_is_below_peak_generation() -> None:
    # Tutto acceso al massimo: 2 + 1,5 + 0,75 + 4 + 3 = 11,25 W, circa metà
    # dei 20,7 W di picco dei pannelli.
    power = LOADS.consumption(
        np.full(3, 1e-3), np.full(3, 0.2), transmitting=True, imaging=True
    )
    assert power.total_w == pytest.approx(11.25)
