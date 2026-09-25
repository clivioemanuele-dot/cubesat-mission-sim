"""Test delle coppie di disturbo: gradiente gravitazionale e dipolo magnetico."""

import math

import numpy as np
import pytest

from cubesat_sim.attitude.disturbances import gravity_gradient_torque, magnetic_torque
from cubesat_sim.core.constants import EARTH_MU_M3_S2, EARTH_RADIUS_M

INERTIA = np.array([0.042, 0.042, 0.007])  # scenario di riferimento [kg m^2]
RADIUS_M = EARTH_RADIUS_M + 500e3
ZERO = np.zeros(3)


# --- Gradiente gravitazionale ---


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_gravity_gradient_vanishes_on_principal_axes(axis: int) -> None:
    # Un asse principale lungo la verticale locale: posizione di equilibrio.
    position = np.zeros(3)
    position[axis] = RADIUS_M
    torque = gravity_gradient_torque(position, INERTIA)
    np.testing.assert_allclose(torque, ZERO, atol=1e-20)


def test_gravity_gradient_maximum_matches_formula() -> None:
    # Verticale a 45 gradi tra gli assi x e z: coppia massima,
    # 3 mu / (2 r^3) (I_x - I_z) = 6,43e-8 N m per il nostro 3U a 500 km.
    tilt = math.radians(45.0)
    position = RADIUS_M * np.array([math.sin(tilt), 0.0, math.cos(tilt)])
    torque = gravity_gradient_torque(position, INERTIA)
    expected = 1.5 * EARTH_MU_M3_S2 / RADIUS_M**3 * (0.042 - 0.007)
    np.testing.assert_allclose(torque, [0.0, expected, 0.0], atol=1e-20)
    assert expected == pytest.approx(6.43e-8, rel=2e-3)


@pytest.mark.parametrize(
    ("body_axis", "tilt_toward", "stable"),
    [
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], True),  # asse z, inerzia minima
        ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], False),  # asse x, inerzia massima
    ],
)
def test_gravity_gradient_stability(
    body_axis: list[float], tilt_toward: list[float], stable: bool
) -> None:
    # Asse del corpo inclinato di 5 gradi rispetto alla verticale. La coppia è
    # "di richiamo" se fa ruotare l'asse verso la verticale, cioè se ha una
    # componente positiva lungo (asse x verticale).
    axis, toward = np.array(body_axis), np.array(tilt_toward)
    tilt = math.radians(5.0)
    vertical = math.cos(tilt) * axis + math.sin(tilt) * toward
    torque = gravity_gradient_torque(RADIUS_M * vertical, INERTIA)
    restoring = bool(torque @ np.cross(axis, vertical) > 0)
    assert restoring is stable


def test_gravity_gradient_scales_with_inverse_cube() -> None:
    position = RADIUS_M * np.array([0.6, 0.0, 0.8])
    near = gravity_gradient_torque(position, INERTIA)
    far = gravity_gradient_torque(2 * position, INERTIA)
    np.testing.assert_allclose(far, near / 8)


def test_gravity_gradient_same_toward_and_away_from_earth() -> None:
    # La coppia dipende da r_hat due volte: nadir e zenit sono equivalenti.
    position = RADIUS_M * np.array([0.6, 0.0, 0.8])
    toward = gravity_gradient_torque(position, INERTIA)
    away = gravity_gradient_torque(-position, INERTIA)
    np.testing.assert_allclose(away, toward)


# --- Dipolo magnetico ---


def test_magnetic_torque_value() -> None:
    # Dipolo residuo del nostro 3U (5 mA m^2 lungo z) in un campo di
    # 30 microtesla lungo x: coppia di 0,15 micronewton-metro lungo y.
    dipole = np.array([0.0, 0.0, 0.005])
    field = np.array([30e-6, 0.0, 0.0])
    np.testing.assert_allclose(magnetic_torque(dipole, field), [0.0, 1.5e-7, 0.0])


def test_magnetic_torque_is_perpendicular_to_field() -> None:
    # Un dipolo non può produrre coppia lungo il campo: è il limite dei
    # magnetorquer, che in ogni istante controllano solo due assi.
    rng = np.random.default_rng(5)
    for _ in range(20):
        dipole = rng.normal(size=3)
        field = rng.normal(size=3) * 1e-5
        torque = magnetic_torque(dipole, field)
        assert torque @ field == pytest.approx(0.0, abs=1e-22)
