"""Test di rotazioni elementari, tempo e rotazione terrestre."""

import datetime as dt
import math
from collections.abc import Callable

import numpy as np
import pytest

from cubesat_sim.core.constants import (
    EARTH_ROTATION_RATE_RAD_S,
    JULIAN_DATE_J2000,
    SECONDS_PER_DAY,
)
from cubesat_sim.core.rotations import rot_x, rot_z
from cubesat_sim.core.types import FloatArray
from cubesat_sim.environment.earth import (
    earth_rotation_angle,
    eci_to_ecef_matrix,
    julian_date,
)

J2000 = dt.datetime(2000, 1, 1, 12, 0, tzinfo=dt.UTC)


@pytest.mark.parametrize("rotation", [rot_x, rot_z])
@pytest.mark.parametrize("angle", [0.0, 0.3, -1.2, 2.5])
def test_rotation_is_proper(
    rotation: Callable[[float], FloatArray], angle: float
) -> None:
    # Una rotazione è ortogonale (conserva le lunghezze) e ha determinante +1.
    matrix = rotation(angle)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_rot_z_is_passive() -> None:
    # Sistema ruotato di +90 gradi attorno a z: l'asse x di partenza è visto
    # lungo -y.
    result = rot_z(math.pi / 2) @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(result, [0.0, -1.0, 0.0], atol=1e-12)


def test_rot_x_is_passive() -> None:
    # Sistema ruotato di +90 gradi attorno a x: l'asse y di partenza è visto
    # lungo -z.
    result = rot_x(math.pi / 2) @ np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-12)


def test_julian_date_of_j2000() -> None:
    assert julian_date(J2000) == JULIAN_DATE_J2000


def test_julian_date_adds_elapsed_time() -> None:
    assert julian_date(J2000, SECONDS_PER_DAY) == pytest.approx(JULIAN_DATE_J2000 + 1)


def test_julian_date_requires_timezone() -> None:
    with pytest.raises(ValueError, match="fuso orario"):
        julian_date(dt.datetime(2000, 1, 1, 12, 0))


def test_earth_rotation_angle_at_j2000() -> None:
    # Valore di riferimento IERS: 0,7790572732640 giri = 280,4606 gradi.
    angle_deg = math.degrees(earth_rotation_angle(JULIAN_DATE_J2000))
    assert angle_deg == pytest.approx(280.4606, abs=1e-4)


def test_earth_rotation_rate_matches_wgs84() -> None:
    # Due fonti indipendenti (IERS e WGS-84) devono dare la stessa velocità.
    step_s = 1000.0
    angle_1 = earth_rotation_angle(JULIAN_DATE_J2000)
    angle_2 = earth_rotation_angle(JULIAN_DATE_J2000 + step_s / SECONDS_PER_DAY)
    rate = ((angle_2 - angle_1) % (2 * math.pi)) / step_s
    assert rate == pytest.approx(EARTH_ROTATION_RATE_RAD_S, rel=1e-6)


def test_inertial_direction_drifts_west() -> None:
    # La Terra ruota verso est: una direzione fissa nello spazio, vista dalla
    # Terra, si sposta verso ovest (la sua longitudine diminuisce).
    direction = np.array([1.0, 0.0, 0.0])
    one_minute = 60.0 / SECONDS_PER_DAY
    before = eci_to_ecef_matrix(JULIAN_DATE_J2000) @ direction
    after = eci_to_ecef_matrix(JULIAN_DATE_J2000 + one_minute) @ direction
    assert math.atan2(after[1], after[0]) < math.atan2(before[1], before[0])


def test_earth_rotation_keeps_polar_axis() -> None:
    polar_axis = np.array([0.0, 0.0, 1.0])
    matrix = eci_to_ecef_matrix(2_460_000.0)
    np.testing.assert_allclose(matrix @ polar_axis, polar_axis)
