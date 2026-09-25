"""Test del campo magnetico terrestre (dipolo inclinato IGRF)."""

import math

import numpy as np
import pytest

from cubesat_sim.core.constants import (
    EARTH_RADIUS_M,
    IGRF_G10_T,
    IGRF_G11_T,
    IGRF_H11_T,
)
from cubesat_sim.core.vectors import unit
from cubesat_sim.environment.magnetic import (
    IGRF_REFERENCE_RADIUS_M,
    magnetic_field_ecef,
    magnetic_field_eci,
)

DIPOLE = np.array([IGRF_G11_T, IGRF_H11_T, IGRF_G10_T])
DIPOLE_STRENGTH_T = float(np.linalg.norm(DIPOLE))
ORBIT_RADIUS_M = EARTH_RADIUS_M + 500e3


def test_field_on_geomagnetic_equator() -> None:
    # Perpendicolarmente al dipolo, alla quota di riferimento, il campo vale
    # l'intensità del dipolo: circa 29,7 microtesla.
    direction = unit(np.cross(DIPOLE, [0.0, 0.0, 1.0]))
    field = magnetic_field_ecef(IGRF_REFERENCE_RADIUS_M * direction)
    assert np.linalg.norm(field) == pytest.approx(29.73e-6, rel=1e-3)


def test_geomagnetic_north_pole() -> None:
    # Il polo nord geomagnetico (IGRF-14, 2025) è a circa 80,8 N e 72,8 O: lì il
    # campo vale il doppio e punta verso il basso, dentro la Terra.
    pole = -unit(DIPOLE)
    assert math.degrees(math.asin(pole[2])) == pytest.approx(80.8, abs=0.1)
    assert math.degrees(math.atan2(pole[1], pole[0])) == pytest.approx(-72.8, abs=0.1)
    field = magnetic_field_ecef(IGRF_REFERENCE_RADIUS_M * pole)
    assert np.linalg.norm(field) == pytest.approx(2 * DIPOLE_STRENGTH_T)
    assert field @ pole < 0


def test_field_at_orbit_altitude_matches_assumptions() -> None:
    # docs/assumptions.md: a 500 km il campo vale tra 20 e 50 microtesla.
    rng = np.random.default_rng(seed=1)
    for _ in range(200):
        direction = unit(rng.normal(size=3))
        magnitude = np.linalg.norm(magnetic_field_ecef(ORBIT_RADIUS_M * direction))
        assert 20e-6 < magnitude < 50e-6


def test_field_decays_with_cube_of_distance() -> None:
    position = np.array([ORBIT_RADIUS_M, 0.0, 0.0])
    near = magnetic_field_ecef(position)
    far = magnetic_field_ecef(2 * position)
    np.testing.assert_allclose(far, near / 8)


def test_field_points_north_at_equator() -> None:
    field = magnetic_field_ecef(np.array([EARTH_RADIUS_M, 0.0, 0.0]))
    assert field[2] > 0


def test_field_at_rome_is_realistic() -> None:
    # A Roma l'intensità misurata è di circa 46 microtesla: il dipolo deve dare
    # lo stesso valore entro il 10 %.
    lat, lon = math.radians(41.9), math.radians(12.5)
    direction = np.array(
        [
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ]
    )
    magnitude = np.linalg.norm(magnetic_field_ecef(EARTH_RADIUS_M * direction))
    assert magnitude == pytest.approx(46e-6, rel=0.1)


def test_field_in_eci_rotates_with_earth() -> None:
    # In un punto fisso dello spazio il campo si ripete dopo un giorno siderale
    # (0,99727 giorni), ma non dopo mezza giornata.
    position = np.array([ORBIT_RADIUS_M, 0.0, 0.0])
    jd = 2_461_306.5  # 2026-09-23 00:00 UTC
    stellar_day = 1 / 1.00273781191135448
    start = magnetic_field_eci(position, jd)
    after_one_day = magnetic_field_eci(position, jd + stellar_day)
    after_half_day = magnetic_field_eci(position, jd + 0.5)
    np.testing.assert_allclose(after_one_day, start, rtol=1e-6, atol=1e-12)
    assert not np.allclose(after_half_day, start, rtol=1e-2, atol=0.0)
