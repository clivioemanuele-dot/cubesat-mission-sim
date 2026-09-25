"""Test della stazione di terra, della visibilità e del punto sotto il satellite."""

import math
from pathlib import Path

import numpy as np
import pytest

from cubesat_sim.core.config import MissionConfig, load_config
from cubesat_sim.core.constants import EARTH_FLATTENING, EARTH_RADIUS_M
from cubesat_sim.core.types import FloatArray
from cubesat_sim.core.vectors import angle_between, unit
from cubesat_sim.environment.earth import eci_to_ecef_matrix, julian_date
from cubesat_sim.environment.ground import (
    GroundStation,
    geodetic_to_ecef,
    is_over_region,
    subsatellite_point,
)
from cubesat_sim.environment.orbit import CircularOrbit

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"


@pytest.fixture(scope="module")
def config() -> MissionConfig:
    return load_config(BASELINE)


@pytest.fixture(scope="module")
def station(config: MissionConfig) -> GroundStation:
    return GroundStation.from_config(config.ground_station)


def sky_point(
    station: GroundStation, elevation_deg: float, distance_m: float = 1.0e6
) -> FloatArray:
    """Punto nel cielo della stazione, verso est, all'elevazione data."""
    east = unit(np.cross([0.0, 0.0, 1.0], station.up_ecef))
    elevation = math.radians(elevation_deg)
    direction = math.cos(elevation) * east + math.sin(elevation) * station.up_ecef
    return station.position_ecef + distance_m * direction


def pass_durations_s(visible: list[bool], step_s: float) -> list[float]:
    """Durata di ogni passaggio: sequenze consecutive di campioni visibili."""
    durations: list[float] = []
    run = 0
    for sample in [*visible, False]:
        if sample:
            run += 1
        elif run:
            durations.append(run * step_s)
            run = 0
    return durations


# --- Coordinate ---


@pytest.mark.parametrize(
    ("latitude_deg", "expected"),
    [
        (0.0, [EARTH_RADIUS_M, 0.0, 0.0]),  # equatore, meridiano di Greenwich
        (90.0, [0.0, 0.0, EARTH_RADIUS_M * (1 - EARTH_FLATTENING)]),  # polo nord
    ],
)
def test_geodetic_to_ecef(latitude_deg: float, expected: list[float]) -> None:
    position = geodetic_to_ecef(math.radians(latitude_deg), 0.0, 0.0)
    np.testing.assert_allclose(position, expected, atol=1e-6)


def test_vertical_is_tilted_from_radial_direction(station: GroundStation) -> None:
    # Sull'ellissoide la verticale non passa per il centro della Terra: a Roma
    # (41,9 N) si discosta dalla direzione radiale di circa 0,19 gradi.
    tilt = angle_between(station.up_ecef, station.position_ecef)
    assert math.degrees(tilt) == pytest.approx(0.19, abs=0.005)


@pytest.mark.parametrize(
    ("position", "expected_deg"),
    [
        ([7.0e6, 0.0, 0.0], (0.0, 0.0)),
        ([0.0, 5.0e6, 5.0e6], (45.0, 90.0)),
    ],
)
def test_subsatellite_point(
    position: list[float], expected_deg: tuple[float, float]
) -> None:
    latitude, longitude = subsatellite_point(np.array(position))
    result_deg = (math.degrees(latitude), math.degrees(longitude))
    assert result_deg == pytest.approx(expected_deg)


@pytest.mark.parametrize(
    ("latitude_deg", "longitude_deg", "expected"),
    [
        (41.9, 12.5, True),  # Roma
        (38.7, -9.1, True),  # Lisbona, vicino al bordo ovest
        (40.7, -74.0, False),  # New York
        (64.1, -21.9, False),  # Reykjavik
    ],
)
def test_imaging_region(
    config: MissionConfig, latitude_deg: float, longitude_deg: float, expected: bool
) -> None:
    latitude, longitude = math.radians(latitude_deg), math.radians(longitude_deg)
    assert is_over_region(latitude, longitude, config.imaging) is expected


# --- Visibilità ---


@pytest.mark.parametrize("elevation_deg", [90.0, 45.0, 0.0, -30.0])
def test_elevation(station: GroundStation, elevation_deg: float) -> None:
    measured = station.elevation_rad(sky_point(station, elevation_deg))
    assert math.degrees(measured) == pytest.approx(elevation_deg, abs=1e-9)


@pytest.mark.parametrize(("elevation_deg", "expected"), [(10.5, True), (9.5, False)])
def test_visibility_threshold(
    station: GroundStation, elevation_deg: float, expected: bool
) -> None:
    assert station.is_visible(sky_point(station, elevation_deg)) is expected


def test_station_rotates_with_earth_in_eci(station: GroundStation) -> None:
    # In ECI la stazione ruota attorno all'asse z: quota z e distanza dal centro
    # restano costanti, la posizione cambia.
    jd = 2_461_306.5
    now = station.position_eci(jd)
    six_hours_later = station.position_eci(jd + 0.25)
    assert now[2] == pytest.approx(six_hours_later[2])
    assert np.linalg.norm(now) == pytest.approx(np.linalg.norm(six_hours_later))
    assert angle_between(now, six_hours_later) > math.radians(10)


def test_passes_over_rome_in_24_hours(
    config: MissionConfig, station: GroundStation
) -> None:
    # Validazione di docs/assumptions.md: passaggi sopra 10 gradi, il più lungo
    # entro la durata massima teorica di circa 7,4 minuti. Campioni ogni 10 s.
    epoch = config.simulation.start_epoch
    orbit = CircularOrbit.from_config(config.orbit, epoch)
    step_s = 10.0
    visible: list[bool] = []
    for k in range(round(config.simulation.duration_s / step_s)):
        position_eci, _ = orbit.state_eci(k * step_s)
        to_ecef = eci_to_ecef_matrix(julian_date(epoch, k * step_s))
        visible.append(station.is_visible(to_ecef @ position_eci))
    durations = pass_durations_s(visible, step_s)
    assert 2 <= len(durations) <= 6
    assert 5 * 60 < max(durations) <= 7.4 * 60 + step_s
