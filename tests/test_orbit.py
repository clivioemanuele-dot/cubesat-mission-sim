"""Test dell'orbita circolare, del riferimento LVLH, dell'angolo beta e dell'eclissi."""

import math
from pathlib import Path

import numpy as np
import pytest

from cubesat_sim.core.config import MissionConfig, load_config
from cubesat_sim.core.constants import (
    ASTRONOMICAL_UNIT_M,
    EARTH_J2,
    EARTH_MU_M3_S2,
    EARTH_RADIUS_M,
    TROPICAL_YEAR_S,
)
from cubesat_sim.core.vectors import angle_between, unit
from cubesat_sim.environment.earth import julian_date
from cubesat_sim.environment.orbit import (
    CircularOrbit,
    beta_angle,
    eci_to_lvlh_matrix,
)
from cubesat_sim.environment.sun import is_in_eclipse, sun_position_eci

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"


@pytest.fixture(scope="module")
def config() -> MissionConfig:
    return load_config(BASELINE)


@pytest.fixture(scope="module")
def orbit(config: MissionConfig) -> CircularOrbit:
    return CircularOrbit.from_config(config.orbit, config.simulation.start_epoch)


def analytic_eclipse_s(orbit: CircularOrbit, beta: float) -> float:
    """Durata dell'eclissi con ombra cilindrica, da formula chiusa [s]."""
    critical_beta = math.asin(EARTH_RADIUS_M / orbit.radius_m)
    if abs(beta) >= critical_beta:
        return 0.0
    half_chord = math.sqrt(orbit.radius_m**2 - EARTH_RADIUS_M**2)
    ratio = half_chord / (orbit.radius_m * math.cos(beta))
    return orbit.period_s / math.pi * math.acos(ratio)


# --- Utilità vettoriali ---


def test_unit_has_norm_one() -> None:
    assert np.linalg.norm(unit(np.array([3.0, -4.0, 12.0]))) == pytest.approx(1.0)


def test_unit_of_zero_vector_is_rejected() -> None:
    with pytest.raises(ValueError, match="vettore nullo"):
        unit(np.zeros(3))


@pytest.mark.parametrize(
    ("b", "expected_rad"),
    [
        ([1.0, 0.0, 0.0], 0.0),
        ([-1.0, 0.0, 0.0], math.pi),
        ([0.0, 2.0, 0.0], math.pi / 2),
        ([1.0, 1e-9, 0.0], 1e-9),  # angolo minuscolo: acos qui perderebbe precisione
    ],
)
def test_angle_between(b: list[float], expected_rad: float) -> None:
    angle = angle_between(np.array([1.0, 0.0, 0.0]), np.array(b))
    assert angle == pytest.approx(expected_rad, abs=1e-15)


# --- Orbita ---


def test_period_matches_assumptions(orbit: CircularOrbit) -> None:
    # docs/assumptions.md, sezione 14: 94,6 minuti.
    assert orbit.period_s / 60.0 == pytest.approx(94.62, abs=0.01)


def test_satellite_returns_after_one_period(orbit: CircularOrbit) -> None:
    start, _ = orbit.state_eci(0.0)
    after, _ = orbit.state_eci(orbit.period_s)
    np.testing.assert_allclose(after, start, atol=1e-3)


@pytest.mark.parametrize("elapsed_s", [0.0, 1234.5, 40_000.0])
def test_circular_motion(orbit: CircularOrbit, elapsed_s: float) -> None:
    position, velocity = orbit.state_eci(elapsed_s)
    radius = EARTH_RADIUS_M + 500e3
    assert np.linalg.norm(position) == pytest.approx(radius)
    assert np.linalg.norm(velocity) == pytest.approx(math.sqrt(EARTH_MU_M3_S2 / radius))
    assert position @ velocity == pytest.approx(0.0, abs=1e-3)


def test_velocity_is_derivative_of_position(orbit: CircularOrbit) -> None:
    step_s = 0.01
    before, _ = orbit.state_eci(100.0 - step_s)
    after, _ = orbit.state_eci(100.0 + step_s)
    _, velocity = orbit.state_eci(100.0)
    derivative = (after - before) / (2 * step_s)
    np.testing.assert_allclose(derivative, velocity, rtol=1e-6, atol=1e-6)


def test_inclination(orbit: CircularOrbit) -> None:
    position, velocity = orbit.state_eci(500.0)
    normal = np.cross(position, velocity)
    inclination = angle_between(normal, np.array([0.0, 0.0, 1.0]))
    assert math.degrees(inclination) == pytest.approx(97.4)


def test_inclination_is_sun_synchronous(config: MissionConfig) -> None:
    # J2 fa ruotare il piano orbitale; l'orbita è eliosincrona se ruota come il
    # Sole apparente (360 gradi in un anno tropico).
    radius = EARTH_RADIUS_M + config.orbit.altitude_m
    sun_rate = 2 * math.pi / TROPICAL_YEAR_S
    nodal_factor = 3 * EARTH_J2 * EARTH_RADIUS_M**2 * math.sqrt(EARTH_MU_M3_S2)
    cos_i = -2 * sun_rate * radius**3.5 / nodal_factor
    expected_deg = config.orbit.inclination_deg
    assert math.degrees(math.acos(cos_i)) == pytest.approx(expected_deg, abs=0.05)


def test_descending_node_local_time(
    config: MissionConfig, orbit: CircularOrbit
) -> None:
    # Al nodo discendente (argomento di latitudine 180 gradi) l'ora solare
    # locale deve essere 10:30.
    angle_to_node = (math.pi - orbit.initial_argument_of_latitude_rad) % (2 * math.pi)
    elapsed_s = angle_to_node / orbit.mean_motion_rad_s
    position, velocity = orbit.state_eci(elapsed_s)
    assert velocity[2] < 0  # il satellite scende verso sud
    sun = sun_position_eci(julian_date(config.simulation.start_epoch, elapsed_s))
    hour_angle = math.atan2(position[1], position[0]) - math.atan2(sun[1], sun[0])
    local_time_h = (12.0 + math.degrees(hour_angle) / 15.0) % 24.0
    assert local_time_h == pytest.approx(10.5, abs=1 / 60)


def test_beta_angle_at_epoch(config: MissionConfig, orbit: CircularOrbit) -> None:
    # docs/assumptions.md: angolo beta di circa 22 gradi all'equinozio.
    position, velocity = orbit.state_eci(0.0)
    sun = sun_position_eci(julian_date(config.simulation.start_epoch))
    beta_deg = math.degrees(beta_angle(position, velocity, sun))
    assert beta_deg == pytest.approx(22.3, abs=0.1)


@pytest.mark.parametrize("raan_deg", [0.0, 30.0, 60.0, 80.0])
def test_eclipse_duration_matches_analytic(raan_deg: float) -> None:
    # Sole fisso lungo +x e nodo ascendente a raan_deg dal Sole: beta vale
    # asin(sin i * sin raan), cioè circa 0, 30, 59 e 78 gradi (senza eclissi).
    # Il tempo in ombra, contato secondo per secondo, deve coincidere con la
    # formula analitica.
    sun = np.array([ASTRONOMICAL_UNIT_M, 0.0, 0.0])
    orbit_case = CircularOrbit(
        radius_m=EARTH_RADIUS_M + 500e3,
        inclination_rad=math.radians(97.4),
        raan_rad=math.radians(raan_deg),
        initial_argument_of_latitude_rad=0.0,
    )
    step_s = 1.0
    steps = round(orbit_case.period_s / step_s)
    in_shadow = [
        is_in_eclipse(orbit_case.state_eci(k * step_s)[0], sun) for k in range(steps)
    ]
    shadow_s = step_s * sum(in_shadow)
    position, velocity = orbit_case.state_eci(0.0)
    beta = beta_angle(position, velocity, sun)
    assert shadow_s == pytest.approx(analytic_eclipse_s(orbit_case, beta), abs=3.0)


# --- Riferimento LVLH ---


def test_lvlh_axes(orbit: CircularOrbit) -> None:
    position, velocity = orbit.state_eci(2000.0)
    matrix = eci_to_lvlh_matrix(position, velocity)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(matrix @ unit(position), [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(matrix @ unit(velocity), [1.0, 0.0, 0.0], atol=1e-12)
