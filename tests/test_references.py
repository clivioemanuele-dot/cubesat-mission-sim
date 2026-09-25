"""Test degli assetti di riferimento e del vettore di rotazione."""

import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from cubesat_sim.attitude.references import (
    PointingAxes,
    nadir_pointing_attitude,
    reference_rate,
    station_pointing_attitude,
    sun_pointing_attitude,
    two_axis_attitude,
)
from cubesat_sim.core.config import load_config
from cubesat_sim.core.quaternions import (
    quat_from_axis_angle,
    quat_from_rotation_vector,
    quat_multiply,
    quat_to_matrix,
    quat_to_rotation_vector,
    random_quaternion,
)
from cubesat_sim.core.types import FloatArray
from cubesat_sim.core.vectors import unit
from cubesat_sim.environment.earth import eci_to_ecef_matrix, julian_date
from cubesat_sim.environment.ground import GroundStation
from cubesat_sim.environment.orbit import CircularOrbit
from cubesat_sim.environment.sun import sun_position_eci

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"
X_AXIS = np.array([1.0, 0.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])


class Scene(NamedTuple):
    """Geometria dello scenario di riferimento a un istante."""

    axes: PointingAxes
    sun: FloatArray
    position: FloatArray
    velocity: FloatArray
    station: FloatArray


def baseline_scene(elapsed_s: float) -> Scene:
    """Assi, Sole, satellite e stazione dello scenario di riferimento."""
    config = load_config(BASELINE)
    epoch = config.simulation.start_epoch
    jd = julian_date(epoch, elapsed_s)
    orbit = CircularOrbit.from_config(config.orbit, epoch)
    position, velocity = orbit.state_eci(elapsed_s)
    station = GroundStation.from_config(config.ground_station).position_eci(jd)
    axes = PointingAxes.from_config(config)
    return Scene(axes, sun_position_eci(jd), position, velocity, station)


def rate_deg_s(q_now: FloatArray, q_next: FloatArray) -> float:
    """Velocità del riferimento tra due istanti a 1 s di distanza [gradi/s]."""
    return math.degrees(float(np.linalg.norm(reference_rate(q_now, q_next, 1.0))))


# --- Vettore di rotazione ---


def test_rotation_vector_round_trip() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        vector = unit(rng.normal(size=3)) * rng.uniform(0.0, 0.999 * math.pi)
        recovered = quat_to_rotation_vector(quat_from_rotation_vector(vector))
        np.testing.assert_allclose(recovered, vector, atol=1e-12)
    zero = quat_to_rotation_vector(quat_from_rotation_vector(np.zeros(3)))
    np.testing.assert_array_equal(zero, np.zeros(3))


def test_rotation_vector_takes_shortest_rotation() -> None:
    # q e -q sono la stessa rotazione: il vettore di rotazione è lo stesso.
    q = quat_from_axis_angle(np.array([1.0, 2.0, -1.0]), 0.5)
    np.testing.assert_allclose(
        quat_to_rotation_vector(-q), quat_to_rotation_vector(q), atol=1e-15
    )


# --- Allineamento di due assi ---


def test_primary_axis_is_aligned_exactly() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        target, secondary = rng.normal(size=3), rng.normal(size=3)
        fallback = np.cross(target, secondary)
        q = two_axis_attitude(Z_AXIS, target, X_AXIS, secondary, fallback)
        np.testing.assert_allclose(quat_to_matrix(q) @ Z_AXIS, unit(target), atol=1e-12)


def test_secondary_axis_is_as_close_as_possible() -> None:
    # L'asse secondario finisce nel piano delle due direzioni, dalla parte
    # della direzione secondaria: è il più vicino possibile, dato il vincolo
    # sull'asse primario.
    rng = np.random.default_rng(2)
    for _ in range(20):
        target, secondary = rng.normal(size=3), rng.normal(size=3)
        plane_normal = np.cross(target, secondary)
        q = two_axis_attitude(Z_AXIS, target, X_AXIS, secondary, plane_normal)
        x_in_eci = quat_to_matrix(q) @ X_AXIS
        assert x_in_eci @ plane_normal == pytest.approx(0.0, abs=1e-12)
        assert x_in_eci @ secondary > 0


def test_parallel_targets_use_fallback() -> None:
    # Direzioni parallele: la secondaria non fissa nulla, si usa la riserva.
    target = np.array([0.0, 0.0, 1.0])
    fallback = np.array([0.0, 1.0, 0.0])
    q = two_axis_attitude(Z_AXIS, target, X_AXIS, 2 * target, fallback)
    matrix = quat_to_matrix(q)
    np.testing.assert_allclose(matrix @ Z_AXIS, target, atol=1e-12)
    np.testing.assert_allclose(matrix @ X_AXIS, fallback, atol=1e-12)


# --- Riferimenti dei modi, nello scenario di riferimento ---


def test_sun_pointing_reference() -> None:
    scene = baseline_scene(1000.0)
    q = sun_pointing_attitude(scene.axes, scene.sun, scene.position, scene.velocity)
    matrix = quat_to_matrix(q)
    np.testing.assert_allclose(matrix @ scene.axes.sun, unit(scene.sun), atol=1e-12)
    camera = matrix @ scene.axes.camera
    assert camera @ -scene.position > 0  # fotocamera dalla parte della Terra


def test_nadir_reference() -> None:
    scene = baseline_scene(1000.0)
    q = nadir_pointing_attitude(scene.axes, scene.sun, scene.position, scene.velocity)
    matrix = quat_to_matrix(q)
    nadir = unit(-scene.position)
    np.testing.assert_allclose(matrix @ scene.axes.camera, nadir, atol=1e-12)
    panels = matrix @ scene.axes.sun
    assert panels @ scene.sun >= 0  # pannelli dalla parte del Sole


def test_station_reference() -> None:
    scene = baseline_scene(1000.0)
    q = station_pointing_attitude(
        scene.axes, scene.sun, scene.position, scene.velocity, scene.station
    )
    matrix = quat_to_matrix(q)
    line_of_sight = unit(scene.station - scene.position)
    np.testing.assert_allclose(matrix @ scene.axes.antenna, line_of_sight, atol=1e-12)
    # Pannelli nel piano (antenna, normale all'orbita), dalla parte del Sole.
    normal = unit(np.cross(scene.position, scene.velocity))
    if normal @ scene.sun < 0:
        normal = -normal
    panels = matrix @ scene.axes.sun
    plane_normal = np.cross(line_of_sight, normal)
    assert panels @ plane_normal == pytest.approx(0.0, abs=1e-12)
    assert panels @ normal > 0


def test_station_reference_rate_stays_low_in_evening_pass() -> None:
    # Passaggio serale su Roma nello scenario di riferimento (circa 20:38 UTC,
    # in eclissi): la direzione della stazione arriva a circa 10 gradi da
    # quella del Sole. Con il Sole come seconda direzione il riferimento
    # ruoterebbe attorno all'antenna a oltre 2 gradi/s, sopra la soglia del
    # DETUMBLE; con la normale all'orbita resta sotto 1 grado/s.
    config = load_config(BASELINE)
    epoch = config.simulation.start_epoch
    orbit = CircularOrbit.from_config(config.orbit, epoch)
    station = GroundStation.from_config(config.ground_station)
    axes = PointingAxes.from_config(config)

    def references(elapsed_s: float) -> tuple[FloatArray, FloatArray]:
        """Riferimento adottato e riferimento con il Sole come seconda direzione."""
        jd = julian_date(epoch, elapsed_s)
        position, velocity = orbit.state_eci(elapsed_s)
        sun, target = sun_position_eci(jd), station.position_eci(jd)
        adopted = station_pointing_attitude(axes, sun, position, velocity, target)
        sun_based = two_axis_attitude(
            axes.antenna, target - position, axes.sun, sun, velocity
        )
        return adopted, sun_based

    adopted_rates: list[float] = []
    sun_based_rates: list[float] = []
    for elapsed_s in range(74_000, 75_000):
        position, _ = orbit.state_eci(elapsed_s)
        jd = julian_date(epoch, elapsed_s)
        if not station.is_visible(eci_to_ecef_matrix(jd) @ position):
            continue
        now, following = references(elapsed_s), references(elapsed_s + 1)
        adopted_rates.append(rate_deg_s(now[0], following[0]))
        sun_based_rates.append(rate_deg_s(now[1], following[1]))

    assert len(adopted_rates) > 300  # il passaggio dura circa 7 minuti
    assert max(adopted_rates) < 1.0
    assert max(sun_based_rates) > 2.0


def test_reference_rate_of_uniform_rotation() -> None:
    # Riferimento che ruota a velocità costante: la stima restituisce proprio
    # quella velocità, qualunque sia il segno del quaternione.
    q_now = random_quaternion(np.random.default_rng(3))
    omega = np.array([0.01, -0.02, 0.005])
    q_next = quat_multiply(q_now, quat_from_rotation_vector(omega * 1.0))
    np.testing.assert_allclose(reference_rate(q_now, q_next, 1.0), omega, atol=1e-14)
    np.testing.assert_allclose(reference_rate(q_now, -q_next, 1.0), omega, atol=1e-14)
