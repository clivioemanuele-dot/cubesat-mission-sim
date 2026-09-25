"""Test del controllore B-dot, fino al detumble in anello chiuso."""

import math
from pathlib import Path

import numpy as np
import pytest

from cubesat_sim.attitude.actuators import Magnetorquers
from cubesat_sim.attitude.bdot import BdotController, bdot_gain
from cubesat_sim.attitude.disturbances import external_torque_function, magnetic_torque
from cubesat_sim.attitude.dynamics import AttitudeState, propagate_attitude
from cubesat_sim.attitude.sensors import Sensors
from cubesat_sim.core.config import load_config
from cubesat_sim.core.constants import EARTH_MU_M3_S2, EARTH_RADIUS_M
from cubesat_sim.core.quaternions import (
    quat_from_axis_angle,
    quat_to_matrix,
    random_quaternion,
)
from cubesat_sim.core.vectors import unit
from cubesat_sim.environment.earth import julian_date
from cubesat_sim.environment.magnetic import magnetic_field_eci
from cubesat_sim.environment.orbit import CircularOrbit

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"
FIELD = np.array([2e-5, -1e-5, 3e-5])


def detumble_rates(duration_s: float, *, controlled: bool) -> list[float]:
    """Simula il rilascio con lo scenario di riferimento, con o senza B-dot.

    Ambiente, sensori con rumore, magnetorquer con saturazione, dipolo
    residuo e gradiente gravitazionale sono tutti attivi. Restituisce la
    velocità angolare vera [rad/s] alla fine di ogni periodo di controllo.
    """
    config = load_config(BASELINE)
    epoch = config.simulation.start_epoch
    orbit = CircularOrbit.from_config(config.orbit, epoch)
    inertia = np.array(config.satellite.inertia_kg_m2)
    residual_dipole = np.array(config.satellite.residual_dipole_am2)
    period_s = config.control.control_period_s
    step_s = config.simulation.integration_step_s

    rng = np.random.default_rng(config.simulation.seed)
    sensors = Sensors(config.sensors, rng)
    torquers = Magnetorquers(config.actuators)
    gain = bdot_gain(
        orbit.mean_motion_rad_s, orbit.inclination_rad, float(inertia.min())
    )
    controller = BdotController(gain, period_s)

    omega = unit(rng.normal(size=3)) * config.initial_state.max_rate_rad_s
    state = AttitudeState(random_quaternion(rng), omega, np.zeros(3)).to_vector()

    rates: list[float] = []
    for k in range(round(duration_s / period_s)):
        elapsed_s = k * period_s
        position, _ = orbit.state_eci(elapsed_s)
        field = magnetic_field_eci(position, julian_date(epoch, elapsed_s))
        measured = sensors.measure(state, field)
        dipole = torquers.command(controller.dipole(measured.magnetic_field)).dipole
        if not controlled:
            dipole = np.zeros(3)
        torque = external_torque_function(
            position, field, dipole + residual_dipole, inertia
        )
        for _ in range(round(period_s / step_s)):
            state = propagate_attitude(state, step_s, inertia, np.zeros(3), torque)
        angular_velocity = AttitudeState.from_vector(state).angular_velocity
        rates.append(float(np.linalg.norm(angular_velocity)))
    return rates


def test_gain_for_baseline() -> None:
    # k = 2 n (1 + sin i) I_min = 2 * 1,107e-3 * 1,992 * 0,007 = 3,086e-5 N m s
    mean_motion = math.sqrt(EARTH_MU_M3_S2 / (EARTH_RADIUS_M + 500e3) ** 3)
    gain = bdot_gain(mean_motion, math.radians(97.4), 0.007)
    assert gain == pytest.approx(3.086e-5, rel=1e-3)


def test_first_measurement_gives_no_dipole() -> None:
    controller = BdotController(gain_nms=3e-5, control_period_s=1.0)
    np.testing.assert_array_equal(controller.dipole(FIELD), np.zeros(3))


def test_bdot_torque_opposes_rotation() -> None:
    # Corpo che ruota a omega: il campo, fisso nello spazio, visto dal corpo
    # ruota con -omega. La coppia m x B deve opporsi alla rotazione (potenza
    # omega . tau negativa): il B-dot toglie energia.
    omega = np.array([0.05, -0.1, 0.08])
    angle = -float(np.linalg.norm(omega)) * 1.0  # rotazione in un periodo di 1 s
    field_after = quat_to_matrix(quat_from_axis_angle(omega, angle)) @ FIELD
    controller = BdotController(gain_nms=3e-5, control_period_s=1.0)
    controller.dipole(FIELD)
    torque = magnetic_torque(controller.dipole(field_after), field_after)
    assert omega @ torque < 0


def test_reset_forgets_previous_measurement() -> None:
    controller = BdotController(gain_nms=3e-5, control_period_s=1.0)
    controller.dipole(FIELD)
    controller.reset()
    new_field = np.array([0.0, 2e-5, 1e-5])
    np.testing.assert_array_equal(controller.dipole(new_field), np.zeros(3))


def test_bdot_detumbles_the_satellite() -> None:
    # Validazione del piano: dal rilascio a 10 gradi/s, in mezz'orbita il
    # B-dot porta la velocità angolare sotto la metà. Senza controllo, nello
    # stesso tempo, resta sopra il 90 %.
    duration_s = 2850.0
    initial_rate = math.radians(10.0)
    with_bdot = detumble_rates(duration_s, controlled=True)
    without_control = detumble_rates(duration_s, controlled=False)
    assert with_bdot[-1] < 0.5 * initial_rate
    assert without_control[-1] > 0.9 * initial_rate
