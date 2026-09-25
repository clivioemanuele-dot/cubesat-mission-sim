"""Test del controllore di puntamento PD, fino all'inseguimento in anello chiuso."""

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from cubesat_sim.attitude.actuators import ReactionWheels
from cubesat_sim.attitude.disturbances import external_torque_function
from cubesat_sim.attitude.dynamics import AttitudeState, propagate_attitude
from cubesat_sim.attitude.pointing import PointingController, pointing_error
from cubesat_sim.attitude.references import (
    PointingAxes,
    nadir_pointing_attitude,
    reference_rate,
)
from cubesat_sim.attitude.sensors import Sensors
from cubesat_sim.core.config import load_config
from cubesat_sim.core.quaternions import (
    quat_angle,
    quat_from_axis_angle,
    quat_identity,
    quat_multiply,
    quat_to_rotation_vector,
)
from cubesat_sim.core.types import FloatArray
from cubesat_sim.environment.earth import julian_date
from cubesat_sim.environment.magnetic import magnetic_field_eci
from cubesat_sim.environment.orbit import CircularOrbit
from cubesat_sim.environment.sun import sun_position_eci

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"
CONFIG = load_config(BASELINE)
INERTIA = np.array(CONFIG.satellite.inertia_kg_m2)
PERIOD_S = CONFIG.control.control_period_s
STEP_S = CONFIG.simulation.integration_step_s
MAX_RATE = CONFIG.control.max_slew_rate_rad_s
X_AXIS = np.array([1.0, 0.0, 0.0])
ZERO = np.zeros(3)


def new_controller() -> PointingController:
    """Controllore con i parametri dello scenario di riferimento."""
    return PointingController(
        INERTIA,
        CONFIG.control.pd_natural_frequency_rad_s,
        CONFIG.control.pd_damping_ratio,
        MAX_RATE,
    )


def no_torque(_state: FloatArray) -> FloatArray:
    """Nessuna coppia esterna."""
    return ZERO


def apply_for_one_period(
    state: FloatArray,
    body_torque: FloatArray,
    wheels: ReactionWheels,
    external_torque: Callable[[FloatArray], FloatArray],
) -> tuple[FloatArray, bool]:
    """Applica un comando per un periodo di controllo; indica se c'è saturazione."""
    momentum = AttitudeState.from_vector(state).wheel_momentum
    command = wheels.command(body_torque, momentum)
    for _ in range(round(PERIOD_S / STEP_S)):
        state = propagate_attitude(
            state, STEP_S, INERTIA, command.torque, external_torque
        )
    return state, command.torque_saturated or command.momentum_saturated


# --- Legge di controllo ---


def test_gains_from_bandwidth_and_damping() -> None:
    controller = new_controller()
    np.testing.assert_allclose(controller.kp, INERTIA * 0.1**2)
    np.testing.assert_allclose(controller.kd, 2 * 0.7 * 0.1 * INERTIA)


def test_no_error_no_torque() -> None:
    q = quat_from_axis_angle(np.array([1.0, -1.0, 2.0]), 0.7)
    torque = new_controller().torque(q, ZERO, q, ZERO)
    np.testing.assert_allclose(torque, ZERO, atol=1e-18)


def test_torque_opposes_small_error() -> None:
    # Corpo ruotato di 1 grado attorno a x rispetto al riferimento: la coppia
    # lo riporta indietro e vale esattamente -K_p * angolo.
    angle = math.radians(1.0)
    attitude = quat_from_axis_angle(X_AXIS, angle)
    torque = new_controller().torque(attitude, ZERO, quat_identity(), ZERO)
    expected = [-INERTIA[0] * 0.1**2 * angle, 0.0, 0.0]
    np.testing.assert_allclose(torque, expected, atol=1e-15)


def test_large_error_commands_limited_rate() -> None:
    # Con 90 gradi di errore il PD classico chiederebbe una velocità di
    # 0,071 * 1,57 = 0,11 rad/s (6,4 gradi/s). Con il limite la velocità
    # comandata vale omega_max: da fermo, la coppia è K_d * omega_max.
    attitude = quat_from_axis_angle(X_AXIS, math.pi / 2)
    torque = new_controller().torque(attitude, ZERO, quat_identity(), ZERO)
    expected = [-2 * 0.7 * 0.1 * INERTIA[0] * MAX_RATE, 0.0, 0.0]
    np.testing.assert_allclose(torque, expected, atol=1e-15)


def test_following_a_moving_reference_needs_no_torque() -> None:
    # Il corpo coincide con il riferimento e ruota come lui: nessuna coppia.
    # Senza la velocità del riferimento il controllore frenerebbe il satellite.
    q = quat_from_axis_angle(np.array([0.0, 1.0, 1.0]), 0.3)
    rate = np.array([0.0, -1.1e-3, 0.0])
    controller = new_controller()
    np.testing.assert_allclose(controller.torque(q, rate, q, rate), ZERO, atol=1e-18)
    assert np.linalg.norm(controller.torque(q, rate, q, ZERO)) > 0


def test_pointing_error_geometry() -> None:
    # Asse z del corpo (assetto identità) e bersaglio a 45 gradi nel piano y-z.
    z_axis = np.array([0.0, 0.0, 1.0])
    error = pointing_error(quat_identity(), z_axis, np.array([0.0, 1.0, 1.0]))
    assert math.degrees(error) == pytest.approx(45.0)


# --- Anello chiuso ---


def test_step_response_settles_as_designed() -> None:
    # Validazione del piano: 10 gradi di errore attorno a x, riferimento fermo,
    # sensori perfetti, nessun disturbo. Con omega_n = 0,1 rad/s e zeta = 0,7
    # la teoria prevede assestamento al 2 % in circa 60 s e una
    # sovraelongazione di circa il 5 %.
    initial_angle = math.radians(10.0)
    attitude = quat_from_axis_angle(X_AXIS, initial_angle)
    state = AttitudeState(attitude, ZERO, ZERO).to_vector()
    controller = new_controller()
    wheels = ReactionWheels(CONFIG.actuators, PERIOD_S)
    errors: list[float] = []  # errore lungo x, con segno, ogni secondo
    for _ in range(200):
        truth = AttitudeState.from_vector(state)
        errors.append(float(quat_to_rotation_vector(truth.attitude)[0]))
        torque = controller.torque(
            truth.attitude, truth.angular_velocity, quat_identity(), ZERO
        )
        state, _ = apply_for_one_period(state, torque, wheels, no_torque)

    band = 0.02 * initial_angle
    outside = [k for k, error in enumerate(errors) if abs(error) > band]
    settling_time_s = (outside[-1] + 1) * PERIOD_S
    assert 45.0 < settling_time_s < 75.0
    overshoot = -min(errors) / initial_angle
    assert 0.0 < overshoot < 0.10


def test_large_slew_respects_rate_limit_and_converges() -> None:
    # Validazione del limite: 170 gradi di errore attorno a un asse obliquo,
    # sensori perfetti, nessun disturbo. La velocità non supera omega_max
    # (1 grado/s, sotto la soglia DETUMBLE di 2 gradi/s), le ruote non
    # saturano e in 400 s l'errore scende sotto 0,1 gradi: circa 160 s di
    # rotazione a velocità costante, poi l'assestamento del PD.
    start = quat_from_axis_angle(np.array([1.0, 1.0, 1.0]), math.radians(170.0))
    state = AttitudeState(start, ZERO, ZERO).to_vector()
    controller = new_controller()
    wheels = ReactionWheels(CONFIG.actuators, PERIOD_S)
    rates: list[float] = []
    saturated = False
    for _ in range(400):
        truth = AttitudeState.from_vector(state)
        torque = controller.torque(
            truth.attitude, truth.angular_velocity, quat_identity(), ZERO
        )
        state, limited = apply_for_one_period(state, torque, wheels, no_torque)
        saturated = saturated or limited
        omega = AttitudeState.from_vector(state).angular_velocity
        rates.append(float(np.linalg.norm(omega)))

    assert max(rates) < 1.05 * MAX_RATE
    assert not saturated
    final_attitude = AttitudeState.from_vector(state).attitude
    assert math.degrees(quat_angle(final_attitude)) < 0.1


def test_nadir_tracking_with_noise_and_disturbances() -> None:
    # Scenario di riferimento completo: rumore dei sensori, gradiente
    # gravitazionale e dipolo residuo. Si parte con 5 gradi di errore; dopo
    # 5 minuti per assestarsi, nei 5 minuti successivi l'errore di puntamento
    # della fotocamera resta sotto 0,5 gradi e le ruote non saturano.
    epoch = CONFIG.simulation.start_epoch
    orbit = CircularOrbit.from_config(CONFIG.orbit, epoch)
    axes = PointingAxes.from_config(CONFIG)
    residual_dipole = np.array(CONFIG.satellite.residual_dipole_am2)
    sensors = Sensors(CONFIG.sensors, np.random.default_rng(CONFIG.simulation.seed))
    wheels = ReactionWheels(CONFIG.actuators, PERIOD_S)
    controller = new_controller()

    def reference_at(elapsed_s: float) -> FloatArray:
        position, velocity = orbit.state_eci(elapsed_s)
        sun = sun_position_eci(julian_date(epoch, elapsed_s))
        return nadir_pointing_attitude(axes, sun, position, velocity)

    start = quat_multiply(reference_at(0.0), quat_from_axis_angle(X_AXIS, 0.087))
    state = AttitudeState(start, ZERO, ZERO).to_vector()
    errors_deg: list[float] = []
    saturated = False
    for k in range(600):
        elapsed_s = k * PERIOD_S
        position, _ = orbit.state_eci(elapsed_s)
        field = magnetic_field_eci(position, julian_date(epoch, elapsed_s))
        q_ref = reference_at(elapsed_s)
        rate_ref = reference_rate(q_ref, reference_at(elapsed_s + PERIOD_S), PERIOD_S)
        measured = sensors.measure(state, field)
        torque = controller.torque(
            measured.attitude, measured.angular_velocity, q_ref, rate_ref
        )
        external = external_torque_function(position, field, residual_dipole, INERTIA)
        state, limited = apply_for_one_period(state, torque, wheels, external)
        saturated = saturated or limited
        if k >= 300:
            attitude = AttitudeState.from_vector(state).attitude
            nadir = -orbit.state_eci(elapsed_s + PERIOD_S)[0]
            error = pointing_error(attitude, axes.camera, nadir)
            errors_deg.append(math.degrees(error))

    assert max(errors_deg) < 0.5
    assert not saturated
