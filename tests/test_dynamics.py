"""Test della dinamica d'assetto: leggi di conservazione e casi noti."""

import numpy as np
import pytest

from cubesat_sim.attitude.dynamics import (
    AttitudeState,
    angular_momentum_eci,
    attitude_derivative,
    rotational_energy,
)
from cubesat_sim.core.quaternions import (
    quat_derivative,
    quat_identity,
    quat_to_matrix,
    random_quaternion,
)
from cubesat_sim.core.types import FloatArray

INERTIA = np.array([0.042, 0.042, 0.007])  # scenario di riferimento [kg m^2]
ZERO = np.zeros(3)


def random_state(seed: int) -> FloatArray:
    """Stato casuale riproducibile: assetto qualsiasi, rotazione e ruote attive."""
    rng = np.random.default_rng(seed)
    state = AttitudeState(
        attitude=random_quaternion(rng),
        angular_velocity=rng.normal(scale=0.1, size=3),
        wheel_momentum=rng.normal(scale=0.005, size=3),
    )
    return state.to_vector()


def rate_of(derivative: FloatArray) -> AttitudeState:
    """Rende leggibile una derivata: ha la stessa struttura dello stato."""
    return AttitudeState.from_vector(derivative)


def momentum_rate_eci(state: FloatArray, derivative: FloatArray) -> FloatArray:
    """Derivata del momento angolare totale in ECI.

    H_eci = R (I omega + h), quindi dH_eci/dt = R (omega x H_body + I domega + dh).
    """
    s, d = AttitudeState.from_vector(state), rate_of(derivative)
    momentum_body = INERTIA * s.angular_velocity + s.wheel_momentum
    rate_body = (
        np.cross(s.angular_velocity, momentum_body)
        + INERTIA * d.angular_velocity
        + d.wheel_momentum
    )
    return quat_to_matrix(s.attitude) @ rate_body


def test_state_vector_round_trip() -> None:
    vector = random_state(seed=0)
    rebuilt = AttitudeState.from_vector(vector).to_vector()
    np.testing.assert_array_equal(rebuilt, vector)


def test_quaternion_part_is_kinematics() -> None:
    state = random_state(seed=1)
    derivative = rate_of(attitude_derivative(state, INERTIA, ZERO, ZERO))
    s = AttitudeState.from_vector(state)
    expected = quat_derivative(s.attitude, s.angular_velocity)
    np.testing.assert_allclose(derivative.attitude, expected)


def test_energy_is_constant_without_torques() -> None:
    # Senza coppie esterne e senza coppia sulle ruote, la derivata dell'energia
    # di rotazione, omega . (I domega/dt), è nulla.
    for seed in range(10):
        state = random_state(seed)
        omega = AttitudeState.from_vector(state).angular_velocity
        omega_dot = rate_of(attitude_derivative(state, INERTIA, ZERO, ZERO))
        power = omega @ (INERTIA * omega_dot.angular_velocity)
        assert power == pytest.approx(0.0, abs=1e-15)


def test_internal_torques_do_not_change_total_momentum() -> None:
    # Le ruote scambiano momento con il corpo, ma il totale in ECI resta
    # costante: la coppia delle ruote è interna al satellite.
    rng = np.random.default_rng(10)
    for seed in range(10):
        state = random_state(seed)
        wheel_torque = rng.normal(scale=1e-3, size=3)
        derivative = attitude_derivative(state, INERTIA, wheel_torque, ZERO)
        np.testing.assert_allclose(
            momentum_rate_eci(state, derivative), ZERO, atol=1e-14
        )


def test_external_torque_changes_total_momentum() -> None:
    # dH_eci/dt = coppia esterna, espressa in ECI.
    state = random_state(seed=3)
    external = np.array([1e-6, -2e-6, 3e-6])
    derivative = attitude_derivative(state, INERTIA, ZERO, external)
    expected = quat_to_matrix(AttitudeState.from_vector(state).attitude) @ external
    np.testing.assert_allclose(
        momentum_rate_eci(state, derivative), expected, atol=1e-15
    )


def test_spin_about_principal_axis_is_steady() -> None:
    state = AttitudeState(quat_identity(), np.array([0.0, 0.0, 0.2]), ZERO).to_vector()
    derivative = rate_of(attitude_derivative(state, INERTIA, ZERO, ZERO))
    np.testing.assert_allclose(derivative.angular_velocity, ZERO, atol=1e-18)
    assert rotational_energy(state, INERTIA) == pytest.approx(0.5 * 0.007 * 0.2**2)
    momentum = angular_momentum_eci(state, INERTIA)
    np.testing.assert_allclose(momentum, [0.0, 0.0, 0.007 * 0.2])


def test_wheel_torque_reacts_on_body() -> None:
    # Il motore accelera la ruota in +z: il corpo, fermo, accelera in -z.
    state = AttitudeState(quat_identity(), ZERO, ZERO).to_vector()
    wheel_torque = np.array([0.0, 0.0, 1e-3])
    derivative = rate_of(attitude_derivative(state, INERTIA, wheel_torque, ZERO))
    np.testing.assert_allclose(derivative.angular_velocity, [0.0, 0.0, -1e-3 / 0.007])
    np.testing.assert_allclose(derivative.wheel_momentum, wheel_torque)


def test_gyroscopic_coupling() -> None:
    # Corpo in rotazione attorno a x, ruote con momento lungo z: il termine
    # -omega x h accelera il corpo attorno a y.
    state = AttitudeState(
        attitude=quat_identity(),
        angular_velocity=np.array([0.1, 0.0, 0.0]),
        wheel_momentum=np.array([0.0, 0.0, 0.01]),
    ).to_vector()
    derivative = rate_of(attitude_derivative(state, INERTIA, ZERO, ZERO))
    expected = [0.0, 0.001 / 0.042, 0.0]
    np.testing.assert_allclose(derivative.angular_velocity, expected, atol=1e-15)
