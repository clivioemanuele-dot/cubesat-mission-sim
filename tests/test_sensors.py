"""Test dei sensori: misure con rumore, riproducibilità e statistica."""

import math

import numpy as np
import pytest

from cubesat_sim.attitude.dynamics import AttitudeState
from cubesat_sim.attitude.sensors import Sensors
from cubesat_sim.core.config import SensorsConfig
from cubesat_sim.core.quaternions import (
    quat_angle,
    quat_conjugate,
    quat_from_axis_angle,
    quat_multiply,
    quat_to_matrix,
)
from cubesat_sim.core.types import FloatArray

FIELD_ECI = np.array([20e-6, -10e-6, 30e-6])
NOISY = SensorsConfig(
    attitude_noise_deg=0.02, rate_noise_deg_s=0.01, magnetometer_noise_nt=50.0
)
PERFECT = SensorsConfig(
    attitude_noise_deg=0.0, rate_noise_deg_s=0.0, magnetometer_noise_nt=0.0
)


def true_state() -> FloatArray:
    """Stato vero di prova: assetto generico, rotazione lenta, ruote ferme."""
    return AttitudeState(
        attitude=quat_from_axis_angle(np.array([1.0, 2.0, 3.0]), 0.8),
        angular_velocity=np.array([0.01, -0.02, 0.03]),
        wheel_momentum=np.zeros(3),
    ).to_vector()


def test_perfect_sensors_return_the_truth() -> None:
    state = true_state()
    truth = AttitudeState.from_vector(state)
    measured = Sensors(PERFECT, np.random.default_rng(0)).measure(state, FIELD_ECI)
    np.testing.assert_allclose(measured.attitude, truth.attitude)
    np.testing.assert_allclose(measured.angular_velocity, truth.angular_velocity)
    expected_field = quat_to_matrix(truth.attitude).T @ FIELD_ECI
    np.testing.assert_allclose(measured.magnetic_field, expected_field)


def test_magnetometer_measures_in_body_frame() -> None:
    # Corpo ruotato di 90 gradi attorno a z: un campo lungo x di ECI è visto
    # lungo -y del corpo.
    attitude = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.pi / 2)
    state = AttitudeState(attitude, np.zeros(3), np.zeros(3)).to_vector()
    sensors = Sensors(PERFECT, np.random.default_rng(0))
    measured = sensors.measure(state, np.array([1e-5, 0.0, 0.0]))
    np.testing.assert_allclose(measured.magnetic_field, [0.0, -1e-5, 0.0], atol=1e-20)


def test_same_seed_gives_same_measurements() -> None:
    state = true_state()
    first = Sensors(NOISY, np.random.default_rng(7)).measure(state, FIELD_ECI)
    second = Sensors(NOISY, np.random.default_rng(7)).measure(state, FIELD_ECI)
    other = Sensors(NOISY, np.random.default_rng(8)).measure(state, FIELD_ECI)
    np.testing.assert_array_equal(first.attitude, second.attitude)
    np.testing.assert_array_equal(first.angular_velocity, second.angular_velocity)
    assert not np.array_equal(first.angular_velocity, other.angular_velocity)


def test_measured_attitude_is_unit_quaternion() -> None:
    sensors = Sensors(NOISY, np.random.default_rng(1))
    for _ in range(100):
        measured = sensors.measure(true_state(), FIELD_ECI)
        assert np.linalg.norm(measured.attitude) == pytest.approx(1.0)


def test_noise_statistics_match_configuration() -> None:
    # Su 2000 misure: errori a media nulla e deviazione standard pari a quella
    # della configurazione, entro il 5 %. Per l'assetto l'angolo d'errore
    # totale ha valore quadratico medio sigma * sqrt(3), perché gli assi sono 3.
    sensors = Sensors(NOISY, np.random.default_rng(2))
    state = true_state()
    truth = AttitudeState.from_vector(state)
    true_field = quat_to_matrix(truth.attitude).T @ FIELD_ECI
    samples = [sensors.measure(state, FIELD_ECI) for _ in range(2000)]

    rate_errors = np.array(
        [m.angular_velocity - truth.angular_velocity for m in samples]
    )
    assert np.std(rate_errors) == pytest.approx(NOISY.rate_noise_rad_s, rel=0.05)
    assert abs(np.mean(rate_errors)) < 0.1 * NOISY.rate_noise_rad_s

    field_errors = np.array([m.magnetic_field - true_field for m in samples])
    assert np.std(field_errors) == pytest.approx(NOISY.magnetometer_noise_t, rel=0.05)

    inverse_truth = quat_conjugate(truth.attitude)
    angles = np.array(
        [quat_angle(quat_multiply(inverse_truth, m.attitude)) for m in samples]
    )
    rms_angle = math.sqrt(float(np.mean(angles**2)))
    expected_rms = NOISY.attitude_noise_rad * math.sqrt(3)
    assert rms_angle == pytest.approx(expected_rms, rel=0.05)
