"""Test degli attuatori: limiti di coppia, di momento e di dipolo."""

import numpy as np
import pytest

from cubesat_sim.attitude.actuators import Magnetorquers, ReactionWheels
from cubesat_sim.core.config import ActuatorsConfig

CONFIG = ActuatorsConfig(
    wheel_max_torque_mnm=1.0,
    wheel_max_momentum_mnms=10.0,
    wheel_max_speed_rpm=6000.0,
    magnetorquer_max_dipole_am2=0.2,
)
MAX_TORQUE = 1e-3  # N m
MAX_MOMENTUM = 1e-2  # N m s
PERIOD_S = 1.0
ZERO = np.zeros(3)


@pytest.fixture
def wheels() -> ReactionWheels:
    return ReactionWheels(CONFIG, control_period_s=PERIOD_S)


# --- Ruote di reazione ---


def test_wheels_within_limits(wheels: ReactionWheels) -> None:
    # Entro i limiti: sulle ruote agisce la coppia opposta a quella chiesta
    # sul corpo.
    body_torque = np.array([2e-4, -5e-4, 1e-4])
    command = wheels.command(body_torque, ZERO)
    np.testing.assert_allclose(command.torque, -body_torque)
    assert not command.torque_saturated
    assert not command.momentum_saturated


def test_wheel_torque_saturation_keeps_direction(wheels: ReactionWheels) -> None:
    # Richiesta oltre 1 mN m: il vettore si riduce in proporzione e conserva
    # la direzione.
    command = wheels.command(np.array([2e-3, 1e-3, 0.0]), ZERO)
    np.testing.assert_allclose(command.torque, [-1e-3, -0.5e-3, 0.0])
    assert command.torque_saturated


@pytest.mark.parametrize(
    ("body_torque_x", "expected_wheel_torque_x"),
    [
        (-5e-4, 0.0),  # accelererebbe ancora la ruota: bloccata
        (5e-4, -5e-4),  # la rallenta: passa
    ],
)
def test_saturated_wheel_cannot_accelerate_further(
    wheels: ReactionWheels, body_torque_x: float, expected_wheel_torque_x: float
) -> None:
    momentum = np.array([MAX_MOMENTUM, 0.0, 0.0])
    command = wheels.command(np.array([body_torque_x, 0.0, 0.0]), momentum)
    assert command.torque[0] == pytest.approx(expected_wheel_torque_x)


def test_wheel_torque_leaves_momentum_within_limit(wheels: ReactionWheels) -> None:
    # Margine residuo di 0,2 mN m s: in un periodo di controllo la ruota può
    # ricevere al massimo 0,2 mN m, così arriva esattamente al limite.
    momentum = np.array([MAX_MOMENTUM - 2e-4, 0.0, 0.0])
    command = wheels.command(np.array([-1e-3, 0.0, 0.0]), momentum)
    assert command.torque[0] == pytest.approx(2e-4)
    assert command.momentum_saturated


def test_limits_hold_in_random_cases(wheels: ReactionWheels) -> None:
    # 500 casi casuali: dopo un periodo di controllo nessuna ruota supera il
    # momento massimo e nessuna coppia supera il limite.
    rng = np.random.default_rng(3)
    for _ in range(500):
        momentum = rng.uniform(-MAX_MOMENTUM, MAX_MOMENTUM, size=3)
        body_torque = rng.normal(scale=2e-3, size=3)
        command = wheels.command(body_torque, momentum)
        after = momentum + command.torque * PERIOD_S
        assert np.all(np.abs(after) <= MAX_MOMENTUM + 1e-15)
        assert np.all(np.abs(command.torque) <= MAX_TORQUE + 1e-15)


# --- Magnetorquer ---


def test_magnetorquers_within_limit() -> None:
    dipole = np.array([0.1, -0.05, 0.2])
    command = Magnetorquers(CONFIG).command(dipole)
    np.testing.assert_allclose(command.dipole, dipole)
    assert not command.saturated


def test_magnetorquer_saturation_keeps_direction() -> None:
    command = Magnetorquers(CONFIG).command(np.array([0.4, -0.2, 0.1]))
    np.testing.assert_allclose(command.dipole, [0.2, -0.1, 0.05])
    assert command.saturated
