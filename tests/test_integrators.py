"""Test dell'integratore RK4 e della propagazione dell'assetto nel tempo."""

import math

import numpy as np
import pytest

from cubesat_sim.attitude.dynamics import (
    AttitudeState,
    angular_momentum_eci,
    propagate_attitude,
    rotational_energy,
)
from cubesat_sim.core.integrators import rk4_step
from cubesat_sim.core.quaternions import quat_identity, random_quaternion
from cubesat_sim.core.types import FloatArray

INERTIA = np.array([0.042, 0.042, 0.007])  # scenario di riferimento [kg m^2]
DISTINCT_INERTIA = np.array([0.01, 0.02, 0.03])  # tre inerzie diverse
ZERO = np.zeros(3)
STEP_S = 0.25


def oscillator(state: FloatArray) -> FloatArray:
    """Oscillatore armonico x'' = -x, scritto come sistema del primo ordine."""
    return np.array([state[1], -state[0]])


def oscillator_error(steps_per_period: int) -> float:
    """Errore dopo un periodo, partendo da x = 1, v = 0 (soluzione: cos t)."""
    step = 2 * math.pi / steps_per_period
    state = np.array([1.0, 0.0])
    for _ in range(steps_per_period):
        state = rk4_step(oscillator, state, step)
    return float(np.linalg.norm(state - np.array([1.0, 0.0])))


def no_torque(_state: FloatArray) -> FloatArray:
    """Nessuna coppia esterna."""
    return ZERO


def simulate(
    state: FloatArray,
    duration_s: float,
    inertia: FloatArray = INERTIA,
    wheel_torque: FloatArray = ZERO,
) -> list[FloatArray]:
    """Propaga l'assetto senza coppie esterne; restituisce tutti gli stati."""
    history = [state]
    for _ in range(round(duration_s / STEP_S)):
        state = propagate_attitude(state, STEP_S, inertia, wheel_torque, no_torque)
        history.append(state)
    return history


def tumbling_state(seed: int) -> FloatArray:
    """Satellite appena rilasciato: 10 gradi/s in direzione casuale, ruote cariche."""
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=3)
    omega *= math.radians(10.0) / np.linalg.norm(omega)
    wheels = np.array([1e-4, -2e-4, 0.5e-4])
    return AttitudeState(random_quaternion(rng), omega, wheels).to_vector()


# --- RK4 su un problema con soluzione esatta ---


def test_rk4_on_harmonic_oscillator() -> None:
    # 100 passi per periodo: dopo un periodo intero l'errore è sotto 1e-5.
    assert oscillator_error(100) < 1e-5


def test_rk4_is_fourth_order() -> None:
    # Dimezzando il passo, l'errore globale si riduce di circa 2^4 = 16 volte.
    ratio = oscillator_error(50) / oscillator_error(100)
    assert 14.0 < ratio < 18.0


# --- Leggi di conservazione nel tempo ---


def test_torque_free_motion_conserves_momentum_and_energy() -> None:
    # 10 minuti di rotazione libera a 10 gradi/s: momento angolare in ECI ed
    # energia di rotazione si conservano; la norma del quaternione resta 1.
    history = simulate(tumbling_state(seed=0), duration_s=600.0)
    first, last = history[0], history[-1]
    momentum_start = angular_momentum_eci(first, INERTIA)
    momentum_drift = angular_momentum_eci(last, INERTIA) - momentum_start
    assert np.linalg.norm(momentum_drift) < 1e-5 * np.linalg.norm(momentum_start)
    energy_start = rotational_energy(first, INERTIA)
    assert rotational_energy(last, INERTIA) == pytest.approx(energy_start, rel=1e-5)
    attitude = AttitudeState.from_vector(last).attitude
    assert np.linalg.norm(attitude) == pytest.approx(1.0, abs=1e-12)


def test_wheels_exchange_momentum_with_body() -> None:
    # Coppia costante sulle ruote per 10 minuti: il loro momento cresce di
    # coppia x tempo, mentre il momento totale in ECI resta costante.
    wheel_torque = np.array([1e-6, 0.0, -1e-6])
    start = tumbling_state(seed=1)
    history = simulate(start, duration_s=600.0, wheel_torque=wheel_torque)
    first, last = (
        AttitudeState.from_vector(start),
        AttitudeState.from_vector(history[-1]),
    )
    expected_wheels = first.wheel_momentum + wheel_torque * 600.0
    np.testing.assert_allclose(last.wheel_momentum, expected_wheels, rtol=1e-9)
    momentum_start = angular_momentum_eci(start, INERTIA)
    momentum_drift = angular_momentum_eci(history[-1], INERTIA) - momentum_start
    assert np.linalg.norm(momentum_drift) < 1e-5 * np.linalg.norm(momentum_start)


# --- Effetto Dzhanibekov (teorema della racchetta da tennis) ---


def test_spin_about_intermediate_axis_is_unstable() -> None:
    # Rotazione attorno all'asse di inerzia intermedio (y): una perturbazione
    # minima cresce finché il corpo si capovolge e omega_y cambia segno.
    omega = np.array([1e-3, 0.5, 1e-3])
    state = AttitudeState(quat_identity(), omega, ZERO).to_vector()
    history = simulate(state, duration_s=100.0, inertia=DISTINCT_INERTIA)
    spin = [AttitudeState.from_vector(s).angular_velocity[1] for s in history]
    assert min(spin) < 0.0


def test_spin_about_major_axis_is_stable() -> None:
    # Rotazione attorno all'asse di inerzia massimo (z): la perturbazione resta
    # piccola e la rotazione non cambia.
    omega = np.array([1e-3, 1e-3, 0.5])
    state = AttitudeState(quat_identity(), omega, ZERO).to_vector()
    history = simulate(state, duration_s=100.0, inertia=DISTINCT_INERTIA)
    spin = [AttitudeState.from_vector(s).angular_velocity[2] for s in history]
    assert min(spin) > 0.99 * 0.5
