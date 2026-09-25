"""Test dei quaternioni di assetto e della loro convenzione."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from cubesat_sim.core.quaternions import (
    quat_angle,
    quat_conjugate,
    quat_derivative,
    quat_from_axis_angle,
    quat_from_matrix,
    quat_identity,
    quat_multiply,
    quat_to_matrix,
    random_quaternion,
)
from cubesat_sim.core.rotations import rot_x, rot_z
from cubesat_sim.core.types import FloatArray

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])


def random_quaternions(count: int, seed: int = 0) -> list[FloatArray]:
    """Quaternioni casuali riproducibili."""
    rng = np.random.default_rng(seed)
    return [random_quaternion(rng) for _ in range(count)]


def same_rotation(p: FloatArray, q: FloatArray) -> bool:
    """Vero se p e q rappresentano la stessa rotazione (q e -q coincidono)."""
    return bool(np.allclose(p, q, atol=1e-12) or np.allclose(p, -q, atol=1e-12))


# --- Convenzione ---


def test_convention_body_to_eci() -> None:
    # Corpo ruotato di 90 gradi attorno all'asse z di ECI: l'asse x del corpo
    # punta lungo l'asse y di ECI.
    q = quat_from_axis_angle(Z_AXIS, math.pi / 2)
    np.testing.assert_allclose(quat_to_matrix(q) @ X_AXIS, Y_AXIS, atol=1e-12)


@pytest.mark.parametrize(("axis", "rotation"), [(X_AXIS, rot_x), (Z_AXIS, rot_z)])
def test_consistent_with_passive_rotations(
    axis: FloatArray, rotation: Callable[[float], FloatArray]
) -> None:
    # R(q) è attiva, rot_x e rot_z sono passive: una è la trasposta dell'altra.
    angle = 0.7
    q = quat_from_axis_angle(axis, angle)
    np.testing.assert_allclose(quat_to_matrix(q), rotation(angle).T, atol=1e-12)


# --- Algebra ---


def test_identity_and_inverse() -> None:
    for q in random_quaternions(20):
        np.testing.assert_allclose(quat_multiply(q, quat_identity()), q, atol=1e-15)
        product = quat_multiply(q, quat_conjugate(q))
        np.testing.assert_allclose(product, quat_identity(), atol=1e-12)


def test_product_is_not_commutative() -> None:
    p = quat_from_axis_angle(X_AXIS, math.pi / 2)
    q = quat_from_axis_angle(Z_AXIS, math.pi / 2)
    assert not np.allclose(quat_multiply(p, q), quat_multiply(q, p))


def test_matrix_is_proper_rotation() -> None:
    for q in random_quaternions(20):
        matrix = quat_to_matrix(q)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_product_composes_rotations() -> None:
    # R(p * q) = R(p) @ R(q): le rotazioni si compongono come i pedici.
    p, q = random_quaternions(2, seed=1)
    composed = quat_to_matrix(quat_multiply(p, q))
    np.testing.assert_allclose(
        composed, quat_to_matrix(p) @ quat_to_matrix(q), atol=1e-12
    )


# --- Da matrice a quaternione ---


def test_matrix_round_trip_random() -> None:
    for q in random_quaternions(50, seed=2):
        assert same_rotation(quat_from_matrix(quat_to_matrix(q)), q)


@pytest.mark.parametrize("axis", [X_AXIS, Y_AXIS, Z_AXIS, np.array([1.0, 1.0, 1.0])])
def test_matrix_round_trip_half_turn(axis: FloatArray) -> None:
    # Rotazioni di 180 gradi: w = 0, il caso in cui i metodi ingenui falliscono.
    q = quat_from_axis_angle(axis, math.pi)
    assert same_rotation(quat_from_matrix(quat_to_matrix(q)), q)


@pytest.mark.parametrize("angle", [0.0, 0.1, 2.0, math.pi])
def test_rotation_angle(angle: float) -> None:
    q = quat_from_axis_angle(np.array([1.0, -2.0, 0.5]), angle)
    assert quat_angle(q) == pytest.approx(angle, abs=1e-12)
    assert quat_angle(-q) == pytest.approx(angle, abs=1e-12)


# --- Cinematica ---


def test_kinematics_matches_exact_rotation() -> None:
    # Rotazione a velocità costante attorno all'asse z del corpo:
    # q(t) = q0 * [cos(r t / 2), 0, 0, sin(r t / 2)]. La derivata calcolata
    # deve coincidere con la derivata numerica di questa soluzione esatta.
    q0 = random_quaternions(1, seed=3)[0]
    rate = 0.3

    def exact(t: float) -> FloatArray:
        return quat_multiply(q0, quat_from_axis_angle(Z_AXIS, rate * t))

    step = 1e-5
    numeric = (exact(2.0 + step) - exact(2.0 - step)) / (2 * step)
    computed = quat_derivative(exact(2.0), rate * Z_AXIS)
    np.testing.assert_allclose(computed, numeric, atol=1e-9)


def test_kinematics_preserves_norm() -> None:
    # La derivata è perpendicolare a q: al primo ordine la norma non cambia.
    q = random_quaternions(1, seed=4)[0]
    derivative = quat_derivative(q, np.array([0.1, -0.2, 0.3]))
    assert q @ derivative == pytest.approx(0.0, abs=1e-15)


def test_random_quaternion_is_reproducible_and_unit() -> None:
    first = random_quaternion(np.random.default_rng(42))
    second = random_quaternion(np.random.default_rng(42))
    np.testing.assert_array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0)
