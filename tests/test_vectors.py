"""Test delle operazioni sui vettori 3D, confrontate con numpy."""

import math

import numpy as np
import pytest

from cubesat_sim.core.vectors import angle_between, cross, norm, unit


def test_cross_matches_numpy_exactly() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.normal(size=3), rng.normal(size=3)
        np.testing.assert_array_equal(cross(a, b), np.cross(a, b))


def test_norm_matches_numpy_exactly() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        vector = rng.normal(size=3) * 10.0 ** rng.uniform(-8.0, 8.0)
        assert norm(vector) == float(np.linalg.norm(vector))


def test_unit_vector_and_zero_vector() -> None:
    np.testing.assert_allclose(unit(np.array([3.0, 0.0, 4.0])), [0.6, 0.0, 0.8])
    with pytest.raises(ValueError, match="vettore nullo"):
        unit(np.zeros(3))


def test_angle_between_small_and_large_angles() -> None:
    # atan2 resta accurato anche per un nanoradiante, dove acos fallirebbe.
    x_axis = np.array([1.0, 0.0, 0.0])
    tiny = 1e-9
    assert angle_between(x_axis, np.array([1.0, tiny, 0.0])) == pytest.approx(tiny)
    assert angle_between(x_axis, -x_axis) == pytest.approx(math.pi)
