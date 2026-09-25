"""Rotazioni elementari del sistema di riferimento.

Convenzione "passiva": la matrice non muove il vettore, cambia il sistema in
cui è espresso. Se il sistema B si ottiene ruotando A di un angolo theta
attorno a un asse, allora v_B = rot(theta) @ v_A.
"""

import math

import numpy as np

from cubesat_sim.core.types import FloatArray


def rot_x(angle: float) -> FloatArray:
    """Rotazione passiva attorno all'asse x.

    Args:
        angle: angolo di cui ruota il sistema di riferimento [rad].

    Returns:
        Matrice 3x3 che esprime nel sistema ruotato un vettore dato nel
        sistema di partenza.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ],
        dtype=np.float64,
    )


def rot_z(angle: float) -> FloatArray:
    """Rotazione passiva attorno all'asse z.

    Args:
        angle: angolo di cui ruota il sistema di riferimento [rad].

    Returns:
        Matrice 3x3 che esprime nel sistema ruotato un vettore dato nel
        sistema di partenza.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
