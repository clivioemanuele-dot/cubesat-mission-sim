"""Operazioni elementari sui vettori 3D.

Prodotto vettoriale e norma sono scritti per vettori di 3 componenti. Le
funzioni generali di numpy (np.cross, np.linalg.norm) accettano array di
qualunque forma e, per un solo vettore, spendono quasi tutto il tempo in
controlli: nel ciclo di simulazione, dove vengono chiamate centinaia di
migliaia di volte, queste versioni costano circa 25 volte meno (prodotto
vettoriale) e 1,5 volte meno (norma), con risultati identici bit per bit.
"""

import math

import numpy as np

from cubesat_sim.core.types import FloatArray


def cross(a: FloatArray, b: FloatArray) -> FloatArray:
    """Prodotto vettoriale di due vettori 3D.

    Args:
        a: primo vettore.
        b: secondo vettore.

    Returns:
        Il vettore a x b.
    """
    a1, a2, a3 = a.tolist()
    b1, b2, b3 = b.tolist()
    return np.array(
        [a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1], dtype=np.float64
    )


def norm(vector: FloatArray) -> float:
    """Norma euclidea (lunghezza) di un vettore.

    Args:
        vector: vettore.

    Returns:
        La norma del vettore.
    """
    return math.sqrt(float(vector @ vector))


def unit(vector: FloatArray) -> FloatArray:
    """Versore con la stessa direzione del vettore.

    Args:
        vector: vettore non nullo.

    Returns:
        Vettore di norma 1.

    Raises:
        ValueError: se il vettore è nullo.
    """
    length = norm(vector)
    if length == 0.0:
        msg = "impossibile normalizzare un vettore nullo"
        raise ValueError(msg)
    return vector / length


def angle_between(a: FloatArray, b: FloatArray) -> float:
    """Angolo tra due vettori non nulli.

    Usa atan2 invece di acos: resta accurato anche per angoli molto piccoli,
    come gli errori di puntamento.

    Args:
        a: primo vettore.
        b: secondo vettore.

    Returns:
        Angolo tra 0 e pi [rad].
    """
    return math.atan2(norm(cross(a, b)), float(a @ b))
