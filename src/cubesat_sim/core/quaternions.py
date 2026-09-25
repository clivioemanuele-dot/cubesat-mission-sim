"""Quaternioni di assetto.

Convenzione del progetto:
    - ordine [w, x, y, z], con la parte scalare w per prima;
    - prodotto di Hamilton, indicato qui con *;
    - il quaternione di assetto q descrive il corpo rispetto a ECI: la matrice
      R(q) porta le componenti di un vettore dal corpo a ECI,
      v_eci = R(q) @ v_body, e quindi v_body = R(q).T @ v_eci;
    - le rotazioni si compongono come i pedici: q_ac = q_ab * q_bc;
    - cinematica con la velocità angolare espressa nel corpo:
      dq/dt = 0,5 * q * [0, omega].

R(q) è una rotazione attiva; le matrici di rotations.py sono passive: per la
stessa rotazione di un angolo theta attorno a un asse, R(q) = rot(theta).T.

Le funzioni chiamate a ogni passo dell'integratore estraggono le componenti
con tolist(): i conti su numeri Python sono molto più rapidi che su singoli
elementi di un array numpy, e i risultati sono identici.
"""

import math

import numpy as np

from cubesat_sim.core.types import FloatArray
from cubesat_sim.core.vectors import unit

_CONJUGATE_SIGNS = np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float64)


def quat_identity() -> FloatArray:
    """Quaternione identità: nessuna rotazione."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quat_from_axis_angle(axis: FloatArray, angle: float) -> FloatArray:
    """Quaternione della rotazione di un angolo attorno a un asse.

    Args:
        axis: asse di rotazione, non nullo (viene normalizzato).
        angle: angolo di rotazione, secondo la regola della mano destra [rad].

    Returns:
        Quaternione unitario [w, x, y, z].
    """
    x, y, z = math.sin(angle / 2) * unit(axis)
    return np.array([math.cos(angle / 2), x, y, z], dtype=np.float64)


def quat_multiply(p: FloatArray, q: FloatArray) -> FloatArray:
    """Prodotto di Hamilton p * q.

    Compone le rotazioni: R(p * q) = R(p) @ R(q).

    Args:
        p: primo quaternione [w, x, y, z].
        q: secondo quaternione [w, x, y, z].

    Returns:
        Il prodotto p * q.
    """
    pw, px, py, pz = p.tolist()
    qw, qx, qy, qz = q.tolist()
    return np.array(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ],
        dtype=np.float64,
    )


def quat_conjugate(q: FloatArray) -> FloatArray:
    """Coniugato di q: per un quaternione unitario è la rotazione inversa.

    Args:
        q: quaternione [w, x, y, z].

    Returns:
        Il coniugato [w, -x, -y, -z].
    """
    return q * _CONJUGATE_SIGNS


def quat_normalize(q: FloatArray) -> FloatArray:
    """Quaternione di norma 1 con la stessa direzione di q.

    Args:
        q: quaternione non nullo.

    Returns:
        Quaternione unitario.
    """
    return q / float(np.linalg.norm(q))


def quat_to_matrix(q: FloatArray) -> FloatArray:
    """Matrice di rotazione del quaternione unitario q.

    Args:
        q: quaternione unitario [w, x, y, z].

    Returns:
        Matrice 3x3 R(q). Per l'assetto: v_eci = R(q) @ v_body.
    """
    w, x, y, z = q.tolist()
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_from_matrix(matrix: FloatArray) -> FloatArray:
    """Quaternione unitario di una matrice di rotazione (metodo di Shepperd).

    Si parte dal termine più grande tra la traccia e gli elementi diagonali:
    così non si divide mai per numeri piccoli, nemmeno a 180 gradi.

    Args:
        matrix: matrice di rotazione 3x3, nella convenzione di quat_to_matrix.

    Returns:
        Quaternione unitario [w, x, y, z] con w >= 0.
    """
    r = matrix
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    if trace >= max(r[0, 0], r[1, 1], r[2, 2]):
        s = 2.0 * math.sqrt(1.0 + trace)  # s = 4 w
        components = [
            0.25 * s,
            (r[2, 1] - r[1, 2]) / s,
            (r[0, 2] - r[2, 0]) / s,
            (r[1, 0] - r[0, 1]) / s,
        ]
    elif r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])  # s = 4 x
        components = [
            (r[2, 1] - r[1, 2]) / s,
            0.25 * s,
            (r[0, 1] + r[1, 0]) / s,
            (r[0, 2] + r[2, 0]) / s,
        ]
    elif r[1, 1] >= r[2, 2]:
        s = 2.0 * math.sqrt(1.0 - r[0, 0] + r[1, 1] - r[2, 2])  # s = 4 y
        components = [
            (r[0, 2] - r[2, 0]) / s,
            (r[0, 1] + r[1, 0]) / s,
            0.25 * s,
            (r[1, 2] + r[2, 1]) / s,
        ]
    else:
        s = 2.0 * math.sqrt(1.0 - r[0, 0] - r[1, 1] + r[2, 2])  # s = 4 z
        components = [
            (r[1, 0] - r[0, 1]) / s,
            (r[0, 2] + r[2, 0]) / s,
            (r[1, 2] + r[2, 1]) / s,
            0.25 * s,
        ]
    q = quat_normalize(np.array(components, dtype=np.float64))
    return q if q[0] >= 0.0 else -q


def quat_derivative(q: FloatArray, omega_body: FloatArray) -> FloatArray:
    """Derivata temporale del quaternione di assetto.

    Args:
        q: quaternione di assetto [w, x, y, z].
        omega_body: velocità angolare del corpo rispetto a ECI, espressa negli
            assi del corpo [rad/s].

    Returns:
        dq/dt = 0,5 * q * [0, omega] [1/s].
    """
    wx, wy, wz = omega_body.tolist()
    return 0.5 * quat_multiply(q, np.array([0.0, wx, wy, wz], dtype=np.float64))


def quat_angle(q: FloatArray) -> float:
    """Angolo della rotazione rappresentata da q.

    q e -q rappresentano la stessa rotazione: si usa il valore assoluto di w.

    Args:
        q: quaternione unitario [w, x, y, z].

    Returns:
        Angolo tra 0 e pi [rad].
    """
    return 2.0 * math.atan2(float(np.linalg.norm(q[1:])), abs(float(q[0])))


def quat_from_rotation_vector(rotation_vector: FloatArray) -> FloatArray:
    """Quaternione della rotazione descritta da un vettore di rotazione.

    Args:
        rotation_vector: asse di rotazione moltiplicato per l'angolo [rad].
            Il vettore nullo corrisponde all'identità.

    Returns:
        Quaternione unitario [w, x, y, z].
    """
    angle = float(np.linalg.norm(rotation_vector))
    if angle == 0.0:
        return quat_identity()
    return quat_from_axis_angle(rotation_vector, angle)


def quat_to_rotation_vector(q: FloatArray) -> FloatArray:
    """Vettore di rotazione (asse per angolo) del quaternione unitario q.

    Tra q e -q, che rappresentano la stessa rotazione, si sceglie la rotazione
    più breve: l'angolo è tra 0 e pi.

    Args:
        q: quaternione unitario [w, x, y, z].

    Returns:
        Vettore di rotazione [rad]: il suo modulo è l'angolo.
    """
    w = float(q[0])
    vector = q[1:] if w >= 0.0 else -q[1:]
    sin_half = float(np.linalg.norm(vector))
    if sin_half == 0.0:
        return np.zeros(3)
    angle = 2.0 * math.atan2(sin_half, abs(w))
    return vector * (angle / sin_half)


def random_quaternion(rng: np.random.Generator) -> FloatArray:
    """Quaternione unitario casuale, uniforme su tutte le rotazioni.

    Una distribuzione gaussiana in 4 dimensioni ha la stessa probabilità in
    ogni direzione: normalizzata, dà una rotazione uniformemente casuale.

    Args:
        rng: generatore di numeri casuali (con seme esplicito).

    Returns:
        Quaternione unitario [w, x, y, z].
    """
    return quat_normalize(rng.normal(size=4))
