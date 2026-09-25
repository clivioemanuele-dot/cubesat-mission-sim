"""Dinamica d'assetto: corpo rigido con tre ruote di reazione.

Stato: vettore di 10 numeri, tutti espressi negli assi del corpo:
    q      quaternione di assetto [w, x, y, z] (convenzione in core/quaternions);
    omega  velocità angolare del corpo rispetto a ECI [rad/s];
    h      momento angolare delle ruote rispetto al corpo [N m s].

Equazioni di Eulero con le ruote:
    dq/dt           = 0,5 * q * [0, omega]
    I * domega/dt   = tau_ext - tau_w - omega x (I omega + h)
    dh/dt           = tau_w
dove tau_w è la coppia che i motori applicano alle ruote: sul corpo agisce la
reazione -tau_w. Il momento angolare totale I omega + h cambia solo per le
coppie esterne tau_ext.

Ipotesi: ruote allineate con gli assi principali del corpo; inerzia del
satellite diagonale e costante (ruote comprese); trascurato l'accoppiamento
dovuto all'inerzia propria delle ruote (1,6e-5 kg m^2, contro 0,007-0,042 kg m^2
del corpo).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import numpy as np

from cubesat_sim.core.integrators import rk4_step
from cubesat_sim.core.quaternions import quat_derivative, quat_normalize, quat_to_matrix
from cubesat_sim.core.types import FloatArray
from cubesat_sim.core.vectors import cross

STATE_SIZE = 10
_Q = slice(0, 4)
_OMEGA = slice(4, 7)
_WHEELS = slice(7, 10)


@dataclass(frozen=True, eq=False)
class AttitudeState:
    """Stato d'assetto in forma leggibile.

    L'integratore lavora su un vettore di 10 numeri; questa classe lo rende
    leggibile e lo ricostruisce.

    Attributes:
        attitude: quaternione di assetto [w, x, y, z].
        angular_velocity: velocità angolare del corpo, assi corpo [rad/s].
        wheel_momentum: momento angolare delle ruote, assi corpo [N m s].
    """

    attitude: FloatArray
    angular_velocity: FloatArray
    wheel_momentum: FloatArray

    @classmethod
    def from_vector(cls, vector: FloatArray) -> Self:
        """Ricostruisce lo stato da un vettore di 10 numeri.

        Args:
            vector: vettore [q, omega, h].

        Returns:
            Lo stato, con copie indipendenti dei tre blocchi.
        """
        return cls(
            attitude=vector[_Q].copy(),
            angular_velocity=vector[_OMEGA].copy(),
            wheel_momentum=vector[_WHEELS].copy(),
        )

    def to_vector(self) -> FloatArray:
        """Vettore di 10 numeri [q, omega, h] per l'integratore."""
        vector = np.empty(STATE_SIZE)
        vector[_Q] = self.attitude
        vector[_OMEGA] = self.angular_velocity
        vector[_WHEELS] = self.wheel_momentum
        return vector


def attitude_derivative(
    state: FloatArray,
    inertia: FloatArray,
    wheel_torque: FloatArray,
    external_torque: FloatArray,
) -> FloatArray:
    """Derivata temporale dello stato d'assetto.

    Args:
        state: vettore di stato [q, omega, h].
        inertia: momenti principali di inerzia [kg m^2].
        wheel_torque: coppia dei motori sulle ruote, assi corpo [N m].
        external_torque: coppie esterne sul satellite, assi corpo [N m].

    Returns:
        Derivata del vettore di stato.
    """
    q = state[_Q]
    omega = state[_OMEGA]
    total_momentum = inertia * omega + state[_WHEELS]
    net_torque = external_torque - wheel_torque - cross(omega, total_momentum)
    derivative = np.empty(STATE_SIZE)
    derivative[_Q] = quat_derivative(q, omega)
    derivative[_OMEGA] = net_torque / inertia
    derivative[_WHEELS] = wheel_torque
    return derivative


def angular_momentum_eci(state: FloatArray, inertia: FloatArray) -> FloatArray:
    """Momento angolare totale (corpo e ruote) espresso in ECI.

    Args:
        state: vettore di stato [q, omega, h].
        inertia: momenti principali di inerzia [kg m^2].

    Returns:
        Momento angolare in ECI [N m s].
    """
    momentum_body = inertia * state[_OMEGA] + state[_WHEELS]
    return quat_to_matrix(state[_Q]) @ momentum_body


def rotational_energy(state: FloatArray, inertia: FloatArray) -> float:
    """Energia cinetica di rotazione del corpo, 0,5 omega^T I omega.

    Args:
        state: vettore di stato [q, omega, h].
        inertia: momenti principali di inerzia [kg m^2].

    Returns:
        Energia [J].
    """
    omega = state[_OMEGA]
    return 0.5 * float(omega @ (inertia * omega))


def propagate_attitude(
    state: FloatArray,
    step_s: float,
    inertia: FloatArray,
    wheel_torque: FloatArray,
    external_torque: Callable[[FloatArray], FloatArray],
) -> FloatArray:
    """Avanza lo stato d'assetto di un passo con RK4.

    La coppia sulle ruote resta costante durante il passo (comando del
    controllo a tempo discreto). Le coppie esterne dipendono dall'assetto e
    vengono ricalcolate a ogni stadio di RK4. Alla fine del passo il
    quaternione è rinormalizzato, per eliminare la lenta deriva numerica della
    sua norma.

    Args:
        state: vettore di stato [q, omega, h] all'inizio del passo.
        step_s: durata del passo [s].
        inertia: momenti principali di inerzia [kg m^2].
        wheel_torque: coppia dei motori sulle ruote, assi corpo [N m].
        external_torque: funzione che, dato il vettore di stato, restituisce
            le coppie esterne in assi corpo [N m].

    Returns:
        Vettore di stato alla fine del passo.
    """

    def derivative(y: FloatArray) -> FloatArray:
        return attitude_derivative(y, inertia, wheel_torque, external_torque(y))

    new_state = rk4_step(derivative, state, step_s)
    new_state[_Q] = quat_normalize(new_state[_Q])
    return new_state
