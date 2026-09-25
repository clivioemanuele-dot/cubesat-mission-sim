"""Coppie esterne: gradiente gravitazionale e dipolo magnetico.

Gradiente gravitazionale: la gravità è più forte sulla parte del satellite più
vicina alla Terra. Su un corpo con inerzie diverse questo produce la coppia
    tau = 3 mu / r^3 * r_hat x (I r_hat),
con r_hat versore dal centro della Terra al satellite, espresso nel corpo. È
nulla quando un asse principale è verticale; l'equilibrio con l'asse di
inerzia minima verticale è stabile.

Dipolo magnetico: un dipolo m immerso nel campo B subisce la coppia
tau = m x B, sempre perpendicolare a B. La stessa formula vale per il dipolo
residuo del satellite (disturbo) e per quello comandato ai magnetorquer.

Resistenza aerodinamica e pressione solare sono trascurate (docs/assumptions.md).
"""

from collections.abc import Callable

from cubesat_sim.core.constants import EARTH_MU_M3_S2
from cubesat_sim.core.quaternions import quat_to_matrix
from cubesat_sim.core.types import FloatArray
from cubesat_sim.core.vectors import cross, norm

# Il quaternione di assetto occupa le prime 4 componenti del vettore di stato
# (attitude/dynamics.py).
_ATTITUDE = slice(0, 4)


def gravity_gradient_torque(
    position_body: FloatArray, inertia: FloatArray
) -> FloatArray:
    """Coppia di gradiente gravitazionale.

    Args:
        position_body: posizione del satellite rispetto al centro della Terra,
            espressa negli assi del corpo [m].
        inertia: momenti principali di inerzia [kg m^2].

    Returns:
        Coppia in assi corpo [N m].
    """
    radius = norm(position_body)
    r_hat = position_body / radius
    return (3.0 * EARTH_MU_M3_S2 / radius**3) * cross(r_hat, inertia * r_hat)


def magnetic_torque(dipole_body: FloatArray, field_body: FloatArray) -> FloatArray:
    """Coppia su un dipolo magnetico immerso in un campo, tau = m x B.

    Args:
        dipole_body: dipolo magnetico in assi corpo [A m^2].
        field_body: campo magnetico in assi corpo [T].

    Returns:
        Coppia in assi corpo [N m].
    """
    return cross(dipole_body, field_body)


def external_torque_function(
    position_eci: FloatArray,
    field_eci: FloatArray,
    dipole_body: FloatArray,
    inertia: FloatArray,
) -> Callable[[FloatArray], FloatArray]:
    """Coppie esterne totali in funzione dello stato, per un periodo di controllo.

    Posizione e campo in ECI restano costanti durante il periodo; l'assetto
    cambia a ogni stadio di RK4, quindi il passaggio nel corpo avviene dentro
    la funzione restituita.

    Args:
        position_eci: posizione del satellite in ECI [m].
        field_eci: campo magnetico vero in ECI [T].
        dipole_body: dipolo magnetico totale del satellite, assi corpo [A m^2]:
            magnetorquer più dipolo residuo.
        inertia: momenti principali di inerzia [kg m^2].

    Returns:
        Funzione che, dato il vettore di stato, restituisce la coppia esterna
        in assi corpo [N m]: gradiente gravitazionale più coppia magnetica.
    """

    def torque(state: FloatArray) -> FloatArray:
        to_body = quat_to_matrix(state[_ATTITUDE]).T
        gravity = gravity_gradient_torque(to_body @ position_eci, inertia)
        return gravity + magnetic_torque(dipole_body, to_body @ field_eci)

    return torque
