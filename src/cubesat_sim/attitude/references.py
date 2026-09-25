"""Assetti di riferimento: dove deve puntare il satellite in ogni modo.

Ogni riferimento allinea due assi del corpo a due direzioni in ECI:
    l'asse primario coincide esattamente con la direzione primaria;
    l'asse secondario è il più vicino possibile alla direzione secondaria, e
    fissa la rotazione rimasta libera attorno all'asse primario.

Riferimenti per modo (assi del corpo dalla configurazione):
    SUN_POINTING e SAFE: pannelli verso il Sole; fotocamera il più possibile
        verso la Terra, per rendere brevi i passaggi agli altri modi;
    NADIR: fotocamera verso il centro della Terra; pannelli il più possibile
        verso il Sole, per produrre energia anche mentre si fotografa;
    DOWNLINK: antenna verso la stazione; pannelli il più possibile verso la
        normale al piano dell'orbita, dalla parte del Sole (il motivo è
        spiegato in station_pointing_attitude).
Il DETUMBLE non ha riferimento: il B-dot frena la rotazione e basta.

Se le due direzioni sono parallele, la secondaria non fissa nulla: si usa una
direzione di riserva (la normale al piano dell'orbita, oppure la velocità nel
DOWNLINK).

Vicino al parallelismo il riferimento resta definito ma ruota velocemente
attorno all'asse primario. Con l'orbita di riferimento SUN_POINTING e NADIR non
ne soffrono: Sole e nadir restano sempre ad almeno |beta| (circa 22 gradi) di
distanza, e il riferimento non supera 0,2 gradi/s.
"""

import math
from dataclasses import dataclass
from typing import Self

import numpy as np

from cubesat_sim.core.config import MissionConfig
from cubesat_sim.core.quaternions import (
    quat_conjugate,
    quat_from_matrix,
    quat_multiply,
    quat_to_rotation_vector,
)
from cubesat_sim.core.types import FloatArray
from cubesat_sim.core.vectors import angle_between, cross, unit

# Sotto questo angolo due direzioni sono considerate parallele [rad].
_PARALLEL_TOLERANCE_RAD = 1e-6


@dataclass(frozen=True, eq=False)
class PointingAxes:
    """Assi del corpo usati per il puntamento.

    Attributes:
        sun: normale dei pannelli, da puntare verso il Sole.
        camera: direzione di vista della fotocamera.
        antenna: asse dell'antenna.
    """

    sun: FloatArray
    camera: FloatArray
    antenna: FloatArray

    @classmethod
    def from_config(cls, config: MissionConfig) -> Self:
        """Legge gli assi dalla configurazione.

        Args:
            config: configurazione della missione.

        Returns:
            Gli assi di puntamento, in assi corpo.
        """
        return cls(
            sun=np.array(config.solar_array.sun_pointing_axis_body),
            camera=np.array(config.payload.camera_boresight_body),
            antenna=np.array(config.payload.antenna_boresight_body),
        )


def _triad(first: FloatArray, second: FloatArray) -> FloatArray:
    """Terna ortonormale destra costruita da due direzioni non parallele.

    Colonne: first normalizzato; la normale al piano (first, second); il terzo
    asse che completa la terna.
    """
    t1 = unit(first)
    t2 = unit(cross(first, second))
    t3 = cross(t1, t2)
    return np.column_stack([t1, t2, t3])


def two_axis_attitude(
    primary_body: FloatArray,
    primary_target: FloatArray,
    secondary_body: FloatArray,
    secondary_target: FloatArray,
    fallback_target: FloatArray,
) -> FloatArray:
    """Assetto che allinea due assi del corpo a due direzioni in ECI.

    Args:
        primary_body: asse del corpo da allineare esattamente.
        primary_target: direzione primaria in ECI.
        secondary_body: asse del corpo da avvicinare il più possibile alla
            direzione secondaria; non parallelo a primary_body.
        secondary_target: direzione secondaria in ECI.
        fallback_target: direzione di riserva, usata se la secondaria è
            parallela alla primaria; non parallela a primary_target.

    Returns:
        Quaternione di assetto [w, x, y, z]: v_eci = R(q) @ v_body.
    """
    secondary = secondary_target
    angle = angle_between(primary_target, secondary_target)
    if min(angle, math.pi - angle) < _PARALLEL_TOLERANCE_RAD:
        secondary = fallback_target
    body = _triad(primary_body, secondary_body)
    inertial = _triad(primary_target, secondary)
    return quat_from_matrix(inertial @ body.T)


def sun_pointing_attitude(
    axes: PointingAxes,
    sun_eci: FloatArray,
    position_eci: FloatArray,
    velocity_eci: FloatArray,
) -> FloatArray:
    """Riferimento di SUN_POINTING e SAFE: pannelli verso il Sole.

    Args:
        axes: assi di puntamento del corpo.
        sun_eci: direzione (o posizione) del Sole in ECI.
        position_eci: posizione del satellite in ECI [m].
        velocity_eci: velocità del satellite in ECI [m/s].

    Returns:
        Quaternione di assetto di riferimento.
    """
    orbit_normal = cross(position_eci, velocity_eci)
    return two_axis_attitude(
        axes.sun, sun_eci, axes.camera, -position_eci, orbit_normal
    )


def nadir_pointing_attitude(
    axes: PointingAxes,
    sun_eci: FloatArray,
    position_eci: FloatArray,
    velocity_eci: FloatArray,
) -> FloatArray:
    """Riferimento di NADIR: fotocamera verso il centro della Terra.

    Args:
        axes: assi di puntamento del corpo.
        sun_eci: direzione (o posizione) del Sole in ECI.
        position_eci: posizione del satellite in ECI [m].
        velocity_eci: velocità del satellite in ECI [m/s].

    Returns:
        Quaternione di assetto di riferimento.
    """
    orbit_normal = cross(position_eci, velocity_eci)
    return two_axis_attitude(
        axes.camera, -position_eci, axes.sun, sun_eci, orbit_normal
    )


def station_pointing_attitude(
    axes: PointingAxes,
    sun_eci: FloatArray,
    position_eci: FloatArray,
    velocity_eci: FloatArray,
    station_eci: FloatArray,
) -> FloatArray:
    """Riferimento di DOWNLINK: antenna verso la stazione di terra.

    La rotazione attorno all'asse dell'antenna non conta per il collegamento:
    la fissa la normale al piano dell'orbita, presa dalla parte del Sole, a cui
    si avvicinano i pannelli. Il Sole non va bene come seconda direzione: nei
    passaggi notturni la direzione della stazione gli passa vicina (fino a
    10 gradi nello scenario di riferimento) e il riferimento ruoterebbe
    attorno all'antenna a oltre 2 gradi/s. La stazione si vede sempre entro
    circa 64 gradi dal nadir, quindi ad almeno 26 gradi dalla normale
    all'orbita: il riferimento resta sotto 1 grado/s. Il costo è energetico e
    piccolo: nei passaggi diurni i pannelli vedono il Sole a 90 - |beta| gradi.

    Args:
        axes: assi di puntamento del corpo.
        sun_eci: direzione (o posizione) del Sole in ECI.
        position_eci: posizione del satellite in ECI [m].
        velocity_eci: velocità del satellite in ECI [m/s].
        station_eci: posizione della stazione in ECI [m].

    Returns:
        Quaternione di assetto di riferimento.
    """
    orbit_normal = cross(position_eci, velocity_eci)
    if float(orbit_normal @ sun_eci) < 0.0:
        orbit_normal = -orbit_normal
    line_of_sight = station_eci - position_eci
    return two_axis_attitude(
        axes.antenna, line_of_sight, axes.sun, orbit_normal, velocity_eci
    )


def reference_rate(
    q_now: FloatArray, q_next: FloatArray, interval_s: float
) -> FloatArray:
    """Velocità angolare del riferimento, da due riferimenti successivi.

    Nell'intervallo il riferimento ruota di q_now^-1 * q_next: il vettore di
    rotazione diviso per l'intervallo è la velocità angolare, espressa negli
    assi del riferimento. Il segno dei quaternioni è indifferente.

    Args:
        q_now: riferimento all'istante attuale.
        q_next: riferimento dopo l'intervallo.
        interval_s: intervallo tra i due riferimenti [s].

    Returns:
        Velocità angolare del riferimento, negli assi del riferimento [rad/s].
    """
    delta = quat_multiply(quat_conjugate(q_now), q_next)
    return quat_to_rotation_vector(delta) / interval_s
