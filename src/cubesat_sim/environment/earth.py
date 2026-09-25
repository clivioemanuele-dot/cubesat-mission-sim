"""Tempo e rotazione terrestre.

Sistemi di riferimento:
    ECI: inerziale, centrato nella Terra, asse z verso il polo nord, asse x
        verso l'equinozio di J2000. Precessione e nutazione sono trascurate:
        l'errore resta sotto 0,5 gradi fino al 2030.
    ECEF: solidale con la Terra, asse x verso il meridiano di Greenwich.
        Ruota rispetto a ECI attorno all'asse z comune.

Scale di tempo: UTC, UT1 e TT sono considerate coincidenti. Le differenze
(meno di 70 secondi) sono trascurabili per questo modello.
"""

import datetime as dt
import math

from cubesat_sim.core.constants import (
    JULIAN_DATE_J2000,
    JULIAN_DATE_UNIX_EPOCH,
    SECONDS_PER_DAY,
)
from cubesat_sim.core.rotations import rot_z
from cubesat_sim.core.types import FloatArray

# Angolo di rotazione terrestre (IERS Conventions 2010), in giri:
# ERA = ERA_J2000 + ERA_RATE * (JD - JD_J2000).
_ERA_AT_J2000_REV = 0.7790572732640
_ERA_RATE_REV_PER_DAY = 1.00273781191135448


def julian_date(epoch: dt.datetime, elapsed_s: float = 0.0) -> float:
    """Data giuliana dell'istante epoch + elapsed_s.

    Args:
        epoch: istante di riferimento, con fuso orario.
        elapsed_s: secondi trascorsi dall'istante di riferimento.

    Returns:
        Data giuliana [giorni].

    Raises:
        ValueError: se epoch non ha il fuso orario.
    """
    if epoch.tzinfo is None:
        msg = "l'istante di riferimento deve avere il fuso orario"
        raise ValueError(msg)
    unix_s = epoch.timestamp() + elapsed_s
    return JULIAN_DATE_UNIX_EPOCH + unix_s / SECONDS_PER_DAY


def earth_rotation_angle(jd: float) -> float:
    """Angolo di rotazione della Terra rispetto al riferimento inerziale.

    Args:
        jd: data giuliana [giorni].

    Returns:
        Angolo tra 0 e 2 pi [rad].
    """
    revolutions = _ERA_AT_J2000_REV + _ERA_RATE_REV_PER_DAY * (jd - JULIAN_DATE_J2000)
    return 2.0 * math.pi * (revolutions % 1.0)


def eci_to_ecef_matrix(jd: float) -> FloatArray:
    """Matrice di passaggio da ECI a ECEF.

    Args:
        jd: data giuliana [giorni].

    Returns:
        Matrice 3x3 tale che v_ecef = matrice @ v_eci.
    """
    return rot_z(earth_rotation_angle(jd))
