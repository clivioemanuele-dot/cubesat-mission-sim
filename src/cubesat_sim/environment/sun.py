"""Posizione del Sole ed eclissi.

Sole: formula a bassa precisione dell'Astronomical Almanac, con un errore di
circa 0,01 gradi tra il 1950 e il 2050. Il risultato è espresso in ECI.

Eclissi: modello a ombra cilindrica. La Terra, sferica con il raggio
equatoriale, proietta un cilindro d'ombra parallelo alla direzione del Sole.
La penombra (circa 10 s a ogni ingresso e uscita in orbita bassa) è trascurata.
"""

import math

import numpy as np

from cubesat_sim.core.constants import (
    ASTRONOMICAL_UNIT_M,
    EARTH_RADIUS_M,
    JULIAN_DATE_J2000,
)
from cubesat_sim.core.types import FloatArray

# Formula a bassa precisione per il Sole (Astronomical Almanac, sezione C):
# angoli in gradi, tempo in giorni da J2000.
_LONGITUDE_J2000_DEG = 280.460  # longitudine media
_LONGITUDE_RATE_DEG_DAY = 0.9856474
_ANOMALY_J2000_DEG = 357.528  # anomalia media
_ANOMALY_RATE_DEG_DAY = 0.9856003
_CENTER_1_DEG = 1.915  # equazione del centro, primo termine
_CENTER_2_DEG = 0.020  # equazione del centro, secondo termine
_OBLIQUITY_J2000_DEG = 23.439  # inclinazione dell'eclittica sull'equatore
_OBLIQUITY_RATE_DEG_DAY = -4.0e-7
_DISTANCE_0_AU = 1.00014  # distanza Terra-Sole: termine costante
_DISTANCE_1_AU = -0.01671  # termine in cos(anomalia)
_DISTANCE_2_AU = -0.00014  # termine in cos(2 anomalia)


def sun_position_eci(jd: float) -> FloatArray:
    """Posizione del Sole in ECI.

    Args:
        jd: data giuliana [giorni].

    Returns:
        Vettore dal centro della Terra al centro del Sole [m].
    """
    days = jd - JULIAN_DATE_J2000
    mean_anomaly = math.radians(_ANOMALY_J2000_DEG + _ANOMALY_RATE_DEG_DAY * days)
    longitude = math.radians(
        _LONGITUDE_J2000_DEG
        + _LONGITUDE_RATE_DEG_DAY * days
        + _CENTER_1_DEG * math.sin(mean_anomaly)
        + _CENTER_2_DEG * math.sin(2.0 * mean_anomaly)
    )
    obliquity = math.radians(_OBLIQUITY_J2000_DEG + _OBLIQUITY_RATE_DEG_DAY * days)
    distance_au = (
        _DISTANCE_0_AU
        + _DISTANCE_1_AU * math.cos(mean_anomaly)
        + _DISTANCE_2_AU * math.cos(2.0 * mean_anomaly)
    )
    direction = np.array(
        [
            math.cos(longitude),
            math.cos(obliquity) * math.sin(longitude),
            math.sin(obliquity) * math.sin(longitude),
        ],
        dtype=np.float64,
    )
    return distance_au * ASTRONOMICAL_UNIT_M * direction


def is_in_eclipse(position_eci: FloatArray, sun_position: FloatArray) -> bool:
    """Indica se il satellite è nell'ombra cilindrica della Terra.

    Args:
        position_eci: posizione del satellite in ECI [m].
        sun_position: posizione del Sole in ECI [m].

    Returns:
        True se il satellite è in ombra.
    """
    sun_unit = sun_position / np.linalg.norm(sun_position)
    along_sun = float(position_eci @ sun_unit)
    distance_from_axis = float(np.linalg.norm(position_eci - along_sun * sun_unit))
    return along_sun < 0.0 and distance_from_axis < EARTH_RADIUS_M
