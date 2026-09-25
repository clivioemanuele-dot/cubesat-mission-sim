"""Stazione di terra, visibilità e punto sotto il satellite.

La stazione è sull'ellissoide WGS-84, in coordinate geodetiche. L'elevazione
del satellite è misurata rispetto all'orizzonte locale, cioè al piano
perpendicolare alla verticale geodetica.

Il punto sotto il satellite è calcolato su una Terra sferica (latitudine
geocentrica): la differenza dalla latitudine geodetica, al massimo 0,2 gradi,
è trascurabile per la traccia a terra e per la regione di acquisizione.
"""

import math
from dataclasses import dataclass
from typing import Self

import numpy as np

from cubesat_sim.core.config import GroundStationConfig, ImagingConfig
from cubesat_sim.core.constants import EARTH_FLATTENING, EARTH_RADIUS_M
from cubesat_sim.core.types import FloatArray
from cubesat_sim.core.vectors import angle_between
from cubesat_sim.environment.earth import eci_to_ecef_matrix

_ECCENTRICITY_SQ = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)


def geodetic_to_ecef(
    latitude_rad: float, longitude_rad: float, altitude_m: float
) -> FloatArray:
    """Posizione ECEF di un punto in coordinate geodetiche WGS-84.

    Args:
        latitude_rad: latitudine geodetica [rad].
        longitude_rad: longitudine, positiva verso est [rad].
        altitude_m: quota sopra l'ellissoide [m].

    Returns:
        Posizione in ECEF [m].
    """
    sin_lat, cos_lat = math.sin(latitude_rad), math.cos(latitude_rad)
    # Raggio di curvatura dell'ellissoide nella direzione est-ovest.
    normal_radius = EARTH_RADIUS_M / math.sqrt(1.0 - _ECCENTRICITY_SQ * sin_lat**2)
    horizontal = (normal_radius + altitude_m) * cos_lat
    return np.array(
        [
            horizontal * math.cos(longitude_rad),
            horizontal * math.sin(longitude_rad),
            (normal_radius * (1.0 - _ECCENTRICITY_SQ) + altitude_m) * sin_lat,
        ],
        dtype=np.float64,
    )


def geodetic_up(latitude_rad: float, longitude_rad: float) -> FloatArray:
    """Versore della verticale locale (normale all'ellissoide) in ECEF.

    Args:
        latitude_rad: latitudine geodetica [rad].
        longitude_rad: longitudine, positiva verso est [rad].

    Returns:
        Versore verso l'alto in ECEF.
    """
    cos_lat = math.cos(latitude_rad)
    return np.array(
        [
            cos_lat * math.cos(longitude_rad),
            cos_lat * math.sin(longitude_rad),
            math.sin(latitude_rad),
        ],
        dtype=np.float64,
    )


def subsatellite_point(position_ecef: FloatArray) -> tuple[float, float]:
    """Latitudine geocentrica e longitudine del punto sotto il satellite.

    Args:
        position_ecef: posizione del satellite in ECEF [m].

    Returns:
        Coppia (latitudine, longitudine) [rad], longitudine tra -pi e pi.
    """
    x, y, z = (float(component) for component in position_ecef)
    return math.atan2(z, math.hypot(x, y)), math.atan2(y, x)


def is_over_region(
    latitude_rad: float, longitude_rad: float, region: ImagingConfig
) -> bool:
    """Indica se un punto è dentro la regione di acquisizione.

    Args:
        latitude_rad: latitudine del punto [rad].
        longitude_rad: longitudine del punto, tra -pi e pi [rad].
        region: sezione [imaging] della configurazione.

    Returns:
        True se il punto è dentro il rettangolo della regione.
    """
    lat_min, lat_max = region.latitude_range_rad
    lon_min, lon_max = region.longitude_range_rad
    return lat_min <= latitude_rad <= lat_max and lon_min <= longitude_rad <= lon_max


@dataclass(frozen=True, eq=False)
class GroundStation:
    """Stazione di terra, fissa sulla Terra.

    Attributes:
        name: nome della stazione.
        position_ecef: posizione in ECEF [m].
        up_ecef: versore della verticale locale in ECEF.
        min_elevation_rad: elevazione minima di utilizzo [rad].
    """

    name: str
    position_ecef: FloatArray
    up_ecef: FloatArray
    min_elevation_rad: float

    @classmethod
    def from_config(cls, station: GroundStationConfig) -> Self:
        """Costruisce la stazione dai parametri di configurazione.

        Args:
            station: sezione [ground_station] della configurazione.

        Returns:
            La stazione di terra.
        """
        latitude, longitude = station.latitude_rad, station.longitude_rad
        return cls(
            name=station.name,
            position_ecef=geodetic_to_ecef(latitude, longitude, station.altitude_m),
            up_ecef=geodetic_up(latitude, longitude),
            min_elevation_rad=station.min_elevation_rad,
        )

    def elevation_rad(self, satellite_ecef: FloatArray) -> float:
        """Elevazione del satellite sull'orizzonte della stazione.

        Args:
            satellite_ecef: posizione del satellite in ECEF [m].

        Returns:
            Angolo tra -pi/2 e pi/2 [rad].
        """
        line_of_sight = satellite_ecef - self.position_ecef
        return math.pi / 2 - angle_between(line_of_sight, self.up_ecef)

    def is_visible(self, satellite_ecef: FloatArray) -> bool:
        """Indica se il satellite è sopra l'elevazione minima.

        Args:
            satellite_ecef: posizione del satellite in ECEF [m].

        Returns:
            True se il satellite è utilizzabile dalla stazione.
        """
        return self.elevation_rad(satellite_ecef) >= self.min_elevation_rad

    def position_eci(self, jd: float) -> FloatArray:
        """Posizione della stazione in ECI.

        Args:
            jd: data giuliana [giorni].

        Returns:
            Posizione in ECI [m].
        """
        return eci_to_ecef_matrix(jd).T @ self.position_ecef
