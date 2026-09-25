"""Orbita circolare, riferimento orbitale LVLH e angolo beta.

L'orbita è kepleriana e circolare, senza perturbazioni: niente J2 e niente
resistenza aerodinamica. In 24 ore la rotazione del piano orbitale dovuta a J2
(circa 1 grado) è trascurata.

Riferimento LVLH (Local Vertical, Local Horizontal):
    z verso il centro della Terra (nadir);
    y opposto al momento angolare dell'orbita;
    x completa la terna destra: in un'orbita circolare è lungo la velocità.
"""

import datetime as dt
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Self

import numpy as np

from cubesat_sim.core.config import OrbitConfig
from cubesat_sim.core.constants import EARTH_MU_M3_S2, EARTH_RADIUS_M, SECONDS_PER_DAY
from cubesat_sim.core.rotations import rot_x, rot_z
from cubesat_sim.core.types import FloatArray
from cubesat_sim.core.vectors import angle_between, unit
from cubesat_sim.environment.earth import julian_date
from cubesat_sim.environment.sun import sun_position_eci

_HALF_DAY_S = SECONDS_PER_DAY / 2.0


def raan_from_descending_node_time(
    ltdn_s: float, sun_right_ascension_rad: float
) -> float:
    """Ascensione retta del nodo ascendente dall'ora locale del nodo discendente.

    L'ora solare locale di un punto dell'equatore celeste vale 12 h più il suo
    angolo dal Sole, a 15 gradi all'ora. Il nodo ascendente è a 12 h dal
    discendente.

    Args:
        ltdn_s: ora solare locale al nodo discendente [s dalla mezzanotte].
        sun_right_ascension_rad: ascensione retta del Sole [rad].

    Returns:
        Ascensione retta del nodo ascendente, tra 0 e 2 pi [rad].
    """
    ltan_s = (ltdn_s + _HALF_DAY_S) % SECONDS_PER_DAY
    angle_from_sun = 2.0 * math.pi * (ltan_s - _HALF_DAY_S) / SECONDS_PER_DAY
    return (sun_right_ascension_rad + angle_from_sun) % (2.0 * math.pi)


@dataclass(frozen=True)
class CircularOrbit:
    """Orbita circolare kepleriana.

    Attributes:
        radius_m: raggio dell'orbita [m].
        inclination_rad: inclinazione [rad].
        raan_rad: ascensione retta del nodo ascendente [rad].
        initial_argument_of_latitude_rad: angolo dal nodo ascendente
            all'istante iniziale [rad].
    """

    radius_m: float
    inclination_rad: float
    raan_rad: float
    initial_argument_of_latitude_rad: float

    @classmethod
    def from_config(cls, orbit: OrbitConfig, epoch: dt.datetime) -> Self:
        """Costruisce l'orbita dai parametri di configurazione.

        Il piano orbitale è orientato in base alla posizione del Sole
        all'istante iniziale e all'ora locale del nodo discendente.

        Args:
            orbit: sezione [orbit] della configurazione.
            epoch: istante iniziale della simulazione.

        Returns:
            L'orbita, con il satellite nella posizione iniziale all'epoca.
        """
        sun = sun_position_eci(julian_date(epoch))
        raan = raan_from_descending_node_time(
            orbit.descending_node_local_time_s, math.atan2(sun[1], sun[0])
        )
        return cls(
            radius_m=EARTH_RADIUS_M + orbit.altitude_m,
            inclination_rad=orbit.inclination_rad,
            raan_rad=raan,
            initial_argument_of_latitude_rad=orbit.initial_argument_of_latitude_rad,
        )

    @cached_property
    def mean_motion_rad_s(self) -> float:
        """Velocità angolare lungo l'orbita [rad/s]."""
        return math.sqrt(EARTH_MU_M3_S2 / self.radius_m**3)

    @property
    def period_s(self) -> float:
        """Periodo orbitale [s]."""
        return 2.0 * math.pi / self.mean_motion_rad_s

    @cached_property
    def orbit_to_eci(self) -> FloatArray:
        """Matrice dal piano orbitale (x sul nodo ascendente, z normale) a ECI."""
        return (rot_x(self.inclination_rad) @ rot_z(self.raan_rad)).T

    def state_eci(self, elapsed_s: float) -> tuple[FloatArray, FloatArray]:
        """Posizione [m] e velocità [m/s] del satellite in ECI.

        Args:
            elapsed_s: secondi trascorsi dall'istante iniziale.

        Returns:
            Coppia (posizione, velocità).
        """
        u = self.initial_argument_of_latitude_rad + self.mean_motion_rad_s * elapsed_s
        cos_u, sin_u = math.cos(u), math.sin(u)
        radius = self.radius_m
        speed = radius * self.mean_motion_rad_s
        in_plane_position = np.array([radius * cos_u, radius * sin_u, 0.0])
        in_plane_velocity = np.array([-speed * sin_u, speed * cos_u, 0.0])
        matrix = self.orbit_to_eci
        return matrix @ in_plane_position, matrix @ in_plane_velocity


def eci_to_lvlh_matrix(position: FloatArray, velocity: FloatArray) -> FloatArray:
    """Matrice di passaggio da ECI al riferimento LVLH.

    Args:
        position: posizione del satellite in ECI [m].
        velocity: velocità del satellite in ECI [m/s].

    Returns:
        Matrice 3x3 tale che v_lvlh = matrice @ v_eci. Le righe sono gli assi
        LVLH espressi in ECI.
    """
    z_axis = -unit(position)
    y_axis = -unit(np.cross(position, velocity))
    x_axis = np.cross(y_axis, z_axis)
    return np.vstack([x_axis, y_axis, z_axis])


def beta_angle(
    position: FloatArray, velocity: FloatArray, sun_position: FloatArray
) -> float:
    """Angolo beta: elevazione del Sole rispetto al piano dell'orbita.

    È positivo se il Sole sta dalla parte del momento angolare dell'orbita.

    Args:
        position: posizione del satellite in ECI [m].
        velocity: velocità del satellite in ECI [m/s].
        sun_position: posizione del Sole in ECI [m].

    Returns:
        Angolo tra -pi/2 e pi/2 [rad].
    """
    normal = np.cross(position, velocity)
    return math.pi / 2 - angle_between(normal, sun_position)
