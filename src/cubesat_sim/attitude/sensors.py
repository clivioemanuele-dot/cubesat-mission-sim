"""Sensori d'assetto: misure con rumore dello stato vero.

Il controllo vede solo queste misure, mai lo stato vero simulato: è la
separazione tra stato vero, stato misurato e comandi richiesta dal progetto.

Modelli (rumore gaussiano bianco, indipendente per asse, con la deviazione
standard della configurazione; nessun bias e nessun filtro di stima):
    star tracker: q_misurato = q_vero * dq, con dq piccola rotazione casuale
        attorno a un asse del corpo;
    giroscopio: omega_misurata = omega_vera + rumore;
    magnetometro: campo vero espresso nel corpo + rumore. Il campo prodotto
        dai magnetorquer sul magnetometro è trascurato.

Il rumore viene da un generatore numpy con seme esplicito: a parità di seme le
misure sono identiche.
"""

from dataclasses import dataclass

import numpy as np

from cubesat_sim.attitude.dynamics import AttitudeState
from cubesat_sim.core.config import SensorsConfig
from cubesat_sim.core.quaternions import (
    quat_from_rotation_vector,
    quat_multiply,
    quat_to_matrix,
)
from cubesat_sim.core.types import FloatArray


@dataclass(frozen=True, eq=False)
class Measurements:
    """Misure dei sensori a un istante: tutto ciò che il controllo può vedere.

    Attributes:
        attitude: quaternione di assetto misurato [w, x, y, z].
        angular_velocity: velocità angolare misurata, assi corpo [rad/s].
        magnetic_field: campo magnetico misurato, assi corpo [T].
    """

    attitude: FloatArray
    angular_velocity: FloatArray
    magnetic_field: FloatArray


class Sensors:
    """Star tracker, giroscopio e magnetometro con rumore gaussiano."""

    def __init__(self, config: SensorsConfig, rng: np.random.Generator) -> None:
        """Prepara i sensori.

        Args:
            config: sezione [sensors] della configurazione.
            rng: generatore di numeri casuali con seme esplicito.
        """
        self._attitude_sigma = config.attitude_noise_rad
        self._rate_sigma = config.rate_noise_rad_s
        self._field_sigma = config.magnetometer_noise_t
        self._rng = rng

    def measure(self, state: FloatArray, field_eci: FloatArray) -> Measurements:
        """Misura lo stato vero.

        Args:
            state: vettore di stato vero [q, omega, h].
            field_eci: campo magnetico vero in ECI, in tesla.

        Returns:
            Le misure dei tre sensori.
        """
        truth = AttitudeState.from_vector(state)
        attitude_noise = self._rng.normal(scale=self._attitude_sigma, size=3)
        rate_noise = self._rng.normal(scale=self._rate_sigma, size=3)
        field_noise = self._rng.normal(scale=self._field_sigma, size=3)
        field_body = quat_to_matrix(truth.attitude).T @ field_eci
        attitude_error = quat_from_rotation_vector(attitude_noise)
        return Measurements(
            attitude=quat_multiply(truth.attitude, attitude_error),
            angular_velocity=truth.angular_velocity + rate_noise,
            magnetic_field=field_body + field_noise,
        )
