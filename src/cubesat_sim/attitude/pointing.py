"""Controllore di puntamento: PD sul quaternione d'errore, con le ruote.

Errore d'assetto: q_err = q_rif^-1 * q_misurato, la rotazione che porta dal
riferimento al corpo. Il suo vettore di rotazione phi (asse per angolo) è
l'errore angolare, negli assi del corpo.

Errore di velocità: omega_misurata - omega_rif, con la velocità del riferimento
riportata negli assi del corpo. Senza questo termine il satellite inseguirebbe
un bersaglio in movimento con un ritardo di circa 2 zeta omega_rif / omega_n.

Legge di controllo: PD con limite di velocità (coppia richiesta sul corpo,
per asse):
    omega_cmd = -(omega_n / (2 zeta)) * phi,   ridotta in modulo a omega_max
    tau = -K_d * (omega - omega_rif - omega_cmd)
    K_p = I omega_n^2,   K_d = 2 zeta omega_n I
Finché |omega_cmd| < omega_max la legge coincide con il PD classico
tau = -K_p * phi - K_d * (omega - omega_rif): ogni asse si comporta come un
sistema del secondo ordine con pulsazione naturale omega_n e smorzamento zeta.
Con i valori di riferimento questo vale per errori sotto circa 14 gradi.

Per errori più grandi il satellite ruota a velocità costante omega_max attorno
all'asse dell'errore, il percorso angolare più breve, e poi si assesta come
sopra. Senza il limite una manovra di 180 gradi arriverebbe a circa 8 gradi/s:
supererebbe la soglia di ingresso in DETUMBLE e caricherebbe le ruote al 60 %.

Il controllore usa solo misure (star tracker e giroscopio), mai lo stato vero.
Termini giroscopici e accelerazione del riferimento non sono compensati: alle
velocità in gioco, sotto 1 grado/s, il loro effetto è trascurabile.
"""

import numpy as np

from cubesat_sim.core.quaternions import (
    quat_conjugate,
    quat_multiply,
    quat_to_matrix,
    quat_to_rotation_vector,
)
from cubesat_sim.core.types import FloatArray
from cubesat_sim.core.vectors import angle_between


class PointingController:
    """Controllore PD di puntamento con le ruote di reazione.

    Attributes:
        kp: guadagno proporzionale per asse, I omega_n^2 [N m/rad].
        kd: guadagno derivativo per asse, 2 zeta omega_n I [N m s/rad].
    """

    def __init__(
        self,
        inertia: FloatArray,
        natural_frequency_rad_s: float,
        damping_ratio: float,
        max_slew_rate_rad_s: float,
    ) -> None:
        """Ricava i guadagni da banda e smorzamento.

        Args:
            inertia: momenti principali di inerzia [kg m^2].
            natural_frequency_rad_s: pulsazione naturale desiderata [rad/s].
            damping_ratio: smorzamento desiderato.
            max_slew_rate_rad_s: velocità angolare massima comandata nelle
                manovre [rad/s].
        """
        self.kp = inertia * natural_frequency_rad_s**2
        self.kd = 2.0 * damping_ratio * natural_frequency_rad_s * inertia
        # K_p / K_d, uguale per tutti gli assi: trasforma l'errore angolare
        # nella velocità comandata.
        self._rate_gain = natural_frequency_rad_s / (2.0 * damping_ratio)
        self._max_rate = max_slew_rate_rad_s

    def torque(
        self,
        measured_attitude: FloatArray,
        measured_rate: FloatArray,
        reference_attitude: FloatArray,
        reference_rate: FloatArray,
    ) -> FloatArray:
        """Coppia richiesta sul corpo.

        Args:
            measured_attitude: quaternione di assetto misurato.
            measured_rate: velocità angolare misurata, assi corpo [rad/s].
            reference_attitude: quaternione di assetto di riferimento.
            reference_rate: velocità angolare del riferimento, nei suoi assi
                [rad/s].

        Returns:
            Coppia richiesta sul corpo, assi corpo [N m], prima della
            saturazione.
        """
        error = quat_multiply(quat_conjugate(reference_attitude), measured_attitude)
        rate_command = -self._rate_gain * quat_to_rotation_vector(error)
        command_norm = float(np.linalg.norm(rate_command))
        if command_norm > self._max_rate:
            rate_command = rate_command * (self._max_rate / command_norm)
        reference_rate_body = quat_to_matrix(error).T @ reference_rate
        rate_error = measured_rate - reference_rate_body
        return -self.kd * (rate_error - rate_command)


def pointing_error(
    attitude: FloatArray, body_axis: FloatArray, target_eci: FloatArray
) -> float:
    """Angolo tra un asse del corpo e la direzione su cui deve puntare.

    Ignora la rotazione attorno all'asse stesso: è l'errore che conta per una
    fotocamera o per un'antenna.

    Args:
        attitude: quaternione di assetto.
        body_axis: asse dello strumento, assi corpo.
        target_eci: direzione del bersaglio in ECI.

    Returns:
        Errore di puntamento, tra 0 e pi [rad].
    """
    return angle_between(quat_to_matrix(attitude) @ body_axis, target_eci)
