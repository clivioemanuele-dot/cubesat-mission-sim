"""Controllore B-dot per smorzare la rotazione iniziale (detumble).

Principio: se il satellite ruota, il campo magnetico terrestre visto dal corpo
cambia direzione. Il B-dot comanda ai magnetorquer un dipolo opposto a questa
variazione: la coppia m x B che ne risulta frena la rotazione, qualunque sia
l'assetto. Servono solo magnetometro e magnetorquer.

Legge (forma normalizzata):
    m = -(k / |B|) * db/dt,   con b = B / |B|
con la derivata stimata dalla differenza di due misure consecutive. La coppia
risultante vale circa -k * omega_perp: frena la rotazione perpendicolare al
campo.

Guadagno, ricavato da orbita e inerzia (Avanzini e Giulietti, 2012):
    k = 2 n (1 + sin i) I_min
con n velocità angolare orbitale, i inclinazione dell'orbita (usata come
approssimazione dell'inclinazione rispetto all'equatore geomagnetico) e I_min
momento di inerzia minimo.

La velocità angolare non va a zero: tende a circa il doppio della velocità
orbitale (0,13 gradi/s), perché il campo ruota anche per il moto lungo l'orbita.
"""

import math

import numpy as np

from cubesat_sim.core.types import FloatArray


def bdot_gain(
    mean_motion_rad_s: float, inclination_rad: float, min_inertia_kg_m2: float
) -> float:
    """Guadagno del B-dot, k = 2 n (1 + sin i) I_min.

    Args:
        mean_motion_rad_s: velocità angolare orbitale [rad/s].
        inclination_rad: inclinazione dell'orbita [rad].
        min_inertia_kg_m2: momento di inerzia minimo del satellite [kg m^2].

    Returns:
        Guadagno k [N m s].
    """
    return (
        2.0 * mean_motion_rad_s * (1.0 + math.sin(inclination_rad)) * min_inertia_kg_m2
    )


class BdotController:
    """Controllore B-dot a tempo discreto, con memoria della misura precedente."""

    def __init__(self, gain_nms: float, control_period_s: float) -> None:
        """Prepara il controllore.

        Args:
            gain_nms: guadagno k [N m s], da bdot_gain.
            control_period_s: intervallo tra due misure consecutive [s].
        """
        self._gain = gain_nms
        self._period_s = control_period_s
        self._previous_direction: FloatArray | None = None

    def reset(self) -> None:
        """Dimentica la misura precedente: da chiamare all'ingresso in DETUMBLE."""
        self._previous_direction = None

    def dipole(self, measured_field_body: FloatArray) -> FloatArray:
        """Dipolo richiesto ai magnetorquer.

        Alla prima chiamata, o dopo un reset, non esiste ancora una misura
        precedente: il dipolo è nullo.

        Args:
            measured_field_body: campo magnetico misurato, assi corpo [T].

        Returns:
            Dipolo richiesto, assi corpo [A m^2], prima della saturazione.
        """
        field_norm = float(np.linalg.norm(measured_field_body))
        direction = measured_field_body / field_norm
        previous = self._previous_direction
        self._previous_direction = direction
        if previous is None:
            return np.zeros(3)
        direction_rate = (direction - previous) / self._period_s
        return -(self._gain / field_norm) * direction_rate
