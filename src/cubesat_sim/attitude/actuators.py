"""Attuatori d'assetto con saturazione: ruote di reazione e magnetorquer.

Ruote di reazione: il controllo chiede una coppia sul corpo; i motori
applicano alle ruote la coppia opposta. Due limiti fisici per ogni ruota:
    coppia massima: se una componente la supera, il vettore è ridotto in
        proporzione, così la direzione della coppia si conserva;
    momento angolare massimo: una ruota al limite non può accelerare oltre in
        quel verso. La coppia è limitata in modo che, mantenuta per un periodo
        di controllo, il momento resti entro il limite.

Magnetorquer: il controllo chiede un dipolo magnetico; se una componente
supera il massimo, il vettore è ridotto in proporzione. La coppia prodotta è
m x B (attitude/disturbances.py), calcolata con il campo vero.
"""

from dataclasses import dataclass

import numpy as np

from cubesat_sim.core.config import ActuatorsConfig
from cubesat_sim.core.types import FloatArray


def _scale_to_limit(vector: FloatArray, limit: float) -> tuple[FloatArray, bool]:
    """Riduce il vettore in proporzione se una sua componente supera il limite.

    Args:
        vector: vettore richiesto.
        limit: valore massimo ammesso per ogni componente.

    Returns:
        Coppia (vettore limitato, True se è stato ridotto).
    """
    largest = float(np.max(np.abs(vector)))
    if largest <= limit:
        return vector, False
    return vector * (limit / largest), True


@dataclass(frozen=True, eq=False)
class WheelCommand:
    """Coppia effettiva sulle ruote, dopo i limiti.

    Attributes:
        torque: coppia dei motori sulle ruote, assi corpo [N m]. Sul corpo
            agisce la coppia opposta.
        torque_saturated: True se la coppia richiesta superava il massimo.
        momentum_saturated: True se il limite di momento ha ridotto la coppia.
    """

    torque: FloatArray
    torque_saturated: bool
    momentum_saturated: bool


@dataclass(frozen=True, eq=False)
class DipoleCommand:
    """Dipolo effettivo dei magnetorquer, dopo il limite.

    Attributes:
        dipole: dipolo magnetico, assi corpo [A m^2].
        saturated: True se il dipolo richiesto superava il massimo.
    """

    dipole: FloatArray
    saturated: bool


class ReactionWheels:
    """Tre ruote di reazione allineate con gli assi del corpo."""

    def __init__(self, config: ActuatorsConfig, control_period_s: float) -> None:
        """Prepara le ruote.

        Args:
            config: sezione [actuators] della configurazione.
            control_period_s: durata di validità di ogni comando [s].
        """
        self._max_torque = config.wheel_max_torque_nm
        self._max_momentum = config.wheel_max_momentum_nms
        self._period_s = control_period_s

    def command(
        self, body_torque: FloatArray, wheel_momentum: FloatArray
    ) -> WheelCommand:
        """Coppia sulle ruote che realizza, entro i limiti, quella chiesta sul corpo.

        Args:
            body_torque: coppia richiesta dal controllo sul corpo, assi corpo [N m].
            wheel_momentum: momento angolare attuale delle ruote [N m s].

        Returns:
            Il comando effettivo, con l'indicazione dei limiti raggiunti.
        """
        torque, torque_saturated = _scale_to_limit(-body_torque, self._max_torque)
        upper = (self._max_momentum - wheel_momentum) / self._period_s
        lower = (-self._max_momentum - wheel_momentum) / self._period_s
        limited = np.clip(torque, lower, upper)
        return WheelCommand(
            torque=limited,
            torque_saturated=torque_saturated,
            momentum_saturated=bool(np.any(limited != torque)),
        )


class Magnetorquers:
    """Tre magnetorquer allineati con gli assi del corpo."""

    def __init__(self, config: ActuatorsConfig) -> None:
        """Prepara i magnetorquer.

        Args:
            config: sezione [actuators] della configurazione.
        """
        self._max_dipole = config.magnetorquer_max_dipole_am2

    def command(self, dipole: FloatArray) -> DipoleCommand:
        """Dipolo effettivo, entro il limite.

        Args:
            dipole: dipolo richiesto dal controllo, assi corpo [A m^2].

        Returns:
            Il comando effettivo, con l'indicazione del limite raggiunto.
        """
        limited, saturated = _scale_to_limit(dipole, self._max_dipole)
        return DipoleCommand(dipole=limited, saturated=saturated)
