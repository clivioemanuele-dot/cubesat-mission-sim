"""Consumi elettrici dei sottosistemi.

    computer di bordo ed elettronica ADCS: sempre accesi;
    ruote di reazione: potenza massima per ruota moltiplicata per la frazione
        di coppia usata (|coppia| / coppia massima), sommata sulle tre ruote;
    magnetorquer: lo stesso, con il dipolo;
    trasmettitore: acceso solo durante lo scarico dati;
    fotocamera: accesa solo durante l'acquisizione.

Quali carichi sono accesi lo decide il gestore dei modi (in SAFE trasmettitore
e fotocamera sono spenti): qui si calcola solo quanto consumano.

Il consumo delle ruote è un modello semplice e dichiarato: proporzionale alla
coppia, senza dipendenza dalla velocità di rotazione e senza attriti.
"""

from dataclasses import dataclass

import numpy as np

from cubesat_sim.core.config import ActuatorsConfig, LoadsConfig
from cubesat_sim.core.types import FloatArray


@dataclass(frozen=True)
class PowerConsumption:
    """Consumi per sottosistema.

    Attributes:
        platform_w: computer di bordo ed elettronica ADCS [W].
        actuators_w: ruote di reazione e magnetorquer [W].
        transmitter_w: trasmettitore [W].
        camera_w: fotocamera [W].
    """

    platform_w: float
    actuators_w: float
    transmitter_w: float
    camera_w: float

    @property
    def total_w(self) -> float:
        """Consumo totale [W]."""
        return self.platform_w + self.actuators_w + self.transmitter_w + self.camera_w


class PowerLoads:
    """Modello dei consumi dei sottosistemi."""

    def __init__(self, loads: LoadsConfig, actuators: ActuatorsConfig) -> None:
        """Prepara il modello dalla configurazione.

        Args:
            loads: sezione [loads] della configurazione.
            actuators: sezione [actuators], per i limiti di coppia e di dipolo.
        """
        self._loads = loads
        self._max_torque = actuators.wheel_max_torque_nm
        self._max_dipole = actuators.magnetorquer_max_dipole_am2

    def consumption(
        self,
        wheel_torque: FloatArray,
        dipole: FloatArray,
        *,
        transmitting: bool,
        imaging: bool,
    ) -> PowerConsumption:
        """Consumi con gli attuatori e i carichi indicati.

        Args:
            wheel_torque: coppia effettiva sulle ruote, assi corpo [N m].
            dipole: dipolo effettivo dei magnetorquer, assi corpo [A m^2].
            transmitting: True se il trasmettitore è acceso.
            imaging: True se la fotocamera è accesa.

        Returns:
            I consumi per sottosistema.
        """
        loads = self._loads
        wheel_use = float(np.sum(np.abs(wheel_torque))) / self._max_torque
        torquer_use = float(np.sum(np.abs(dipole))) / self._max_dipole
        return PowerConsumption(
            platform_w=loads.obc_w + loads.adcs_base_w,
            actuators_w=loads.wheel_max_power_w * wheel_use
            + loads.magnetorquer_max_power_w * torquer_use,
            transmitter_w=loads.transmitter_w if transmitting else 0.0,
            camera_w=loads.camera_w if imaging else 0.0,
        )
