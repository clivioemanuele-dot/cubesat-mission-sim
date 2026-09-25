"""Batteria: energia immagazzinata, con rendimenti di carica e scarica.

La potenza netta al bus è la produzione solare meno i consumi, costante
durante ogni aggiornamento.
    Positiva: la batteria si carica e trattiene la frazione eta_carica. Se è
        piena, l'eccesso non si può immagazzinare ed è registrato come
        sprecato (nella realtà i pannelli vengono scollegati).
    Negativa: la batteria si scarica e, per fornire E al bus, cede
        E / eta_scarica. Se è vuota, la parte di consumo non coperta è
        registrata come non fornita.

Bilancio, verificato nei test, per ogni aggiornamento:
    energia prodotta - energia consumata = variazione della batteria
        + perdite di conversione + energia sprecata - energia non fornita

Stato di carica (SoC): energia immagazzinata divisa per la capacità, tra 0 e 1.
Nessun modello di tensione, temperatura o invecchiamento.
"""

from dataclasses import dataclass

from cubesat_sim.core.config import BatteryConfig


@dataclass(frozen=True)
class BatteryUpdate:
    """Esito di un aggiornamento della batteria.

    Attributes:
        stored_change_j: variazione dell'energia immagazzinata [J].
        loss_j: energia persa nella conversione, in carica o in scarica [J].
        wasted_j: energia prodotta ma non immagazzinabile, batteria piena [J].
        unmet_j: consumo non coperto, batteria vuota [J].
    """

    stored_change_j: float
    loss_j: float
    wasted_j: float
    unmet_j: float


class Battery:
    """Batteria con capacità fissa e rendimenti costanti.

    Attributes:
        capacity_j: capacità [J].
        energy_j: energia immagazzinata [J], tra 0 e capacity_j.
    """

    def __init__(self, config: BatteryConfig) -> None:
        """Prepara la batteria con lo stato di carica iniziale della configurazione.

        Args:
            config: sezione [battery] della configurazione.
        """
        self.capacity_j = config.capacity_j
        self.energy_j = config.initial_soc * config.capacity_j
        self._charge_efficiency = config.charge_efficiency
        self._discharge_efficiency = config.discharge_efficiency

    @property
    def state_of_charge(self) -> float:
        """Stato di carica, tra 0 e 1."""
        return self.energy_j / self.capacity_j

    def update(self, net_power_w: float, duration_s: float) -> BatteryUpdate:
        """Applica una potenza netta costante per un intervallo di tempo.

        Args:
            net_power_w: produzione meno consumi, al bus [W].
            duration_s: durata dell'intervallo [s].

        Returns:
            L'esito dell'aggiornamento, con perdite, sprechi e consumi non coperti.
        """
        bus_energy_j = net_power_w * duration_s
        before_j = self.energy_j
        if bus_energy_j >= 0.0:
            room_j = (self.capacity_j - self.energy_j) / self._charge_efficiency
            accepted_j = min(bus_energy_j, room_j)
            self.energy_j = min(
                self.energy_j + self._charge_efficiency * accepted_j, self.capacity_j
            )
            return BatteryUpdate(
                stored_change_j=self.energy_j - before_j,
                loss_j=(1.0 - self._charge_efficiency) * accepted_j,
                wasted_j=bus_energy_j - accepted_j,
                unmet_j=0.0,
            )
        demand_j = -bus_energy_j
        delivered_j = min(demand_j, self.energy_j * self._discharge_efficiency)
        drawn_j = delivered_j / self._discharge_efficiency
        self.energy_j = max(self.energy_j - drawn_j, 0.0)
        return BatteryUpdate(
            stored_change_j=self.energy_j - before_j,
            loss_j=drawn_j - delivered_j,
            wasted_j=0.0,
            unmet_j=demand_j - delivered_j,
        )
