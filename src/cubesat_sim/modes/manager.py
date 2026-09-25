"""Gestore dei modi di bordo: macchina a stati con priorità e isteresi.

A ogni passo di controllo il gestore riceve le condizioni misurate a bordo e
decide il modo. Regole:

- priorità decrescente: SAFE, DETUMBLE, DOWNLINK, NADIR, SUN_POINTING;
- isteresi su batteria e rotazione: la condizione si attiva oltre una soglia
  e si spegne solo oltre una seconda soglia, così il modo non oscilla quando
  la grandezza resta vicina al limite;
- tempo di conferma: un nuovo modo viene adottato solo se è richiesto senza
  interruzioni per ``confirmation_time_s``; una condizione breve non basta;
- ogni transizione viene registrata con istante, modi e causa.

Il gestore vede solo grandezze misurate, mai lo stato vero, e decide soltanto
il modo: quali carichi accendere e quale legge di controllo usare lo stabilisce
il ciclo di simulazione in base al modo.
"""

import math
from dataclasses import dataclass
from enum import StrEnum

from cubesat_sim.core.config import ModesConfig

_PERCENT_PER_FRACTION = 100.0


class Mode(StrEnum):
    """Modi di bordo, elencati in ordine di priorità decrescente.

    Ogni modo è anche una stringa (per esempio ``"SAFE"``), così si salva nei
    risultati senza conversioni.
    """

    SAFE = "SAFE"
    DETUMBLE = "DETUMBLE"
    DOWNLINK = "DOWNLINK"
    NADIR = "NADIR"
    SUN_POINTING = "SUN_POINTING"


# Posizione di ogni modo nell'elenco: numero più piccolo, priorità più alta.
_RANK: dict[Mode, int] = {mode: rank for rank, mode in enumerate(Mode)}


@dataclass(frozen=True)
class ModeInputs:
    """Condizioni misurate a bordo che il gestore usa per decidere.

    Attributes:
        state_of_charge: carica della batteria come frazione tra 0 e 1.
        angular_rate_rad_s: norma della velocità angolare misurata [rad/s].
        station_visible: la stazione di terra è sopra l'elevazione minima.
        imaging_window: il satellite è sopra la regione da fotografare e il
            suolo sotto di lui è illuminato.
    """

    state_of_charge: float
    angular_rate_rad_s: float
    station_visible: bool
    imaging_window: bool


@dataclass(frozen=True)
class ModeTransition:
    """Una transizione di modo registrata.

    Attributes:
        time_s: istante della transizione dall'inizio della simulazione [s].
        previous: modo di partenza.
        current: modo di arrivo.
        cause: descrizione leggibile del motivo.
    """

    time_s: float
    previous: Mode
    current: Mode
    cause: str


class ModeManager:
    """Macchina a stati dei modi di bordo.

    Due condizioni hanno memoria (isteresi) e vengono aggiornate a ogni
    chiamata, qualunque sia il modo: batteria scarica e rotazione elevata.
    Il modo richiesto è quello a priorità più alta tra le condizioni attive e
    diventa il modo attuale dopo il tempo di conferma.
    """

    def __init__(self, config: ModesConfig) -> None:
        """Crea il gestore nel modo DETUMBLE.

        Dopo il rilascio il satellite parte sempre smorzando la rotazione: è
        la sequenza standard dei CubeSat. Se la rotazione è già bassa, il
        gestore passa al modo successivo dopo il tempo di conferma.

        Args:
            config: soglie e tempo di conferma dei modi.
        """
        self._config = config
        self._mode = Mode.DETUMBLE
        self._low_battery = False
        self._tumbling = True
        self._candidate: Mode | None = None
        self._candidate_since_s = 0.0
        self._transitions: list[ModeTransition] = []

    @property
    def mode(self) -> Mode:
        """Modo attuale."""
        return self._mode

    @property
    def tumbling(self) -> bool:
        """Rotazione sopra soglia, valutata con isteresi in qualunque modo.

        Serve in SAFE: se il satellite ruota ancora velocemente, il puntamento
        al Sole con le ruote non è possibile e va usato il B-dot.
        """
        return self._tumbling

    @property
    def transitions(self) -> tuple[ModeTransition, ...]:
        """Transizioni registrate finora, in ordine di tempo."""
        return tuple(self._transitions)

    def update(self, time_s: float, inputs: ModeInputs) -> Mode:
        """Aggiorna il modo con le condizioni misurate all'istante ``time_s``.

        Args:
            time_s: istante dall'inizio della simulazione [s].
            inputs: condizioni misurate a bordo.

        Returns:
            Il modo da applicare fino alla prossima chiamata.
        """
        self._update_conditions(inputs)
        wanted = self._wanted_mode(inputs)
        if wanted is self._mode:
            self._candidate = None
            return self._mode
        if wanted is not self._candidate:
            self._candidate = wanted
            self._candidate_since_s = time_s
        if time_s - self._candidate_since_s >= self._config.confirmation_time_s:
            self._switch(time_s, wanted, inputs)
        return self._mode

    def _update_conditions(self, inputs: ModeInputs) -> None:
        """Aggiorna con isteresi le condizioni di batteria scarica e rotazione."""
        config = self._config
        if inputs.state_of_charge < config.safe_enter_soc:
            self._low_battery = True
        elif inputs.state_of_charge >= config.safe_exit_soc:
            self._low_battery = False
        if inputs.angular_rate_rad_s > config.detumble_enter_rate_rad_s:
            self._tumbling = True
        elif inputs.angular_rate_rad_s < config.detumble_exit_rate_rad_s:
            self._tumbling = False

    def _wanted_mode(self, inputs: ModeInputs) -> Mode:
        """Restituisce il modo a priorità più alta tra le condizioni attive."""
        if self._low_battery:
            return Mode.SAFE
        if self._tumbling:
            return Mode.DETUMBLE
        if inputs.station_visible:
            return Mode.DOWNLINK
        if inputs.imaging_window:
            return Mode.NADIR
        return Mode.SUN_POINTING

    def _switch(self, time_s: float, new_mode: Mode, inputs: ModeInputs) -> None:
        """Adotta il nuovo modo e registra la transizione con la sua causa."""
        transition = ModeTransition(
            time_s=time_s,
            previous=self._mode,
            current=new_mode,
            cause=_cause(self._mode, new_mode, inputs),
        )
        self._transitions.append(transition)
        self._mode = new_mode
        self._candidate = None


def _cause(previous: Mode, current: Mode, inputs: ModeInputs) -> str:
    """Descrive il motivo di una transizione.

    Se il nuovo modo ha priorità più alta si riporta la condizione che lo
    attiva; altrimenti la condizione che ha fatto uscire dal modo precedente.
    """
    soc_pct = inputs.state_of_charge * _PERCENT_PER_FRACTION
    rate_deg_s = math.degrees(inputs.angular_rate_rad_s)
    entering = {
        Mode.SAFE: f"batteria scarica (carica al {soc_pct:.1f} %)",
        Mode.DETUMBLE: f"rotazione elevata ({rate_deg_s:.2f} °/s)",
        Mode.DOWNLINK: "stazione di terra visibile",
        Mode.NADIR: "inizio della finestra di acquisizione",
        Mode.SUN_POINTING: "nessuna condizione attiva",
    }
    leaving = {
        Mode.SAFE: f"batteria ricaricata (carica al {soc_pct:.1f} %)",
        Mode.DETUMBLE: f"rotazione smorzata ({rate_deg_s:.2f} °/s)",
        Mode.DOWNLINK: "stazione di terra non più visibile",
        Mode.NADIR: "fine della finestra di acquisizione",
        Mode.SUN_POINTING: "nessuna condizione attiva",
    }
    if _RANK[current] < _RANK[previous]:
        return entering[current]
    return leaving[previous]
