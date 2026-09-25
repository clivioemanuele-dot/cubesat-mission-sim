"""Testi esplicativi della dashboard, costruiti dai risultati salvati.

Funzioni pure che restituiscono testo Markdown: la pagina (app.py) decide
soltanto dove mostrarlo. I testi descrivono ciò che il simulatore ha già
deciso e salvato, con i numeri della configurazione: qui non si ricalcola
nessuna logica di bordo.
"""

import datetime as dt
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import pandas as pd

from cubesat_sim.core.config import MissionConfig
from cubesat_sim.core.constants import EARTH_ROTATION_RATE_RAD_S
from cubesat_sim.environment.orbit import CircularOrbit
from cubesat_sim.modes.manager import Mode

_S_PER_MIN = 60
_MIN_PER_H = 60
_BIT_PER_GBIT = 1.0e9
_PERCENT = 100.0
_HALF_TURN_DEG = 180.0
# Oltre questa carica la batteria è piena: l'energia in più non entra.
_FULL_CHARGE = 0.999
# Tempo di assestamento al 2 % di un sistema del secondo ordine: 4 / (zeta wn).
_SETTLING_FACTOR = 4.0


@dataclass(frozen=True)
class _Instant:
    """Grandezze salvate di un istante, nelle unità usate nei testi.

    Gli errori valgono NaN quando non sono definiti: in DETUMBLE non c'è un
    puntamento, e l'errore dell'antenna esiste solo con la stazione visibile.
    """

    time_s: float
    mode: Mode
    rate_deg_s: float
    pointing_error_deg: float
    antenna_error_deg: float
    in_eclipse: bool
    imaging_window: bool
    state_of_charge: float
    solar_w: float
    consumed_w: float
    stored_gbit: float
    wheel_fill: float

    @classmethod
    def from_series(
        cls, series: pd.DataFrame, index: int, config: MissionConfig
    ) -> Self:
        """Legge la riga ``index`` delle serie temporali salvate."""
        row = series.iloc[index]
        momentum = max(abs(float(row[f"wheel_momentum_{a}_nms"])) for a in "xyz")
        return cls(
            time_s=float(row["time_s"]),
            mode=Mode(str(row["mode"])),
            rate_deg_s=math.degrees(float(row["angular_rate_rad_s"])),
            pointing_error_deg=math.degrees(float(row["pointing_error_rad"])),
            antenna_error_deg=math.degrees(float(row["antenna_error_rad"])),
            in_eclipse=bool(row["in_eclipse"]),
            imaging_window=bool(row["imaging_window"]),
            state_of_charge=float(row["state_of_charge"]),
            solar_w=float(row["solar_power_w"]),
            consumed_w=float(row["consumed_power_w"]),
            stored_gbit=float(row["data_stored_bit"]) / _BIT_PER_GBIT,
            wheel_fill=momentum / config.actuators.wheel_max_momentum_nms,
        )


def _duration(seconds: float) -> str:
    """Durata leggibile: secondi, minuti oppure ore e minuti."""
    if seconds < _S_PER_MIN:
        return f"{seconds:.0f} s"
    minutes = round(seconds / _S_PER_MIN)
    if minutes < _MIN_PER_H:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, _MIN_PER_H)
    return f"{hours} h {minutes:02d} min"


def _clock(config: MissionConfig, time_s: float) -> str:
    """Ora UTC di un istante della simulazione."""
    moment = config.simulation.start_epoch + dt.timedelta(seconds=time_s)
    return f"{moment.astimezone(dt.UTC):%H:%M:%S} UTC"


def _detumble(now: _Instant, config: MissionConfig) -> str:
    modes = config.modes
    return (
        "**DETUMBLE: smorzamento della rotazione.** Il satellite ruota a "
        f"{now.rate_deg_s:.2f} °/s. I magnetorquer applicano la legge B-dot: "
        "creano un dipolo magnetico che si oppone alla variazione del campo "
        "terrestre misurato, e la coppia che nasce frena la rotazione. Le ruote "
        "non lavorano e non c'è un assetto da inseguire, quindi l'errore di "
        "puntamento non è definito. Il modo finisce "
        f"{modes.confirmation_time_s:g} s dopo che la velocità misurata è scesa "
        f"sotto {modes.detumble_exit_rate_deg_s:g} °/s, e torna solo sopra "
        f"{modes.detumble_enter_rate_deg_s:g} °/s: la distanza tra le due soglie "
        "(isteresi) evita cambi di modo continui."
    )


def _safe(now: _Instant, config: MissionConfig) -> str:
    modes = config.modes
    text = (
        "**SAFE: batteria scarica.** È scattato quando la carica è scesa sotto "
        f"il {modes.safe_enter_soc_pct:g} %: fotocamera e trasmettitore sono "
        "spenti e il satellite orienta i pannelli verso il Sole per ricaricarsi "
        "(se ruota ancora troppo in fretta, prima smorza la rotazione con il "
        f"B-dot). Errore dei pannelli verso il Sole: {now.pointing_error_deg:.1f}°. "
        f"Si esce solo sopra il {modes.safe_exit_soc_pct:g} %: con due soglie "
        "diverse (isteresi) il modo non si accende e spegne di continuo."
    )
    if now.in_eclipse:
        text += (
            " Ora il satellite è in ombra: la carica scende fino all'uscita "
            "dall'eclissi."
        )
    return text


def _downlink(now: _Instant, config: MissionConfig) -> str:
    payload, station = config.payload, config.ground_station
    half_deg = math.degrees(payload.antenna_half_beamwidth_rad)
    text = (
        f"**DOWNLINK: scarico dati verso {station.name}.** La stazione vede il "
        f"satellite sopra {station.min_elevation_deg:g}° di elevazione. Le ruote "
        "orientano l'antenna verso la stazione e la inseguono per tutto il "
        "passaggio, più in fretta quando il satellite le passa sopra. "
    )
    if math.isnan(now.antenna_error_deg):
        text += (
            "La stazione è appena scesa sotto l'elevazione minima: il modo cambia "
            f"dopo il tempo di conferma di {config.modes.confirmation_time_s:g} s."
        )
    elif now.antenna_error_deg <= half_deg:
        text += (
            f"Errore dell'antenna {now.antenna_error_deg:.1f}°, entro metà del "
            f"fascio ({half_deg:g}°): i dati scendono a "
            f"{payload.downlink_rate_mbps:g} Mbit/s."
        )
    else:
        text += (
            f"Errore dell'antenna {now.antenna_error_deg:.1f}°, oltre metà del "
            f"fascio ({half_deg:g}°): il trasmettitore è acceso "
            f"({config.loads.transmitter_w:g} W) ma i dati non passano. Succede "
            "durante la manovra iniziale, finché l'antenna non raggiunge la "
            "stazione."
        )
    return text + f" In memoria restano {now.stored_gbit:.3f} Gbit."


def _nadir(now: _Instant, config: MissionConfig) -> str:
    payload = config.payload
    text = f"**NADIR: acquisizione immagini su {config.imaging.region_name}.** "
    if not now.imaging_window:
        text += (
            "La finestra di acquisizione si è appena chiusa: il modo cambia dopo "
            f"il tempo di conferma di {config.modes.confirmation_time_s:g} s. "
        )
    return text + (
        "La fotocamera punta il centro della Terra (errore "
        f"{now.pointing_error_deg:.2f}°) e registra "
        f"{payload.camera_data_rate_mbps:g} Mbit/s, con "
        f"{config.loads.camera_w:g} W di consumo in più. In memoria "
        f"{now.stored_gbit:.3f} Gbit su {payload.storage_capacity_gbit:g}. Con "
        "la fotocamera verso la Terra i pannelli non guardano più il Sole e la "
        "potenza prodotta cala: è il costo misurato dallo studio di compromesso 1."
    )


def _sun_pointing(now: _Instant, config: MissionConfig) -> str:
    text = (
        "**SUN_POINTING: pannelli verso il Sole.** È il modo di riposo, scelto "
        "quando nessun altro è richiesto: batteria carica, rotazione smorzata, "
        "stazione non visibile e nessuna finestra di acquisizione. Le ruote "
        "tengono i pannelli verso il Sole con un errore di "
        f"{now.pointing_error_deg:.2f}°, per produrre la massima potenza."
    )
    if now.in_eclipse:
        text += (
            " Ora il satellite è in ombra: mantiene comunque l'assetto, così "
            "all'uscita dall'eclissi i pannelli producono subito."
        )
    return text


# Stessa firma per tutti i modi, anche se non tutti usano la configurazione.
_MODE_TEXTS: dict[Mode, Callable[[_Instant, MissionConfig], str]] = {
    Mode.SAFE: _safe,
    Mode.DETUMBLE: _detumble,
    Mode.DOWNLINK: _downlink,
    Mode.NADIR: _nadir,
    Mode.SUN_POINTING: _sun_pointing,
}


def _power(now: _Instant) -> str:
    net_w = now.solar_w - now.consumed_w
    if net_w < 0.0:
        balance = f"la batteria copre i {-net_w:.1f} W mancanti"
    elif now.state_of_charge >= _FULL_CHARGE:
        balance = "la batteria è piena e l'energia in più va persa"
    else:
        balance = f"i {net_w:.1f} W in più ricaricano la batteria"
    light = "in eclissi" if now.in_eclipse else "al Sole"
    return (
        f"**Potenza** ({light}): i pannelli producono {now.solar_w:.1f} W e i "
        f"carichi ne consumano {now.consumed_w:.1f} W: {balance}."
    )


def _wheels(now: _Instant) -> str:
    return (
        f"**Ruote**: la più carica è al {_PERCENT * now.wheel_fill:.0f} % del suo "
        "momento angolare massimo. I disturbi esterni lo fanno crescere e qui non "
        "viene scaricato (limite dichiarato): al 100 % la ruota satura."
    )


def _transitions(
    now: _Instant, config: MissionConfig, transitions: list[dict[str, Any]]
) -> str:
    past = [t for t in transitions if float(t["time_s"]) <= now.time_s]
    future = [t for t in transitions if float(t["time_s"]) > now.time_s]
    if past:
        last = past[-1]
        last_s = float(last["time_s"])
        text = (
            f"**Ultima transizione** alle {_clock(config, last_s)}, "
            f"{_duration(now.time_s - last_s)} fa: {last['previous']} → "
            f"{last['current']}; causa: {last['cause']}."
        )
    else:
        text = (
            "**Nessuna transizione finora**: il satellite è nel modo iniziale "
            "dal rilascio."
        )
    if not future:
        return text + " Nessun'altra transizione fino alla fine della simulazione."
    following = future[0]
    following_s = float(following["time_s"])
    return text + (
        f" **Prossima** alle {_clock(config, following_s)}, tra "
        f"{_duration(following_s - now.time_s)}: {following['previous']} → "
        f"{following['current']}; causa: {following['cause']}."
    )


def current_activity(
    series: pd.DataFrame,
    index: int,
    config: MissionConfig,
    transitions: list[dict[str, Any]],
) -> str:
    """Spiega che cosa sta facendo il satellite nell'istante scelto.

    Args:
        series: serie temporali salvate dalla simulazione.
        index: riga di ``series`` dell'istante scelto.
        config: configurazione della simulazione.
        transitions: transizioni di modo salvate nei metadati.

    Returns:
        Testo Markdown: il modo con i numeri dell'istante, il bilancio di
        potenza, le ruote (tranne che in DETUMBLE), l'ultima e la prossima
        transizione.
    """
    now = _Instant.from_series(series, index, config)
    paragraphs = [_MODE_TEXTS[now.mode](now, config), _power(now)]
    if now.mode is not Mode.DETUMBLE:
        paragraphs.append(_wheels(now))
    paragraphs.append(_transitions(now, config, transitions))
    return "\n\n".join(paragraphs)


def ground_track_help(config: MissionConfig) -> str:
    """Come leggere la traccia a terra."""
    orbit, station = config.orbit, config.ground_station
    period_s = CircularOrbit.from_config(orbit, config.simulation.start_epoch).period_s
    shift_deg = math.degrees(EARTH_ROTATION_RATE_RAD_S * period_s)
    max_latitude_deg = min(
        orbit.inclination_deg, _HALF_TURN_DEG - orbit.inclination_deg
    )
    return (
        "La mappa mostra il punto della superficie che si trova sotto il "
        "satellite. Linea grigia: traccia dell'intera simulazione; linea blu: "
        "ultima orbita fino all'istante scelto; cerchio bianco: il satellite.\n\n"
        f"- Un'orbita a {orbit.altitude_km:g} km dura "
        f"{period_s / _S_PER_MIN:.1f} min. Nel frattempo la Terra ruota verso est "
        f"di {shift_deg:.1f}°: per questo ogni passaggio cade più a ovest del "
        "precedente.\n"
        f"- Con un'inclinazione di {orbit.inclination_deg:g}° l'orbita è quasi "
        f"polare e arriva a {max_latitude_deg:.1f}° di latitudine nord e sud. È "
        "eliosincrona: il satellite attraversa l'equatore verso sud sempre alla "
        f"stessa ora solare locale ({orbit.descending_node_local_time:%H:%M}), "
        "quindi con un'illuminazione simile a ogni passaggio.\n"
        f"- Punti verde acqua: istanti in cui {station.name} (triangolo) vede il "
        f"satellite sopra {station.min_elevation_deg:g}° di elevazione. Solo lì "
        "si possono scaricare i dati.\n"
        "- Rettangolo arancione: la regione di acquisizione "
        f"({config.imaging.region_name}). La fotocamera lavora quando il "
        "satellite la sorvola ed è al Sole.\n\n"
        "Rotella per lo zoom, trascinamento per spostarsi: la vista resta quando "
        "si muove il cursore del tempo."
    )


def attitude_help(config: MissionConfig) -> str:
    """Come leggere la vista 3D dell'assetto."""
    x_cm, y_cm, z_cm = config.satellite.size_cm
    payload = config.payload
    if payload.camera_boresight_body == payload.antenna_boresight_body:
        colors = (
            "pannelli e Sole in arancione; fotocamera e antenna stanno sulla "
            "stessa faccia e hanno un'unica freccia blu, da allineare con la Terra "
            "(blu) in NADIR e con la stazione (verde acqua) in DOWNLINK"
        )
    else:
        colors = (
            "pannelli e Sole in arancione, fotocamera e Terra in blu, antenna e "
            "stazione in verde acqua"
        )
    return (
        "Il satellite visto da chi gli vola accanto, nel riferimento orbitale "
        "(LVLH): si muove con il satellite e tiene la Terra sempre in basso. "
        "Assi: *moto* nel verso della velocità, *zenit* verso l'alto, *normale* "
        "perpendicolare al piano dell'orbita.\n\n"
        f"- Parallelepipedo grigio: il corpo del CubeSat ({x_cm:g} x {y_cm:g} x "
        f"{z_cm:g} cm); superfici blu: le due ali solari (disegno indicativo).\n"
        "- Frecce spesse: assi del satellite, che ruotano con l'assetto vero. "
        "Frecce sottili: direzioni del Sole, della Terra e della stazione "
        "(quest'ultima solo quando è visibile). In eclissi la legenda lo segnala: "
        "la freccia indica comunque dove si trova il Sole.\n"
        f"- Lo stesso colore indica la coppia da allineare: {colors}. Quando il "
        "puntamento del modo è preciso, la freccia spessa copre quella sottile.\n"
        "- In DETUMBLE il satellite ruota su sé stesso: le frecce spesse cambiano "
        "direzione a ogni istante.\n\n"
        "L'assetto è salvato come quaternione: quattro numeri che descrivono una "
        "rotazione senza le singolarità degli angoli di Eulero. Trascinando con "
        "il mouse si ruota la vista."
    )


def timeline_help(config: MissionConfig) -> str:
    """Come leggere i grafici dell'andamento nel tempo."""
    modes, loads, control = config.modes, config.loads, config.control
    payload, station = config.payload, config.ground_station
    half_deg = math.degrees(payload.antenna_half_beamwidth_rad)
    settling_s = _SETTLING_FACTOR / (
        control.pd_damping_ratio * control.pd_natural_frequency_rad_s
    )
    return (
        "Cinque grafici con lo stesso asse dei tempi; la linea verticale bianca "
        "è l'istante scelto con il cursore.\n\n"
        "- **Modo**: lo stato operativo del satellite, scelto ogni secondo dal "
        "gestore dei modi. Decide dove punta, quali apparati sono accesi e quale "
        "controllo lavora. Se più condizioni sono vere insieme vince il modo più "
        "in alto, cioè il primo di questo elenco; il cambio avviene solo se la "
        f"condizione dura almeno {modes.confirmation_time_s:g} s (tempo di "
        "conferma), così una misura rumorosa non basta. Un modo mai usato non "
        "compare sull'asse.\n"
        f"    - **SAFE**: batteria sotto il {modes.safe_enter_soc_pct:g} %. "
        "Fotocamera e trasmettitore spenti, pannelli verso il Sole finché la "
        f"carica non torna sopra il {modes.safe_exit_soc_pct:g} %.\n"
        "    - **DETUMBLE**: rotazione sopra "
        f"{modes.detumble_enter_rate_deg_s:g} °/s, come subito dopo il rilascio. "
        "I magnetorquer la frenano con il campo magnetico terrestre fino a sotto "
        f"{modes.detumble_exit_rate_deg_s:g} °/s.\n"
        f"    - **DOWNLINK**: {station.name} vede il satellite sopra "
        f"{station.min_elevation_deg:g}° di elevazione. Antenna verso la "
        f"stazione, dati a terra a {payload.downlink_rate_mbps:g} Mbit/s.\n"
        "    - **NADIR**: il satellite sorvola la regione di acquisizione "
        f"({config.imaging.region_name}) ed è al Sole. Fotocamera verso la Terra "
        "per acquisire immagini.\n"
        "    - **SUN_POINTING**: nessuna delle condizioni precedenti. Pannelli "
        "verso il Sole per produrre il massimo di energia (modo di riposo).\n"
        "- **Batteria**: stato di carica. Linee rosse tratteggiate: si entra in "
        f"SAFE sotto il {modes.safe_enter_soc_pct:g} % e se ne esce sopra il "
        f"{modes.safe_exit_soc_pct:g} %.\n"
        "- **Potenza**: prodotta dai pannelli (arancione) e consumata dai carichi "
        "(blu). La produzione va a zero in eclissi e cala quando i pannelli non "
        "guardano il Sole, come in NADIR e in DOWNLINK. Consumo di base "
        f"{loads.obc_w + loads.adcs_base_w:g} W (computer di bordo e ADCS), più "
        f"{loads.camera_w:g} W con la fotocamera, {loads.transmitter_w:g} W con il "
        "trasmettitore e una quota che cresce con l'uso di ruote e magnetorquer.\n"
        "- **Errore di puntamento** (scala logaritmica): angolo tra l'asse da "
        "puntare nel modo e la sua direzione. È l'errore vero: il controllore "
        "vede solo l'assetto misurato, con un rumore di "
        f"{config.sensors.attitude_noise_deg:g}°. Assente in DETUMBLE. I picchi "
        "cadono ai cambi di modo: il nuovo riferimento è lontano e le manovre "
        f"sono limitate a {control.max_slew_rate_deg_s:g} °/s. Alla fine della "
        f"manovra il controllo PD (banda {control.pd_natural_frequency_rad_s:g} "
        f"rad/s, smorzamento {control.pd_damping_ratio:g}) assesta l'errore in "
        f"circa {settling_s:.0f} s. Sotto la linea tratteggiata ({half_deg:g}°, "
        "metà del fascio) l'antenna può scaricare dati.\n"
        "- **Velocità delle ruote**: una ruota per asse del corpo. Quando una "
        "ruota accelera, il corpo ruota nel verso opposto: così le ruote "
        "orientano il satellite. Linee rosse tratteggiate: il limite di "
        f"±{config.actuators.wheel_max_speed_rpm:g} giri/min, dove la "
        "ruota satura. I disturbi esterni (gradiente gravitazionale, dipolo "
        "magnetico residuo) accumulano momento nelle ruote, che qui non viene "
        "scaricato.\n\n"
        "Trascinando si ingrandisce un intervallo; doppio clic per tornare alla "
        "vista intera."
    )


def metrics_help(config: MissionConfig) -> str:
    """Come leggere le metriche della missione e gli studi di compromesso."""
    half_deg = math.degrees(config.payload.antenna_half_beamwidth_rad)
    return (
        "- **Detumble**: tempo dal rilascio al primo modo di puntamento "
        "(SUN_POINTING, NADIR o DOWNLINK), cioè quando la rotazione è smorzata e "
        "il controllo passa alle ruote. Se nel frattempo scatta il SAFE, l'attesa "
        "è compresa.\n"
        "- **Carica minima** e **Tempo in SAFE**: il profilo di missione è "
        "sostenibile se la carica resta sopra il "
        f"{config.modes.safe_enter_soc_pct:g} % e il SAFE non scatta mai.\n"
        "- **Collegamento disponibile**: parte del tempo di visibilità della "
        f"stazione in cui l'antenna è entro metà fascio ({half_deg:g}°). Misura "
        "quanto l'ADCS sfrutta ogni passaggio.\n"
        "- **Dati scaricati / acquisiti**: la differenza resta nella memoria di "
        f"bordo ({config.payload.storage_capacity_gbit:g} Gbit), o va persa se la "
        "memoria è piena.\n"
        "- **Saturazioni delle ruote**: episodi in cui almeno una ruota ha "
        "raggiunto la coppia o il momento massimo.\n"
        "- **Errore di puntamento per modo**: medio, 95° percentile (il 95 % del "
        "tempo l'errore è minore) e massimo, di solito all'inizio di una "
        "manovra.\n"
        "- **Energia per orbita completa**: netta = prodotta - consumata. Se "
        "resta positiva orbita dopo orbita, il bilancio energetico regge.\n"
        "- **Studi di compromesso**: (1) quanta energia costa puntare la "
        "fotocamera verso la Terra invece dei pannelli verso il Sole, con gli "
        "assetti ideali lungo un'orbita; (2) la batteria più piccola che, con gli "
        "stessi consumi, avrebbe evitato il SAFE."
    )


def monte_carlo_help(config: MissionConfig) -> str:
    """Come leggere la campagna Monte Carlo del detumble.

    Args:
        config: configurazione dello scenario di riferimento, quella usata
            dalla campagna.
    """
    inertia = config.satellite.inertia_kg_m2
    return (
        "La durata del detumble dipende dalle condizioni al rilascio, che sono "
        "casuali. La campagna ripete lo scenario di riferimento cambiando solo il "
        "seme, il numero che inizializza il generatore casuale: stesso seme, "
        "stessa corsa.\n\n"
        "- Cambiano l'assetto e l'asse di rotazione al rilascio e il rumore dei "
        "sensori; la velocità al rilascio è sempre la massima "
        f"({config.initial_state.max_rate_deg_s:g} °/s).\n"
        "- Ogni barra conta i rilasci che hanno completato il detumble in "
        "quell'intervallo di tempo.\n"
        "- Linea bianca tratteggiata: 95° percentile. Un requisito di missione si "
        "fissa su questo valore e non sulla media: copre quasi tutti i casi "
        "senza farsi dominare dal peggiore.\n"
        "- Linea arancione: la simulazione mostrata sopra. È confrontabile solo "
        "se ha la stessa velocità al rilascio.\n"
        "- I rilasci più rapidi ruotano soprattutto attorno all'asse lungo, dove "
        f"l'inerzia è {max(inertia) / min(inertia):.0f} volte minore: a parità di "
        "velocità c'è meno momento angolare da smorzare."
    )
