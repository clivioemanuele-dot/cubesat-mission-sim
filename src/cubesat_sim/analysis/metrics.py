"""Metriche di missione, calcolate dai risultati salvati di una simulazione.

Metriche richieste dal progetto:
    - tempo di detumble;
    - tempo trascorso in ogni modo, compreso il SAFE;
    - errore di puntamento per modo: medio, 95° percentile e massimo;
    - episodi di saturazione delle ruote e momento massimo accumulato;
    - disponibilità del collegamento: quanto del tempo in cui la stazione è
      visibile l'antenna la punta davvero entro metà fascio. È il punto in cui
      le prestazioni del controllo d'assetto diventano dati a terra;
    - bilancio energetico per orbita e carica minima della batteria;
    - dati acquisiti, scaricati, rimasti a bordo e persi.

Tutto viene dai file salvati: non serve rilanciare la simulazione. I tempi dei
modi vengono dalle transizioni registrate, al secondo; le altre grandezze dalle
serie temporali, campionate ogni output_step_s. I totali cumulati sono quelli
dell'ultimo istante salvato: l'ultimo passo di uscita (10 s su 24 h) non entra.

Unità comode, indicate dal suffisso: ore, gradi, Wh, Gbit.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cubesat_sim.core.config import MissionConfig
from cubesat_sim.core.results import SimulationResults
from cubesat_sim.environment.orbit import CircularOrbit
from cubesat_sim.modes.manager import Mode

_S_PER_H = 3600.0
_J_PER_WH = 3600.0
_BIT_PER_GBIT = 1.0e9
_PERCENTILE = 95.0
# Modi in cui il satellite punta con le ruote: si entra solo a rotazione smorzata.
_POINTING_MODES = frozenset({Mode.DOWNLINK, Mode.NADIR, Mode.SUN_POINTING})
_WHEEL_COLUMNS = [
    "wheel_momentum_x_nms",
    "wheel_momentum_y_nms",
    "wheel_momentum_z_nms",
]


@dataclass(frozen=True)
class PointingStats:
    """Errore di puntamento vero dell'asse principale di un modo.

    Attributes:
        mode: nome del modo.
        mean_deg: errore medio [gradi].
        p95_deg: 95° percentile: per il 95 % del tempo l'errore è minore [gradi].
        max_deg: errore massimo, di solito all'inizio della manovra [gradi].
    """

    mode: str
    mean_deg: float
    p95_deg: float
    max_deg: float


@dataclass(frozen=True)
class OrbitEnergy:
    """Bilancio energetico di un'orbita completa.

    Attributes:
        orbit: numero dell'orbita, a partire da 1.
        start_h: inizio dell'orbita dall'inizio della simulazione [h].
        generated_wh: energia prodotta dai pannelli [Wh].
        consumed_wh: energia chiesta dai carichi [Wh].
        net_wh: prodotta meno consumata [Wh].
        min_state_of_charge: carica minima della batteria nell'orbita, tra 0 e 1.
    """

    orbit: int
    start_h: float
    generated_wh: float
    consumed_wh: float
    net_wh: float
    min_state_of_charge: float


@dataclass(frozen=True)
class MissionMetrics:
    """Metriche di una simulazione di missione.

    Attributes:
        duration_h: durata simulata [h].
        detumble_time_h: istante del primo ingresso in un modo di puntamento
            (SUN_POINTING, NADIR o DOWNLINK) [h]: la rotazione è smorzata e il
            controllo passa alle ruote. None se non avviene. Se il SAFE scatta
            durante il detumble, comprende l'attesa in SAFE.
        mode_hours: ore trascorse in ogni modo; i modi mai usati valgono 0.
        pointing: errore di puntamento per modo, in ordine di priorità; il
            DETUMBLE non ha puntamento e non compare.
        wheel_saturation_events: episodi con almeno una ruota al limite.
        wheel_saturation_s: tempo totale con almeno una ruota al limite [s].
        max_wheel_momentum_fraction: momento massimo di una ruota, come
            frazione del momento massimo ammesso.
        station_visible_h: ore con la stazione sopra l'elevazione minima.
        link_availability: frazione del tempo di visibilità in cui il satellite
            è in DOWNLINK con l'antenna entro metà fascio, tra 0 e 1.
        orbits: bilancio energetico di ogni orbita completa.
        min_state_of_charge: carica minima della batteria, tra 0 e 1.
        data_acquired_gbit: dati acquisiti dalla fotocamera [Gbit].
        data_downlinked_gbit: dati scaricati a terra [Gbit].
        data_on_board_gbit: dati rimasti in memoria alla fine [Gbit].
        data_lost_gbit: dati persi per memoria piena [Gbit].
    """

    duration_h: float
    detumble_time_h: float | None
    mode_hours: dict[str, float]
    pointing: tuple[PointingStats, ...]
    wheel_saturation_events: int
    wheel_saturation_s: float
    max_wheel_momentum_fraction: float
    station_visible_h: float
    link_availability: float
    orbits: tuple[OrbitEnergy, ...]
    min_state_of_charge: float
    data_acquired_gbit: float
    data_downlinked_gbit: float
    data_on_board_gbit: float
    data_lost_gbit: float


def compute_metrics(results: SimulationResults) -> MissionMetrics:
    """Calcola le metriche di missione dai risultati di una simulazione.

    Args:
        results: serie temporali e metadati, appena calcolati o riletti da
            disco.

    Returns:
        Le metriche della missione.
    """
    config = MissionConfig.model_validate(results.metadata["config"])
    series = results.timeseries
    transitions: list[dict[str, Any]] = results.metadata["transitions"]
    orbit = CircularOrbit.from_config(config.orbit, config.simulation.start_epoch)
    last = series.iloc[-1]
    wheel_momentum = series[_WHEEL_COLUMNS].abs().to_numpy(dtype=float)
    return MissionMetrics(
        duration_h=config.simulation.duration_h,
        detumble_time_h=_detumble_time_h(transitions),
        mode_hours=_mode_hours(
            str(series["mode"].iloc[0]), transitions, config.simulation.duration_s
        ),
        pointing=_pointing_stats(series),
        wheel_saturation_events=_saturation_events(series),
        wheel_saturation_s=float(last["wheel_saturation_time_s"]),
        max_wheel_momentum_fraction=float(wheel_momentum.max())
        / config.actuators.wheel_max_momentum_nms,
        station_visible_h=_visible_hours(series, config.simulation.output_step_s),
        link_availability=_link_availability(
            series, config.payload.antenna_half_beamwidth_rad
        ),
        orbits=_orbit_energy(series, orbit.period_s),
        min_state_of_charge=float(series["state_of_charge"].min()),
        data_acquired_gbit=float(last["data_acquired_bit"]) / _BIT_PER_GBIT,
        data_downlinked_gbit=float(last["data_downlinked_bit"]) / _BIT_PER_GBIT,
        data_on_board_gbit=float(last["data_stored_bit"]) / _BIT_PER_GBIT,
        data_lost_gbit=float(last["data_lost_bit"]) / _BIT_PER_GBIT,
    )


def _detumble_time_h(transitions: list[dict[str, Any]]) -> float | None:
    """Istante del primo ingresso in un modo di puntamento [h], o None.

    Non basta la prima uscita dal DETUMBLE: se la batteria si scarica, il
    gestore passa in SAFE mentre il satellite ruota ancora.
    """
    for transition in transitions:
        if transition["current"] in _POINTING_MODES:
            return float(transition["time_s"]) / _S_PER_H
    return None


def _mode_hours(
    initial_mode: str, transitions: list[dict[str, Any]], duration_s: float
) -> dict[str, float]:
    """Ore trascorse in ogni modo, dalle transizioni registrate."""
    seconds = {str(mode): 0.0 for mode in Mode}
    mode, start_s = initial_mode, 0.0
    for transition in transitions:
        seconds[mode] += float(transition["time_s"]) - start_s
        mode, start_s = str(transition["current"]), float(transition["time_s"])
    seconds[mode] += duration_s - start_s
    return {name: value / _S_PER_H for name, value in seconds.items()}


def _pointing_stats(series: pd.DataFrame) -> tuple[PointingStats, ...]:
    """Errore di puntamento medio, 95° percentile e massimo per ogni modo."""
    stats: list[PointingStats] = []
    for mode in Mode:
        in_mode = series["mode"] == str(mode)
        errors = series.loc[in_mode, "pointing_error_rad"].to_numpy(dtype=float)
        errors = np.degrees(errors[~np.isnan(errors)])
        if errors.size == 0:
            continue
        stats.append(
            PointingStats(
                mode=str(mode),
                mean_deg=float(errors.mean()),
                p95_deg=float(np.percentile(errors, _PERCENTILE)),
                max_deg=float(errors.max()),
            )
        )
    return tuple(stats)


def _saturation_events(series: pd.DataFrame) -> int:
    """Numero di episodi di saturazione delle ruote.

    Un episodio è una sequenza di intervalli di uscita consecutivi in cui il
    tempo di saturazione cumulato cresce.
    """
    cumulative = series["wheel_saturation_time_s"].to_numpy(dtype=float)
    saturated = np.diff(cumulative) > 0.0
    previous = np.concatenate(([False], saturated[:-1]))
    return int(np.count_nonzero(saturated & ~previous))


def _visible_hours(series: pd.DataFrame, output_step_s: float) -> float:
    """Ore con la stazione visibile, dagli istanti di uscita."""
    visible = series["station_visible"].to_numpy(dtype=bool)
    return int(np.count_nonzero(visible)) * output_step_s / _S_PER_H


def _link_availability(series: pd.DataFrame, half_beamwidth_rad: float) -> float:
    """Frazione del tempo di visibilità con il collegamento disponibile."""
    visible = series["station_visible"].to_numpy(dtype=bool)
    if not visible.any():
        return 0.0
    downlink = (series["mode"] == str(Mode.DOWNLINK)).to_numpy(dtype=bool)
    antenna_error = series["antenna_error_rad"].to_numpy(dtype=float)
    in_beam = visible & downlink & (antenna_error <= half_beamwidth_rad)
    return int(np.count_nonzero(in_beam)) / int(np.count_nonzero(visible))


def _orbit_energy(series: pd.DataFrame, period_s: float) -> tuple[OrbitEnergy, ...]:
    """Bilancio energetico di ogni orbita completa.

    Le energie cumulate sono interpolate linearmente agli istanti di inizio e
    fine orbita, che non coincidono con gli istanti di uscita.
    """
    time_s = series["time_s"].to_numpy(dtype=float)
    generated_j = series["energy_generated_j"].to_numpy(dtype=float)
    consumed_j = series["energy_consumed_j"].to_numpy(dtype=float)
    soc = series["state_of_charge"].to_numpy(dtype=float)
    orbits: list[OrbitEnergy] = []
    for k in range(int(time_s[-1] // period_s)):
        bounds = [k * period_s, (k + 1) * period_s]
        generated_wh = float(np.diff(np.interp(bounds, time_s, generated_j))[0])
        consumed_wh = float(np.diff(np.interp(bounds, time_s, consumed_j))[0])
        generated_wh /= _J_PER_WH
        consumed_wh /= _J_PER_WH
        inside = (time_s >= bounds[0]) & (time_s <= bounds[1])
        orbits.append(
            OrbitEnergy(
                orbit=k + 1,
                start_h=bounds[0] / _S_PER_H,
                generated_wh=generated_wh,
                consumed_wh=consumed_wh,
                net_wh=generated_wh - consumed_wh,
                min_state_of_charge=float(soc[inside].min()),
            )
        )
    return tuple(orbits)
