"""Studi di compromesso della missione.

Studio 1, puntare la Terra o il Sole: quanta energia si perde, in un'orbita,
tenendo la fotocamera verso la Terra (riferimento NADIR) invece dei pannelli
verso il Sole (riferimento SUN_POINTING). È un calcolo geometrico sugli assetti
di riferimento ideali, lungo un'orbita a partire dall'istante iniziale: non
dipende dal controllo né dalla batteria e non richiede una simulazione.

Studio 2, dimensionamento della batteria: la capacità minima che evita il
SAFE nello scenario di riferimento. Finché il SAFE non scatta, la capacità
della batteria non cambia nulla del resto della missione: assetto, potenza
prodotta e consumi restano identici. Basta quindi rigiocare il modello della
batteria sui profili di energia di una simulazione già fatta, con capacità
diverse, e cercare per bisezione la più piccola che tiene la carica sopra la
soglia del SAFE. Ogni prova costa millisecondi invece di una corsa di 24 h. Il
risultato va confermato con una simulazione completa, sopra e sotto la
capacità trovata (docs/results.md).

Approssimazione del rigioco: in ogni intervallo di uscita (10 s) conta solo
l'energia netta, quindi un intervallo a cavallo dell'ingresso in eclissi non
distingue la parte in carica da quella in scarica.
"""

from dataclasses import dataclass

import numpy as np

from cubesat_sim.attitude.references import (
    PointingAxes,
    nadir_pointing_attitude,
    sun_pointing_attitude,
)
from cubesat_sim.core.config import BatteryConfig, MissionConfig
from cubesat_sim.core.quaternions import quat_to_matrix
from cubesat_sim.core.results import SimulationResults
from cubesat_sim.core.types import FloatArray
from cubesat_sim.environment.earth import julian_date
from cubesat_sim.environment.orbit import CircularOrbit
from cubesat_sim.environment.sun import is_in_eclipse, sun_position_eci
from cubesat_sim.modes.manager import Mode
from cubesat_sim.resources.battery import Battery
from cubesat_sim.resources.solar import SolarArray

_S_PER_MIN = 60.0
_J_PER_WH = 3600.0

# Ricerca della capacità minima: intervallo iniziale, come multiplo della
# capacità configurata, e precisione della bisezione.
_MAX_CAPACITY_FACTOR = 10.0
_CAPACITY_TOLERANCE_WH = 0.01


@dataclass(frozen=True)
class PointingEnergyTrade:
    """Energia prodotta in un'orbita con due assetti di riferimento.

    Attributes:
        period_min: periodo orbitale [min].
        sunlit_fraction: frazione dell'orbita al Sole, tra 0 e 1.
        sun_pointing_wh: energia per orbita puntando i pannelli al Sole [Wh].
        nadir_pointing_wh: energia per orbita puntando la fotocamera a terra [Wh].
        loss_wh: energia persa per orbita puntando la Terra [Wh].
        loss_fraction: perdita relativa, tra 0 e 1.
        sun_pointing_power_w: potenza media al Sole in SUN_POINTING [W].
        nadir_pointing_power_w: potenza media al Sole in NADIR [W].
    """

    period_min: float
    sunlit_fraction: float
    sun_pointing_wh: float
    nadir_pointing_wh: float
    loss_wh: float
    loss_fraction: float
    sun_pointing_power_w: float
    nadir_pointing_power_w: float


def pointing_energy_trade(config: MissionConfig) -> PointingEnergyTrade:
    """Studio 1: energia per orbita puntando il Sole o la Terra.

    La potenza è campionata ogni output_step_s lungo la prima orbita, con
    l'assetto di riferimento ideale di ciascun modo (errore di puntamento
    nullo).

    Args:
        config: configurazione della missione.

    Returns:
        L'energia prodotta nei due casi e la perdita.
    """
    epoch = config.simulation.start_epoch
    orbit = CircularOrbit.from_config(config.orbit, epoch)
    axes = PointingAxes.from_config(config)
    solar = SolarArray(config.solar_array)
    step_s = config.simulation.output_step_s
    sun_w: list[float] = []
    nadir_w: list[float] = []
    lit: list[bool] = []
    for elapsed_s in np.arange(0.0, orbit.period_s, step_s):
        position, velocity = orbit.state_eci(float(elapsed_s))
        sun = sun_position_eci(julian_date(epoch, float(elapsed_s)))
        in_eclipse = is_in_eclipse(position, sun)
        sun_from_satellite = sun - position
        for attitude, powers in (
            (sun_pointing_attitude(axes, sun, position, velocity), sun_w),
            (nadir_pointing_attitude(axes, sun, position, velocity), nadir_w),
        ):
            sun_body = quat_to_matrix(attitude).T @ sun_from_satellite
            powers.append(solar.power(sun_body, in_eclipse))
        lit.append(not in_eclipse)
    sunlit = np.array(lit)
    sun_array, nadir_array = np.array(sun_w), np.array(nadir_w)
    # Energia = potenza media sui campioni per la durata dell'orbita.
    sun_wh = float(sun_array.mean()) * orbit.period_s / _J_PER_WH
    nadir_wh = float(nadir_array.mean()) * orbit.period_s / _J_PER_WH
    return PointingEnergyTrade(
        period_min=orbit.period_s / _S_PER_MIN,
        sunlit_fraction=float(sunlit.mean()),
        sun_pointing_wh=sun_wh,
        nadir_pointing_wh=nadir_wh,
        loss_wh=sun_wh - nadir_wh,
        loss_fraction=(sun_wh - nadir_wh) / sun_wh,
        sun_pointing_power_w=float(sun_array[sunlit].mean()),
        nadir_pointing_power_w=float(nadir_array[sunlit].mean()),
    )


@dataclass(frozen=True)
class BatterySizingTrade:
    """Dimensionamento della batteria nello scenario simulato.

    Attributes:
        configured_capacity_wh: capacità della configurazione [Wh].
        safe_threshold: soglia di ingresso in SAFE, tra 0 e 1.
        min_capacity_wh: capacità minima che tiene la carica sopra la soglia
            del SAFE per tutta la simulazione [Wh].
        max_drawdown_wh: energia massima prelevata dalla batteria tra un
            massimo di carica e il minimo successivo, con la capacità
            configurata [Wh]. Tipicamente un'eclisse.
        max_depth_of_discharge: max_drawdown_wh diviso la capacità
            configurata: la profondità di scarica massima, tra 0 e 1.
    """

    configured_capacity_wh: float
    safe_threshold: float
    min_capacity_wh: float
    max_drawdown_wh: float
    max_depth_of_discharge: float


def battery_sizing_trade(results: SimulationResults) -> BatterySizingTrade:
    """Studio 2: capacità minima della batteria che evita il SAFE.

    Args:
        results: risultati di una simulazione in cui il SAFE non è mai
            scattato, così che i profili di energia valgano per ogni capacità
            che non lo fa scattare.

    Returns:
        La capacità minima e la profondità di scarica con la capacità
        configurata.

    Raises:
        ValueError: se la simulazione è entrata in SAFE, oppure se nemmeno una
            batteria dieci volte più grande eviterebbe il SAFE.
    """
    config = MissionConfig.model_validate(results.metadata["config"])
    series = results.timeseries
    if (series["mode"] == str(Mode.SAFE)).any():
        msg = "la simulazione è entrata in SAFE: i suoi profili non sono validi"
        raise ValueError(msg)
    durations_s = np.diff(series["time_s"].to_numpy(dtype=float))
    net_energy_j = np.diff(
        series["energy_generated_j"].to_numpy(dtype=float)
        - series["energy_consumed_j"].to_numpy(dtype=float)
    )
    battery = config.battery
    threshold = config.modes.safe_enter_soc

    def avoids_safe(capacity_wh: float) -> bool:
        min_soc, _ = _replay(battery, capacity_wh, net_energy_j, durations_s)
        return min_soc >= threshold

    low_wh, high_wh = 0.0, battery.capacity_wh * _MAX_CAPACITY_FACTOR
    if not avoids_safe(high_wh):
        msg = "nemmeno una batteria dieci volte più grande evita il SAFE"
        raise ValueError(msg)
    while high_wh - low_wh > _CAPACITY_TOLERANCE_WH:
        middle_wh = (low_wh + high_wh) / 2.0
        if avoids_safe(middle_wh):
            high_wh = middle_wh
        else:
            low_wh = middle_wh
    _, drawdown_wh = _replay(battery, battery.capacity_wh, net_energy_j, durations_s)
    return BatterySizingTrade(
        configured_capacity_wh=battery.capacity_wh,
        safe_threshold=threshold,
        min_capacity_wh=high_wh,
        max_drawdown_wh=drawdown_wh,
        max_depth_of_discharge=drawdown_wh / battery.capacity_wh,
    )


def _replay(
    battery: BatteryConfig,
    capacity_wh: float,
    net_energy_j: FloatArray,
    durations_s: FloatArray,
) -> tuple[float, float]:
    """Rigioca la batteria sui profili di energia netta al bus.

    Args:
        battery: sezione [battery] della configurazione.
        capacity_wh: capacità da provare [Wh].
        net_energy_j: energia prodotta meno consumata in ogni intervallo [J].
        durations_s: durata di ogni intervallo [s].

    Returns:
        Coppia (carica minima tra 0 e 1, prelievo massimo tra un massimo di
        carica e il minimo successivo [Wh]).
    """
    model = Battery(battery.model_copy(update={"capacity_wh": capacity_wh}))
    min_soc = model.state_of_charge
    peak_j = model.energy_j
    drawdown_j = 0.0
    for energy_j, duration_s in zip(net_energy_j, durations_s, strict=True):
        model.update(float(energy_j / duration_s), float(duration_s))
        min_soc = min(min_soc, model.state_of_charge)
        peak_j = max(peak_j, model.energy_j)
        drawdown_j = max(drawdown_j, peak_j - model.energy_j)
    return min_soc, drawdown_j / _J_PER_WH
