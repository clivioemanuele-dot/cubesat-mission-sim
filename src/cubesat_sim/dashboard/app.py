"""Dashboard Streamlit: riproduce una simulazione e ne mostra le metriche.

Avvio, dalla cartella del progetto:
    uv run python -m streamlit run src/cubesat_sim/dashboard/app.py

Mostra lo scenario di riferimento salvato in results/baseline, oppure una
nuova simulazione breve (al massimo MAX_RERUN_H ore) con alcuni parametri
cambiati dalla barra laterale. La nuova simulazione resta in memoria e non
viene salvata. In fondo alla pagina, la campagna Monte Carlo del detumble
salvata in results/monte_carlo. Sotto ogni grafico, un riquadro apribile
spiega come leggerlo. Sola visualizzazione: la logica di bordo sta nei
sottopacchetti del simulatore. Tutti i dati mostrati sono simulati.
"""

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from cubesat_sim.analysis.metrics import MissionMetrics, compute_metrics
from cubesat_sim.analysis.monte_carlo import (
    RUNS_FILE,
    DetumbleStatistics,
    load_runs,
    summarize,
)
from cubesat_sim.analysis.trade_studies import (
    BatterySizingTrade,
    PointingEnergyTrade,
    battery_sizing_trade,
    pointing_energy_trade,
)
from cubesat_sim.core.config import MissionConfig, load_config
from cubesat_sim.core.results import (
    SIMULATED_NOTICE,
    TIMESERIES_FILE,
    SimulationResults,
)
from cubesat_sim.dashboard.explanations import (
    attitude_help,
    current_activity,
    ground_track_help,
    metrics_help,
    monte_carlo_help,
    timeline_help,
)
from cubesat_sim.dashboard.figures import (
    attitude_figure,
    detumble_figure,
    ground_track_figure,
    timeline_figure,
)
from cubesat_sim.modes.manager import Mode
from cubesat_sim.simulation.runner import run_simulation

PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "baseline"
DEFAULT_MONTE_CARLO_DIR = PROJECT_DIR / "results" / "monte_carlo"
BASELINE_CONFIG = PROJECT_DIR / "config" / "baseline.toml"

# Durata massima di una nuova simulazione dalla dashboard: circa 45 s di
# calcolo sul PC di sviluppo. Le 24 ore si simulano dalla riga di comando.
MAX_RERUN_H = 6.0
_MIN_RERUN_H = 0.5
_DEFAULT_RERUN_H = 3.0
# Intervalli ammessi nel modulo della nuova simulazione.
_CAPACITY_RANGE_WH = (1.0, 100.0)
_RELEASE_RATE_RANGE_DEG_S = (0.0, 20.0)
_PERCENT = 100.0

SAVED_SOURCE = "Scenario di riferimento (salvato)"
NEW_SOURCE = "Nuova simulazione"
_PARAMETERS_KEY = "new_simulation_parameters"

# Aspetto della pagina: cielo stellato disegnato solo con gradienti CSS (niente
# immagini esterne) e riquadri semitrasparenti sopra lo sfondo. Ogni strato di
# stelle si ripete su una piastrella di lato diverso, così il motivo non si nota.
# Nessun effetto sui dati; i selettori sono quelli degli elementi di Streamlit.
_SPACE_CSS = """
<style>
[data-testid="stAppViewContainer"] {
  background-color: #070b16;
  background-image:
    radial-gradient(1px 1px at 25px 35px, rgba(255, 255, 255, 0.9), transparent),
    radial-gradient(1px 1px at 140px 90px, rgba(255, 255, 255, 0.6), transparent),
    radial-gradient(1.5px 1.5px at 260px 180px, rgba(255, 255, 255, 0.8), transparent),
    radial-gradient(1px 1px at 330px 40px, rgba(190, 210, 255, 0.8), transparent),
    radial-gradient(1px 1px at 80px 250px, rgba(255, 255, 255, 0.5), transparent),
    radial-gradient(1.5px 1.5px at 200px 320px, rgba(255, 236, 214, 0.7), transparent),
    radial-gradient(ellipse 60% 45% at 12% 0%, rgba(57, 135, 229, 0.16), transparent),
    radial-gradient(ellipse 50% 40% at 92% 100%, rgba(144, 133, 233, 0.12),
      transparent);
  background-size: 310px 310px, 370px 370px, 430px 430px, 490px 490px,
    550px 550px, 610px 610px, 100% 100%, 100% 100%;
  background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background-color: rgba(15, 22, 41, 0.92); }
[data-testid="stMetric"] {
  background-color: rgba(15, 22, 41, 0.85);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  padding: 12px 16px;
}
[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }
</style>
"""

_RUN_COMMAND = (
    "uv run python -m cubesat_sim.simulation config/baseline.toml results/baseline"
)
_MONTE_CARLO_COMMAND = (
    "uv run python -m cubesat_sim.analysis.monte_carlo config/baseline.toml "
    "results/monte_carlo"
)


# Risultati e analisi di una simulazione, come li restituiscono le funzioni in
# cache: solo tipi definiti nei moduli del pacchetto. st.cache_data li salva con
# pickle, che ritrova ogni classe tramite il modulo in cui è definita. Streamlit
# esegue questo file come modulo __main__ e lo sostituisce a ogni esecuzione,
# anche fra visitatori collegati insieme: una classe definita qui, come
# _Mission, non si potrebbe salvare in modo affidabile.
_Analysis = tuple[
    SimulationResults,
    MissionConfig,
    MissionMetrics,
    PointingEnergyTrade,
    BatterySizingTrade | str,
]


@dataclass(frozen=True, eq=False)
class _Mission:
    """Una simulazione con le analisi già calcolate.

    Si costruisce fuori dalle funzioni in cache (vedi ``_Analysis``).

    Attributes:
        description: da dove vengono i risultati, per la didascalia.
        results: serie temporali e metadati.
        config: configurazione della simulazione.
        metrics: metriche della missione.
        energy: studio 1, puntare la Terra o il Sole.
        battery: studio 2, capacità della batteria; oppure il motivo per cui
            lo studio non si può fare (per esempio, il SAFE è scattato).
    """

    description: str
    results: SimulationResults
    config: MissionConfig
    metrics: MissionMetrics
    energy: PointingEnergyTrade
    battery: BatterySizingTrade | str


def _analyse(results: SimulationResults) -> _Analysis:
    """Calcola metriche e studi di compromesso di una simulazione."""
    config = MissionConfig.model_validate(results.metadata["config"])
    battery: BatterySizingTrade | str
    try:
        battery = battery_sizing_trade(results)
    except ValueError as error:
        battery = str(error)
    return (
        results,
        config,
        compute_metrics(results),
        pointing_energy_trade(config),
        battery,
    )


@st.cache_data(show_spinner=False)
def _load_saved(directory: str, modified_ns: int) -> _Analysis:
    """Legge e analizza i risultati salvati, una volta sola.

    La data di modifica del CSV (``modified_ns``) fa parte della chiave della
    cache: se la simulazione viene rilanciata, i file vengono riletti.
    """
    return _analyse(SimulationResults.load(Path(directory)))


@st.cache_data(show_spinner=False)
def _simulate(hours: float, capacity_wh: float, release_rate_deg_s: float) -> _Analysis:
    """Esegue e analizza una nuova simulazione; stessi parametri, stessa corsa.

    Parte dallo scenario di riferimento, con il suo seme, e cambia solo i
    parametri indicati. L'effetto del seme si studia con il Monte Carlo.
    """
    data = load_config(BASELINE_CONFIG).model_dump(mode="json")
    data["simulation"]["duration_h"] = hours
    data["battery"]["capacity_wh"] = capacity_wh
    data["initial_state"]["max_rate_deg_s"] = release_rate_deg_s
    return _analyse(run_simulation(MissionConfig.model_validate(data)))


def _choose_mission(results_dir: Path) -> _Mission | None:
    """Barra laterale: scenario salvato oppure nuova simulazione.

    Returns:
        La simulazione da mostrare, oppure None se non ce n'è ancora una (la
        pagina spiega che cosa fare).
    """
    st.sidebar.header("Simulazione")
    source = st.sidebar.radio("Risultati da mostrare", [SAVED_SOURCE, NEW_SOURCE])
    if source == SAVED_SOURCE:
        csv_path = results_dir / TIMESERIES_FILE
        if not csv_path.is_file():
            st.error(
                f"Nessun risultato in {results_dir}. "
                f"Lanciare prima, dalla cartella del progetto: {_RUN_COMMAND}"
            )
            return None
        # Solo il nome della cartella: il percorso completo mostrerebbe il nome
        # utente del PC o del server.
        return _Mission(
            f"Scenario salvato (cartella {results_dir.name}).",
            *_load_saved(str(results_dir), csv_path.stat().st_mtime_ns),
        )

    baseline = load_config(BASELINE_CONFIG)
    with st.sidebar.form("new_simulation"):
        hours = st.number_input(
            "Durata [h]",
            min_value=_MIN_RERUN_H,
            max_value=MAX_RERUN_H,
            value=_DEFAULT_RERUN_H,
            step=_MIN_RERUN_H,
        )
        capacity_wh = st.number_input(
            "Capacità della batteria [Wh]",
            min_value=_CAPACITY_RANGE_WH[0],
            max_value=_CAPACITY_RANGE_WH[1],
            value=baseline.battery.capacity_wh,
        )
        release_rate = st.number_input(
            "Rotazione massima al rilascio [°/s]",
            min_value=_RELEASE_RATE_RANGE_DEG_S[0],
            max_value=_RELEASE_RATE_RANGE_DEG_S[1],
            value=baseline.initial_state.max_rate_deg_s,
        )
        if st.form_submit_button("Simula"):
            st.session_state[_PARAMETERS_KEY] = (
                float(hours),
                float(capacity_wh),
                float(release_rate),
            )
    parameters = st.session_state.get(_PARAMETERS_KEY)
    if parameters is None:
        st.info(
            "Scegli i parametri nella barra laterale e premi Simula. "
            f"Durata massima {MAX_RERUN_H:g} h: le 24 h si simulano dalla riga "
            "di comando."
        )
        return None
    # Valori dell'ultima pressione di Simula, non quelli ancora nel modulo.
    run_h, run_capacity_wh, run_rate_deg_s = parameters
    with st.spinner(f"Simulazione di {run_h:g} h in corso..."):
        analysis = _simulate(run_h, run_capacity_wh, run_rate_deg_s)
    return _Mission(
        f"Nuova simulazione, non salvata: batteria da {run_capacity_wh:g} Wh, "
        f"rotazione al rilascio fino a {run_rate_deg_s:g} °/s.",
        *analysis,
    )


def _show_metrics(mission: _Mission) -> None:
    """Metriche della missione e studi di compromesso."""
    metrics = mission.metrics
    st.subheader("Metriche della missione")
    st.expander("Come leggere le metriche").markdown(metrics_help(mission.config))
    top, bottom = st.columns(3), st.columns(3)
    top[0].metric(
        "Detumble",
        "non completato"
        if metrics.detumble_time_h is None
        else f"{metrics.detumble_time_h:.2f} h",
    )
    top[1].metric("Carica minima", f"{metrics.min_state_of_charge:.1%}")
    top[2].metric("Tempo in SAFE", f"{metrics.mode_hours[str(Mode.SAFE)]:.2f} h")
    bottom[0].metric(
        "Collegamento disponibile",
        f"{metrics.link_availability:.0%}",
        help="Parte del tempo di visibilità della stazione con l'antenna "
        "puntata entro metà fascio.",
    )
    bottom[1].metric(
        "Dati scaricati / acquisiti [Gbit]",
        f"{metrics.data_downlinked_gbit:.3f} / {metrics.data_acquired_gbit:.3f}",
    )
    bottom[2].metric("Saturazioni delle ruote", str(metrics.wheel_saturation_events))

    left, right = st.columns(2)
    left.markdown("**Errore di puntamento per modo [°]**")
    if metrics.pointing:
        pointing = pd.DataFrame(
            {
                "Modo": item.mode,
                "Medio": item.mean_deg,
                "95° percentile": item.p95_deg,
                "Massimo": item.max_deg,
            }
            for item in metrics.pointing
        )
        left.dataframe(pointing.round(2), hide_index=True)
    else:
        left.caption("Nessun modo con un assetto di riferimento.")
    right.markdown("**Energia per orbita completa [Wh]**")
    if metrics.orbits:
        orbits = pd.DataFrame(
            {
                "Orbita": orbit.orbit,
                "Inizio [h]": orbit.start_h,
                "Prodotta": orbit.generated_wh,
                "Consumata": orbit.consumed_wh,
                "Netta": orbit.net_wh,
                "Carica minima [%]": _PERCENT * orbit.min_state_of_charge,
            }
            for orbit in metrics.orbits
        )
        right.dataframe(orbits.round(2), hide_index=True)
    else:
        right.caption("Nessuna orbita completa.")

    st.markdown("**Studi di compromesso**")
    energy = mission.energy
    st.markdown(
        "1. Puntare la Terra invece del Sole, con gli assetti ideali lungo la "
        f"prima orbita: {energy.nadir_pointing_wh:.1f} Wh per orbita invece di "
        f"{energy.sun_pointing_wh:.1f}, si perdono **{energy.loss_wh:.1f} Wh "
        f"({energy.loss_fraction:.0%})**."
    )
    battery = mission.battery
    if isinstance(battery, str):
        st.markdown(f"2. Capacità minima della batteria: non calcolabile ({battery}).")
    else:
        st.markdown(
            "2. Capacità minima della batteria senza SAFE: "
            f"**{battery.min_capacity_wh:.2f} Wh** (configurata "
            f"{battery.configured_capacity_wh:g} Wh). Prelievo massimo "
            f"{battery.max_drawdown_wh:.2f} Wh, profondità di scarica "
            f"{battery.max_depth_of_discharge:.1%}."
        )


@st.cache_data(show_spinner=False)
def _load_campaign(
    directory: str, modified_ns: int
) -> tuple[pd.DataFrame, DetumbleStatistics]:
    """Legge la campagna Monte Carlo salvata, una volta sola.

    Come in ``_load_saved``, la data di modifica fa parte della chiave.
    """
    table = load_runs(Path(directory))
    return table, summarize(table)


def _show_monte_carlo(directory: Path, current_h: float | None) -> None:
    """Statistica del detumble su molti rilasci casuali."""
    st.subheader("Detumble su molti rilasci (Monte Carlo)")
    runs_path = directory / RUNS_FILE
    if not runs_path.is_file():
        st.info(
            f"Campagna Monte Carlo non trovata in {directory}. Per crearla, "
            f"dalla cartella del progetto: {_MONTE_CARLO_COMMAND}"
        )
        return
    table, statistics = _load_campaign(str(directory), runs_path.stat().st_mtime_ns)
    columns = st.columns(3)
    columns[0].metric("Detumble medio", f"{statistics.mean_h:.2f} h")
    columns[1].metric(
        "95° percentile",
        f"{statistics.p95_h:.2f} h",
        help="Il 95 % dei rilasci completa il detumble prima di questo tempo.",
    )
    columns[2].metric("Detumble massimo", f"{statistics.max_h:.2f} h")
    st.plotly_chart(detumble_figure(table, statistics.p95_h, current_h), theme=None)
    st.caption(
        f"{statistics.runs} rilasci dello scenario di riferimento con semi "
        f"diversi, {statistics.completed} completati entro la finestra simulata."
    )
    st.expander("Come leggere il Monte Carlo").markdown(
        monte_carlo_help(load_config(BASELINE_CONFIG))
    )


def render(results_dir: Path, monte_carlo_dir: Path) -> None:
    """Disegna la pagina.

    Args:
        results_dir: cartella dei risultati salvati (``timeseries.csv`` e
            ``metadata.json``) dello scenario di riferimento.
        monte_carlo_dir: cartella della campagna Monte Carlo salvata.
    """
    st.set_page_config(page_title="CubeSat 3U - simulazione", layout="wide")
    st.markdown(_SPACE_CSS, unsafe_allow_html=True)
    st.warning(f"**SIMULAZIONE** - {SIMULATED_NOTICE}")
    st.title("Simulatore di missione CubeSat 3U")

    mission = _choose_mission(results_dir)
    if mission is None:
        st.stop()
    config = mission.config
    series = mission.results.timeseries
    times_s = series["time_s"].to_numpy()

    # Istanti in UTC senza fuso orario: il cursore li mostra così come sono.
    start = config.simulation.start_epoch.astimezone(dt.UTC).replace(tzinfo=None)
    end = start + dt.timedelta(seconds=float(times_s[-1]))
    st.caption(
        f"Seme {config.simulation.seed}, "
        f"{config.simulation.duration_h:g} h dal {start:%d/%m/%Y %H:%M} UTC, "
        f"stazione di terra {config.ground_station.name}. {mission.description}"
    )
    selected: dt.datetime = st.slider(
        "Istante (UTC)",
        min_value=start,
        max_value=end,
        value=start,
        step=dt.timedelta(seconds=config.simulation.output_step_s),
        format="DD/MM/YYYY HH:mm:ss",
    )
    # Ultimo campione salvato non successivo all'istante scelto.
    elapsed_s = (selected - start).total_seconds()
    index = int(np.searchsorted(times_s, elapsed_s, side="right")) - 1
    row = series.iloc[index]

    # Due righe da tre: in una riga sola i nomi lunghi dei modi verrebbero tagliati.
    top, bottom = st.columns(3), st.columns(3)
    top[0].metric("Modo", str(row["mode"]))
    top[1].metric("Batteria", f"{row['state_of_charge']:.1%}")
    # In DETUMBLE non c'è un assetto di riferimento: l'errore non è definito.
    error_rad = float(row["pointing_error_rad"])
    top[2].metric(
        "Errore di puntamento",
        "n.d." if math.isnan(error_rad) else f"{math.degrees(error_rad):.1f}°",
    )
    bottom[0].metric(
        "Latitudine, longitudine",
        f"{math.degrees(row['latitude_rad']):.1f}°, "
        f"{math.degrees(row['longitude_rad']):.1f}°",
    )
    bottom[1].metric("Illuminazione", "eclissi" if row["in_eclipse"] else "Sole")
    bottom[2].metric(
        "Stazione di terra", "visibile" if row["station_visible"] else "non visibile"
    )

    st.subheader("Cosa sta facendo il satellite")
    transitions = mission.results.metadata["transitions"]
    st.container(border=True).markdown(
        current_activity(series, index, config, transitions)
    )

    left, right = st.columns([3, 2])
    left.subheader("Traccia a terra")
    left.plotly_chart(ground_track_figure(series, config, index), theme=None)
    left.expander("Come leggere la mappa").markdown(ground_track_help(config))
    right.subheader("Assetto")
    right.plotly_chart(attitude_figure(series, config, index), theme=None)
    right.expander("Come leggere la vista 3D").markdown(attitude_help(config))

    st.subheader("Andamento nel tempo")
    st.plotly_chart(timeline_figure(series, config, index), theme=None)
    st.expander("Come leggere questi grafici").markdown(timeline_help(config))

    _show_metrics(mission)
    _show_monte_carlo(monte_carlo_dir, mission.metrics.detumble_time_h)


if __name__ == "__main__":
    render(DEFAULT_RESULTS_DIR, DEFAULT_MONTE_CARLO_DIR)
