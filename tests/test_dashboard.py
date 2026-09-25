"""Test della dashboard: la pagina si apre e mostra l'istante scelto."""

import datetime as dt
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from cubesat_sim.analysis.monte_carlo import save_campaign
from cubesat_sim.attitude.references import (
    PointingAxes,
    nadir_pointing_attitude,
    station_pointing_attitude,
    sun_pointing_attitude,
)
from cubesat_sim.core.config import MissionConfig, load_config
from cubesat_sim.core.results import SIMULATED_NOTICE, SimulationResults
from cubesat_sim.core.types import FloatArray
from cubesat_sim.core.vectors import angle_between
from cubesat_sim.dashboard.app import NEW_SOURCE
from cubesat_sim.dashboard.figures import (
    attitude_figure,
    detumble_figure,
    ground_track_figure,
    timeline_figure,
)
from cubesat_sim.environment.earth import julian_date
from cubesat_sim.environment.ground import GroundStation
from cubesat_sim.environment.orbit import CircularOrbit
from cubesat_sim.environment.sun import sun_position_eci
from cubesat_sim.simulation.runner import run_simulation

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"

# Tempo massimo per un'esecuzione della pagina [s]: largo, per i PC lenti.
TIMEOUT_S = 30.0


@pytest.fixture(autouse=True)
def restore_main_module() -> Iterator[None]:
    # AppTest esegue la pagina come modulo __main__ e non rimette al suo posto
    # quello di pytest. Su Windows i processi paralleli avviati dopo (per
    # esempio dai test del Monte Carlo) rieseguirebbero lo script della pagina
    # e si interromperebbero.
    main_module = sys.modules["__main__"]
    yield
    sys.modules["__main__"] = main_module


def page(results_dir: str, monte_carlo_dir: str) -> None:
    # AppTest esegue questa funzione come uno script a sé stante: per questo
    # gli import stanno qui dentro e gli argomenti sono semplici stringhe.
    from pathlib import Path

    from cubesat_sim.dashboard.app import render

    render(Path(results_dir), Path(monte_carlo_dir))


def open_page(results_dir: Path, monte_carlo_dir: Path | None = None) -> AppTest:
    """Esegue la pagina una volta, come alla prima apertura nel browser.

    Senza cartella del Monte Carlo la pagina ne cerca una che non esiste.
    """
    campaign = monte_carlo_dir or results_dir / "monte_carlo_assente"
    app = AppTest.from_function(
        page, args=(str(results_dir), str(campaign)), default_timeout=TIMEOUT_S
    )
    return app.run()


@pytest.fixture(scope="module")
def results_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Tre minuti dal rilascio bastano per provare la pagina.
    data = load_config(BASELINE).model_dump(mode="json")
    data["simulation"]["duration_h"] = 0.05
    directory = tmp_path_factory.mktemp("results")
    run_simulation(MissionConfig.model_validate(data)).save(directory)
    return directory


# Campagna Monte Carlo inventata: quattro corse, una non finita.
CAMPAIGN = pd.DataFrame(
    {"seed": [1, 2, 3, 4], "detumble_time_h": [1.5, 2.0, 2.5, math.nan]}
)


@pytest.fixture(scope="module")
def monte_carlo_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("monte_carlo")
    save_campaign(CAMPAIGN, 3.0, directory)
    return directory


def test_page_declares_simulated_data(results_dir: Path) -> None:
    app = open_page(results_dir)
    assert not app.exception
    banner = app.warning[0].value
    assert "SIMULAZIONE" in banner
    assert SIMULATED_NOTICE in banner
    # All'istante iniziale il satellite è appena stato rilasciato: in DETUMBLE
    # non c'è un assetto di riferimento e l'errore di puntamento non è definito.
    assert app.metric[0].value == "DETUMBLE"
    assert app.metric[2].value == "n.d."
    assert len(app.get("plotly_chart")) == 3
    # Metriche della missione: in tre minuti il detumble non si completa.
    metrics = {metric.label: metric.value for metric in app.metric[6:]}
    assert metrics["Detumble"] == "non completato"
    assert metrics["Tempo in SAFE"] == "0.00 h"
    # Senza campagna Monte Carlo la pagina spiega come crearla.
    assert "monte_carlo" in app.info[0].value


def test_page_shows_monte_carlo(results_dir: Path, monte_carlo_dir: Path) -> None:
    app = open_page(results_dir, monte_carlo_dir)
    assert not app.exception
    assert len(app.get("plotly_chart")) == 4
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Detumble medio"] == "2.00 h"
    assert metrics["Detumble massimo"] == "2.50 h"
    assert "3 completati" in app.caption[-1].value


def position(series: pd.DataFrame, index: int) -> str:
    """Latitudine e longitudine del campione, scritte come nella pagina."""
    latitude = math.degrees(series["latitude_rad"].iloc[index])
    longitude = math.degrees(series["longitude_rad"].iloc[index])
    return f"{latitude:.1f}°, {longitude:.1f}°"


def test_slider_selects_saved_sample(results_dir: Path) -> None:
    # La posizione cambia di circa 0,6° ogni 10 s: distingue ogni campione.
    series = SimulationResults.load(results_dir).timeseries
    app = open_page(results_dir)
    start = app.slider[0].value
    assert isinstance(start, dt.datetime)
    assert app.metric[3].value == position(series, 0)

    app = app.slider[0].set_value(start + dt.timedelta(seconds=90)).run()
    assert not app.exception
    assert app.metric[3].value == position(series, 9)

    # In fondo al cursore si vede l'ultimo campione salvato.
    end = start + dt.timedelta(seconds=float(series["time_s"].iloc[-1]))
    app = app.slider[0].set_value(end).run()
    assert app.metric[3].value == position(series, -1)


def test_missing_results_are_explained(tmp_path: Path) -> None:
    app = open_page(tmp_path / "non_esiste")
    assert not app.exception
    assert "Nessun risultato" in app.error[0].value
    assert not app.slider


def traces(figure: Any) -> dict[str, Any]:
    """Tracce della figura, per nome (senza il testo tra parentesi)."""
    return {trace.name.split(" (")[0]: trace for trace in figure.data}


def test_ground_track_shows_satellite_station_and_region(results_dir: Path) -> None:
    results = SimulationResults.load(results_dir)
    config = MissionConfig.model_validate(results.metadata["config"])
    series = results.timeseries
    found = traces(ground_track_figure(series, config, 9))

    satellite = found["Satellite"]
    assert satellite.lat[0] == pytest.approx(
        math.degrees(series["latitude_rad"].iloc[9])
    )
    assert satellite.lon[0] == pytest.approx(
        math.degrees(series["longitude_rad"].iloc[9])
    )
    # Simulazione più corta di un'orbita: la scia parte dall'inizio e finisce
    # sul satellite.
    trail = found["Ultima orbita"]
    assert len(trail.lat) == 10
    assert (trail.lat[-1], trail.lon[-1]) == (satellite.lat[0], satellite.lon[0])

    station = found["Stazione di terra"]
    assert (station.lat[0], station.lon[0]) == (
        config.ground_station.latitude_deg,
        config.ground_station.longitude_deg,
    )
    region = found["Regione di acquisizione"]
    assert (region.lat[0], region.lon[0]) == (region.lat[-1], region.lon[-1])
    assert min(region.lat) == config.imaging.latitude_min_deg
    assert max(region.lat) == config.imaging.latitude_max_deg
    assert min(region.lon) == config.imaging.longitude_min_deg
    assert max(region.lon) == config.imaging.longitude_max_deg


def test_ground_track_trail_is_one_orbit() -> None:
    # Serie inventata di tre ore: conta solo la durata della scia.
    config = load_config(BASELINE)
    step_s = config.simulation.output_step_s
    time_s = np.arange(0.0, 3.0 * 3600.0, step_s)
    series = pd.DataFrame(
        {
            "time_s": time_s,
            "latitude_rad": 0.0,
            "longitude_rad": np.linspace(-3.0, 3.0, time_s.size),
            "station_visible": False,
        }
    )
    found = traces(ground_track_figure(series, config, time_s.size - 1))
    period_s = CircularOrbit.from_config(
        config.orbit, config.simulation.start_epoch
    ).period_s
    span_s = step_s * (len(found["Ultima orbita"].lat) - 1)
    assert period_s - step_s <= span_s < period_s


def arrow(figure: Any, name: str) -> FloatArray:
    """Versore della freccia con il nome indicato."""
    (line,) = [t for t in figure.data if t.type == "scatter3d" and t.name == name]
    tip = np.array([line.x[1], line.y[1], line.z[1]], dtype=np.float64)
    return tip / np.linalg.norm(tip)


def reference_attitudes(config: MissionConfig, elapsed_s: float) -> dict[str, Any]:
    """Assetti di riferimento dei modi all'istante indicato."""
    epoch = config.simulation.start_epoch
    position, velocity = CircularOrbit.from_config(config.orbit, epoch).state_eci(
        elapsed_s
    )
    jd = julian_date(epoch, elapsed_s)
    sun = sun_position_eci(jd)
    station = GroundStation.from_config(config.ground_station).position_eci(jd)
    axes = PointingAxes.from_config(config)
    return {
        "sun": sun_pointing_attitude(axes, sun, position, velocity),
        "nadir": nadir_pointing_attitude(axes, sun, position, velocity),
        "station": station_pointing_attitude(axes, sun, position, velocity, station),
    }


@pytest.mark.parametrize(
    ("reference", "body_arrow", "external_arrow"),
    [
        ("sun", "Pannelli", "Sole"),
        ("nadir", "Fotocamera e antenna", "Terra (nadir)"),
        ("station", "Fotocamera e antenna", "Stazione (Roma)"),
    ],
)
def test_attitude_arrows_match_reference(
    reference: str, body_arrow: str, external_arrow: str
) -> None:
    # Con l'assetto di riferimento di un modo, la freccia del corpo coincide
    # con la direzione esterna da puntare: controlla quaternione, rotazioni e
    # modelli dell'ambiente usati dalla vista 3D.
    config = load_config(BASELINE)
    elapsed_s = 36_320.0  # 10:05 UTC: passaggio diurno su Roma
    q = reference_attitudes(config, elapsed_s)[reference]
    series = pd.DataFrame(
        {
            "time_s": [elapsed_s],
            "attitude_w": [q[0]],
            "attitude_x": [q[1]],
            "attitude_y": [q[2]],
            "attitude_z": [q[3]],
            "in_eclipse": [False],
            "station_visible": [True],
        }
    )
    figure = attitude_figure(series, config, 0)
    assert (
        angle_between(arrow(figure, body_arrow), arrow(figure, external_arrow)) < 1e-9
    )
    # La Terra sta sempre in basso nella vista.
    np.testing.assert_allclose(arrow(figure, "Terra (nadir)"), [0, 0, -1], atol=1e-12)


def test_attitude_shows_station_only_when_visible(results_dir: Path) -> None:
    # Nei primi tre minuti Roma non è visibile.
    results = SimulationResults.load(results_dir)
    config = MissionConfig.model_validate(results.metadata["config"])
    names = {
        trace.name for trace in attitude_figure(results.timeseries, config, 0).data
    }
    assert {"Pannelli", "Fotocamera e antenna", "Sole (in eclissi)"} <= names
    assert "Stazione (Roma)" not in names


def test_timeline_marks_instant_and_converts_units(results_dir: Path) -> None:
    results = SimulationResults.load(results_dir)
    config = MissionConfig.model_validate(results.metadata["config"])
    series = results.timeseries.copy()
    # Ruota X al momento angolare massimo: deve girare alla velocità massima.
    series["wheel_momentum_x_nms"] = config.actuators.wheel_max_momentum_nms
    figure = timeline_figure(series, config, 9)
    found = traces(figure)

    np.testing.assert_allclose(found["Ruota X"].y, config.actuators.wheel_max_speed_rpm)
    np.testing.assert_allclose(found["Carica"].y, 100.0 * series["state_of_charge"])
    # Modi dal meno al più prioritario, dal basso verso l'alto.
    assert figure.layout.yaxis.categoryarray == (
        "SUN_POINTING",
        "NADIR",
        "DOWNLINK",
        "DETUMBLE",
        "SAFE",
    )
    # Una linea verticale per ciascuno dei cinque grafici, all'istante scelto.
    instants = [shape.x0 for shape in figure.layout.shapes if shape.x0 == shape.x1]
    assert instants == [pytest.approx(90.0 / 3600.0)] * 5


def test_new_simulation_from_sidebar(results_dir: Path) -> None:
    app = open_page(results_dir)
    app = app.sidebar.radio[0].set_value(NEW_SOURCE).run()
    # Prima di premere Simula non parte nulla: la pagina spiega che cosa fare.
    assert "premi Simula" in app.info[0].value
    assert not app.slider

    # Batteria da 1 Wh: nell'eclissi iniziale la carica scende sotto il 30 %.
    app.sidebar.number_input[0].set_value(0.5)  # durata [h]
    app.sidebar.number_input[1].set_value(1.0)  # capacità della batteria [Wh]
    app = app.sidebar.button[0].click().run()
    assert not app.exception
    caption = app.caption[0].value
    assert "0.5 h dal" in caption
    assert "Nuova simulazione, non salvata: batteria da 1 Wh" in caption
    assert app.slider
    metrics = {metric.label: metric.value for metric in app.metric[6:]}
    assert metrics["Tempo in SAFE"] != "0.00 h"
    # Con il SAFE nei profili lo studio 2 non si può fare, e la pagina lo dice.
    assert any("non calcolabile" in text.value for text in app.markdown)


def test_detumble_histogram_marks_percentile_and_mission() -> None:
    figure = detumble_figure(CAMPAIGN, p95_h=2.45, current_h=1.73)
    (histogram,) = figure.data
    # La corsa non finita non entra nell'istogramma.
    np.testing.assert_allclose(histogram.x, [1.5, 2.0, 2.5])
    assert [shape.x0 for shape in figure.layout.shapes] == [2.45, 1.73]
    without_mission = detumble_figure(CAMPAIGN, p95_h=2.45, current_h=None)
    assert [shape.x0 for shape in without_mission.layout.shapes] == [2.45]
