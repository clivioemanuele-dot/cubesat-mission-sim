"""Grafici plotly della dashboard.

Funzioni pure: ricevono i risultati già calcolati e la configurazione e
restituiscono una figura. Nessuna logica di bordo: solo conversioni di unità
e scelte grafiche. La vista 3D ricalcola all'istante scelto i modelli
analitici dell'ambiente (orbita, Sole, stazione): sono deterministici e danno
gli stessi valori usati dalla simulazione.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cubesat_sim.core.config import MissionConfig
from cubesat_sim.core.quaternions import quat_to_matrix
from cubesat_sim.core.types import FloatArray
from cubesat_sim.core.vectors import unit
from cubesat_sim.environment.earth import julian_date
from cubesat_sim.environment.ground import GroundStation
from cubesat_sim.environment.orbit import CircularOrbit, eci_to_lvlh_matrix
from cubesat_sim.environment.sun import sun_position_eci
from cubesat_sim.modes.manager import Mode

# Tema scuro "spazio". Colori delle serie: i primi tre del set categorico per
# fondo scuro (blu, arancione, acqua), verificati con il controllo di
# separabilità per la visione dei colori, anche con daltonismo. Oltre tre
# elementi la differenza è affidata a forma, spessore o etichetta, non al solo
# colore.
_SURFACE = "#0f1629"
_INK = "#e6e9f2"
_INK_MUTED = "#8b93b0"
_GRID = "#222b45"
_AXIS = "#333d5c"
_BLUE = "#3987e5"
_ORANGE = "#d95926"
_AQUA = "#199e70"
_CRITICAL = "#d03b3b"
_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Punti per lato del rettangolo della regione di acquisizione: i lati seguono
# meridiani e paralleli invece degli archi di cerchio massimo tra gli angoli.
_REGION_EDGE_POINTS = 30

# Vista 3D dell'assetto. Riferimento di visualizzazione: LVLH ruotato di 180°
# attorno a x, così che z punti verso lo zenit (in alto sullo schermo) e y
# lungo la normale all'orbita.
_LVLH_TO_VIEW = np.diag([1.0, -1.0, -1.0])
# Lunghezze [m], scelte per un corpo lungo 34 cm: le frecce del corpo sono
# più corte e più spesse di quelle esterne, così restano visibili entrambe
# quando sono allineate.
_BODY_ARROW_M = 0.3
_EXTERNAL_ARROW_M = 0.42
_ARROW_HEAD_M = 0.05
_SCENE_HALF_WIDTH_M = 0.45
_M_PER_CM = 0.01

# Grafici nel tempo: conversioni di unità per la lettura.
_S_PER_H = 3600.0
_PERCENT = 100.0
_RPM_PER_RAD_S = 60.0 / (2.0 * math.pi)
# Errore di puntamento su scala logaritmica, da 0,01° a 200°: si leggono sia
# il puntamento fine sia le manovre.
_ERROR_RANGE_LOG_DEG = (-2.0, math.log10(200.0))
_ROW_HEIGHT_PX = 170

# Istogramma del Monte Carlo: larghezza delle classi [h], 6 minuti.
_DETUMBLE_BIN_H = 0.1


def _apply_theme(figure: go.Figure, height: int, top_margin: int) -> None:
    """Fondo, testo e assi del tema scuro, uguali per tutti i grafici."""
    figure.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": top_margin, "b": 8},
        paper_bgcolor=_SURFACE,
        plot_bgcolor=_SURFACE,
        font={"family": _FONT, "color": _INK, "size": 13},
        hoverlabel={"bgcolor": _SURFACE, "bordercolor": _AXIS, "font_color": _INK},
    )
    # automargin: i margini crescono quanto serve per etichette e titoli degli assi.
    axes = {"gridcolor": _GRID, "linecolor": _AXIS, "zerolinecolor": _AXIS}
    figure.update_xaxes(automargin=True, **axes)
    figure.update_yaxes(automargin=True, **axes)


def _region_outline(config: MissionConfig) -> tuple[FloatArray, FloatArray]:
    """Contorno chiuso della regione di acquisizione [gradi].

    Il contorno è percorso in senso orario (ovest, nord, est, sud): plotly
    colora l'interno di un poligono sulla sfera solo con questo verso.
    """
    imaging = config.imaging
    south, north = imaging.latitude_min_deg, imaging.latitude_max_deg
    west, east = imaging.longitude_min_deg, imaging.longitude_max_deg
    along_lat = np.linspace(south, north, _REGION_EDGE_POINTS)
    along_lon = np.linspace(west, east, _REGION_EDGE_POINTS)
    latitude = np.concatenate(
        [
            along_lat,
            np.full_like(along_lon, north),
            along_lat[::-1],
            np.full_like(along_lon, south),
        ]
    )
    longitude = np.concatenate(
        [
            np.full_like(along_lat, west),
            along_lon,
            np.full_like(along_lat, east),
            along_lon[::-1],
        ]
    )
    return latitude, longitude


def ground_track_figure(
    series: pd.DataFrame, config: MissionConfig, index: int
) -> go.Figure:
    """Traccia a terra con stazione, regione di acquisizione e satellite.

    Args:
        series: serie temporali salvate dalla simulazione.
        config: configurazione della simulazione.
        index: riga di ``series`` dell'istante scelto.

    Returns:
        Mappa del mondo: l'intera traccia in grigio, l'ultima orbita fino
        all'istante scelto in blu, i punti con la stazione visibile in verde
        acqua.
    """
    latitude = np.degrees(series["latitude_rad"].to_numpy())
    longitude = np.degrees(series["longitude_rad"].to_numpy())
    time_s = series["time_s"].to_numpy()
    period_s = CircularOrbit.from_config(
        config.orbit, config.simulation.start_epoch
    ).period_s
    trail = (time_s > time_s[index] - period_s) & (time_s <= time_s[index])
    visible = series["station_visible"].to_numpy(dtype=bool)
    region_lat, region_lon = _region_outline(config)
    station = config.ground_station

    figure = go.Figure()
    figure.add_scattergeo(
        lat=latitude,
        lon=longitude,
        mode="lines",
        line={"color": _AXIS, "width": 1},
        name="Traccia a terra",
        hoverinfo="skip",
    )
    figure.add_scattergeo(
        lat=region_lat,
        lon=region_lon,
        mode="lines",
        fill="toself",
        fillcolor="rgba(217, 89, 38, 0.18)",
        line={"color": _ORANGE, "width": 1},
        name=f"Regione di acquisizione ({config.imaging.region_name})",
        hoverinfo="skip",
    )
    figure.add_scattergeo(
        lat=latitude[visible],
        lon=longitude[visible],
        mode="markers",
        marker={"color": _AQUA, "size": 4},
        name="Stazione visibile",
        hoverinfo="skip",
    )
    figure.add_scattergeo(
        lat=latitude[trail],
        lon=longitude[trail],
        mode="lines",
        line={"color": _BLUE, "width": 2},
        name="Ultima orbita",
        hoverinfo="skip",
    )
    figure.add_scattergeo(
        lat=[station.latitude_deg],
        lon=[station.longitude_deg],
        mode="markers+text",
        marker={
            "color": _INK,
            "size": 11,
            "symbol": "triangle-up",
            "line": {"color": _SURFACE, "width": 1},
        },
        text=[station.name],
        textposition="bottom center",
        textfont={"color": _INK, "size": 12},
        name=f"Stazione di terra ({station.name})",
    )
    figure.add_scattergeo(
        lat=[latitude[index]],
        lon=[longitude[index]],
        mode="markers",
        marker={
            "color": _INK,
            "size": 13,
            "line": {"color": _BLUE, "width": 3},
        },
        name="Satellite",
    )
    figure.update_geos(
        projection_type="natural earth",
        bgcolor=_SURFACE,
        showland=True,
        landcolor="#1b2542",
        showocean=True,
        oceancolor="#0b1226",
        showlakes=False,
        showcountries=True,
        countrycolor="#34406a",
        coastlinecolor="#4a5687",
        showframe=False,
    )
    _apply_theme(figure, height=450, top_margin=8)
    figure.update_layout(
        legend={"orientation": "h", "y": -0.02, "font": {"color": _INK_MUTED}},
        # Lo zoom scelto con il mouse resta quando si sposta il cursore.
        uirevision="ground_track",
    )
    return figure


def _box_vertices(size_m: FloatArray) -> FloatArray:
    """Gli 8 vertici di un parallelepipedo centrato nell'origine, per righe."""
    signs = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float64,
    )
    return 0.5 * signs * size_m


def _wing_vertices(size_m: FloatArray, side: int) -> FloatArray:
    """I 4 vertici di un'ala, nel piano della faccia +X.

    Disegno indicativo: la configurazione dà area e orientamento dei pannelli,
    non la posizione. Le ali sono disegnate grandi come la faccia +X,
    incernierate ai suoi bordi e aperte lungo +Y (side=1) o -Y (side=-1).
    """
    half_x, half_y, half_z = 0.5 * size_m
    near, far = side * half_y, side * 3.0 * half_y
    return np.array(
        [
            [half_x, near, -half_z],
            [half_x, far, -half_z],
            [half_x, far, half_z],
            [half_x, near, half_z],
        ]
    )


def _add_arrow(
    figure: go.Figure,
    direction: FloatArray,
    color: str,
    name: str,
    *,
    body: bool,
) -> None:
    """Freccia dall'origine lungo una direzione (versore).

    Le frecce del corpo (body=True) sono più corte e più spesse di quelle
    esterne.
    """
    length, width = (_BODY_ARROW_M, 8) if body else (_EXTERNAL_ARROW_M, 4)
    tip = length * direction
    figure.add_scatter3d(
        x=[0.0, tip[0]],
        y=[0.0, tip[1]],
        z=[0.0, tip[2]],
        mode="lines",
        line={"color": color, "width": width},
        name=name,
        hoverinfo="name",
    )
    figure.add_cone(
        x=[tip[0]],
        y=[tip[1]],
        z=[tip[2]],
        u=[direction[0]],
        v=[direction[1]],
        w=[direction[2]],
        anchor="tip",
        sizemode="absolute",
        sizeref=_ARROW_HEAD_M,
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        hoverinfo="skip",
    )


def attitude_figure(
    series: pd.DataFrame, config: MissionConfig, index: int
) -> go.Figure:
    """Assetto del satellite visto nel riferimento orbitale.

    Vista di chi vola accanto al satellite: x nel verso del moto, z verso lo
    zenit (la Terra sta in basso), y lungo la normale all'orbita. Le frecce
    del corpo (pannelli, fotocamera, antenna) ruotano con l'assetto vero
    salvato; quelle esterne (Sole, Terra, stazione) vengono dai modelli
    analitici dell'ambiente, ricalcolati all'istante scelto.

    Args:
        series: serie temporali salvate dalla simulazione.
        config: configurazione della simulazione.
        index: riga di ``series`` dell'istante scelto.

    Returns:
        Vista 3D con il corpo, le ali e le frecce di puntamento.
    """
    row = series.iloc[index]
    elapsed_s = float(row["time_s"])
    epoch = config.simulation.start_epoch
    orbit = CircularOrbit.from_config(config.orbit, epoch)
    position, velocity = orbit.state_eci(elapsed_s)
    jd = julian_date(epoch, elapsed_s)
    eci_to_view = _LVLH_TO_VIEW @ eci_to_lvlh_matrix(position, velocity)
    attitude = np.array(
        [row["attitude_w"], row["attitude_x"], row["attitude_y"], row["attitude_z"]],
        dtype=np.float64,
    )
    body_to_view = eci_to_view @ quat_to_matrix(attitude)

    size_m = np.array(config.satellite.size_cm, dtype=np.float64) * _M_PER_CM
    body = _box_vertices(size_m) @ body_to_view.T
    figure = go.Figure()
    figure.add_mesh3d(
        x=body[:, 0],
        y=body[:, 1],
        z=body[:, 2],
        alphahull=0,
        color="#c3c2b7",
        flatshading=True,
        name="Corpo 3U",
        hoverinfo="name",
    )
    for side in (1, -1):
        wing = _wing_vertices(size_m, side) @ body_to_view.T
        figure.add_mesh3d(
            x=wing[:, 0],
            y=wing[:, 1],
            z=wing[:, 2],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="#1c5cab",
            name="Ala solare",
            hoverinfo="name",
        )

    solar, payload = config.solar_array, config.payload
    # Ogni asse del corpo ha il colore della direzione esterna a cui deve
    # puntare: pannelli e Sole in arancione, fotocamera e Terra in blu,
    # antenna e stazione in acqua. Lo spessore distingue corpo ed esterno.
    body_arrows = [
        (solar.sun_pointing_axis_body, _ORANGE, "pannelli"),
        (payload.camera_boresight_body, _BLUE, "fotocamera"),
        (payload.antenna_boresight_body, _AQUA, "antenna"),
    ]
    # Assi del corpo coincidenti (nello scenario di riferimento fotocamera e
    # antenna): una sola freccia con tutti i nomi, invece di frecce sovrapposte.
    groups: dict[tuple[float, float, float], tuple[str, list[str]]] = {}
    for axis, color, name in body_arrows:
        groups.setdefault(axis, (color, []))[1].append(name)
    for axis, (color, names) in groups.items():
        direction = body_to_view @ np.array(axis, dtype=np.float64)
        name = " e ".join(names).capitalize()
        _add_arrow(figure, direction, color, name, body=True)

    sun = eci_to_view @ unit(sun_position_eci(jd))
    sun_name = "Sole (in eclissi)" if row["in_eclipse"] else "Sole"
    _add_arrow(figure, sun, _ORANGE, sun_name, body=False)
    nadir = eci_to_view @ unit(-position)
    _add_arrow(figure, nadir, _BLUE, "Terra (nadir)", body=False)
    if row["station_visible"]:
        station = GroundStation.from_config(config.ground_station)
        line_of_sight = eci_to_view @ unit(station.position_eci(jd) - position)
        name = f"Stazione ({config.ground_station.name})"
        _add_arrow(figure, line_of_sight, _AQUA, name, body=False)

    axis_range = [-_SCENE_HALF_WIDTH_M, _SCENE_HALF_WIDTH_M]
    axis_style = {
        "range": axis_range,
        "showticklabels": False,
        "backgroundcolor": _SURFACE,
        "gridcolor": _GRID,
        "zerolinecolor": _AXIS,
        "showbackground": True,
    }
    figure.update_scenes(
        xaxis={**axis_style, "title": {"text": "moto", "font": {"color": _INK_MUTED}}},
        yaxis={
            **axis_style,
            "title": {"text": "normale", "font": {"color": _INK_MUTED}},
        },
        zaxis={**axis_style, "title": {"text": "zenit", "font": {"color": _INK_MUTED}}},
        bgcolor=_SURFACE,
        aspectmode="cube",
        camera={"eye": {"x": 1.0, "y": -1.8, "z": 0.8}},
    )
    _apply_theme(figure, height=450, top_margin=48)
    figure.update_layout(
        legend={
            "orientation": "h",
            "y": 1.0,
            "yanchor": "bottom",
            "font": {"color": _INK_MUTED, "size": 12},
        },
        # La rotazione scelta con il mouse resta quando si sposta il cursore.
        uirevision="attitude",
    )
    return figure


def timeline_figure(
    series: pd.DataFrame, config: MissionConfig, index: int
) -> go.Figure:
    """Andamento nel tempo di modi, batteria, potenza, puntamento e ruote.

    Args:
        series: serie temporali salvate dalla simulazione.
        config: configurazione della simulazione.
        index: riga di ``series`` dell'istante scelto, segnato da una linea
            verticale.

    Returns:
        Cinque grafici sovrapposti con lo stesso asse dei tempi, in ore
        dall'inizio della simulazione. I grafici con una sola grandezza sono
        descritti dal titolo; quelli con più grandezze hanno una legenda
        propria.
    """
    hours = series["time_s"].to_numpy() / _S_PER_H
    modes, wheels = config.modes, config.actuators
    titles = [
        "Modo",
        "Batteria [%]",
        "Potenza [W]",
        "Errore di puntamento [°]",
        "Velocità delle ruote [giri/min]",
    ]
    figure = make_subplots(
        rows=len(titles),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=titles,
    )
    threshold = {"dash": "dot", "width": 1}

    # Modi in ordine di priorità, dal più basso (in basso) al più alto.
    order = [mode.value for mode in reversed(Mode)]
    figure.add_scatter(
        x=hours,
        y=series["mode"],
        mode="lines",
        line={"shape": "hv", "color": _INK, "width": 1.5},
        name="Modo",
        showlegend=False,
        row=1,
        col=1,
    )
    figure.update_yaxes(categoryorder="array", categoryarray=order, row=1, col=1)

    figure.add_scatter(
        x=hours,
        y=series["state_of_charge"] * _PERCENT,
        line={"color": _BLUE, "width": 2},
        name="Carica",
        showlegend=False,
        row=2,
        col=1,
    )
    for level, label in [
        (modes.safe_enter_soc_pct, "ingresso SAFE"),
        (modes.safe_exit_soc_pct, "uscita SAFE"),
    ]:
        figure.add_hline(
            y=level,
            line={**threshold, "color": _CRITICAL},
            annotation_text=label,
            annotation_position="bottom right",
            annotation_font_color=_INK_MUTED,
            row=2,
            col=1,
            exclude_empty_subplots=False,
        )
    figure.update_yaxes(range=[0, _PERCENT], row=2, col=1)

    figure.add_scatter(
        x=hours,
        y=series["solar_power_w"],
        line={"color": _ORANGE, "width": 1.5},
        name="Prodotta",
        row=3,
        col=1,
    )
    figure.add_scatter(
        x=hours,
        y=series["consumed_power_w"],
        line={"color": _BLUE, "width": 1.5},
        name="Consumata",
        row=3,
        col=1,
    )

    figure.add_scatter(
        x=hours,
        y=np.degrees(series["pointing_error_rad"].to_numpy()),
        line={"color": _BLUE, "width": 1},
        name="Errore di puntamento",
        showlegend=False,
        row=4,
        col=1,
    )
    figure.add_hline(
        y=math.degrees(config.payload.antenna_half_beamwidth_rad),
        line={**threshold, "color": _INK_MUTED},
        annotation_text="metà fascio dell'antenna",
        annotation_position="bottom right",
        annotation_font_color=_INK_MUTED,
        row=4,
        col=1,
        exclude_empty_subplots=False,
    )
    figure.update_yaxes(type="log", range=_ERROR_RANGE_LOG_DEG, row=4, col=1)

    for axis, color in zip("xyz", [_BLUE, _ORANGE, _AQUA], strict=True):
        momentum = series[f"wheel_momentum_{axis}_nms"].to_numpy()
        figure.add_scatter(
            x=hours,
            y=momentum / wheels.wheel_inertia_kg_m2 * _RPM_PER_RAD_S,
            line={"color": color, "width": 1.5},
            name=f"Ruota {axis.upper()}",
            legend="legend2",
            row=5,
            col=1,
        )
    for sign in (1, -1):
        figure.add_hline(
            y=sign * wheels.wheel_max_speed_rpm,
            line={**threshold, "color": _CRITICAL},
            row=5,
            col=1,
            exclude_empty_subplots=False,
        )

    figure.add_vline(
        x=hours[index],
        line={"color": _INK, "width": 1},
        opacity=0.8,
        row="all",
        col=1,
        exclude_empty_subplots=False,
    )
    figure.update_xaxes(title_text="Ore dall'inizio della simulazione", row=5, col=1)
    _apply_theme(figure, height=_ROW_HEIGHT_PX * len(titles), top_margin=32)
    # Legende dentro i due grafici con più grandezze, in alto a destra.
    legend_style = {
        "orientation": "h",
        "x": 1.0,
        "xanchor": "right",
        "yanchor": "bottom",
        "bgcolor": "rgba(0, 0, 0, 0)",
        "font": {"color": _INK_MUTED},
    }
    figure.update_layout(
        legend={**legend_style, "y": figure.layout.yaxis3.domain[1]},
        legend2={**legend_style, "y": figure.layout.yaxis5.domain[1]},
        hovermode="x unified",
        # Lo zoom scelto con il mouse resta quando si sposta il cursore.
        uirevision="timeline",
    )
    return figure


def detumble_figure(
    table: pd.DataFrame, p95_h: float, current_h: float | None
) -> go.Figure:
    """Istogramma dei tempi di detumble della campagna Monte Carlo.

    Args:
        table: una riga per corsa, con la colonna ``detumble_time_h`` (NaN se
            il detumble non finisce entro la finestra simulata).
        p95_h: 95° percentile dei tempi di detumble [h].
        current_h: detumble della simulazione mostrata nella pagina [h], o
            None se non è completato.

    Returns:
        Istogramma con il 95° percentile e, se c'è, la simulazione mostrata.
    """
    times_h = table["detumble_time_h"].dropna().to_numpy(dtype=float)
    figure = go.Figure()
    figure.add_histogram(
        x=times_h,
        xbins={"size": _DETUMBLE_BIN_H},
        marker={"color": _BLUE},
        name="Rilasci",
    )
    figure.add_vline(
        x=p95_h,
        line={"dash": "dash", "color": _INK, "width": 1},
        annotation_text="95 %",
        annotation_position="top right",
        annotation_font_color=_INK,
    )
    if current_h is not None:
        figure.add_vline(
            x=current_h,
            line={"color": _ORANGE, "width": 2},
            annotation_text="simulazione mostrata",
            annotation_position="top left",
            annotation_font_color=_ORANGE,
        )
    figure.update_xaxes(title_text="Durata del detumble [h]")
    figure.update_yaxes(title_text="Rilasci")
    _apply_theme(figure, height=320, top_margin=32)
    figure.update_layout(showlegend=False, bargap=0.08)
    return figure
