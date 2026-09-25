"""Test dei testi esplicativi della dashboard."""

import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from cubesat_sim.core.config import MissionConfig, load_config
from cubesat_sim.dashboard.explanations import (
    attitude_help,
    current_activity,
    ground_track_help,
    metrics_help,
    monte_carlo_help,
    timeline_help,
)
from cubesat_sim.modes.manager import Mode

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"
CONFIG = load_config(BASELINE)

# Due transizioni inventate; la simulazione parte alle 00:00 UTC.
TRANSITIONS = [
    {
        "time_s": 6243.0,  # 01:44:03 UTC
        "previous": "DETUMBLE",
        "current": "SUN_POINTING",
        "cause": "rotazione smorzata (0.51 °/s)",
    },
    {
        "time_s": 35_951.0,  # 09:59:11 UTC
        "previous": "SUN_POINTING",
        "current": "NADIR",
        "cause": "inizio della finestra di acquisizione",
    },
]


def explain(transitions: list[dict[str, Any]] | None = None, **changes: Any) -> str:
    """Testo dell'istante per una riga inventata; ``changes`` cambia i valori."""
    row: dict[str, Any] = {
        "time_s": 0.0,
        "mode": "SUN_POINTING",
        "angular_rate_rad_s": math.radians(0.1),
        "pointing_error_rad": math.radians(0.05),
        "antenna_error_rad": math.nan,
        "in_eclipse": False,
        "imaging_window": False,
        "state_of_charge": 0.8,
        "solar_power_w": 10.0,
        "consumed_power_w": 4.0,
        "data_stored_bit": 2.5e9,
        # Ruota Y al 40 % del momento massimo (10 mN m s).
        "wheel_momentum_x_nms": 0.0,
        "wheel_momentum_y_nms": -0.004,
        "wheel_momentum_z_nms": 0.001,
    }
    row.update(changes)
    return current_activity(pd.DataFrame([row]), 0, CONFIG, transitions or [])


def has_nan(text: str) -> bool:
    """Vero se nel testo è finito un valore non definito."""
    return re.search(r"\bnan\b", text) is not None


@pytest.mark.parametrize("mode", list(Mode))
def test_every_mode_is_explained(mode: Mode) -> None:
    text = explain(mode=str(mode))
    assert text.startswith(f"**{mode}:")
    assert "**Potenza**" in text
    assert not has_nan(text)


def test_detumble_explains_bdot_and_thresholds() -> None:
    text = explain(
        mode="DETUMBLE",
        angular_rate_rad_s=math.radians(10.0),
        pointing_error_rad=math.nan,
    )
    assert "ruota a 10.00 °/s" in text
    assert "B-dot" in text
    assert "sotto 0.5 °/s" in text
    assert "sopra 2 °/s" in text
    # In DETUMBLE le ruote non lavorano: il loro paragrafo non c'è.
    assert "**Ruote**" not in text
    assert "Nessuna transizione finora" in text
    assert not has_nan(text)


@pytest.mark.parametrize(
    ("antenna_error_deg", "expected"),
    [
        (5.0, "i dati scendono a 1 Mbit/s"),
        (30.0, "i dati non passano"),
        (math.nan, "appena scesa sotto l'elevazione minima"),
    ],
)
def test_downlink_depends_on_antenna_error(
    antenna_error_deg: float, expected: str
) -> None:
    # Fascio di 40°: i dati passano entro 20° dalla stazione.
    text = explain(mode="DOWNLINK", antenna_error_rad=math.radians(antenna_error_deg))
    assert expected in text
    assert not has_nan(text)


def test_nadir_names_region_and_trade_study() -> None:
    text = explain(mode="NADIR", imaging_window=True)
    assert "acquisizione immagini su Europa" in text
    assert "registra 2 Mbit/s" in text
    assert "2.500 Gbit su 8" in text
    assert "studio di compromesso 1" in text
    assert "appena chiusa" not in text
    assert "appena chiusa" in explain(mode="NADIR", imaging_window=False)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, "i 6.0 W in più ricaricano la batteria"),
        (
            {"solar_power_w": 0.0, "consumed_power_w": 2.0, "in_eclipse": True},
            "la batteria copre i 2.0 W mancanti",
        ),
        ({"state_of_charge": 1.0}, "la batteria è piena"),
    ],
)
def test_power_balance(changes: dict[str, Any], expected: str) -> None:
    assert expected in explain(**changes)


def test_wheel_fill_uses_most_loaded_wheel() -> None:
    assert "al 40 % del suo momento" in explain()


@pytest.mark.parametrize(
    ("time_s", "ago"),
    [(6288.0, "45 s fa"), (7200.0, "16 min fa"), (13_743.0, "2 h 05 min fa")],
)
def test_last_transition_and_elapsed_time(time_s: float, ago: str) -> None:
    text = explain(TRANSITIONS, time_s=time_s)
    assert f"alle 01:44:03 UTC, {ago}: DETUMBLE → SUN_POINTING" in text
    assert "**Prossima** alle 09:59:11 UTC" in text


def test_next_transition_and_end_of_simulation() -> None:
    before = explain(TRANSITIONS, time_s=7200.0)
    assert "tra 7 h 59 min: SUN_POINTING → NADIR" in before
    after = explain(TRANSITIONS, time_s=40_000.0)
    assert "SUN_POINTING → NADIR" in after
    assert "Nessun'altra transizione" in after


def test_ground_track_help_uses_orbit() -> None:
    text = ground_track_help(CONFIG)
    # Periodo a 500 km: 94.6 min; in quel tempo la Terra ruota di 23.7° rispetto
    # alle stelle. Con 97.4° di inclinazione si arriva a 180 - 97.4 = 82.6°.
    assert "dura 94.6 min" in text
    assert "di 23.7°" in text
    assert "82.6° di latitudine" in text
    assert "(10:30)" in text
    assert "Roma (triangolo)" in text
    assert "(Europa)" in text


def with_antenna(boresight: list[float]) -> MissionConfig:
    """Scenario di riferimento con un altro asse dell'antenna."""
    data = CONFIG.model_dump(mode="json")
    data["payload"]["antenna_boresight_body"] = boresight
    return MissionConfig.model_validate(data)


def test_attitude_help_follows_body_axes() -> None:
    shared = attitude_help(CONFIG)
    assert "10 x 10 x 34 cm" in shared
    assert "unica freccia blu" in shared
    separate = attitude_help(with_antenna([0.0, 0.0, -1.0]))
    assert "antenna e stazione in verde acqua" in separate


def test_timeline_help_uses_thresholds_and_gains() -> None:
    text = timeline_help(CONFIG)
    # Ogni modo ha la sua riga, con le condizioni che lo attivano.
    for mode in Mode:
        assert f"**{mode}**:" in text
    assert "rotazione sopra 2 °/s" in text
    assert "Roma vede il satellite sopra 10°" in text
    assert "regione di acquisizione (Europa)" in text
    assert "sotto il 30 %" in text
    assert "sopra il 50 %" in text
    assert "Consumo di base 2 W" in text
    # Assestamento al 2 %: 4 / (0.7 x 0.1 rad/s) = 57 s.
    assert "circa 57 s" in text
    assert "(20°, metà del fascio)" in text
    assert "±6000 giri/min" in text


@pytest.mark.parametrize(
    ("help_text", "expected"),
    [
        (metrics_help, "(8 Gbit)"),
        (monte_carlo_help, "6 volte minore"),
        (monte_carlo_help, "(10 °/s)"),
    ],
)
def test_other_help_texts_use_configuration(
    help_text: Callable[[MissionConfig], str], expected: str
) -> None:
    assert expected in help_text(CONFIG)


@pytest.mark.parametrize(
    "help_text",
    [ground_track_help, attitude_help, timeline_help, metrics_help, monte_carlo_help],
)
def test_help_texts_are_complete(help_text: Callable[[MissionConfig], str]) -> None:
    # Nessun segnaposto rimasto e nessun valore non definito.
    text = help_text(CONFIG)
    assert "{" not in text
    assert not has_nan(text)
