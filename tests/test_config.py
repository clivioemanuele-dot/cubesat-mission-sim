"""Test della configurazione: lettura dello scenario, conversioni SI e validazione."""

import datetime as dt
import math
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from cubesat_sim.core.config import MissionConfig, load_config

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"


def baseline_data() -> dict[str, Any]:
    """Legge lo scenario di riferimento come dizionario, senza validarlo."""
    with BASELINE.open("rb") as file:
        return tomllib.load(file)


def test_baseline_loads_and_converts_to_si() -> None:
    config = load_config(BASELINE)
    assert config.orbit.altitude_m == 500_000.0
    assert config.orbit.inclination_rad == pytest.approx(math.radians(97.4))
    assert config.orbit.descending_node_local_time_s == 10.5 * 3600
    assert config.battery.capacity_j == 40.0 * 3600
    assert config.battery.initial_soc == pytest.approx(0.70)
    assert config.actuators.wheel_max_torque_nm == pytest.approx(1.0e-3)
    assert config.control.max_slew_rate_rad_s == pytest.approx(math.radians(1.0))
    expected_inertia = 1.0e-2 / (6000.0 * 2.0 * math.pi / 60.0)
    assert config.actuators.wheel_inertia_kg_m2 == pytest.approx(expected_inertia)
    assert config.payload.antenna_half_beamwidth_rad == pytest.approx(
        math.radians(20.0)
    )
    assert config.payload.storage_capacity_bit == 8.0e9
    assert len(config.solar_array.panels) == 6


def test_json_round_trip() -> None:
    # La configurazione salvata in JSON con i risultati deve ricaricarsi identica.
    config = load_config(BASELINE)
    assert MissionConfig.model_validate(config.model_dump(mode="json")) == config


INVALID_VALUES = [
    # (sezione, campo, valore sbagliato, testo atteso nel messaggio di errore)
    ("orbit", "altitud_km", 500.0, "orbit.altitud_km"),
    ("orbit", "inclination_deg", 200.0, "orbit.inclination_deg"),
    ("battery", "capacity_wh", -40.0, "battery.capacity_wh"),
    ("battery", "capacity_wh", math.nan, "battery.capacity_wh"),
    ("solar_array", "cell_efficiency", 1.3, "solar_array.cell_efficiency"),
    ("satellite", "size_cm", [10.0, 10.0], "satellite.size_cm"),
    ("satellite", "inertia_kg_m2", [0.01, 0.01, 0.05], "non è fisicamente possibile"),
    ("payload", "camera_boresight_body", [1.0, 1.0, 0.0], "versore"),
    ("simulation", "start_epoch", dt.datetime(2026, 9, 23), "simulation.start_epoch"),
    ("simulation", "output_step_s", 10.1, "output_step_s"),
    ("control", "control_period_s", 0.3, "control.control_period_s"),
    ("control", "max_slew_rate_deg_s", 2.5, "max_slew_rate_deg_s deve essere minore"),
    ("modes", "safe_enter_soc_pct", 60.0, "safe_exit_soc_pct deve superare"),
    ("modes", "detumble_exit_rate_deg_s", 3.0, "detumble_enter_rate_deg_s"),
    ("imaging", "latitude_min_deg", 65.0, "latitude_min_deg deve essere minore"),
]


@pytest.mark.parametrize(("section", "key", "value", "message"), INVALID_VALUES)
def test_invalid_value_is_rejected(
    section: str, key: str, value: object, message: str
) -> None:
    data = baseline_data()
    data[section][key] = value
    with pytest.raises(ValidationError, match=re.escape(message)):
        MissionConfig.model_validate(data)


def test_missing_field_is_rejected() -> None:
    data = baseline_data()
    del data["battery"]["capacity_wh"]
    with pytest.raises(ValidationError, match=re.escape("battery.capacity_wh")):
        MissionConfig.model_validate(data)


def test_duplicate_panel_names_are_rejected() -> None:
    data = baseline_data()
    data["solar_array"]["panels"][1]["name"] = "wing_1"
    with pytest.raises(ValidationError, match="nomi dei pannelli ripetuti"):
        MissionConfig.model_validate(data)


def test_config_is_immutable() -> None:
    config = load_config(BASELINE)
    with pytest.raises(ValidationError, match="frozen"):
        config.orbit.altitude_km = 600.0
