"""Test del salvataggio e della lettura dei risultati."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cubesat_sim.core.config import MissionConfig, load_config
from cubesat_sim.core.results import (
    METADATA_FILE,
    SIMULATED_NOTICE,
    TIMESERIES_FILE,
    SimulationResults,
)

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"
CONFIG = load_config(BASELINE)


def sample_results() -> SimulationResults:
    timeseries = pd.DataFrame(
        {
            "time_s": [0.0, 10.0, 20.0],
            "mode": ["DETUMBLE", "DETUMBLE", "SUN_POINTING"],
            "state_of_charge": [0.7, 0.699812345678, 0.6995],
            "in_eclipse": [False, True, True],
        }
    )
    metadata: dict[str, Any] = {
        "seed": CONFIG.simulation.seed,
        "config": CONFIG.model_dump(mode="json"),
        "transitions": [
            {
                "time_s": 10.0,
                "previous": "DETUMBLE",
                "current": "SUN_POINTING",
                "cause": "rotazione smorzata (0.10 °/s)",
            }
        ],
    }
    return SimulationResults(timeseries=timeseries, metadata=metadata)


def test_round_trip(tmp_path: Path) -> None:
    results = sample_results()
    results.save(tmp_path)
    loaded = SimulationResults.load(tmp_path)
    pd.testing.assert_frame_equal(loaded.timeseries, results.timeseries, rtol=1e-9)
    assert loaded.metadata == {"notice": SIMULATED_NOTICE, **results.metadata}


def test_saved_config_is_enough_to_rerun(tmp_path: Path) -> None:
    sample_results().save(tmp_path)
    loaded = SimulationResults.load(tmp_path)
    assert MissionConfig.model_validate(loaded.metadata["config"]) == CONFIG


def test_files_declare_simulated_data(tmp_path: Path) -> None:
    sample_results().save(tmp_path)
    csv_lines = (tmp_path / TIMESERIES_FILE).read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == f"# {SIMULATED_NOTICE}"
    assert csv_lines[1] == "time_s,mode,state_of_charge,in_eclipse"
    json_text = (tmp_path / METADATA_FILE).read_text(encoding="utf-8")
    assert SIMULATED_NOTICE in json_text


def test_saving_is_deterministic_with_unix_line_endings(tmp_path: Path) -> None:
    sample_results().save(tmp_path / "runs" / "first")
    sample_results().save(tmp_path / "runs" / "second")
    for name in (TIMESERIES_FILE, METADATA_FILE):
        first = (tmp_path / "runs" / "first" / name).read_bytes()
        assert first == (tmp_path / "runs" / "second" / name).read_bytes()
        assert b"\r\n" not in first


def test_invalid_metadata_leaves_no_files(tmp_path: Path) -> None:
    results = SimulationResults(
        timeseries=sample_results().timeseries,
        metadata={"seed": np.int64(42)},
    )
    target = tmp_path / "run"
    with pytest.raises(TypeError, match="not JSON serializable"):
        results.save(target)
    assert not target.exists()
