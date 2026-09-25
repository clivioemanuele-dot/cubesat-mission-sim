"""Test della riga di comando del simulatore."""

from pathlib import Path

import pytest

from cubesat_sim.core.results import SIMULATED_NOTICE, SimulationResults
from cubesat_sim.simulation.__main__ import main

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"


def test_command_line_runs_and_saves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "run"
    assert main([str(BASELINE), str(output), "--hours", "0.05"]) == 0
    results = SimulationResults.load(output)
    assert len(results.timeseries) == 18  # 180 s, un'uscita ogni 10 s
    assert results.metadata["config"]["simulation"]["duration_h"] == 0.05
    assert SIMULATED_NOTICE in capsys.readouterr().out


def test_missing_config_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        main([str(tmp_path / "missing.toml"), str(tmp_path / "run")])
