"""Test della riga di comando dell'analisi."""

import json
from pathlib import Path

import pytest

from cubesat_sim.analysis.__main__ import ANALYSIS_FILE, main
from cubesat_sim.core.config import MissionConfig, load_config
from cubesat_sim.core.results import SIMULATED_NOTICE
from cubesat_sim.simulation.runner import run_simulation

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"


def test_analysis_prints_and_saves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Tre minuti dal rilascio: il satellite è ancora in DETUMBLE e non c'è
    # nessuna orbita completa, ma l'analisi deve funzionare lo stesso.
    data = load_config(BASELINE).model_dump(mode="json")
    data["simulation"]["duration_h"] = 0.05
    run_simulation(MissionConfig.model_validate(data)).save(tmp_path)

    assert main([str(tmp_path)]) == 0

    printed = capsys.readouterr().out
    assert SIMULATED_NOTICE in printed
    assert "Detumble: mai completato" in printed
    assert "nessuna orbita completa" in printed
    saved = json.loads((tmp_path / ANALYSIS_FILE).read_text(encoding="utf-8"))
    assert saved["notice"] == SIMULATED_NOTICE
    assert set(saved) == {
        "notice",
        "metrics",
        "pointing_energy_trade",
        "battery_sizing_trade",
    }
    assert saved["metrics"]["duration_h"] == 0.05
    assert saved["metrics"]["detumble_time_h"] is None
