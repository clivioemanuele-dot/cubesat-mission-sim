"""Test dell'analisi Monte Carlo del detumble."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cubesat_sim.analysis.metrics import compute_metrics
from cubesat_sim.analysis.monte_carlo import (
    RUNS_FILE,
    SUMMARY_FILE,
    detumble_run,
    load_runs,
    main,
    run_campaign,
    summarize,
)
from cubesat_sim.core.config import MissionConfig, load_config
from cubesat_sim.core.results import SIMULATED_NOTICE
from cubesat_sim.simulation.runner import run_simulation

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"
# Rilascio lento e finestra di 9 minuti, per test rapidi: con questi valori il
# detumble finisce con i semi 1, 3 e 7, non con il seme 2.
SLOW_RELEASE_DEG_S = 0.7
HOURS = 0.15


def slow_release() -> MissionConfig:
    """Scenario di riferimento con rilascio lento."""
    data = load_config(BASELINE).model_dump(mode="json")
    data["initial_state"]["max_rate_deg_s"] = SLOW_RELEASE_DEG_S
    return MissionConfig.model_validate(data)


def test_run_matches_a_single_simulation() -> None:
    config = slow_release()
    run = detumble_run(config, seed=3, hours=HOURS)

    data = config.model_dump(mode="json")
    data["simulation"]["seed"] = 3
    data["simulation"]["duration_h"] = HOURS
    expected = compute_metrics(run_simulation(MissionConfig.model_validate(data)))
    assert run.seed == 3
    assert run.detumble_time_h is not None
    assert run.detumble_time_h == expected.detumble_time_h
    # Al rilascio la velocità vale sempre il massimo: cambia solo l'asse.
    assert math.hypot(*run.release_rate_deg_s) == pytest.approx(SLOW_RELEASE_DEG_S)


def test_parallel_campaign_is_reproducible() -> None:
    # Due processi: risultati nell'ordine dei semi e identici alle corse singole.
    config = slow_release()
    runs = list(run_campaign(config, [3, 7], HOURS, workers=2))
    assert [run.seed for run in runs] == [3, 7]
    assert runs == [detumble_run(config, seed, HOURS) for seed in (3, 7)]
    assert runs[0].release_rate_deg_s != runs[1].release_rate_deg_s


def test_summary_of_completed_runs() -> None:
    table = pd.DataFrame(
        {"seed": [1, 2, 3, 4], "detumble_time_h": [1.0, 2.0, 3.0, math.nan]}
    )
    statistics = summarize(table)
    assert (statistics.runs, statistics.completed) == (4, 3)
    assert statistics.mean_h == pytest.approx(2.0)
    assert statistics.max_h == pytest.approx(3.0)
    assert statistics.p95_h == pytest.approx(np.percentile([1.0, 2.0, 3.0], 95.0))
    with pytest.raises(ValueError, match="nessuna corsa"):
        summarize(table.iloc[3:])


def test_command_line_saves_runs_and_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = BASELINE.read_text(encoding="utf-8")
    slow_text = text.replace("max_rate_deg_s = 10.0", "max_rate_deg_s = 0.7")
    assert slow_text != text
    config_path = tmp_path / "slow.toml"
    config_path.write_text(slow_text, encoding="utf-8")
    output = tmp_path / "campaign"

    args = [str(config_path), str(output), "--runs", "2", "--hours", "0.15"]
    assert main([*args, "--workers", "2"]) == 0

    assert SIMULATED_NOTICE in capsys.readouterr().out
    csv_text = (output / RUNS_FILE).read_text(encoding="utf-8")
    assert csv_text.startswith(f"# {SIMULATED_NOTICE}\n")
    table = load_runs(output)
    assert list(table["seed"]) == [1, 2]
    # Seme 2: detumble non finito entro la finestra, salvato come valore mancante.
    assert not math.isnan(table["detumble_time_h"].iloc[0])
    assert math.isnan(table["detumble_time_h"].iloc[1])
    summary = json.loads((output / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert summary["notice"] == SIMULATED_NOTICE
    assert summary["hours_per_run"] == 0.15
    assert summary["seeds"] == [1, 2]
    assert summary["statistics"]["runs"] == 2
    assert summary["statistics"]["completed"] == 1
