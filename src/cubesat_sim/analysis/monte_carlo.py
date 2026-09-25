"""Analisi Monte Carlo del detumble: molti rilasci casuali, una statistica.

Una sola simulazione dice quanto dura il detumble per un rilascio; molte
simulazioni con semi diversi dicono entro quanto tempo si chiude nella gran
parte dei rilasci. Ogni corsa ripete lo scenario con un seme diverso: cambiano
l'assetto e l'asse di rotazione al rilascio (la velocità resta quella massima,
il caso peggiore) e il rumore dei sensori. Orbita, Sole ed eclissi sono uguali
per tutte le corse. Si simulano solo le prime ore, quanto basta al detumble.

Limite dichiarato: i parametri fisici (inerzia, dipolo residuo, velocità al
rilascio, guadagni) non vengono dispersi.

Uso, dalla cartella del progetto:
    uv run python -m cubesat_sim.analysis.monte_carlo config/baseline.toml
        results/monte_carlo

Le corse girano in parallelo su più processi. I risultati sono dati simulati.
"""

import argparse
import json
import math
import os
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import repeat
from pathlib import Path

import numpy as np
import pandas as pd

from cubesat_sim.analysis.metrics import compute_metrics
from cubesat_sim.core.config import MissionConfig, load_config
from cubesat_sim.core.results import SIMULATED_NOTICE
from cubesat_sim.simulation.runner import run_simulation

RUNS_FILE = "detumble_runs.csv"
SUMMARY_FILE = "summary.json"
DEFAULT_RUNS = 100
# Finestra simulata per ogni corsa: su 100 rilasci il detumble più lungo è di
# 2,3 h. Le corse che non finiscono entro la finestra sono contate a parte.
DEFAULT_HOURS = 3.0
_PERCENTILE = 95.0
_JSON_INDENT = 2


@dataclass(frozen=True)
class DetumbleRun:
    """Una corsa della campagna.

    Attributes:
        seed: seme del generatore casuale.
        detumble_time_h: durata del detumble [h]; None se non finisce entro
            la finestra simulata.
        release_rate_deg_s: velocità angolare al rilascio, assi del corpo
            [gradi/s].
    """

    seed: int
    detumble_time_h: float | None
    release_rate_deg_s: tuple[float, float, float]


@dataclass(frozen=True)
class DetumbleStatistics:
    """Statistica dei tempi di detumble.

    Attributes:
        runs: corse eseguite.
        completed: corse con il detumble finito entro la finestra; le
            statistiche seguenti sono calcolate su queste.
        mean_h: durata media [h].
        p95_h: 95° percentile: il 95 % dei rilasci finisce prima [h].
        max_h: durata massima [h].
    """

    runs: int
    completed: int
    mean_h: float
    p95_h: float
    max_h: float


def detumble_run(config: MissionConfig, seed: int, hours: float) -> DetumbleRun:
    """Esegue una corsa: lo scenario con il seme e la durata indicati.

    Args:
        config: scenario di partenza.
        seed: seme del generatore casuale.
        hours: durata simulata [h].

    Returns:
        Il risultato della corsa.
    """
    data = config.model_dump(mode="json")
    data["simulation"]["seed"] = seed
    data["simulation"]["duration_h"] = hours
    results = run_simulation(MissionConfig.model_validate(data))
    rate = np.degrees(results.metadata["initial_state"]["angular_velocity_rad_s"])
    return DetumbleRun(
        seed=seed,
        detumble_time_h=compute_metrics(results).detumble_time_h,
        release_rate_deg_s=(float(rate[0]), float(rate[1]), float(rate[2])),
    )


def run_campaign(
    config: MissionConfig, seeds: Sequence[int], hours: float, workers: int
) -> Iterator[DetumbleRun]:
    """Esegue le corse in parallelo e le restituisce nell'ordine dei semi.

    Args:
        config: scenario di partenza.
        seeds: semi delle corse.
        hours: durata simulata di ogni corsa [h].
        workers: processi in parallelo.

    Yields:
        Il risultato di ogni corsa, appena disponibile.
    """
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(detumble_run, repeat(config), seeds, repeat(hours))


def runs_table(runs: Sequence[DetumbleRun]) -> pd.DataFrame:
    """Tabella delle corse, una riga per seme; NaN per i detumble non finiti."""
    return pd.DataFrame(
        {
            "seed": run.seed,
            "detumble_time_h": math.nan
            if run.detumble_time_h is None
            else run.detumble_time_h,
            "release_rate_x_deg_s": run.release_rate_deg_s[0],
            "release_rate_y_deg_s": run.release_rate_deg_s[1],
            "release_rate_z_deg_s": run.release_rate_deg_s[2],
        }
        for run in runs
    )


def summarize(table: pd.DataFrame) -> DetumbleStatistics:
    """Statistica dei tempi di detumble delle corse completate.

    Raises:
        ValueError: se nessuna corsa ha completato il detumble.
    """
    times = table["detumble_time_h"].to_numpy(dtype=float)
    completed = times[~np.isnan(times)]
    if completed.size == 0:
        msg = "nessuna corsa ha completato il detumble"
        raise ValueError(msg)
    return DetumbleStatistics(
        runs=int(times.size),
        completed=int(completed.size),
        mean_h=float(completed.mean()),
        p95_h=float(np.percentile(completed, _PERCENTILE)),
        max_h=float(completed.max()),
    )


def save_campaign(
    table: pd.DataFrame, hours: float, directory: Path
) -> DetumbleStatistics:
    """Salva la tabella delle corse (CSV) e il riepilogo (JSON).

    Entrambi i file dichiarano che i dati sono simulati, come i risultati di
    una simulazione.

    Returns:
        La statistica salvata nel riepilogo.
    """
    statistics = summarize(table)
    summary = {
        "notice": SIMULATED_NOTICE,
        "hours_per_run": hours,
        "seeds": [int(table["seed"].min()), int(table["seed"].max())],
        "statistics": asdict(statistics),
    }
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=_JSON_INDENT, ensure_ascii=False) + "\n"
    (directory / SUMMARY_FILE).write_text(text, encoding="utf-8", newline="\n")
    with (directory / RUNS_FILE).open("w", encoding="utf-8", newline="") as file:
        file.write(f"# {SIMULATED_NOTICE}\n")
        table.to_csv(file, index=False, lineterminator="\n")
    return statistics


def load_runs(directory: Path) -> pd.DataFrame:
    """Legge la tabella delle corse salvata da ``save_campaign``."""
    return pd.read_csv(directory / RUNS_FILE, skiprows=1)


def main(argv: list[str] | None = None) -> int:
    """Esegue la campagna Monte Carlo e ne salva i risultati.

    Args:
        argv: argomenti della riga di comando; None per usare quelli del
            processo.

    Returns:
        Codice di uscita del processo: 0 se tutto è andato bene.
    """
    parser = argparse.ArgumentParser(
        prog="python -m cubesat_sim.analysis.monte_carlo",
        description="Monte Carlo del detumble, CubeSat 3U (dati simulati).",
    )
    parser.add_argument("config", type=Path, help="file TOML dello scenario")
    parser.add_argument("output", type=Path, help="cartella dei risultati")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="corse")
    parser.add_argument(
        "--hours", type=float, default=DEFAULT_HOURS, help="ore simulate per corsa"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="processi in parallelo (predefinito: tutti i core)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seeds = range(1, args.runs + 1)
    # La riga di comando è l'unico punto del modulo che scrive a schermo.
    print(SIMULATED_NOTICE)  # noqa: T201
    print(  # noqa: T201
        f"{args.runs} corse da {args.hours:g} h su {args.workers} processi..."
    )
    start = time.perf_counter()
    runs: list[DetumbleRun] = []
    for run in run_campaign(config, seeds, args.hours, args.workers):
        runs.append(run)
        print(f"  corsa {len(runs)}/{args.runs}", end="\r", flush=True)  # noqa: T201
    statistics = save_campaign(runs_table(runs), args.hours, args.output)
    elapsed_s = time.perf_counter() - start
    print(  # noqa: T201
        f"Fatto in {elapsed_s:.0f} s. Detumble: medio {statistics.mean_h:.2f} h, "
        f"95 % {statistics.p95_h:.2f} h, massimo {statistics.max_h:.2f} h "
        f"({statistics.completed} corse su {statistics.runs} completate). "
        f"Risultati in {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
