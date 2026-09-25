"""Riga di comando: esegue uno scenario e salva i risultati.

Uso, dalla cartella del progetto:
    uv run python -m cubesat_sim.simulation config/baseline.toml results/baseline

Con --hours si sostituisce la durata dello scenario, per prove brevi o per
misurare i tempi di calcolo. I risultati (timeseries.csv e metadata.json) sono
dati simulati.
"""

import argparse
import time
from pathlib import Path

from cubesat_sim.core.config import MissionConfig, load_config
from cubesat_sim.core.results import SIMULATED_NOTICE
from cubesat_sim.simulation.runner import run_simulation


def _scenario(path: Path, hours: float | None) -> MissionConfig:
    """Legge lo scenario e, se richiesto, ne cambia la durata."""
    config = load_config(path)
    if hours is None:
        return config
    data = config.model_dump()
    data["simulation"]["duration_h"] = hours
    return MissionConfig.model_validate(data)


def main(argv: list[str] | None = None) -> int:
    """Esegue lo scenario indicato e ne salva i risultati.

    Args:
        argv: argomenti della riga di comando; None per usare quelli del
            processo.

    Returns:
        Codice di uscita del processo: 0 se tutto è andato bene.
    """
    parser = argparse.ArgumentParser(
        prog="python -m cubesat_sim.simulation",
        description="Simulatore di missione CubeSat 3U (dati simulati).",
    )
    parser.add_argument("config", type=Path, help="file TOML dello scenario")
    parser.add_argument("output", type=Path, help="cartella dei risultati")
    parser.add_argument(
        "--hours", type=float, help="durata simulata, al posto di quella del file"
    )
    args = parser.parse_args(argv)

    config = _scenario(args.config, args.hours)
    start = time.perf_counter()
    results = run_simulation(config)
    elapsed_s = time.perf_counter() - start
    results.save(args.output)

    summary = (
        f"Simulate {config.simulation.duration_h:g} h in {elapsed_s:.1f} s: "
        f"{len(results.timeseries)} righe, "
        f"{len(results.metadata['transitions'])} cambi di modo. "
        f"Risultati in {args.output}"
    )
    # La riga di comando è l'unico punto del pacchetto che scrive a schermo.
    print(SIMULATED_NOTICE)  # noqa: T201
    print(summary)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
