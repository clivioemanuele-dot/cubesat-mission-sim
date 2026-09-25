"""Riga di comando: analisi di una simulazione salvata.

Uso, dalla cartella del progetto:
    uv run python -m cubesat_sim.analysis results/baseline

Legge i risultati, calcola metriche e studi di compromesso, li stampa e li
salva in analysis.json nella stessa cartella. Sono dati simulati.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from cubesat_sim.analysis.metrics import MissionMetrics, compute_metrics
from cubesat_sim.analysis.trade_studies import (
    BatterySizingTrade,
    PointingEnergyTrade,
    battery_sizing_trade,
    pointing_energy_trade,
)
from cubesat_sim.core.config import MissionConfig
from cubesat_sim.core.results import SIMULATED_NOTICE, SimulationResults

ANALYSIS_FILE = "analysis.json"
_PERCENT = 100.0
_JSON_INDENT = 2


def main(argv: list[str] | None = None) -> int:
    """Analizza i risultati salvati in una cartella.

    Args:
        argv: argomenti della riga di comando; None per usare quelli del
            processo.

    Returns:
        Codice di uscita del processo: 0 se tutto è andato bene.
    """
    parser = argparse.ArgumentParser(
        prog="python -m cubesat_sim.analysis",
        description="Analisi di missione CubeSat 3U (dati simulati).",
    )
    parser.add_argument("results", type=Path, help="cartella dei risultati")
    args = parser.parse_args(argv)

    results = SimulationResults.load(args.results)
    config = MissionConfig.model_validate(results.metadata["config"])
    metrics = compute_metrics(results)
    energy = pointing_energy_trade(config)
    battery = battery_sizing_trade(results)

    payload = {
        "notice": SIMULATED_NOTICE,
        "metrics": asdict(metrics),
        "pointing_energy_trade": asdict(energy),
        "battery_sizing_trade": asdict(battery),
    }
    text = json.dumps(payload, indent=_JSON_INDENT, ensure_ascii=False) + "\n"
    (args.results / ANALYSIS_FILE).write_text(text, encoding="utf-8", newline="\n")

    # La riga di comando è l'unico punto dell'analisi che scrive a schermo.
    print("\n".join(report_lines(metrics, energy, battery)))  # noqa: T201
    return 0


def report_lines(
    metrics: MissionMetrics, energy: PointingEnergyTrade, battery: BatterySizingTrade
) -> list[str]:
    """Riepilogo leggibile di metriche e studi di compromesso.

    Args:
        metrics: metriche della missione.
        energy: studio 1, puntare la Terra o il Sole.
        battery: studio 2, capacità della batteria.

    Returns:
        Le righe del riepilogo.
    """
    detumble = (
        f"{metrics.detumble_time_h:.2f} h"
        if metrics.detumble_time_h is not None
        else "mai completato"
    )
    lines = [
        SIMULATED_NOTICE,
        "",
        f"Durata simulata: {metrics.duration_h:g} h",
        f"Detumble: {detumble}",
        "Tempo nei modi [h]: "
        + ", ".join(
            f"{mode} {hours:.2f}" for mode, hours in metrics.mode_hours.items()
        ),
        "",
        "Errore di puntamento [gradi]   medio    95%     max",
    ]
    lines += [
        f"  {item.mode:<26}{item.mean_deg:7.2f}{item.p95_deg:8.2f}{item.max_deg:8.2f}"
        for item in metrics.pointing
    ]
    lines += [
        "",
        f"Ruote: {metrics.wheel_saturation_events} episodi di saturazione "
        f"({metrics.wheel_saturation_s:.0f} s), momento massimo al "
        f"{metrics.max_wheel_momentum_fraction * _PERCENT:.0f} % della capacità",
        f"Stazione visibile {metrics.station_visible_h:.2f} h, collegamento "
        f"disponibile per il {metrics.link_availability * _PERCENT:.0f} % del tempo",
        "",
        "Orbita  inizio [h]  prodotta [Wh]  consumata [Wh]  netta [Wh]  carica min",
    ]
    lines += [
        f"  {orbit.orbit:>4}{orbit.start_h:11.2f}{orbit.generated_wh:15.2f}"
        f"{orbit.consumed_wh:16.2f}{orbit.net_wh:12.2f}"
        f"{orbit.min_state_of_charge * _PERCENT:10.1f} %"
        for orbit in metrics.orbits
    ] or ["  nessuna orbita completa"]
    lines += [
        f"Carica minima della batteria: {metrics.min_state_of_charge * _PERCENT:.1f} %",
        "Dati [Gbit]: "
        f"acquisiti {metrics.data_acquired_gbit:.3f}, "
        f"scaricati {metrics.data_downlinked_gbit:.3f}, "
        f"a bordo {metrics.data_on_board_gbit:.3f}, "
        f"persi {metrics.data_lost_gbit:.3f}",
        "",
        "Studio 1, puntare la Terra invece del Sole (per orbita):",
        f"  SUN_POINTING {energy.sun_pointing_wh:.1f} Wh, NADIR "
        f"{energy.nadir_pointing_wh:.1f} Wh: si perdono {energy.loss_wh:.1f} Wh "
        f"({energy.loss_fraction * _PERCENT:.0f} %)",
        "Studio 2, capacità della batteria:",
        f"  minima senza SAFE {battery.min_capacity_wh:.2f} Wh "
        f"(configurata {battery.configured_capacity_wh:g} Wh); prelievo massimo "
        f"{battery.max_drawdown_wh:.2f} Wh, profondità di scarica "
        f"{battery.max_depth_of_discharge * _PERCENT:.1f} %",
    ]
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
