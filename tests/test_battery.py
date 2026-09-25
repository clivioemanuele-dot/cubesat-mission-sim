"""Test della batteria: carica, scarica, limiti e bilancio energetico."""

import math

import numpy as np
import pytest

from cubesat_sim.core.config import BatteryConfig
from cubesat_sim.resources.battery import Battery

CONFIG = BatteryConfig(
    capacity_wh=40.0,
    charge_efficiency=0.95,
    discharge_efficiency=0.95,
    initial_soc_pct=70.0,
)
CAPACITY_J = 40.0 * 3600


def test_initial_state_of_charge() -> None:
    battery = Battery(CONFIG)
    assert battery.state_of_charge == pytest.approx(0.70)
    assert battery.energy_j == pytest.approx(0.70 * CAPACITY_J)


def test_charging_keeps_95_percent() -> None:
    # 10 W per 6 minuti = 3600 J al bus: nella batteria ne entrano 3420 J.
    update = Battery(CONFIG).update(10.0, 360.0)
    assert update.stored_change_j == pytest.approx(3420.0)
    assert update.loss_j == pytest.approx(180.0)


def test_discharging_draws_more_than_delivered() -> None:
    # Per fornire 3600 J al bus la batteria ne cede 3600 / 0,95 = 3789 J.
    update = Battery(CONFIG).update(-10.0, 360.0)
    assert update.stored_change_j == pytest.approx(-3600.0 / 0.95)
    assert update.unmet_j == 0.0


def test_full_battery_wastes_surplus() -> None:
    # 20 W per un giorno: molto più di quanto serve per riempirla.
    battery = Battery(CONFIG)
    update = battery.update(20.0, 86_400.0)
    assert battery.state_of_charge == pytest.approx(1.0)
    assert update.wasted_j > 0.0


def test_empty_battery_reports_unmet_load() -> None:
    # 5 W per un giorno = 432 kJ richiesti; la batteria può fornirne solo
    # 0,70 * 144 kJ * 0,95 = 95,8 kJ. Il resto non è fornito.
    battery = Battery(CONFIG)
    update = battery.update(-5.0, 86_400.0)
    assert battery.state_of_charge == pytest.approx(0.0)
    expected_unmet = 5.0 * 86_400.0 - 0.70 * CAPACITY_J * 0.95
    assert update.unmet_j == pytest.approx(expected_unmet)


def test_energy_balance_closes() -> None:
    # Validazione del piano. Profilo di 10 000 intervalli da 10 s: potenza
    # netta che oscilla tra +25 e -25 W, con rumore, così la batteria arriva
    # sia al pieno sia al vuoto. Energia prodotta - consumata = variazione
    # della batteria + perdite + sprecata - non fornita, e il SoC resta sempre
    # tra 0 e 1.
    rng = np.random.default_rng(0)
    battery = Battery(CONFIG)
    start_j = battery.energy_j
    step_s = 10.0
    net_j = losses_j = wasted_j = unmet_j = 0.0
    for k in range(10_000):
        oscillation = 25.0 * math.sin(2 * math.pi * k * step_s / 20_000.0)
        power_w = oscillation + float(rng.uniform(-5.0, 5.0))
        update = battery.update(power_w, step_s)
        net_j += power_w * step_s
        losses_j += update.loss_j
        wasted_j += update.wasted_j
        unmet_j += update.unmet_j
        assert 0.0 <= battery.state_of_charge <= 1.0

    assert wasted_j > 0.0  # la batteria si è riempita
    assert unmet_j > 0.0  # la batteria si è svuotata
    stored_change_j = battery.energy_j - start_j
    balance_j = stored_change_j + losses_j + wasted_j - unmet_j
    assert net_j == pytest.approx(balance_j, abs=1e-3)
