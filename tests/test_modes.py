"""Test del gestore dei modi: priorità, isteresi, conferma e registro."""

import math
from pathlib import Path

import pytest

from cubesat_sim.core.config import load_config
from cubesat_sim.modes.manager import Mode, ModeInputs, ModeManager

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"
CONFIG = load_config(BASELINE).modes
CONFIRM_S = int(CONFIG.confirmation_time_s)
ENOUGH_S = CONFIRM_S + 1
LONG_S = 10 * CONFIRM_S


def inputs(
    *,
    soc: float = 0.7,
    rate_deg_s: float = 0.1,
    station: bool = False,
    window: bool = False,
) -> ModeInputs:
    return ModeInputs(
        state_of_charge=soc,
        angular_rate_rad_s=math.radians(rate_deg_s),
        station_visible=station,
        imaging_window=window,
    )


class Clock:
    """Chiama il gestore una volta al secondo, come il controllo a 1 Hz."""

    def __init__(self, manager: ModeManager) -> None:
        self.manager = manager
        self.time_s = 0

    def run(self, conditions: ModeInputs, duration_s: int) -> Mode:
        mode = self.manager.mode
        for _ in range(duration_s):
            mode = self.manager.update(float(self.time_s), conditions)
            self.time_s += 1
        return mode


@pytest.fixture
def clock() -> Clock:
    return Clock(ModeManager(CONFIG))


def settle(clock: Clock) -> None:
    """Porta il gestore da DETUMBLE a SUN_POINTING con condizioni tranquille."""
    assert clock.run(inputs(), ENOUGH_S) is Mode.SUN_POINTING


def test_starts_in_detumble() -> None:
    manager = ModeManager(CONFIG)
    assert manager.mode is Mode.DETUMBLE
    assert manager.tumbling
    assert manager.transitions == ()


def test_transition_waits_for_confirmation(clock: Clock) -> None:
    assert clock.run(inputs(), CONFIRM_S) is Mode.DETUMBLE
    assert clock.run(inputs(), 1) is Mode.SUN_POINTING


def test_short_condition_is_ignored(clock: Clock) -> None:
    settle(clock)
    assert clock.run(inputs(station=True), CONFIRM_S // 2) is Mode.SUN_POINTING
    assert clock.run(inputs(), CONFIRM_S) is Mode.SUN_POINTING
    assert len(clock.manager.transitions) == 1


@pytest.mark.parametrize(
    ("conditions", "expected"),
    [
        (inputs(soc=0.2, rate_deg_s=5.0, station=True, window=True), Mode.SAFE),
        (inputs(rate_deg_s=5.0, station=True, window=True), Mode.DETUMBLE),
        (inputs(station=True, window=True), Mode.DOWNLINK),
        (inputs(window=True), Mode.NADIR),
        (inputs(), Mode.SUN_POINTING),
    ],
    ids=["safe", "detumble", "downlink", "nadir", "sun_pointing"],
)
def test_priority_order(clock: Clock, conditions: ModeInputs, expected: Mode) -> None:
    assert clock.run(conditions, ENOUGH_S) is expected


def test_safe_hysteresis(clock: Clock) -> None:
    settle(clock)
    assert clock.run(inputs(soc=0.31), LONG_S) is Mode.SUN_POINTING
    assert clock.run(inputs(soc=0.29), ENOUGH_S) is Mode.SAFE
    assert clock.run(inputs(soc=0.45), LONG_S) is Mode.SAFE
    assert clock.run(inputs(soc=0.55), ENOUGH_S) is Mode.SUN_POINTING


def test_detumble_hysteresis(clock: Clock) -> None:
    settle(clock)
    assert clock.run(inputs(rate_deg_s=1.5), LONG_S) is Mode.SUN_POINTING
    assert clock.run(inputs(rate_deg_s=3.0), ENOUGH_S) is Mode.DETUMBLE
    assert clock.run(inputs(rate_deg_s=1.0), LONG_S) is Mode.DETUMBLE
    assert clock.run(inputs(rate_deg_s=0.3), ENOUGH_S) is Mode.SUN_POINTING


def test_changing_condition_restarts_confirmation(clock: Clock) -> None:
    settle(clock)
    clock.run(inputs(window=True), CONFIRM_S // 2)
    assert clock.run(inputs(station=True), CONFIRM_S) is Mode.SUN_POINTING
    assert clock.run(inputs(station=True), 1) is Mode.DOWNLINK


def test_transitions_are_logged_with_cause(clock: Clock) -> None:
    settle(clock)
    clock.run(inputs(soc=0.25), ENOUGH_S)
    first, second = clock.manager.transitions
    assert first.time_s == float(CONFIRM_S)
    assert first.previous is Mode.DETUMBLE
    assert first.current is Mode.SUN_POINTING
    assert first.cause == "rotazione smorzata (0.10 °/s)"
    assert second.time_s == float(ENOUGH_S + CONFIRM_S)
    assert second.previous is Mode.SUN_POINTING
    assert second.current is Mode.SAFE
    assert second.cause == "batteria scarica (carica al 25.0 %)"


def test_tumbling_is_tracked_in_safe(clock: Clock) -> None:
    settle(clock)
    clock.run(inputs(soc=0.25), ENOUGH_S)
    assert clock.run(inputs(soc=0.25, rate_deg_s=5.0), ENOUGH_S) is Mode.SAFE
    assert clock.manager.tumbling


def test_modes_are_listed_by_priority_as_text() -> None:
    assert [str(mode) for mode in Mode] == [
        "SAFE",
        "DETUMBLE",
        "DOWNLINK",
        "NADIR",
        "SUN_POINTING",
    ]
