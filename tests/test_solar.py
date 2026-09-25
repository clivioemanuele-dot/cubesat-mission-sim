"""Test del generatore solare: potenza in funzione dell'assetto."""

import math
from pathlib import Path

import numpy as np
import pytest

from cubesat_sim.core.config import load_config
from cubesat_sim.core.constants import ASTRONOMICAL_UNIT_M
from cubesat_sim.core.types import FloatArray
from cubesat_sim.core.vectors import unit
from cubesat_sim.resources.solar import SolarArray

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"
ARRAY = SolarArray(load_config(BASELINE).solar_array)
ONE_PANEL_W = 1361.0 * 0.30 * 0.80 * 0.02113  # un pannello di fronte al Sole a 1 UA


def sun_at(direction: list[float], distance_au: float = 1.0) -> FloatArray:
    """Posizione del Sole negli assi del corpo, nella direzione e alla distanza date."""
    return distance_au * ASTRONOMICAL_UNIT_M * unit(np.array(direction))


def test_peak_power_when_pointing_the_sun() -> None:
    # docs/assumptions.md: ali e faccia +X verso il Sole a 1 UA danno 20,7 W.
    power = ARRAY.power(sun_at([1.0, 0.0, 0.0]), in_eclipse=False)
    assert power == pytest.approx(20.71, abs=0.01)


def test_no_power_in_eclipse() -> None:
    assert ARRAY.power(sun_at([1.0, 0.0, 0.0]), in_eclipse=True) == 0.0


def test_sun_behind_the_wings() -> None:
    # Sole da -X: lavora solo la faccia -X del corpo, un pannello su sei.
    power = ARRAY.power(sun_at([-1.0, 0.0, 0.0]), in_eclipse=False)
    assert power == pytest.approx(ONE_PANEL_W)


def test_camera_toward_the_sun_gives_no_power() -> None:
    # Sole lungo +Z, la faccia della fotocamera, senza celle: tutti i pannelli
    # sono di taglio e la potenza è nulla.
    power = ARRAY.power(sun_at([0.0, 0.0, 1.0]), in_eclipse=False)
    assert power == pytest.approx(0.0, abs=1e-12)


def test_cosine_law() -> None:
    # Sole a 60 gradi dall'asse +X: la potenza si dimezza (cos 60 = 0,5).
    angle = math.radians(60.0)
    power = ARRAY.power(sun_at([math.cos(angle), 0.0, math.sin(angle)]), False)
    assert power == pytest.approx(0.5 * 3 * ONE_PANEL_W)


def test_power_scales_with_sun_distance() -> None:
    # Al perielio (0,9833 UA) il flusso solare è più alto del 3,4 %.
    power = ARRAY.power(sun_at([1.0, 0.0, 0.0], distance_au=0.9833), False)
    assert power == pytest.approx(3 * ONE_PANEL_W / 0.9833**2)


def test_power_bounds_for_any_attitude() -> None:
    # Per qualunque direzione del Sole la potenza è tra zero e il massimo
    # geometrico: tre pannelli su +X più uno su +Y o -Y, orientati al meglio,
    # danno sqrt(10) volte un pannello (21,8 W).
    rng = np.random.default_rng(0)
    maximum = math.sqrt(10) * ONE_PANEL_W
    for _ in range(1000):
        power = ARRAY.power(sun_at(rng.normal(size=3).tolist()), in_eclipse=False)
        assert 0.0 <= power <= maximum + 1e-9
