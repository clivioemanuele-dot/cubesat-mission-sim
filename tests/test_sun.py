"""Test della posizione del Sole e del modello di eclissi."""

import datetime as dt
import math

import numpy as np
import pytest

from cubesat_sim.core.constants import ASTRONOMICAL_UNIT_M, EARTH_RADIUS_M
from cubesat_sim.core.types import FloatArray
from cubesat_sim.environment.earth import julian_date
from cubesat_sim.environment.sun import is_in_eclipse, sun_position_eci

ORBIT_RADIUS_M = EARTH_RADIUS_M + 500e3
SUN_ALONG_X = np.array([ASTRONOMICAL_UNIT_M, 0.0, 0.0])


def sun_at(moment: dt.datetime) -> FloatArray:
    """Posizione del Sole in ECI a un istante dato."""
    return sun_position_eci(julian_date(moment))


def declination_deg(sun: FloatArray) -> float:
    """Declinazione: angolo del Sole sopra l'equatore celeste."""
    return math.degrees(math.asin(sun[2] / np.linalg.norm(sun)))


def right_ascension_deg(sun: FloatArray) -> float:
    """Ascensione retta: angolo lungo l'equatore celeste, tra 0 e 360 gradi."""
    return math.degrees(math.atan2(sun[1], sun[0])) % 360.0


def test_sun_at_september_equinox_2026() -> None:
    # Equinozio: 23 settembre 2026, 00:05 UTC. Il Sole attraversa l'equatore
    # scendendo verso sud, in direzione -x (ascensione retta 180 gradi).
    sun = sun_at(dt.datetime(2026, 9, 23, 0, 5, tzinfo=dt.UTC))
    assert declination_deg(sun) == pytest.approx(0.0, abs=0.05)
    assert right_ascension_deg(sun) == pytest.approx(180.0, abs=0.1)


def test_sun_at_june_solstice_2026() -> None:
    # Solstizio: la declinazione è massima e pari all'obliquità (23,436 gradi).
    sun = sun_at(dt.datetime(2026, 6, 21, 8, 24, tzinfo=dt.UTC))
    assert declination_deg(sun) == pytest.approx(23.436, abs=0.005)


@pytest.mark.parametrize(
    ("moment", "expected_au"),
    [
        (dt.datetime(2026, 1, 3, 17, 0, tzinfo=dt.UTC), 0.9833),  # perielio
        (dt.datetime(2026, 7, 6, 17, 0, tzinfo=dt.UTC), 1.0166),  # afelio
    ],
)
def test_sun_distance(moment: dt.datetime, expected_au: float) -> None:
    distance_au = np.linalg.norm(sun_at(moment)) / ASTRONOMICAL_UNIT_M
    assert distance_au == pytest.approx(expected_au, abs=5e-4)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ([ORBIT_RADIUS_M, 0.0, 0.0], False),  # dalla parte del Sole
        ([-ORBIT_RADIUS_M, 0.0, 0.0], True),  # dietro la Terra, sull'asse
        ([0.0, ORBIT_RADIUS_M, 0.0], False),  # di lato, sul terminatore
        ([-ORBIT_RADIUS_M, EARTH_RADIUS_M - 1e3, 0.0], True),  # appena dentro
        ([-ORBIT_RADIUS_M, EARTH_RADIUS_M + 1e3, 0.0], False),  # appena fuori
        ([-ORBIT_RADIUS_M, 0.0, EARTH_RADIUS_M + 1e3], False),  # fuori, sul polo
    ],
)
def test_cylindrical_shadow(position: list[float], expected: bool) -> None:
    assert is_in_eclipse(np.array(position), SUN_ALONG_X) is expected
