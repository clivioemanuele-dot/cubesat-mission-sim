"""Test della memoria dati e dello scarico a terra."""

import math
from pathlib import Path

import numpy as np
import pytest

from cubesat_sim.core.config import load_config
from cubesat_sim.resources.data import DataStorage

BASELINE = Path(__file__).resolve().parents[1] / "config" / "baseline.toml"
PAYLOAD = load_config(BASELINE).payload
MBIT = 1e6


def test_storage_starts_empty() -> None:
    assert DataStorage(PAYLOAD).stored_bits == 0.0


def test_imaging_fills_storage() -> None:
    # 60 s di acquisizione a 2 Mbit/s: 120 Mbit in memoria.
    storage = DataStorage(PAYLOAD)
    update = storage.update(60.0, imaging=True, downlinking=False)
    assert update.acquired_bits == pytest.approx(120 * MBIT)
    assert storage.stored_bits == pytest.approx(120 * MBIT)


def test_downlink_empties_storage() -> None:
    # Dopo 120 Mbit acquisiti, 60 s di scarico a 1 Mbit/s ne lasciano 60.
    storage = DataStorage(PAYLOAD)
    storage.update(60.0, imaging=True, downlinking=False)
    update = storage.update(60.0, imaging=False, downlinking=True)
    assert update.downlinked_bits == pytest.approx(60 * MBIT)
    assert storage.stored_bits == pytest.approx(60 * MBIT)


def test_cannot_downlink_more_than_stored() -> None:
    # 10 Mbit in memoria e 60 s di collegamento: si scaricano solo 10 Mbit.
    storage = DataStorage(PAYLOAD)
    storage.update(5.0, imaging=True, downlinking=False)
    update = storage.update(60.0, imaging=False, downlinking=True)
    assert update.downlinked_bits == pytest.approx(10 * MBIT)
    assert storage.stored_bits == 0.0


@pytest.mark.parametrize(
    ("visible", "error_deg", "expected"),
    [
        (True, 19.0, True),  # antenna entro metà fascio (20 gradi)
        (True, 21.0, False),  # stazione visibile, ma antenna puntata male
        (False, 0.0, False),  # antenna perfetta, ma stazione sotto l'orizzonte
    ],
)
def test_link_needs_visibility_and_pointing(
    visible: bool, error_deg: float, expected: bool
) -> None:
    storage = DataStorage(PAYLOAD)
    available = storage.link_available(
        station_visible=visible, antenna_error_rad=math.radians(error_deg)
    )
    assert available is expected


def test_full_storage_loses_data() -> None:
    # 5000 s di acquisizione = 10 Gbit, contro 8 Gbit di memoria: 2 Gbit persi.
    storage = DataStorage(PAYLOAD)
    update = storage.update(5000.0, imaging=True, downlinking=False)
    assert storage.stored_bits == pytest.approx(storage.capacity_bits)
    assert update.lost_bits == pytest.approx(2000 * MBIT)


def test_data_are_conserved() -> None:
    # 5000 intervalli casuali da 10 s, con più acquisizione che scarico, così la
    # memoria arriva al limite. Dati in memoria = acquisiti - scaricati, e la
    # memoria resta sempre tra zero e la capacità.
    rng = np.random.default_rng(0)
    storage = DataStorage(PAYLOAD)
    acquired_bits = downlinked_bits = lost_bits = 0.0
    for _ in range(5000):
        update = storage.update(
            10.0,
            imaging=bool(rng.random() < 0.3),
            downlinking=bool(rng.random() < 0.3),
        )
        acquired_bits += update.acquired_bits
        downlinked_bits += update.downlinked_bits
        lost_bits += update.lost_bits
        assert 0.0 <= storage.stored_bits <= storage.capacity_bits

    assert lost_bits > 0.0  # la memoria si è riempita
    expected_bits = acquired_bits - downlinked_bits
    assert storage.stored_bits == pytest.approx(expected_bits, rel=1e-12)
