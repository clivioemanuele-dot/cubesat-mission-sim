"""Test di importabilità: ogni sottopacchetto si importa senza errori."""

import importlib

import pytest

SUBPACKAGES = (
    "core",
    "environment",
    "attitude",
    "resources",
    "modes",
    "simulation",
    "analysis",
    "dashboard",
)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_is_importable(name: str) -> None:
    module = importlib.import_module(f"cubesat_sim.{name}")
    # Ogni sottopacchetto dichiara il proprio ruolo nella docstring.
    assert module.__doc__
