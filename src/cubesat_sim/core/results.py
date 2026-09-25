"""Salvataggio e lettura dei risultati di una simulazione.

Una simulazione salvata è una cartella con due file:

- ``timeseries.csv``: le serie temporali, una riga per istante di uscita;
- ``metadata.json``: seed, configurazione completa, eventi e riepiloghi.

Entrambi i file dichiarano in modo visibile che i dati sono simulati: il CSV
nella prima riga, il JSON nel campo ``notice``. I file vengono scritti in UTF-8
con fine riga in stile Unix (LF) e senza data di salvataggio: la stessa
simulazione produce file identici byte per byte su qualunque sistema operativo.

I numeri decimali si scrivono con il formato predefinito di pandas, che tiene
tutte le cifre e il punto decimale: una colonna di decimali resta di decimali
anche quando tutti i suoi valori sono interi (per esempio ``10.0``).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import pandas as pd

SIMULATED_NOTICE = (
    "DATI SIMULATI: prodotti da cubesat-mission-sim, "
    "non provengono da alcun satellite reale."
)
TIMESERIES_FILE = "timeseries.csv"
METADATA_FILE = "metadata.json"

_JSON_INDENT = 2


@dataclass(frozen=True, eq=False)
class SimulationResults:
    """Risultati di una simulazione.

    Attributes:
        timeseries: serie temporali, una colonna per grandezza; l'unità di
            misura è indicata dal suffisso del nome della colonna.
        metadata: seed, configurazione, eventi e riepiloghi. Solo tipi JSON
            nativi di Python: dict, list, str, int, float, bool, None.
    """

    timeseries: pd.DataFrame
    metadata: dict[str, Any]

    def save(self, directory: Path) -> None:
        """Salva i risultati nella cartella indicata, creandola se serve.

        Il JSON viene preparato prima di scrivere qualunque file: se i metadati
        contengono tipi non validi, l'errore arriva prima di lasciare una
        cartella a metà.

        Args:
            directory: cartella di destinazione.

        Raises:
            TypeError: se i metadati contengono tipi non convertibili in JSON.
        """
        payload = {"notice": SIMULATED_NOTICE} | self.metadata
        text = json.dumps(payload, indent=_JSON_INDENT, ensure_ascii=False) + "\n"
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / METADATA_FILE
        json_path.write_text(text, encoding="utf-8", newline="\n")
        csv_path = directory / TIMESERIES_FILE
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            file.write(f"# {SIMULATED_NOTICE}\n")
            self.timeseries.to_csv(file, index=False, lineterminator="\n")

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Legge i risultati salvati con ``save``.

        Args:
            directory: cartella che contiene ``timeseries.csv`` e
                ``metadata.json``.

        Returns:
            I risultati, con i metadati comprensivi del campo ``notice``.
        """
        timeseries = pd.read_csv(directory / TIMESERIES_FILE, skiprows=1)
        text = (directory / METADATA_FILE).read_text(encoding="utf-8")
        metadata: dict[str, Any] = json.loads(text)
        return cls(timeseries=timeseries, metadata=metadata)
