"""Memoria dati di bordo e scarico verso la stazione di terra.

    Acquisizione: con la fotocamera accesa i dati entrano in memoria alla
        velocità della fotocamera. A memoria piena i dati in più sono persi.
    Scarico: i dati escono alla velocità del collegamento solo se il
        collegamento è disponibile, cioè se la stazione è visibile e l'errore
        di puntamento dell'antenna è entro metà del suo fascio. È qui che le
        prestazioni del controllo d'assetto decidono quanti dati arrivano a
        terra.

Conservazione: dati in memoria alla fine = dati all'inizio + acquisiti -
scaricati. Velocità costanti: nessun link budget dettagliato.
"""

from dataclasses import dataclass

from cubesat_sim.core.config import PayloadConfig


@dataclass(frozen=True)
class DataUpdate:
    """Esito di un aggiornamento della memoria.

    Attributes:
        acquired_bits: dati acquisiti ed entrati in memoria [bit].
        downlinked_bits: dati scaricati a terra [bit].
        lost_bits: dati acquisiti ma persi per memoria piena [bit].
    """

    acquired_bits: float
    downlinked_bits: float
    lost_bits: float


class DataStorage:
    """Memoria di massa di bordo, con acquisizione e scarico.

    Attributes:
        capacity_bits: capacità della memoria [bit].
        stored_bits: dati in memoria [bit], tra 0 e capacity_bits.
    """

    def __init__(self, payload: PayloadConfig) -> None:
        """Prepara la memoria, inizialmente vuota.

        Args:
            payload: sezione [payload] della configurazione.
        """
        self.capacity_bits = payload.storage_capacity_bit
        self.stored_bits = 0.0
        self._camera_rate_bps = payload.camera_data_rate_bps
        self._downlink_rate_bps = payload.downlink_rate_bps
        self._half_beamwidth_rad = payload.antenna_half_beamwidth_rad

    def link_available(
        self, *, station_visible: bool, antenna_error_rad: float
    ) -> bool:
        """Indica se il collegamento con la stazione funziona.

        Args:
            station_visible: True se la stazione è sopra l'elevazione minima.
            antenna_error_rad: angolo tra l'asse dell'antenna e la direzione
                della stazione [rad].

        Returns:
            True se la stazione è visibile e l'antenna punta entro metà fascio.
        """
        return station_visible and antenna_error_rad <= self._half_beamwidth_rad

    def update(
        self, duration_s: float, *, imaging: bool, downlinking: bool
    ) -> DataUpdate:
        """Aggiorna la memoria per un intervallo di tempo.

        Args:
            duration_s: durata dell'intervallo [s].
            imaging: True se la fotocamera acquisisce.
            downlinking: True se i dati vengono scaricati (collegamento
                disponibile e trasmettitore acceso).

        Returns:
            L'esito dell'aggiornamento.
        """
        generated_bits = self._camera_rate_bps * duration_s if imaging else 0.0
        acquired_bits = min(generated_bits, self.capacity_bits - self.stored_bits)
        self.stored_bits += acquired_bits
        requested_bits = self._downlink_rate_bps * duration_s if downlinking else 0.0
        downlinked_bits = min(requested_bits, self.stored_bits)
        self.stored_bits -= downlinked_bits
        return DataUpdate(
            acquired_bits=acquired_bits,
            downlinked_bits=downlinked_bits,
            lost_bits=generated_bits - acquired_bits,
        )
