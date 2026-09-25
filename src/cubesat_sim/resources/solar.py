"""Generatore solare: potenza prodotta dai pannelli in funzione dell'assetto.

Ogni pannello piano produce in proporzione all'area che vede il Sole:
    P = S (UA / d)^2 * eta * derating * somma_i A_i * max(0, n_i . s)
con S costante solare a 1 UA, d distanza Terra-Sole, eta efficienza delle
celle, derating perdite di sistema, A_i area e n_i normale del pannello, s
direzione del Sole negli assi del corpo. In eclissi la potenza è nulla.

Trascurati: ombre reciproche tra ali e corpo, perdite aggiuntive con il Sole
radente, luce riflessa e infrarosso dalla Terra (albedo), variazioni di
efficienza con la temperatura (incluse in media nel derating).
"""

import numpy as np

from cubesat_sim.core.config import SolarArrayConfig
from cubesat_sim.core.constants import ASTRONOMICAL_UNIT_M, SOLAR_CONSTANT_W_M2
from cubesat_sim.core.types import FloatArray


class SolarArray:
    """Insieme dei pannelli solari, fissi rispetto al corpo."""

    def __init__(self, config: SolarArrayConfig) -> None:
        """Prepara i pannelli dalla configurazione.

        Args:
            config: sezione [solar_array] della configurazione.
        """
        self._normals = np.array([panel.normal_body for panel in config.panels])
        self._areas_m2 = np.array([panel.area_m2 for panel in config.panels])
        self._efficiency = config.cell_efficiency * config.derating_factor

    def power(self, sun_body: FloatArray, in_eclipse: bool) -> float:
        """Potenza elettrica prodotta.

        Args:
            sun_body: posizione del Sole rispetto al satellite, negli assi del
                corpo [m]. La direzione dà l'illuminazione dei pannelli, il
                modulo la distanza dal Sole.
            in_eclipse: True se il satellite è nell'ombra della Terra.

        Returns:
            Potenza prodotta dai pannelli [W].
        """
        if in_eclipse:
            return 0.0
        distance_m = float(np.linalg.norm(sun_body))
        flux_w_m2 = SOLAR_CONSTANT_W_M2 * (ASTRONOMICAL_UNIT_M / distance_m) ** 2
        cosines = self._normals @ (sun_body / distance_m)
        lit_area_m2 = float(self._areas_m2 @ np.clip(cosines, 0.0, None))
        return flux_w_m2 * self._efficiency * lit_area_m2
