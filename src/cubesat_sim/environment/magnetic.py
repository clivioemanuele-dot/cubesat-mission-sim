"""Campo magnetico terrestre: dipolo inclinato che ruota con la Terra.

Si usano solo i tre termini di dipolo del modello IGRF-14 (epoca 2025). Il
dipolo è centrato nella Terra e fisso in ECEF: in ECI ruota con la Terra. In
orbita bassa l'errore rispetto al modello completo è dell'ordine del 10 %, più
grande sopra l'Anomalia del Sud Atlantico. È adeguato per il detumble e per il
calcolo dei disturbi, non per la navigazione magnetica.

Formula (potenziale IGRF troncato al primo grado):
    B(r) = (a / |r|)^3 * (3 (m . r_hat) r_hat - m),  con m = (g11, h11, g10)
dove a è il raggio di riferimento del modello IGRF.
"""

import numpy as np

from cubesat_sim.core.constants import IGRF_G10_T, IGRF_G11_T, IGRF_H11_T
from cubesat_sim.core.types import FloatArray
from cubesat_sim.environment.earth import eci_to_ecef_matrix

IGRF_REFERENCE_RADIUS_M = 6_371_200.0  # raggio di riferimento del modello IGRF

# Vettore di dipolo in ECEF [T]. Punta verso l'emisfero sud: il polo nord
# geomagnetico si trova nella direzione opposta.
_DIPOLE_ECEF_T = np.array([IGRF_G11_T, IGRF_H11_T, IGRF_G10_T], dtype=np.float64)


def magnetic_field_ecef(position_ecef: FloatArray) -> FloatArray:
    """Campo magnetico in ECEF.

    Args:
        position_ecef: posizione in ECEF [m], non nulla.

    Returns:
        Vettore campo magnetico in ECEF [T].
    """
    radius = float(np.linalg.norm(position_ecef))
    r_hat = position_ecef / radius
    scale = (IGRF_REFERENCE_RADIUS_M / radius) ** 3
    return scale * (3.0 * float(_DIPOLE_ECEF_T @ r_hat) * r_hat - _DIPOLE_ECEF_T)


def magnetic_field_eci(position_eci: FloatArray, jd: float) -> FloatArray:
    """Campo magnetico in ECI.

    Args:
        position_eci: posizione in ECI [m], non nulla.
        jd: data giuliana [giorni].

    Returns:
        Vettore campo magnetico in ECI [T].
    """
    to_ecef = eci_to_ecef_matrix(jd)
    return to_ecef.T @ magnetic_field_ecef(to_ecef @ position_eci)
