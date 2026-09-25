"""Costanti fisiche e astronomiche in unità SI, con la fonte di ciascuna.

Sono grandezze della natura, non scelte di missione: per questo non stanno nel
file di configurazione.
"""

# Tempo.
SECONDS_PER_DAY = 86_400.0
JULIAN_DATE_J2000 = 2_451_545.0  # data giuliana di J2000.0 (2000-01-01 12:00 TT)
JULIAN_DATE_UNIX_EPOCH = 2_440_587.5  # data giuliana di 1970-01-01 00:00 UTC

# Terra: modello WGS-84 (NGA, TR8350.2).
EARTH_MU_M3_S2 = 3.986004418e14  # parametro gravitazionale GM
EARTH_RADIUS_M = 6_378_137.0  # raggio equatoriale
EARTH_FLATTENING = 1.0 / 298.257223563  # schiacciamento dell'ellissoide
EARTH_ROTATION_RATE_RAD_S = 7.292115e-5  # rotazione rispetto alle stelle fisse

# Terra: armonica zonale J2 (modello EGM2008). Usata solo nelle verifiche,
# non nella propagazione dell'orbita.
EARTH_J2 = 1.08262668e-3

# Sole.
SOLAR_CONSTANT_W_M2 = 1361.0  # irradianza solare nominale a 1 UA (IAU 2015, B3)
ASTRONOMICAL_UNIT_M = 1.495978707e11  # unità astronomica (IAU 2012, B2)
TROPICAL_YEAR_S = 365.2422 * SECONDS_PER_DAY  # 360 gradi di moto apparente del Sole

# Campo magnetico terrestre: termini di dipolo del modello IGRF-14 (IAGA),
# epoca 2025, arrotondati al nanotesla. Coefficienti di Gauss in tesla.
IGRF_G10_T = -29_350.0e-9
IGRF_G11_T = -1_410.0e-9
IGRF_H11_T = 4_545.0e-9
