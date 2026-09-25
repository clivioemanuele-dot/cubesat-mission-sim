"""Configurazione della missione: lettura del file TOML e validazione.

Il file TOML usa unità comode, indicate dal suffisso del nome di ogni campo
(km, deg, cm2, wh, ...). Ogni sezione espone le stesse grandezze in unità SI
tramite proprietà con il suffisso SI (m, rad, m2, j, ...): il resto del
simulatore usa solo queste proprietà.

I valori sono validati al caricamento: campi mancanti o sconosciuti, tipi
sbagliati, valori fuori dai limiti e incoerenze tra campi fermano il
programma con un messaggio che indica il campo responsabile.
"""

import datetime as dt
import math
import tomllib
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    model_validator,
)

# Fattori di conversione verso le unità SI.
_M_PER_KM = 1.0e3
_M_PER_CM = 1.0e-2
_M2_PER_CM2 = 1.0e-4
_S_PER_MIN = 60.0
_S_PER_H = 3600.0
_J_PER_WH = 3600.0
_NM_PER_MNM = 1.0e-3  # da mN m a N m
_NMS_PER_MNMS = 1.0e-3  # da mN m s a N m s
_RAD_S_PER_RPM = 2.0 * math.pi / _S_PER_MIN
_T_PER_NT = 1.0e-9
_BPS_PER_MBPS = 1.0e6
_BIT_PER_GBIT = 1.0e9
_FRACTION_PER_PCT = 1.0e-2

# Tolleranze numeriche dei controlli di coerenza.
_UNIT_NORM_TOLERANCE = 1.0e-6
_MULTIPLE_TOLERANCE = 1.0e-9

# Tipi riutilizzati nelle sezioni.
Vector3 = tuple[float, float, float]
PositiveVector3 = tuple[PositiveFloat, PositiveFloat, PositiveFloat]
Fraction = Annotated[float, Field(gt=0.0, le=1.0)]
Percent = Annotated[float, Field(ge=0.0, le=100.0)]
Latitude = Annotated[float, Field(ge=-90.0, le=90.0)]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0)]
NonEmptyStr = Annotated[str, Field(min_length=1)]


def _require_unit_norm(vector: Vector3) -> Vector3:
    """Verifica che il vettore sia un versore (norma unitaria)."""
    norm = math.hypot(*vector)
    if abs(norm - 1.0) > _UNIT_NORM_TOLERANCE:
        msg = f"deve essere un versore (norma 1), norma ricevuta {norm:.6g}"
        raise ValueError(msg)
    return vector


UnitVector3 = Annotated[Vector3, AfterValidator(_require_unit_norm)]


def _require_multiple(value: float, step: float, name: str) -> None:
    """Verifica che value sia un multiplo intero, non nullo, di step."""
    ratio = value / step
    if round(ratio) < 1 or abs(ratio - round(ratio)) > _MULTIPLE_TOLERANCE:
        msg = f"{name} = {value} deve essere un multiplo intero di {step}"
        raise ValueError(msg)


class _Section(BaseModel):
    """Base comune delle sezioni: campi sconosciuti vietati, valori immutabili."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SimulationConfig(_Section):
    """Parametri generali della simulazione."""

    seed: NonNegativeInt
    start_epoch: AwareDatetime
    duration_h: PositiveFloat
    integration_step_s: PositiveFloat
    output_step_s: PositiveFloat

    @property
    def duration_s(self) -> float:
        """Durata simulata [s]."""
        return self.duration_h * _S_PER_H

    @model_validator(mode="after")
    def check_output_step(self) -> Self:
        """Il salvataggio deve avvenire a un numero intero di passi."""
        _require_multiple(self.output_step_s, self.integration_step_s, "output_step_s")
        return self


class InitialStateConfig(_Section):
    """Condizioni iniziali: satellite appena rilasciato, in rotazione casuale."""

    max_rate_deg_s: NonNegativeFloat

    @property
    def max_rate_rad_s(self) -> float:
        """Velocità angolare massima al rilascio [rad/s]."""
        return math.radians(self.max_rate_deg_s)


class SatelliteConfig(_Section):
    """Proprietà di massa e magnetiche del satellite.

    Gli assi principali di inerzia coincidono con gli assi del corpo.
    """

    mass_kg: PositiveFloat
    size_cm: PositiveVector3
    inertia_kg_m2: PositiveVector3
    residual_dipole_am2: Vector3

    @property
    def size_m(self) -> Vector3:
        """Dimensioni lungo gli assi X, Y, Z del corpo [m]."""
        x, y, z = self.size_cm
        return (x * _M_PER_CM, y * _M_PER_CM, z * _M_PER_CM)

    @model_validator(mode="after")
    def check_inertia(self) -> Self:
        """I momenti principali devono rispettare la disuguaglianza triangolare.

        Per ogni corpo rigido reale ciascun momento principale è minore o
        uguale alla somma degli altri due.
        """
        i1, i2, i3 = self.inertia_kg_m2
        if i1 > i2 + i3 or i2 > i1 + i3 or i3 > i1 + i2:
            msg = f"inertia_kg_m2 = {self.inertia_kg_m2} non è fisicamente possibile"
            raise ValueError(msg)
        return self


class OrbitConfig(_Section):
    """Orbita circolare."""

    altitude_km: PositiveFloat
    inclination_deg: Annotated[float, Field(ge=0.0, le=180.0)]
    descending_node_local_time: dt.time
    initial_argument_of_latitude_deg: float

    @property
    def altitude_m(self) -> float:
        """Quota sopra il raggio equatoriale terrestre [m]."""
        return self.altitude_km * _M_PER_KM

    @property
    def inclination_rad(self) -> float:
        """Inclinazione [rad]."""
        return math.radians(self.inclination_deg)

    @property
    def descending_node_local_time_s(self) -> float:
        """Ora solare locale al nodo discendente [s dalla mezzanotte]."""
        t = self.descending_node_local_time
        return t.hour * _S_PER_H + t.minute * _S_PER_MIN + t.second

    @property
    def initial_argument_of_latitude_rad(self) -> float:
        """Argomento di latitudine all'istante iniziale [rad]."""
        return math.radians(self.initial_argument_of_latitude_deg)


class GroundStationConfig(_Section):
    """Stazione di terra."""

    name: NonEmptyStr
    latitude_deg: Latitude
    longitude_deg: Longitude
    altitude_m: float
    min_elevation_deg: Annotated[float, Field(ge=0.0, lt=90.0)]

    @property
    def latitude_rad(self) -> float:
        """Latitudine geodetica [rad]."""
        return math.radians(self.latitude_deg)

    @property
    def longitude_rad(self) -> float:
        """Longitudine, positiva verso est [rad]."""
        return math.radians(self.longitude_deg)

    @property
    def min_elevation_rad(self) -> float:
        """Elevazione minima di utilizzo [rad]."""
        return math.radians(self.min_elevation_deg)


class SensorsConfig(_Section):
    """Rumore di misura dei sensori (deviazione standard, 1 sigma per asse)."""

    attitude_noise_deg: NonNegativeFloat
    rate_noise_deg_s: NonNegativeFloat
    magnetometer_noise_nt: NonNegativeFloat

    @property
    def attitude_noise_rad(self) -> float:
        """Rumore sulla misura d'assetto [rad]."""
        return math.radians(self.attitude_noise_deg)

    @property
    def rate_noise_rad_s(self) -> float:
        """Rumore sulla misura di velocità angolare [rad/s]."""
        return math.radians(self.rate_noise_deg_s)

    @property
    def magnetometer_noise_t(self) -> float:
        """Rumore sulla misura del campo magnetico [T]."""
        return self.magnetometer_noise_nt * _T_PER_NT


class ActuatorsConfig(_Section):
    """Ruote di reazione e magnetorquer, uno per asse del corpo."""

    wheel_max_torque_mnm: PositiveFloat
    wheel_max_momentum_mnms: PositiveFloat
    wheel_max_speed_rpm: PositiveFloat
    magnetorquer_max_dipole_am2: PositiveFloat

    @property
    def wheel_max_torque_nm(self) -> float:
        """Coppia massima di ogni ruota [N m]."""
        return self.wheel_max_torque_mnm * _NM_PER_MNM

    @property
    def wheel_max_momentum_nms(self) -> float:
        """Momento angolare massimo di ogni ruota [N m s]."""
        return self.wheel_max_momentum_mnms * _NMS_PER_MNMS

    @property
    def wheel_max_speed_rad_s(self) -> float:
        """Velocità di ogni ruota al momento angolare massimo [rad/s]."""
        return self.wheel_max_speed_rpm * _RAD_S_PER_RPM

    @property
    def wheel_inertia_kg_m2(self) -> float:
        """Inerzia del rotore di ogni ruota [kg m^2]."""
        return self.wheel_max_momentum_nms / self.wheel_max_speed_rad_s


class ControlConfig(_Section):
    """Controllo d'assetto discreto."""

    control_period_s: PositiveFloat
    pd_natural_frequency_rad_s: PositiveFloat
    pd_damping_ratio: PositiveFloat
    max_slew_rate_deg_s: PositiveFloat

    @property
    def max_slew_rate_rad_s(self) -> float:
        """Velocità angolare massima comandata nelle manovre [rad/s]."""
        return math.radians(self.max_slew_rate_deg_s)


class PanelConfig(_Section):
    """Pannello solare piano, fisso rispetto al corpo."""

    name: NonEmptyStr
    area_cm2: PositiveFloat
    normal_body: UnitVector3

    @property
    def area_m2(self) -> float:
        """Area delle celle [m^2]."""
        return self.area_cm2 * _M2_PER_CM2


class SolarArrayConfig(_Section):
    """Generatore solare: celle, perdite e pannelli."""

    cell_efficiency: Fraction
    derating_factor: Fraction
    sun_pointing_axis_body: UnitVector3
    panels: Annotated[tuple[PanelConfig, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def check_unique_names(self) -> Self:
        """Ogni pannello deve avere un nome diverso."""
        names = [panel.name for panel in self.panels]
        if len(set(names)) != len(names):
            msg = f"nomi dei pannelli ripetuti: {names}"
            raise ValueError(msg)
        return self


class BatteryConfig(_Section):
    """Batteria."""

    capacity_wh: PositiveFloat
    charge_efficiency: Fraction
    discharge_efficiency: Fraction
    initial_soc_pct: Percent

    @property
    def capacity_j(self) -> float:
        """Capacità [J]."""
        return self.capacity_wh * _J_PER_WH

    @property
    def initial_soc(self) -> float:
        """Stato di carica iniziale, frazione tra 0 e 1."""
        return self.initial_soc_pct * _FRACTION_PER_PCT


class LoadsConfig(_Section):
    """Consumi elettrici [W]."""

    obc_w: NonNegativeFloat
    adcs_base_w: NonNegativeFloat
    wheel_max_power_w: NonNegativeFloat
    magnetorquer_max_power_w: NonNegativeFloat
    transmitter_w: NonNegativeFloat
    camera_w: NonNegativeFloat


class PayloadConfig(_Section):
    """Fotocamera, memoria di bordo e antenna di scarico dati."""

    camera_boresight_body: UnitVector3
    camera_data_rate_mbps: PositiveFloat
    storage_capacity_gbit: PositiveFloat
    antenna_boresight_body: UnitVector3
    antenna_beamwidth_deg: Annotated[float, Field(gt=0.0, le=180.0)]
    downlink_rate_mbps: PositiveFloat

    @property
    def camera_data_rate_bps(self) -> float:
        """Dati generati durante l'acquisizione [bit/s]."""
        return self.camera_data_rate_mbps * _BPS_PER_MBPS

    @property
    def storage_capacity_bit(self) -> float:
        """Capacità della memoria di bordo [bit]."""
        return self.storage_capacity_gbit * _BIT_PER_GBIT

    @property
    def antenna_half_beamwidth_rad(self) -> float:
        """Metà dell'ampiezza del fascio dell'antenna [rad]."""
        return math.radians(self.antenna_beamwidth_deg) / 2

    @property
    def downlink_rate_bps(self) -> float:
        """Velocità di scarico dati [bit/s]."""
        return self.downlink_rate_mbps * _BPS_PER_MBPS


class ImagingConfig(_Section):
    """Regione di acquisizione delle immagini, in latitudine e longitudine.

    La regione è un rettangolo che non attraversa l'antimeridiano (180 gradi).
    """

    region_name: NonEmptyStr
    latitude_min_deg: Latitude
    latitude_max_deg: Latitude
    longitude_min_deg: Longitude
    longitude_max_deg: Longitude

    @property
    def latitude_range_rad(self) -> tuple[float, float]:
        """Latitudine minima e massima [rad]."""
        return (
            math.radians(self.latitude_min_deg),
            math.radians(self.latitude_max_deg),
        )

    @property
    def longitude_range_rad(self) -> tuple[float, float]:
        """Longitudine minima e massima [rad]."""
        return (
            math.radians(self.longitude_min_deg),
            math.radians(self.longitude_max_deg),
        )

    @model_validator(mode="after")
    def check_ranges(self) -> Self:
        """I valori minimi devono essere minori dei massimi."""
        if self.latitude_min_deg >= self.latitude_max_deg:
            msg = "latitude_min_deg deve essere minore di latitude_max_deg"
            raise ValueError(msg)
        if self.longitude_min_deg >= self.longitude_max_deg:
            msg = "longitude_min_deg deve essere minore di longitude_max_deg"
            raise ValueError(msg)
        return self


class ModesConfig(_Section):
    """Soglie del gestore dei modi, con isteresi."""

    safe_enter_soc_pct: Percent
    safe_exit_soc_pct: Percent
    detumble_enter_rate_deg_s: PositiveFloat
    detumble_exit_rate_deg_s: PositiveFloat
    confirmation_time_s: NonNegativeFloat

    @property
    def safe_enter_soc(self) -> float:
        """Soglia di ingresso in SAFE, frazione tra 0 e 1."""
        return self.safe_enter_soc_pct * _FRACTION_PER_PCT

    @property
    def safe_exit_soc(self) -> float:
        """Soglia di uscita da SAFE, frazione tra 0 e 1."""
        return self.safe_exit_soc_pct * _FRACTION_PER_PCT

    @property
    def detumble_enter_rate_rad_s(self) -> float:
        """Soglia di ingresso in DETUMBLE [rad/s]."""
        return math.radians(self.detumble_enter_rate_deg_s)

    @property
    def detumble_exit_rate_rad_s(self) -> float:
        """Soglia di uscita da DETUMBLE [rad/s]."""
        return math.radians(self.detumble_exit_rate_deg_s)

    @model_validator(mode="after")
    def check_hysteresis(self) -> Self:
        """Le soglie di ingresso e di uscita devono formare un'isteresi."""
        if self.safe_exit_soc_pct <= self.safe_enter_soc_pct:
            msg = "safe_exit_soc_pct deve superare safe_enter_soc_pct"
            raise ValueError(msg)
        if self.detumble_enter_rate_deg_s <= self.detumble_exit_rate_deg_s:
            msg = "detumble_enter_rate_deg_s deve superare detumble_exit_rate_deg_s"
            raise ValueError(msg)
        return self


class MissionConfig(_Section):
    """Configurazione completa di uno scenario di missione."""

    simulation: SimulationConfig
    initial_state: InitialStateConfig
    satellite: SatelliteConfig
    orbit: OrbitConfig
    ground_station: GroundStationConfig
    sensors: SensorsConfig
    actuators: ActuatorsConfig
    control: ControlConfig
    solar_array: SolarArrayConfig
    battery: BatteryConfig
    loads: LoadsConfig
    payload: PayloadConfig
    imaging: ImagingConfig
    modes: ModesConfig

    @model_validator(mode="after")
    def check_control_period(self) -> Self:
        """Il controllo deve agire a un numero intero di passi dell'integratore."""
        _require_multiple(
            self.control.control_period_s,
            self.simulation.integration_step_s,
            "control.control_period_s",
        )
        return self

    @model_validator(mode="after")
    def check_slew_rate(self) -> Self:
        """Una manovra non deve far scattare il DETUMBLE.

        Se la velocità massima di manovra raggiungesse la soglia di ingresso in
        DETUMBLE, ogni grande manovra riporterebbe il satellite in DETUMBLE e il
        puntamento non si concluderebbe mai.
        """
        if self.control.max_slew_rate_deg_s >= self.modes.detumble_enter_rate_deg_s:
            msg = (
                "control.max_slew_rate_deg_s deve essere minore di "
                "modes.detumble_enter_rate_deg_s"
            )
            raise ValueError(msg)
        return self


def load_config(path: Path) -> MissionConfig:
    """Legge e valida un file di configurazione TOML.

    Args:
        path: percorso del file TOML.

    Returns:
        La configurazione validata, immutabile.

    Raises:
        FileNotFoundError: se il file non esiste.
        tomllib.TOMLDecodeError: se il file non è TOML valido.
        pydantic.ValidationError: se un valore manca, è del tipo sbagliato o
            viola un vincolo.
    """
    with path.open("rb") as file:
        data = tomllib.load(file)
    return MissionConfig.model_validate(data)
