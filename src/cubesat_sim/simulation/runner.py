"""Ciclo principale della simulazione di missione.

A ogni periodo di controllo (1 s nello scenario di riferimento):

1. ambiente all'istante t: orbita, Sole, eclissi, campo magnetico, stazione,
   punto sotto il satellite e finestra di acquisizione;
2. misure dei sensori sullo stato vero;
3. gestore dei modi, con grandezze misurate o note a bordo;
4. comandi, calcolati solo dalle misure: B-dot con i magnetorquer in DETUMBLE
   e in SAFE se il satellite ruota ancora velocemente; altrimenti PD con le
   ruote verso il riferimento del modo;
5. energia e dati nel periodo: pannelli e collegamento radio con l'assetto
   vero, consumi dai comandi, batteria e memoria;
6. dinamica vera: RK4 a passo fisso, con comandi, posizione e campo costanti
   nel periodo.

Carichi per modo: fotocamera accesa per tutto il NADIR, trasmettitore acceso
per tutto il DOWNLINK; i dati scendono a terra solo quando l'antenna punta
davvero entro metà fascio. In SAFE fotocamera e trasmettitore sono spenti.

Conoscenze di bordo considerate esatte (limite dichiarato): posizione
sull'orbita e ora, quindi eclissi e visibilità della stazione; stato di carica
della batteria; momento delle ruote, letto dai loro tachimetri.

Condizioni iniziali (caso peggiore): assetto casuale uniforme, velocità
angolare di modulo pari al massimo della configurazione in direzione casuale,
ruote ferme. Tutto viene dal seme della configurazione.

Uscite: una riga ogni output_step_s. Ogni riga descrive l'istante t: stato
vero, ambiente, modo e comandi di t. I contatori cumulati (energia, dati,
saturazione delle ruote) sommano tutto ciò che è avvenuto prima di t.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from cubesat_sim.attitude.actuators import Magnetorquers, ReactionWheels
from cubesat_sim.attitude.bdot import BdotController, bdot_gain
from cubesat_sim.attitude.disturbances import external_torque_function
from cubesat_sim.attitude.dynamics import AttitudeState, propagate_attitude
from cubesat_sim.attitude.pointing import PointingController, pointing_error
from cubesat_sim.attitude.references import (
    PointingAxes,
    nadir_pointing_attitude,
    reference_rate,
    station_pointing_attitude,
    sun_pointing_attitude,
)
from cubesat_sim.attitude.sensors import Measurements, Sensors
from cubesat_sim.core.config import MissionConfig
from cubesat_sim.core.quaternions import quat_to_matrix, random_quaternion
from cubesat_sim.core.results import SimulationResults
from cubesat_sim.core.types import FloatArray
from cubesat_sim.core.vectors import unit
from cubesat_sim.environment.earth import eci_to_ecef_matrix, julian_date
from cubesat_sim.environment.ground import (
    GroundStation,
    is_over_region,
    subsatellite_point,
)
from cubesat_sim.environment.magnetic import magnetic_field_eci
from cubesat_sim.environment.orbit import CircularOrbit
from cubesat_sim.environment.sun import is_in_eclipse, sun_position_eci
from cubesat_sim.modes.manager import Mode, ModeInputs, ModeManager
from cubesat_sim.resources.battery import Battery
from cubesat_sim.resources.data import DataStorage
from cubesat_sim.resources.loads import PowerLoads
from cubesat_sim.resources.solar import SolarArray

# Tolleranza relativa nel verificare che una durata sia un multiplo del periodo.
_RATIO_TOLERANCE = 1e-9

# Una riga delle serie temporali: nome della colonna e valore.
_Row = dict[str, float | bool | str]


def _whole_steps(duration_s: float, period_s: float, name: str) -> int:
    """Numero di periodi di controllo contenuti in una durata.

    Args:
        duration_s: durata [s].
        period_s: periodo di controllo [s].
        name: nome del parametro, per il messaggio di errore.

    Returns:
        Numero intero di periodi, almeno 1.

    Raises:
        ValueError: se la durata non è un multiplo intero del periodo.
    """
    ratio = duration_s / period_s
    steps = round(ratio)
    if steps < 1 or abs(ratio - steps) > _RATIO_TOLERANCE * ratio:
        msg = f"{name} deve essere un multiplo intero del periodo di controllo"
        raise ValueError(msg)
    return steps


@dataclass(frozen=True, eq=False)
class _Environment:
    """Ambiente a un istante: calcolato una volta, usato da tutto il ciclo.

    Attributes:
        position: posizione del satellite, ECI [m].
        velocity: velocità del satellite, ECI [m/s].
        sun: posizione del Sole, ECI [m].
        field: campo magnetico vero, ECI [T].
        station: posizione della stazione di terra, ECI [m].
        in_eclipse: satellite nell'ombra della Terra.
        station_visible: stazione sopra l'elevazione minima.
        latitude_rad: latitudine del punto sotto il satellite [rad].
        longitude_rad: longitudine del punto sotto il satellite [rad].
        imaging_window: sopra la regione di acquisizione e al Sole.
    """

    position: FloatArray
    velocity: FloatArray
    sun: FloatArray
    field: FloatArray
    station: FloatArray
    in_eclipse: bool
    station_visible: bool
    latitude_rad: float
    longitude_rad: float
    imaging_window: bool


@dataclass(frozen=True, eq=False)
class _Commands:
    """Comandi agli attuatori per un periodo di controllo, dopo i limiti.

    Attributes:
        wheel_torque: coppia dei motori sulle ruote, assi corpo [N m].
        dipole: dipolo dei magnetorquer, assi corpo [A m^2].
        wheel_saturated: una ruota ha raggiunto il limite di coppia o momento.
    """

    wheel_torque: FloatArray
    dipole: FloatArray
    wheel_saturated: bool


@dataclass
class _Totals:
    """Contatori cumulati dall'inizio della simulazione."""

    generated_j: float = 0.0
    consumed_j: float = 0.0
    loss_j: float = 0.0
    wasted_j: float = 0.0
    unmet_j: float = 0.0
    acquired_bit: float = 0.0
    downlinked_bit: float = 0.0
    lost_bit: float = 0.0
    wheel_saturation_s: float = 0.0


class _MissionSimulation:
    """Una simulazione in corso: stato vero, software di bordo e contatori."""

    def __init__(self, config: MissionConfig) -> None:
        """Prepara modelli, controllori e condizioni iniziali dal seme."""
        self._config = config
        self._epoch = config.simulation.start_epoch
        self._period_s = config.control.control_period_s
        self._step_s = config.simulation.integration_step_s
        self._substeps = round(self._period_s / self._step_s)
        self._steps = _whole_steps(
            config.simulation.duration_s, self._period_s, "duration_h"
        )
        self._output_every = _whole_steps(
            config.simulation.output_step_s, self._period_s, "output_step_s"
        )
        self._inertia = np.array(config.satellite.inertia_kg_m2)
        self._residual_dipole = np.array(config.satellite.residual_dipole_am2)

        # Ambiente.
        self._orbit = CircularOrbit.from_config(config.orbit, self._epoch)
        self._station = GroundStation.from_config(config.ground_station)

        # Stato vero iniziale e rumore dei sensori, dallo stesso generatore.
        rng = np.random.default_rng(config.simulation.seed)
        self._initial_attitude = random_quaternion(rng)
        direction = unit(rng.normal(size=3))
        self._initial_rate = direction * config.initial_state.max_rate_rad_s
        self._state = AttitudeState(
            self._initial_attitude, self._initial_rate, np.zeros(3)
        ).to_vector()
        self._sensors = Sensors(config.sensors, rng)

        # Software di bordo: gestore dei modi, controllori, attuatori.
        self._manager = ModeManager(config.modes)
        self._axes = PointingAxes.from_config(config)
        gain = bdot_gain(
            self._orbit.mean_motion_rad_s,
            self._orbit.inclination_rad,
            float(self._inertia.min()),
        )
        self._bdot = BdotController(gain, self._period_s)
        self._bdot_active = False
        self._pointing = PointingController(
            self._inertia,
            config.control.pd_natural_frequency_rad_s,
            config.control.pd_damping_ratio,
            config.control.max_slew_rate_rad_s,
        )
        self._wheels = ReactionWheels(config.actuators, self._period_s)
        self._torquers = Magnetorquers(config.actuators)

        # Risorse di bordo.
        self._solar = SolarArray(config.solar_array)
        self._loads = PowerLoads(config.loads, config.actuators)
        self._battery = Battery(config.battery)
        self._storage = DataStorage(config.payload)

        self._totals = _Totals()
        self._rows: list[_Row] = []

    def run(self) -> SimulationResults:
        """Esegue tutta la simulazione e raccoglie i risultati."""
        now = self._environment(0.0)
        for k in range(self._steps):
            following = self._environment((k + 1) * self._period_s)
            self._control_period(k, now, following)
            now = following
        return SimulationResults(
            timeseries=pd.DataFrame(self._rows), metadata=self._metadata()
        )

    def _environment(self, elapsed_s: float) -> _Environment:
        """Calcola l'ambiente all'istante indicato."""
        jd = julian_date(self._epoch, elapsed_s)
        position, velocity = self._orbit.state_eci(elapsed_s)
        sun = sun_position_eci(jd)
        position_ecef = eci_to_ecef_matrix(jd) @ position
        latitude, longitude = subsatellite_point(position_ecef)
        in_eclipse = is_in_eclipse(position, sun)
        over_region = is_over_region(latitude, longitude, self._config.imaging)
        return _Environment(
            position=position,
            velocity=velocity,
            sun=sun,
            field=magnetic_field_eci(position, jd),
            station=self._station.position_eci(jd),
            in_eclipse=in_eclipse,
            station_visible=self._station.is_visible(position_ecef),
            latitude_rad=latitude,
            longitude_rad=longitude,
            imaging_window=over_region and not in_eclipse,
        )

    def _control_period(
        self, k: int, now: _Environment, following: _Environment
    ) -> None:
        """Un periodo di controllo: decisioni di bordo, risorse, dinamica vera."""
        elapsed_s = k * self._period_s
        truth = AttitudeState.from_vector(self._state)

        # Software di bordo: vede solo misure e grandezze note a bordo.
        measured = self._sensors.measure(self._state, now.field)
        inputs = ModeInputs(
            state_of_charge=self._battery.state_of_charge,
            angular_rate_rad_s=float(np.linalg.norm(measured.angular_velocity)),
            station_visible=now.station_visible,
            imaging_window=now.imaging_window,
        )
        mode = self._manager.update(elapsed_s, inputs)
        commands = self._commands(mode, measured, truth.wheel_momentum, now, following)

        # Fisica: pannelli e collegamento radio dipendono dall'assetto vero.
        imaging = mode is Mode.NADIR
        transmitting = mode is Mode.DOWNLINK
        sun_body = quat_to_matrix(truth.attitude).T @ (now.sun - now.position)
        solar_w = self._solar.power(sun_body, now.in_eclipse)
        consumed_w = self._loads.consumption(
            commands.wheel_torque,
            commands.dipole,
            transmitting=transmitting,
            imaging=imaging,
        ).total_w
        antenna_error = self._antenna_error(truth.attitude, now)
        downlinking = transmitting and self._storage.link_available(
            station_visible=now.station_visible, antenna_error_rad=antenna_error
        )

        if k % self._output_every == 0:
            self._record(
                elapsed_s, mode, now, truth, antenna_error, solar_w, consumed_w
            )

        self._update_resources(
            solar_w, consumed_w, imaging=imaging, downlinking=downlinking
        )
        if commands.wheel_saturated:
            self._totals.wheel_saturation_s += self._period_s

        # Dinamica vera per il periodo, con comandi costanti.
        external = external_torque_function(
            now.position,
            now.field,
            commands.dipole + self._residual_dipole,
            self._inertia,
        )
        for _ in range(self._substeps):
            self._state = propagate_attitude(
                self._state,
                self._step_s,
                self._inertia,
                commands.wheel_torque,
                external,
            )

    def _commands(
        self,
        mode: Mode,
        measured: Measurements,
        wheel_momentum: FloatArray,
        now: _Environment,
        following: _Environment,
    ) -> _Commands:
        """Comandi agli attuatori, calcolati solo da misure e dati di bordo.

        B-dot in DETUMBLE e in SAFE con il satellite ancora in rotazione
        veloce, quando puntare con le ruote non avrebbe senso; il B-dot riparte
        da zero ogni volta che viene riattivato. Negli altri casi PD con le
        ruote verso il riferimento del modo, con la velocità del riferimento
        stimata dall'istante successivo.
        """
        tumbling_in_safe = mode is Mode.SAFE and self._manager.tumbling
        if mode is Mode.DETUMBLE or tumbling_in_safe:
            if not self._bdot_active:
                self._bdot.reset()
            self._bdot_active = True
            requested = self._bdot.dipole(measured.magnetic_field)
            return _Commands(
                wheel_torque=np.zeros(3),
                dipole=self._torquers.command(requested).dipole,
                wheel_saturated=False,
            )
        self._bdot_active = False
        q_ref = self._reference(mode, now)
        q_next = self._reference(mode, following)
        rate_ref = reference_rate(q_ref, q_next, self._period_s)
        torque = self._pointing.torque(
            measured.attitude, measured.angular_velocity, q_ref, rate_ref
        )
        command = self._wheels.command(torque, wheel_momentum)
        return _Commands(
            wheel_torque=command.torque,
            dipole=np.zeros(3),
            wheel_saturated=command.torque_saturated or command.momentum_saturated,
        )

    def _reference(self, mode: Mode, env: _Environment) -> FloatArray:
        """Assetto di riferimento del modo; SAFE punta il Sole come SUN_POINTING."""
        if mode is Mode.NADIR:
            return nadir_pointing_attitude(
                self._axes, env.sun, env.position, env.velocity
            )
        if mode is Mode.DOWNLINK:
            return station_pointing_attitude(
                self._axes, env.sun, env.position, env.velocity, env.station
            )
        return sun_pointing_attitude(self._axes, env.sun, env.position, env.velocity)

    def _pointing_error(
        self, mode: Mode, attitude: FloatArray, env: _Environment
    ) -> float:
        """Errore di puntamento vero dell'asse principale del modo.

        SUN_POINTING e SAFE: pannelli verso il Sole; NADIR: fotocamera verso il
        centro della Terra; DOWNLINK: antenna verso la stazione. In DETUMBLE
        non c'è puntamento: il valore è NaN.
        """
        if mode is Mode.DETUMBLE:
            return math.nan
        if mode is Mode.NADIR:
            return pointing_error(attitude, self._axes.camera, -env.position)
        if mode is Mode.DOWNLINK:
            return pointing_error(
                attitude, self._axes.antenna, env.station - env.position
            )
        return pointing_error(attitude, self._axes.sun, env.sun - env.position)

    def _antenna_error(self, attitude: FloatArray, env: _Environment) -> float:
        """Errore vero dell'antenna verso la stazione; NaN se non è visibile."""
        if not env.station_visible:
            return math.nan
        return pointing_error(attitude, self._axes.antenna, env.station - env.position)

    def _update_resources(
        self, solar_w: float, consumed_w: float, *, imaging: bool, downlinking: bool
    ) -> None:
        """Batteria e memoria per un periodo di controllo, con i contatori."""
        period_s = self._period_s
        energy = self._battery.update(solar_w - consumed_w, period_s)
        data = self._storage.update(period_s, imaging=imaging, downlinking=downlinking)
        totals = self._totals
        totals.generated_j += solar_w * period_s
        totals.consumed_j += consumed_w * period_s
        totals.loss_j += energy.loss_j
        totals.wasted_j += energy.wasted_j
        totals.unmet_j += energy.unmet_j
        totals.acquired_bit += data.acquired_bits
        totals.downlinked_bit += data.downlinked_bits
        totals.lost_bit += data.lost_bits

    def _record(
        self,
        elapsed_s: float,
        mode: Mode,
        now: _Environment,
        truth: AttitudeState,
        antenna_error_rad: float,
        solar_w: float,
        consumed_w: float,
    ) -> None:
        """Aggiunge alle serie temporali la riga dell'istante elapsed_s."""
        q = truth.attitude
        h = truth.wheel_momentum
        totals = self._totals
        self._rows.append(
            {
                "time_s": elapsed_s,
                "mode": str(mode),
                "latitude_rad": now.latitude_rad,
                "longitude_rad": now.longitude_rad,
                "in_eclipse": now.in_eclipse,
                "station_visible": now.station_visible,
                "imaging_window": now.imaging_window,
                "attitude_w": float(q[0]),
                "attitude_x": float(q[1]),
                "attitude_y": float(q[2]),
                "attitude_z": float(q[3]),
                "angular_rate_rad_s": float(np.linalg.norm(truth.angular_velocity)),
                "wheel_momentum_x_nms": float(h[0]),
                "wheel_momentum_y_nms": float(h[1]),
                "wheel_momentum_z_nms": float(h[2]),
                "pointing_error_rad": self._pointing_error(mode, q, now),
                "antenna_error_rad": antenna_error_rad,
                "solar_power_w": solar_w,
                "consumed_power_w": consumed_w,
                "state_of_charge": self._battery.state_of_charge,
                "energy_generated_j": totals.generated_j,
                "energy_consumed_j": totals.consumed_j,
                "energy_loss_j": totals.loss_j,
                "energy_wasted_j": totals.wasted_j,
                "energy_unmet_j": totals.unmet_j,
                "data_stored_bit": self._storage.stored_bits,
                "data_acquired_bit": totals.acquired_bit,
                "data_downlinked_bit": totals.downlinked_bit,
                "data_lost_bit": totals.lost_bit,
                "wheel_saturation_time_s": totals.wheel_saturation_s,
            }
        )

    def _metadata(self) -> dict[str, Any]:
        """Seme, configurazione completa, condizioni iniziali e transizioni."""
        return {
            "seed": self._config.simulation.seed,
            "config": self._config.model_dump(mode="json"),
            "initial_state": {
                "attitude": self._initial_attitude.tolist(),
                "angular_velocity_rad_s": self._initial_rate.tolist(),
            },
            "transitions": [asdict(item) for item in self._manager.transitions],
        }


def run_simulation(config: MissionConfig) -> SimulationResults:
    """Esegue una simulazione completa della missione.

    Args:
        config: configurazione della missione.

    Returns:
        Serie temporali e metadati. Stessa configurazione e stesso seme danno
        risultati identici.

    Raises:
        ValueError: se la durata o il passo di uscita non sono multipli interi
            del periodo di controllo.
    """
    return _MissionSimulation(config).run()
