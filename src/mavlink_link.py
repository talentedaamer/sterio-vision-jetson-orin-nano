"""MAVLink telemetry + command link to the flight controller over UART.

Connects on /dev/ttyTHS1 (Jetson Orin Nano's onboard UART, wired to the
flight controller's telemetry/companion-computer serial port -- requires
SERIALx_PROTOCOL=2 (MAVLink 2) and SERIALx_BAUD matching config.MAVLINK_BAUD
on that port).

All MAVLink message reception happens on a single background thread
(started by connect()); the public get_*() methods return a thread-safe
snapshot of the latest known values and are safe to call from any thread
(e.g. the GStreamer streaming thread via src/probes.py, or the main/GLib
thread). send_velocity_setpoint() is likewise safe to call from any thread.

Three telemetry methods, per the intended usage:
  - get_imu_telemetry() -- IMU-only (roll/pitch/yaw, angular rate, xyz
    accel, groundspeed, relative height). Never needs a GPS fix.
  - get_gps_compass()   -- GPS position + compass heading. Only trust the
    position/heading fields when .has_fix is True.
  - get_telemetry()     -- the main entry point: merges IMU + GPS/compass
    when a GPS fix is available, falls back to IMU-only (gps=None)
    otherwise. This is the fallback logic described in the project spec.
"""
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

from pymavlink import mavutil

from . import config


@dataclass
class ImuTelemetry:
    """IMU-only telemetry -- does not require a GPS fix."""
    timestamp: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    # Gyro angular RATE (deg/s), not angular acceleration -- MAVLink's
    # ATTITUDE message provides rate, not acceleration; differentiate this
    # yourself over time if true angular acceleration is ever needed.
    roll_rate_dps: float
    pitch_rate_dps: float
    yaw_rate_dps: float
    accel_x_mps2: float
    accel_y_mps2: float
    accel_z_mps2: float
    groundspeed_mps: float
    airspeed_mps: float
    relative_altitude_m: float


@dataclass
class GpsCompassTelemetry:
    """GPS + compass. Only meaningful when has_fix is True."""
    has_fix: bool
    fix_type: int
    satellites_visible: int
    hdop: float
    latitude_deg: float
    longitude_deg: float
    altitude_msl_m: float
    heading_deg: float  # magnetometer-derived compass heading (VFR_HUD.heading)


@dataclass
class Telemetry:
    """Merged view returned by get_telemetry(): imu is always populated;
    gps is None when no GPS fix is available (IMU-only fallback)."""
    imu: ImuTelemetry
    gps: Optional[GpsCompassTelemetry]


# SET_POSITION_TARGET_LOCAL_NED type_mask bits (MAV_FRAME docs) -- named
# explicitly rather than a hardcoded magic literal, since getting this
# wrong sends a command with unintended fields active.
_TYPEMASK_X_IGNORE = 1 << 0
_TYPEMASK_Y_IGNORE = 1 << 1
_TYPEMASK_Z_IGNORE = 1 << 2
_TYPEMASK_AX_IGNORE = 1 << 6
_TYPEMASK_AY_IGNORE = 1 << 7
_TYPEMASK_AZ_IGNORE = 1 << 8
_TYPEMASK_YAW_IGNORE = 1 << 10
_VELOCITY_AND_YAW_RATE_ONLY = (
    _TYPEMASK_X_IGNORE | _TYPEMASK_Y_IGNORE | _TYPEMASK_Z_IGNORE
    | _TYPEMASK_AX_IGNORE | _TYPEMASK_AY_IGNORE | _TYPEMASK_AZ_IGNORE
    | _TYPEMASK_YAW_IGNORE
)


class MavlinkLink:
    def __init__(self, device: str = config.MAVLINK_DEVICE, baud: int = config.MAVLINK_BAUD):
        self._device = device
        self._baud = baud
        self._master = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None

        self._attitude = None
        self._raw_imu = None
        self._vfr_hud = None
        self._gps_raw = None
        self._global_pos = None
        self._heartbeat = None
        self._last_heartbeat_time = 0.0

    def connect(self, heartbeat_timeout_s: float = 10.0) -> None:
        self._master = mavutil.mavlink_connection(self._device, baud=self._baud)
        print(f"[mavlink] waiting for heartbeat on {self._device} @ {self._baud}...")
        self._master.wait_heartbeat(timeout=heartbeat_timeout_s)
        print(
            f"[mavlink] connected: system={self._master.target_system} "
            f"component={self._master.target_component}"
        )
        self._stop.clear()
        self._reader_thread = threading.Thread(target=self._read_loop, name="mavlink-reader", daemon=True)
        self._reader_thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
        if self._master is not None:
            self._master.close()

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            msg = self._master.recv_match(blocking=True, timeout=1.0)
            if msg is None:
                continue
            msg_type = msg.get_type()
            with self._lock:
                if msg_type == "ATTITUDE":
                    self._attitude = msg
                elif msg_type in ("RAW_IMU", "SCALED_IMU2"):
                    self._raw_imu = msg
                elif msg_type == "VFR_HUD":
                    self._vfr_hud = msg
                elif msg_type == "GPS_RAW_INT":
                    self._gps_raw = msg
                elif msg_type == "GLOBAL_POSITION_INT":
                    self._global_pos = msg
                elif msg_type == "HEARTBEAT":
                    self._heartbeat = msg
                    self._last_heartbeat_time = time.monotonic()

    def is_link_healthy(self) -> bool:
        return (time.monotonic() - self._last_heartbeat_time) < config.MAVLINK_HEARTBEAT_TIMEOUT_S

    def get_flight_mode(self) -> Optional[str]:
        """Current flight mode string (e.g. 'GUIDED', 'AUTO'), or None if
        no heartbeat has been received yet."""
        if self._master is None or self._heartbeat is None:
            return None
        return self._master.flightmode

    def get_imu_telemetry(self) -> Optional[ImuTelemetry]:
        with self._lock:
            attitude = self._attitude
            raw_imu = self._raw_imu
            vfr_hud = self._vfr_hud
            global_pos = self._global_pos
        if attitude is None or vfr_hud is None:
            return None

        if raw_imu is not None:
            # RAW_IMU/SCALED_IMU2 report acceleration in milli-g.
            g = 9.80665
            ax = raw_imu.xacc / 1000.0 * g
            ay = raw_imu.yacc / 1000.0 * g
            az = raw_imu.zacc / 1000.0 * g
        else:
            ax = ay = az = 0.0

        # VFR_HUD.alt is ambiguous (spec says MSL, but ArduPilot commonly
        # fills it with relative-to-home altitude in practice) --
        # GLOBAL_POSITION_INT.relative_alt is unambiguous and preferred
        # when available. Both come from the EKF, not GPS directly, so
        # relative_alt is typically populated even without a GPS fix.
        if global_pos is not None:
            relative_altitude_m = global_pos.relative_alt / 1000.0
        else:
            relative_altitude_m = vfr_hud.alt

        return ImuTelemetry(
            timestamp=time.time(),
            roll_deg=math.degrees(attitude.roll),
            pitch_deg=math.degrees(attitude.pitch),
            yaw_deg=math.degrees(attitude.yaw),
            roll_rate_dps=math.degrees(attitude.rollspeed),
            pitch_rate_dps=math.degrees(attitude.pitchspeed),
            yaw_rate_dps=math.degrees(attitude.yawspeed),
            accel_x_mps2=ax,
            accel_y_mps2=ay,
            accel_z_mps2=az,
            groundspeed_mps=vfr_hud.groundspeed,
            airspeed_mps=vfr_hud.airspeed,
            relative_altitude_m=relative_altitude_m,
        )

    def get_gps_compass(self) -> Optional[GpsCompassTelemetry]:
        with self._lock:
            gps_raw, global_pos, vfr_hud = self._gps_raw, self._global_pos, self._vfr_hud
        if gps_raw is None:
            return None

        has_fix = gps_raw.fix_type >= 3  # 3D fix -- see MAV_GPS_FIX_TYPE
        return GpsCompassTelemetry(
            has_fix=has_fix,
            fix_type=gps_raw.fix_type,
            satellites_visible=gps_raw.satellites_visible,
            hdop=(gps_raw.eph / 100.0) if gps_raw.eph != 65535 else float("inf"),
            latitude_deg=(global_pos.lat / 1e7) if global_pos else 0.0,
            longitude_deg=(global_pos.lon / 1e7) if global_pos else 0.0,
            altitude_msl_m=(global_pos.alt / 1000.0) if global_pos else 0.0,
            heading_deg=vfr_hud.heading if vfr_hud else 0.0,
        )

    def get_telemetry(self) -> Optional[Telemetry]:
        """Main entry point: IMU merged with GPS/compass when a fix is
        available, IMU-only (gps=None) otherwise."""
        imu = self.get_imu_telemetry()
        if imu is None:
            return None
        gps = self.get_gps_compass()
        if gps is None or not gps.has_fix:
            return Telemetry(imu=imu, gps=None)
        return Telemetry(imu=imu, gps=gps)

    def send_velocity_setpoint(self, vx: float, vy: float, vz: float) -> None:
        """Command a body-frame velocity setpoint in m/s (vx=forward,
        vy=right, vz=down) via SET_POSITION_TARGET_LOCAL_NED. Only takes
        effect while the flight controller is in a mode that accepts
        offboard/guided velocity commands -- see config.FOLLOW_TRIGGER_
        FLIGHT_MODE and src/pid.py (the caller). Thread-safe to call from
        any thread.
        """
        if self._master is None:
            return
        self._master.mav.set_position_target_local_ned_send(
            0,
            self._master.target_system,
            self._master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
            _VELOCITY_AND_YAW_RATE_ONLY,
            0, 0, 0,       # position (ignored)
            vx, vy, vz,    # velocity, body frame
            0, 0, 0,       # acceleration (ignored)
            0, 0,          # yaw (ignored), yaw_rate = 0
        )
