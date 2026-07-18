"""Mission orchestration: gates FOLLOW / ISR / AVOID behavior behind
config.MISSION_MODE (FOLLOW and ISR additionally require the flight
controller's *current* flight mode to match; AVOID does not -- see below).

FOLLOW and ISR never trigger just because this process is running:
- FOLLOW only runs while the flight controller reports
  config.FOLLOW_TRIGGER_FLIGHT_MODE (e.g. GUIDED).
- ISR (not yet implemented -- next milestone, per project staging) will
  only trigger while in config.ISR_TRIGGER_FLIGHT_MODE (e.g. AUTO) AND
  after climbing above config.ISR_TRIGGER_ALTITUDE_M.
- AVOID streams MAVLink OBSTACLE_DISTANCE continuously whenever the
  mavlink link is healthy, with NO flight-mode gate -- unlike FOLLOW, it
  never sends a vehicle-motion command itself (see src/avoidance.py);
  it's sensor telemetry ArduPilot's own OA_TYPE can use across
  GUIDED/AUTO/RTL, not just one specific mode.

update() must be driven periodically from the main thread (see main.py's
GLib.timeout_add wiring), never from the GStreamer streaming thread that
produces detections/obstacle readings.

status_text() is a one-line summary (MAVLink link health, mission mode,
live flight mode, follow/avoid state) -- printed to the console
periodically by update() below, and also wired (via main.py) into
src.probes.register_frame_status_provider() to draw the same line as an
on-screen HUD overlay on the video itself.

update() also prints immediately whenever the link health or the flight
controller's live flight mode actually CHANGES (heartbeat lost/restored,
mode switched via the RC), on top of the periodic status heartbeat --
these are the events worth seeing right away rather than waiting for the
next periodic line.
"""
import time

from . import config
from .mavlink_link import MavlinkLink
from .pid import ObjectFollowController

_STATUS_LOG_INTERVAL_S = 1.0


class Mission:
    def __init__(self, mavlink: MavlinkLink):
        self._mavlink = mavlink
        self.follow = ObjectFollowController(mavlink) if config.MISSION_MODE == "FOLLOW" else None
        self.avoidance = None
        if config.MISSION_MODE == "AVOID":
            # Lazy imports -- calibration/avoidance are only ever touched
            # when AVOID mode is actually configured, same convention as
            # main.py's MISSION_MODE-gated imports.
            from .avoidance import ObstacleAvoidance
            from .calibration import load as load_calibration

            self.avoidance = ObstacleAvoidance(mavlink, load_calibration())
        self._isr_active = False
        self._last_status_log_time = 0.0
        self._last_logged_mode = None
        self._last_logged_link_healthy = None

    def on_detection(self, detection) -> None:
        """Wire this into src.probes.register_detection_listener()."""
        if self.follow is not None:
            self.follow.add_detection(detection)

    def on_obstacle_reading(self, bin_distances_m: list, bin_valid_mask: list) -> None:
        """Wire this into src.probes.register_obstacle_listener()."""
        if self.avoidance is not None:
            self.avoidance.add_bin_distances(bin_distances_m, bin_valid_mask)

    def status_text(self) -> str:
        """One-line status summary -- see module docstring."""
        link_state = "CONNECTED" if self._mavlink.is_link_healthy() else "NO HEARTBEAT"
        flight_mode = self._mavlink.get_flight_mode() or "UNKNOWN"
        if self.follow is not None:
            mission_state = "ACTIVE" if self.follow.active else "STANDBY"
        elif self.avoidance is not None:
            mission_state = "STREAMING" if self.avoidance.streaming else "NO DATA YET"
        elif config.MISSION_MODE == "ISR":
            mission_state = "ACTIVE" if self._isr_active else "STANDBY"
        else:
            mission_state = "-"
        return (
            f"MAVLINK:{link_state}  MODE:{config.MISSION_MODE}  "
            f"FC:{flight_mode}  MISSION:{mission_state}"
        )

    def update(self) -> bool:
        """Call periodically (GLib.timeout_add) from the main thread.
        Returns True so a GLib timeout keeps repeating."""
        mode = self._mavlink.get_flight_mode()
        link_healthy = self._mavlink.is_link_healthy()

        if link_healthy != self._last_logged_link_healthy:
            self._last_logged_link_healthy = link_healthy
            print(f"[mavlink] {'heartbeat OK -- link connected' if link_healthy else 'heartbeat LOST -- link down'}")

        if mode != self._last_logged_mode:
            print(f"[mavlink] flight mode changed: {self._last_logged_mode or '(none)'} -> {mode or '(none)'}")
            self._last_logged_mode = mode

        if config.MISSION_MODE == "FOLLOW" and self.follow is not None:
            should_run = mode == config.FOLLOW_TRIGGER_FLIGHT_MODE
            if should_run and not self.follow.active:
                self.follow.start()
            elif not should_run and self.follow.active:
                self.follow.stop()
            if self.follow.active:
                self.follow.update()

        elif config.MISSION_MODE == "AVOID" and self.avoidance is not None:
            if link_healthy:
                self.avoidance.update()

        elif config.MISSION_MODE == "ISR":
            self._update_isr(mode)

        now = time.monotonic()
        if now - self._last_status_log_time >= _STATUS_LOG_INTERVAL_S:
            self._last_status_log_time = now
            print(f"[status] {self.status_text()}")

        return True

    def _update_isr(self, mode) -> None:
        # Scaffolded, not implemented -- next milestone per project
        # staging (CSV/JSON logging of object + IMU + GPS data). Kept here
        # so main.py's wiring doesn't need to change when it's built.
        if self._isr_active or mode != config.ISR_TRIGGER_FLIGHT_MODE:
            return
        telemetry = self._mavlink.get_telemetry()
        if telemetry and telemetry.imu.relative_altitude_m >= config.ISR_TRIGGER_ALTITUDE_M:
            self._isr_active = True
            print(
                f"[isr] altitude threshold reached "
                f"({telemetry.imu.relative_altitude_m:.1f}m >= {config.ISR_TRIGGER_ALTITUDE_M}m) "
                f"-- logging not yet implemented, see CLAUDE.md/README roadmap"
            )
