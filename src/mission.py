"""Mission orchestration: gates FOLLOW (and, later, ISR) behavior behind
both config.MISSION_MODE AND the flight controller's *current* flight mode.

Neither mission ever triggers just because this process is running:
- FOLLOW only runs while the flight controller reports
  config.FOLLOW_TRIGGER_FLIGHT_MODE (e.g. GUIDED).
- ISR (not yet implemented -- next milestone, per project staging) will
  only trigger while in config.ISR_TRIGGER_FLIGHT_MODE (e.g. AUTO) AND
  after climbing above config.ISR_TRIGGER_ALTITUDE_M.

update() must be driven periodically from the main thread (see main.py's
GLib.timeout_add wiring), never from the GStreamer streaming thread that
produces detections.

status_text() is a one-line summary (MAVLink link health, mission mode,
live flight mode, follow state) -- printed to the console periodically by
update() below, and also wired (via main.py) into
src.probes.register_frame_status_provider() to draw the same line as an
on-screen HUD overlay on the video itself.
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
        self._isr_active = False
        self._last_status_log_time = 0.0

    def on_detection(self, detection) -> None:
        """Wire this into src.probes.register_detection_listener()."""
        if self.follow is not None:
            self.follow.add_detection(detection)

    def status_text(self) -> str:
        """One-line status summary -- see module docstring."""
        link_state = "CONNECTED" if self._mavlink.is_link_healthy() else "NO HEARTBEAT"
        flight_mode = self._mavlink.get_flight_mode() or "UNKNOWN"
        if self.follow is not None:
            mission_state = "ACTIVE" if self.follow.active else "STANDBY"
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

        if config.MISSION_MODE == "FOLLOW" and self.follow is not None:
            should_run = mode == config.FOLLOW_TRIGGER_FLIGHT_MODE
            if should_run and not self.follow.active:
                self.follow.start()
            elif not should_run and self.follow.active:
                self.follow.stop()
            if self.follow.active:
                self.follow.update()

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
