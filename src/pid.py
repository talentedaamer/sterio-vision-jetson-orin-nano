"""Generic PID controller + a drone object-follow controller built on it.

ObjectFollowController converts the most recent Detection's camera-relative
X/Y/Z (see src/distance.py) into a body-frame velocity setpoint sent to the
flight controller via MavlinkLink.send_velocity_setpoint().

SAFETY: this drives real vehicle motion. The gains in src/config.py
(FOLLOW_PID_*) are documented starting points, not validated values --
tune them incrementally: bench test with props off first, then a
supervised, low-altitude tethered GUIDED-mode test, before trusting this
in free flight. config.FOLLOW_DRY_RUN defaults to True, which computes and
logs every setpoint without ever sending it, specifically so the control
logic can be exercised safely before being armed for real.

There is no object tracker yet (see CLAUDE.md/README roadmap), so
add_detection() just keeps the latest detection matching
config.FOLLOW_TARGET_CLASS -- if more than one matching object is in view,
which one gets followed can change frame to frame. Add nvtracker-based
persistent IDs before relying on this with multiple similar objects in
frame.
"""
import time
from typing import Optional

from . import config
from .distance import Detection
from .mavlink_link import MavlinkLink


class PIDController:
    """Standard PID with output clamping and basic integral anti-windup."""

    def __init__(self, kp: float, ki: float, kd: float, output_limit: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self._integral = 0.0
        self._prev_error: Optional[float] = None
        self._prev_time: Optional[float] = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None
        self._prev_time = None

    def update(self, error: float, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else now
        dt = 0.0 if self._prev_time is None else max(now - self._prev_time, 1e-3)

        self._integral += error * dt
        derivative = (
            0.0 if self._prev_error is None or dt == 0.0
            else (error - self._prev_error) / dt
        )

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        clamped = max(-self.output_limit, min(self.output_limit, output))
        if output != clamped:
            # Anti-windup: undo the integration step that pushed us past
            # the clamp, so the integral term doesn't keep growing while
            # saturated.
            self._integral -= error * dt

        self._prev_error = error
        self._prev_time = now
        return clamped


class ObjectFollowController:
    """Holds config.FOLLOW_TARGET_DISTANCE_M from the most recent detection
    of config.FOLLOW_TARGET_CLASS, centered in frame.

    add_detection() is the subscribe callback for
    src.probes.register_detection_listener() -- cheap and thread-safe,
    called from the GStreamer streaming thread; it only caches the latest
    matching detection. update() runs the actual PID computation and
    MAVLink send, and must be called periodically from the main thread
    (see src/mission.py + main.py), never from the streaming thread.
    """

    def __init__(self, mavlink: MavlinkLink):
        self._mavlink = mavlink
        self._lateral_pid = PIDController(*config.FOLLOW_PID_LATERAL, output_limit=config.FOLLOW_MAX_VELOCITY_MPS)
        self._vertical_pid = PIDController(*config.FOLLOW_PID_VERTICAL, output_limit=config.FOLLOW_MAX_VELOCITY_MPS)
        self._forward_pid = PIDController(*config.FOLLOW_PID_FORWARD, output_limit=config.FOLLOW_MAX_VELOCITY_MPS)
        self._latest: Optional[Detection] = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def add_detection(self, detection: Detection) -> None:
        if detection.class_id != config.FOLLOW_TARGET_CLASS:
            return
        self._latest = detection

    def start(self) -> None:
        self._lateral_pid.reset()
        self._vertical_pid.reset()
        self._forward_pid.reset()
        self._latest = None
        self._active = True
        print("[follow] started")

    def stop(self) -> None:
        self._active = False
        self._latest = None
        if not config.FOLLOW_DRY_RUN:
            self._mavlink.send_velocity_setpoint(0.0, 0.0, 0.0)
        print("[follow] stopped")

    def update(self) -> None:
        """Call at config.FOLLOW_UPDATE_INTERVAL_S from the main thread."""
        if not self._active or self._latest is None:
            return

        detection = self._latest
        vy = self._lateral_pid.update(-detection.x_m)                              # center laterally
        vz = self._vertical_pid.update(-detection.y_m)                             # center vertically
        vx = self._forward_pid.update(detection.z_m - config.FOLLOW_TARGET_DISTANCE_M)  # hold standoff distance

        if config.FOLLOW_DRY_RUN:
            print(
                f"[follow] DRY RUN setpoint vx={vx:.2f} vy={vy:.2f} vz={vz:.2f} "
                f"(Z={detection.z_m:.1f}m, target={config.FOLLOW_TARGET_DISTANCE_M}m)"
            )
        else:
            self._mavlink.send_velocity_setpoint(vx, vy, vz)
