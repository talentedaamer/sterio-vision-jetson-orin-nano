"""Monocular distance estimation from a known object height + bbox pixel height.

This is the interim depth estimator for the initial dual-camera streaming
milestone. Now that the two IMX296 cameras are calibrated (see
src/calibration.py), add a disparity-based estimator and switch the call
site in probes.py over to it -- estimate_xyz()'s signature is the only
contract the rest of the pipeline depends on.
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from . import config


@dataclass
class Detection:
    source_id: int
    class_id: int
    label: str
    confidence: float
    left: float
    top: float
    width: float
    height: float
    x_m: float
    y_m: float
    z_m: float


def estimate_xyz(class_id: int, left: float, top: float, width: float, height: float,
                  frame_w: int, frame_h: int) -> tuple[float, float, float]:
    """Monocular X/Y/Z estimate in meters, camera-relative, from bbox pixel height."""
    if height <= 0:
        return 0.0, 0.0, 0.0

    real_h = config.KNOWN_HEIGHTS_M.get(class_id, 1.0)
    cx = left + width / 2.0
    cy = top + height / 2.0

    z = (real_h * config.FOCAL_LENGTH_PX) / height
    x = (cx - frame_w / 2.0) * z / config.FOCAL_LENGTH_PX
    y = (cy - frame_h / 2.0) * z / config.FOCAL_LENGTH_PX
    return x, y, z


def estimate_xyz_stereo(*_args, **_kwargs) -> tuple[float, float, float]:
    """Placeholder for post-calibration stereo disparity depth."""
    raise NotImplementedError(
        "Stereo depth requires camera calibration (intrinsics + baseline). "
        "estimate_xyz() (monocular) is used until calibration is done."
    )


class SmoothedDetection:
    """Averages X/Y/Z over a trailing window (config.DISPLAY_AVERAGE_WINDOW_S)
    and only changes its reported value once every
    config.DISPLAY_UPDATE_INTERVAL_S, instead of every frame -- reduces
    frame-to-frame jitter in the on-screen readout and reports a less noisy
    value than any single frame's raw estimate would be.

    Presentational only: FOLLOW's control loop and the debug plot use the
    raw, un-smoothed per-frame estimate directly (see src/probes.py) -- a
    control loop wants low latency, not a value that can lag up to a
    second behind. This class only feeds the on-screen text label.

    One instance per (source_id, class_id) -- see src/probes.py, which
    creates and reuses these across frames (there's no object tracker yet,
    so this is the same "identity by class, not by object" limitation
    FOLLOW already has -- see CLAUDE.md/README roadmap).
    """

    def __init__(self):
        self._samples: deque = deque()   # (timestamp, x, y, z)
        self._display: Optional[tuple[float, float, float]] = None
        self._last_update = 0.0

    def update(self, x: float, y: float, z: float, now: Optional[float] = None) -> tuple[float, float, float]:
        now = time.monotonic() if now is None else now
        self._samples.append((now, x, y, z))

        cutoff = now - config.DISPLAY_AVERAGE_WINDOW_S
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        if self._display is None or now - self._last_update >= config.DISPLAY_UPDATE_INTERVAL_S:
            n = len(self._samples)
            self._display = (
                sum(s[1] for s in self._samples) / n,
                sum(s[2] for s in self._samples) / n,
                sum(s[3] for s in self._samples) / n,
            )
            self._last_update = now

        return self._display
