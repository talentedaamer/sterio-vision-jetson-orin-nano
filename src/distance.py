"""Distance estimation: monocular (default, always-on) and stereo
(on-demand, real disparity -- see estimate_xyz_stereo() and
src/stereo_depth.py).

estimate_xyz() (monocular, from a known object height + bbox pixel height)
remains the always-on estimator feeding src/probes.py's per-frame overlay,
FOLLOW, and the debug plot -- cheap, no raw pixel access needed, and
accurate enough for those uses. estimate_xyz_stereo() is real disparity-
based depth, but requires rectified pixel data from BOTH cameras (see
src/stereo_depth.py) and is comparatively expensive, so callers invoke it
on demand for one detection at a time (e.g. src/geolocation.py) rather
than every detection every frame.
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


def estimate_xyz_stereo(left: float, top: float, width: float, height: float,
                         disparity_px: float, cx: float, cy: float,
                         focal_length_px: float, baseline_m: float) -> tuple[float, float, float]:
    """X/Y/Z in meters, camera-relative, from a rectified disparity value
    (see src/stereo_depth.py for how disparity_px is obtained -- this
    function is pure math, no image processing).

    Unlike estimate_xyz() (monocular), the bbox here must already be in
    RECTIFIED image coordinates (src/stereo_depth.py's rectify_bbox_left()),
    and cx/cy/focal_length_px must be the RECTIFIED principal point/focal
    length (P1[0,2], P1[1,2], P1[0,0] from src/calibration.py's
    StereoCalibration), not the raw frame center/config.FOCAL_LENGTH_PX --
    rectification can shift the principal point away from the image center.
    """
    if disparity_px <= 0:
        raise ValueError("disparity_px must be positive for a valid stereo depth estimate")

    z = (focal_length_px * baseline_m) / disparity_px
    u = left + width / 2.0
    v = top + height / 2.0
    x = (u - cx) * z / focal_length_px
    y = (v - cy) * z / focal_length_px
    return x, y, z


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
