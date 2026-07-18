"""OBSTACLE_DISTANCE construction + streaming for MISSION_MODE=="AVOID".

NOT a steering controller -- unlike FOLLOW (src/pid.py's
ObjectFollowController, which computes its own velocity setpoints and
sends SET_POSITION_TARGET_LOCAL_NED), this module only builds and sends
MAVLink OBSTACLE_DISTANCE messages from per-bin depth readings
(src/obstacle_depth.py). ArduPilot's own OA_TYPE (BendyRuler/Dijkstra's)
decides whether/how to bend the flight path -- see config.py's "AVOID"
block and CLAUDE.md for the full reasoning and FC-side setup
(PRX1_TYPE, OA_TYPE).

BinDistanceSmoother is the TEMPORAL stage of the RealSense-style filter
chain described in src/obstacle_depth.py -- applied here (per bin, across
cycles) rather than there (per pixel, within one cycle), since it's the
only stage that needs state to persist across calls.

build_obstacle_distance() assumes the camera's optical axis is boresight-
aligned with the vehicle's forward direction -- angle_offset is derived
purely from the rig's own horizontal FOV, with no camera->body mounting
correction applied. Unlike src/extrinsics.py's geolocation chain, there is
no measured mounting-angle correction here yet; if the rig is ever mounted
with a real yaw offset, either add one here or accept a proportional bias
in which angular zone ArduPilot thinks each reading came from.
"""
import time
from typing import Optional

import numpy as np

from . import config
from .calibration import StereoCalibration
from .mavlink_link import MavlinkLink

_MAV_DISTANCE_UNKNOWN_CM = 65535   # MAVLink's "no obstacle/unknown" sentinel
_OBSTACLE_DISTANCE_ARRAY_LEN = 72  # fixed length of the MAVLink OBSTACLE_DISTANCE message


class BinDistanceSmoother:
    """Per-bin exponential moving average across cycles -- the TEMPORAL
    stage (see module docstring). One instance per ObstacleAvoidance (i.e.
    per mission), not per bin -- update() takes the whole
    (bin_distances_m, bin_valid_mask) pair each cycle.

    A bin currently marked invalid reports no-data this cycle (never a
    stale value), even though its internal EMA state is preserved -- so
    smoothing resumes immediately, without a fresh warm-up, once real
    readings return for that bin.
    """

    def __init__(self, num_bins: int = config.AVOID_NUM_BINS):
        self._smoothed: list = [None] * num_bins  # Optional[float] per bin

    def update(self, bin_distances_m: list, bin_valid_mask: list) -> list:
        """Returns smoothed distances, same length as the input -- None
        for any bin currently invalid."""
        alpha = config.AVOID_TEMPORAL_EMA_ALPHA
        for i, (distance, is_valid) in enumerate(zip(bin_distances_m, bin_valid_mask)):
            if not is_valid:
                continue
            if self._smoothed[i] is None:
                self._smoothed[i] = distance
            else:
                self._smoothed[i] = alpha * distance + (1 - alpha) * self._smoothed[i]

        return [
            self._smoothed[i] if bin_valid_mask[i] else None
            for i in range(len(bin_distances_m))
        ]


def _horizontal_fov_deg(calib: StereoCalibration) -> float:
    """Derived from the rig's real rectified calibration
    (2*atan(image_width / (2*fx))), not hardcoded -- same preference this
    project already applied when it sourced FOCAL_LENGTH_PX/
    STEREO_BASELINE_M from real chessboard calibration instead of a
    guess."""
    fx = calib.P1[0, 0]
    return float(np.degrees(2.0 * np.arctan(calib.image_width / (2.0 * fx))))


def build_obstacle_distance(bin_distances_m: list, bin_valid_mask: list,
                             calib: StereoCalibration) -> dict:
    """Pure: maps bin readings to the MAVLink OBSTACLE_DISTANCE message's
    fixed-length distances[72] array. Bins are contiguous starting at
    sector 0 (leftmost bin), so angle_offset/increment_f alone locate them
    -- sectors beyond len(bin_distances_m) stay "no data"
    (_MAV_DISTANCE_UNKNOWN_CM); no need to compute arbitrary global sector
    indices for a full 360deg wrap.

    Returns a dict of keyword arguments matching
    MavlinkLink.send_obstacle_distance()'s signature.
    """
    num_bins = len(bin_distances_m)
    hfov_deg = _horizontal_fov_deg(calib)
    sector_width_deg = hfov_deg / num_bins
    angle_offset_deg = -hfov_deg / 2.0 + sector_width_deg / 2.0

    min_distance_cm = int(round(config.AVOID_MIN_VALID_DEPTH_M * 100))
    max_distance_cm = int(round(config.AVOID_MAX_VALID_DEPTH_M * 100))

    distances_cm = [_MAV_DISTANCE_UNKNOWN_CM] * _OBSTACLE_DISTANCE_ARRAY_LEN
    for i in range(num_bins):
        if not bin_valid_mask[i]:
            continue
        distance_cm = int(round(bin_distances_m[i] * 100))
        distances_cm[i] = max(min_distance_cm, min(max_distance_cm, distance_cm))

    return {
        "distances_cm": distances_cm,
        "increment_f_deg": sector_width_deg,
        "min_distance_cm": min_distance_cm,
        "max_distance_cm": max_distance_cm,
        "angle_offset_deg": angle_offset_deg,
    }


class ObstacleAvoidance:
    """Streams MAVLink OBSTACLE_DISTANCE built from the latest bin
    reading -- mirrors ObjectFollowController's producer/consumer split
    (src/pid.py): add_bin_distances() is the cheap/thread-safe subscribe
    callback for src.probes.register_obstacle_listener(), called from the
    GStreamer streaming thread; update() runs the actual smoothing +
    MAVLink send, and must be called periodically from the main thread
    (see src/mission.py + main.py), never the streaming thread.
    """

    def __init__(self, mavlink: MavlinkLink, calib: StereoCalibration):
        self._mavlink = mavlink
        self._calib = calib
        self._smoother = BinDistanceSmoother()
        self._latest: Optional[tuple] = None  # (bin_distances_m, bin_valid_mask)
        self._last_update_time = 0.0

    @property
    def streaming(self) -> bool:
        """True once at least one obstacle reading has been processed --
        drives Mission.status_text()'s STREAMING/NO DATA YET line."""
        return self._last_update_time > 0.0

    def add_bin_distances(self, bin_distances_m: list, bin_valid_mask: list) -> None:
        self._latest = (bin_distances_m, bin_valid_mask)

    def update(self) -> None:
        """Call at config.AVOID_UPDATE_INTERVAL_S from the main thread."""
        if self._latest is None:
            return

        bin_distances_m, bin_valid_mask = self._latest
        smoothed = self._smoother.update(bin_distances_m, bin_valid_mask)
        smoothed_valid = [v is not None for v in smoothed]
        smoothed_distances = [v if v is not None else 0.0 for v in smoothed]

        message = build_obstacle_distance(smoothed_distances, smoothed_valid, self._calib)
        self._last_update_time = time.monotonic()

        if config.AVOID_DRY_RUN:
            readable = [
                f"{d:.1f}m" if valid else "--"
                for d, valid in zip(smoothed_distances, smoothed_valid)
            ]
            print(f"[avoid] DRY RUN obstacle_distance bins={readable}")
        else:
            self._mavlink.send_obstacle_distance(**message)
