"""Camera detection -> absolute lat/lon/altitude.

Chains: camera frame -> body frame (src/extrinsics.py's fixed mounting
transform) -> local NED frame (the drone's own attitude, from MAVLink) ->
geodetic lat/lon/alt (flat-earth/local-tangent-plane approximation via
pymap3d.ned2geodetic, anchored at the drone's own current GPS fix).

This is a single-frame, single-detection point estimate -- no multi-frame
fusion/smoothing across detections of the same object over time (that's
the explicitly deferred step 8 in the project's build order; worth doing
once an object tracker exists so "the same object" is a well-defined
concept across frames, which it currently isn't -- see CLAUDE.md/README
roadmap).

Accuracy of the result is bounded by, in roughly descending order of
likely impact:
  - src/extrinsics.py's mounting angles, if still at their placeholder
    (zero) values -- validate those FIRST (see that module's docstring)
    before trusting anything from here.
  - GPS horizontal/vertical accuracy itself (a consumer GPS fix is
    routinely several meters off -- this pipeline cannot improve on that,
    it only adds the camera's relative offset on top of wherever the GPS
    fix says the drone is).
  - The distance (Z) estimate feeding this -- stereo (src/stereo_depth.py)
    if available, monocular (src/distance.py estimate_xyz(), which itself
    depends on config.KNOWN_HEIGHTS_M being right for the detected class)
    otherwise.
  - Attitude/GPS timestamp mismatch relative to the camera frame's capture
    time -- reduced but not eliminated by
    MavlinkLink.get_interpolated_attitude(), see this module's
    camera_to_latlon().
"""
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pymap3d
from scipy.spatial.transform import Rotation

from .distance import Detection
from .extrinsics import CameraBodyExtrinsics
from .mavlink_link import InterpolatedAttitude, Telemetry


@dataclass
class GeoPosition:
    latitude_deg: float
    longitude_deg: float
    altitude_msl_m: float
    timestamp: float
    depth_source: str   # "stereo" or "monocular" -- which Z estimate this was computed from


def camera_to_latlon(
    detection: Detection,
    telemetry: Telemetry,
    extrinsics: CameraBodyExtrinsics,
    xyz_cam_override: Optional[tuple] = None,
    attitude_override: Optional[InterpolatedAttitude] = None,
) -> Optional[GeoPosition]:
    """Computes the detected object's absolute lat/lon/altitude.

    xyz_cam_override: pass the (x, y, z) tuple from
    src/stereo_depth.py's estimate_stereo_xyz() when real stereo depth was
    successfully computed for this detection; omit (or pass None, e.g.
    when estimate_stereo_xyz() returned None) to fall back to the
    detection's own monocular x_m/y_m/z_m -- the same values already
    driving the on-screen label/FOLLOW/debug plot.

    attitude_override: pass the result of
    MavlinkLink.get_interpolated_attitude(frame_capture_time) to rotate
    using attitude aligned to the camera frame's actual capture time
    instead of telemetry.imu's roll/pitch/yaw (whatever ATTITUDE message
    happened to arrive most recently relative to whenever
    get_telemetry() was called).

    Returns None (rather than a wrong-looking answer) when telemetry.gps
    is None / has_fix is False -- callers must check GPS fix status via
    telemetry.gps before calling this if they want to distinguish
    "no fix yet" from other failure modes; this function does not.
    """
    if telemetry.gps is None or not telemetry.gps.has_fix:
        return None

    if xyz_cam_override is not None:
        xyz_cam = np.array(xyz_cam_override, dtype=float)
        depth_source = "stereo"
    else:
        xyz_cam = np.array([detection.x_m, detection.y_m, detection.z_m], dtype=float)
        depth_source = "monocular"

    xyz_body = extrinsics.apply(xyz_cam)

    attitude = attitude_override if attitude_override is not None else telemetry.imu
    ned = _rotate_body_to_ned(xyz_body, attitude.roll_deg, attitude.pitch_deg, attitude.yaw_deg)

    lat, lon, alt = pymap3d.ned2geodetic(
        ned[0], ned[1], ned[2],
        telemetry.gps.latitude_deg, telemetry.gps.longitude_deg, telemetry.gps.altitude_msl_m,
    )

    return GeoPosition(
        latitude_deg=lat, longitude_deg=lon, altitude_msl_m=alt,
        timestamp=time.time(), depth_source=depth_source,
    )


def _rotate_body_to_ned(xyz_body: np.ndarray, roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Rotates a body-frame vector (x forward, y right, z down) into local
    NED (north, east, down) using the standard aerospace ZYX
    (yaw, then pitch, then roll) Euler sequence."""
    rotation = Rotation.from_euler("ZYX", [yaw_deg, pitch_deg, roll_deg], degrees=True)
    return rotation.apply(xyz_body)
