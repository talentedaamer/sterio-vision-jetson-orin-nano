"""Loads the static camera -> flight-controller body frame mounting
transform (configs/camera_body_extrinsics.yaml).

PLACEHOLDER VALUES until physically measured/validated on the real rig --
see the comments in that YAML file. src/geolocation.py's camera_to_latlon()
depends entirely on this being right; get the mounting angles wrong and
every computed lat/lon is silently biased by a fixed (not noisy, so not
obviously wrong-looking) amount.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

DEFAULT_EXTRINSICS_PATH = "configs/camera_body_extrinsics.yaml"

# Fixed base rotation for a camera mounted level and forward-facing (zero
# mount_roll/pitch/yaw): camera z (forward, out of lens) -> body x
# (forward); camera x (right) -> body y (right); camera y (down) -> body z
# (down). The measured mount_roll/pitch/yaw on top of this describes any
# deviation from this "ideal" mounting (tilt, twist, etc).
_BASE_CAM_TO_BODY = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


@dataclass
class CameraBodyExtrinsics:
    R_cam2body: np.ndarray   # 3x3 -- rotates a camera-frame vector into body frame
    t_cam2body: np.ndarray   # 3 -- lever-arm, body-frame meters (camera optical center relative to body origin)

    def apply(self, xyz_cam: np.ndarray) -> np.ndarray:
        """Rotates + offsets a camera-frame [X, Y, Z] vector (OpenCV
        convention: x right, y down, z forward -- see Detection.x_m/y_m/z_m
        in src/distance.py) into body frame (x forward, y right, z down)."""
        return self.R_cam2body @ np.asarray(xyz_cam, dtype=float) + self.t_cam2body


def load(path: Optional[str] = None) -> CameraBodyExtrinsics:
    path = path or DEFAULT_EXTRINSICS_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Camera-body extrinsics file not found: {path}. Measure the "
            f"mounting angles (see this module's docstring and the YAML "
            f"file's comments) and fill them in."
        )
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    mount_rotation = Rotation.from_euler(
        "ZYX",
        [
            data.get("mount_yaw_deg", 0.0),
            data.get("mount_pitch_deg", 0.0),
            data.get("mount_roll_deg", 0.0),
        ],
        degrees=True,
    ).as_matrix()

    R_cam2body = mount_rotation @ _BASE_CAM_TO_BODY
    t_cam2body = np.array([
        data.get("lever_arm_x_m", 0.0),
        data.get("lever_arm_y_m", 0.0),
        data.get("lever_arm_z_m", 0.0),
    ])

    return CameraBodyExtrinsics(R_cam2body=R_cam2body, t_cam2body=t_cam2body)
