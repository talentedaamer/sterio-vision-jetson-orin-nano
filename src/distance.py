"""Monocular distance estimation from a known object height + bbox pixel height.

This is the interim depth estimator for the initial dual-camera streaming
milestone. Once the two IMX296 cameras are calibrated (intrinsics + stereo
extrinsics/baseline), add a disparity-based estimator and switch the call
site in probes.py over to it -- estimate_xyz()'s signature is the only
contract the rest of the pipeline depends on.
"""
from dataclasses import dataclass

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
