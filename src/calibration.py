"""Loads the chessboard-derived stereo calibration
(configs/stereo_calibration.yaml) produced by cv2.stereoCalibrate +
cv2.stereoRectify.

That file holds the compact parameters only: image size, both cameras'
intrinsics + distortion, R/T/E/F (stereo extrinsics), R1/R2/P1/P2/Q
(rectification), baseline, focal length, and the calibration's RMS
reprojection error. It does NOT include the precomputed undistort/rectify
remap tables (map_left_1/2, map_right_1/2) that the original, much larger
(~512KB) calibration output also contains -- those are 100% derivable at
runtime from the compact parameters via cv2.initUndistortRectifyMap(),
which StereoCalibration.rectify_maps() below does lazily, so there's no
need to store or commit them. See CLAUDE.md "Stereo calibration" for the
full numbers and how this file was produced.

This module only loads and exposes calibration data -- it does not itself
compute disparity/depth or feed ORB-SLAM3. Those are separate, not-yet-built
steps (see CLAUDE.md roadmap); config.FOCAL_LENGTH_PX/STEREO_BASELINE_M
were updated from this same calibration for the existing monocular
estimator in the meantime.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config

DEFAULT_CALIBRATION_PATH = "configs/stereo_calibration.yaml"


@dataclass
class StereoCalibration:
    image_width: int
    image_height: int
    camera_matrix_left: np.ndarray
    dist_coeffs_left: np.ndarray
    camera_matrix_right: np.ndarray
    dist_coeffs_right: np.ndarray
    R: np.ndarray               # rotation, left camera -> right camera
    T: np.ndarray                # translation, left camera -> right camera (m)
    R1: np.ndarray               # rectification rotation, left
    R2: np.ndarray               # rectification rotation, right
    P1: np.ndarray               # rectified projection matrix, left
    P2: np.ndarray               # rectified projection matrix, right
    Q: np.ndarray                 # disparity-to-depth reprojection matrix
    baseline_m: float
    focal_length_px: float
    stereo_rms_error_px: float

    def rectify_maps(self):
        """Computes (not cached to disk -- cheap, ms-scale) the undistort +
        rectify remap tables for both cameras via cv2.initUndistortRectifyMap.
        Returns ((map_left_x, map_left_y), (map_right_x, map_right_y)),
        each usable directly with cv2.remap()."""
        size = (self.image_width, self.image_height)
        left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_coeffs_left, self.R1, self.P1, size, cv2.CV_32FC1
        )
        right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_coeffs_right, self.R2, self.P2, size, cv2.CV_32FC1
        )
        return left, right


def load(path: Optional[str] = None) -> StereoCalibration:
    path = path or DEFAULT_CALIBRATION_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Stereo calibration file not found: {path}. Run the chessboard "
            f"calibration script and copy its output here (compact "
            f"parameters only -- see this module's docstring)."
        )

    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    try:
        def mat(name):
            return fs.getNode(name).mat()

        return StereoCalibration(
            image_width=int(fs.getNode("image_width").real()),
            image_height=int(fs.getNode("image_height").real()),
            camera_matrix_left=mat("camera_matrix_left"),
            dist_coeffs_left=mat("dist_coeffs_left"),
            camera_matrix_right=mat("camera_matrix_right"),
            dist_coeffs_right=mat("dist_coeffs_right"),
            R=mat("R"),
            T=mat("T"),
            R1=mat("R1"),
            R2=mat("R2"),
            P1=mat("P1"),
            P2=mat("P2"),
            Q=mat("Q"),
            baseline_m=float(fs.getNode("baseline_m").real()),
            focal_length_px=float(fs.getNode("focal_length_px").real()),
            stereo_rms_error_px=float(fs.getNode("stereo_rms_error_px").real()),
        )
    finally:
        fs.release()
