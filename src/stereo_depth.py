"""Real stereo depth for a single detection, computed on demand.

Deliberately NOT a full-frame dense disparity map -- running OpenCV block
matching over the whole 1456x1088 frame, for both cameras, every frame at
60fps would be genuine, sustained CPU load on an ARM CPU, which this
project has otherwise been carefully built to avoid (see CLAUDE.md "GPU
utilization model"). Instead, disparity is computed only for the small
region around ONE detection's bbox, ON DEMAND (e.g. from
src/geolocation.py, once per FOLLOW-locked target or ISR log tick) --
NOT wired into src/probes.py's always-on per-frame loop.

Three stages, kept as separate functions so each is independently
testable/reasoned about:
  1. extract_luma_plane()   -- pulls one camera's raw pixels out of the
     DeepStream NVMM buffer into a CPU-accessible numpy grayscale array.
     UNVERIFIED on real hardware: pyds.get_nvds_buf_surface() is used in
     every official DeepStream Python sample against RGBA buffers (after
     an explicit nvvideoconvert); using it against this pipeline's NV12
     buffers (src/pipeline.py's PGIE src pad, before any conversion) is
     the untested part of this whole module. If it raises/returns an
     unexpected shape on-device, the fix is either converting to RGBA
     first (cheap on GPU via nvvideoconvert, if a spare tee branch is
     added) or reading the NV12 Y-plane at a different offset/stride than
     assumed here.
  2. rectify_and_locate() -- warps a full frame through the calibration's
     rectification maps (src/calibration.py) and maps a detection's bbox
     (in the ORIGINAL, distorted image) into rectified-image coordinates
     via cv2.undistortPoints(..., R=R1, P=P1) -- NOT a plain coordinate
     scale/offset, since undistortion is a nonlinear per-pixel warp.
  3. roi_disparity() -- runs cv2.StereoSGBM on a crop just wide enough to
     cover the bbox + the configured max disparity search range, and
     returns the median disparity over the bbox's own pixels (robust to
     the occasional bad/occluded match a single-pixel disparity lookup
     would be vulnerable to).

Only meaningful for source_id==0 (left camera) detections, matching
src/calibration.py's left-camera-as-reference convention (T/P2 encode the
right camera's offset relative to the left) -- source_id==1 (right
camera) detections are not supported here and should keep using the
monocular estimate (src/distance.py estimate_xyz()) instead.
"""
from typing import Optional

import cv2
import numpy as np

from . import config
from .calibration import StereoCalibration
from .distance import estimate_xyz_stereo


def extract_luma_plane(nv12_frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Given the raw NV12 buffer as a numpy array (see this module's
    docstring re: pyds.get_nvds_buf_surface, UNVERIFIED), returns just the
    Y (luma) plane as a (height, width) grayscale array -- NV12's luma
    plane occupies the first `height` full-width rows, followed by the
    half-height interleaved UV plane, which stereo matching doesn't need.
    """
    if nv12_frame.shape[0] < height or nv12_frame.shape[1] < width:
        raise ValueError(
            f"NV12 buffer {nv12_frame.shape} smaller than expected {(height, width)}"
        )
    return np.ascontiguousarray(nv12_frame[:height, :width])


def rectify_frame(luma_plane: np.ndarray, rect_map: tuple) -> np.ndarray:
    """Applies one camera's undistort+rectify remap (from
    StereoCalibration.rectify_maps()) to a full grayscale frame."""
    map_x, map_y = rect_map
    return cv2.remap(luma_plane, map_x, map_y, cv2.INTER_LINEAR)


def rectify_bbox_left(bbox: tuple, camera_matrix_left: np.ndarray, dist_coeffs_left: np.ndarray,
                       R1: np.ndarray, P1: np.ndarray) -> tuple:
    """Maps a detection bbox (left, top, width, height), in the ORIGINAL
    (distorted, unrectified) left-camera image, into the rectified image's
    pixel coordinates. Returns (left, top, width, height) in rectified
    space, still axis-aligned (a tight approximation -- rectification can
    slightly rotate/skew a box's true footprint, close enough for a
    disparity search window)."""
    left, top, width, height = bbox
    corners = np.array([
        [left, top], [left + width, top],
        [left, top + height], [left + width, top + height],
    ], dtype=np.float32).reshape(-1, 1, 2)

    rectified = cv2.undistortPoints(corners, camera_matrix_left, dist_coeffs_left, R=R1, P=P1)
    xs = rectified[:, 0, 0]
    ys = rectified[:, 0, 1]
    r_left, r_top = float(xs.min()), float(ys.min())
    r_width, r_height = float(xs.max() - r_left), float(ys.max() - r_top)
    return r_left, r_top, r_width, r_height


def roi_disparity(left_rect: np.ndarray, right_rect: np.ndarray, bbox_rect: tuple,
                   max_disparity_px: Optional[int] = None) -> Optional[float]:
    """Median disparity (px) over a detection's bbox, given both cameras'
    ALREADY-RECTIFIED grayscale frames and the bbox in rectified-image
    coordinates (see rectify_bbox_left()). Returns None if the search
    window falls outside the image or no valid (positive) disparity is
    found anywhere in the bbox -- callers should fall back to the
    monocular estimate in that case, not treat it as fatal.
    """
    max_disparity_px = max_disparity_px or config.STEREO_MAX_DISPARITY_PX
    img_h, img_w = left_rect.shape[:2]

    left, top, width, height = (int(round(v)) for v in bbox_rect)
    top = max(0, min(top, img_h - 1))
    height = max(1, min(height, img_h - top))
    if left < 0 or width <= 0 or left + width > img_w:
        return None

    num_disparities = max(16, (max_disparity_px // 16) * 16)
    crop_left = left - num_disparities
    if crop_left < 0:
        # Shrink the search range rather than reading out of bounds --
        # objects near the left edge of the frame simply get a reduced
        # max-depth search window.
        num_disparities = max(16, ((num_disparities + crop_left) // 16) * 16)
        crop_left = left - num_disparities
    crop_width = num_disparities + width
    if crop_left < 0 or crop_left + crop_width > img_w:
        return None

    left_crop = left_rect[top:top + height, crop_left:crop_left + crop_width]
    right_crop = right_rect[top:top + height, crop_left:crop_left + crop_width]
    if left_crop.shape != right_crop.shape or left_crop.size == 0:
        return None

    block_size = config.STEREO_BLOCK_SIZE
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
    )
    disparity_fixed = matcher.compute(left_crop, right_crop)
    disparity_map = disparity_fixed.astype(np.float32) / 16.0  # StereoSGBM returns disparity * 16

    # The detection's own pixels are the rightmost `width` columns of the
    # crop (the crop was built by extending leftward from the bbox's left
    # edge by the search range).
    object_disparities = disparity_map[:, num_disparities:num_disparities + width]
    valid = object_disparities[object_disparities > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def estimate_stereo_xyz(left_nv12: np.ndarray, right_nv12: np.ndarray, bbox_left: tuple,
                         calib: StereoCalibration) -> Optional[tuple]:
    """End-to-end stereo depth for one left-camera detection: extracts +
    rectifies both cameras' luma planes, maps the bbox into rectified
    coordinates, searches for disparity, and converts to camera-relative
    X/Y/Z. Returns None (caller should fall back to estimate_xyz(), the
    monocular estimate) if any stage fails to produce a valid result --
    this is the expected outcome for objects closer than
    config.STEREO_MAX_DISPARITY_PX's minimum depth, near frame edges, or
    over low-texture/occluded regions, not a bug.

    left_nv12/right_nv12: raw NV12 buffers as returned by
    pyds.get_nvds_buf_surface() for source_id 0 and 1 respectively (see
    extract_luma_plane()'s docstring re: this being unverified on real
    hardware). bbox_left: (left, top, width, height) in the ORIGINAL
    (distorted) left-camera image, i.e. straight from obj_meta.rect_params.
    """
    try:
        left_luma = extract_luma_plane(left_nv12, calib.image_width, calib.image_height)
        right_luma = extract_luma_plane(right_nv12, calib.image_width, calib.image_height)
    except ValueError:
        return None

    left_map, right_map = calib.rectify_maps()
    left_rect = rectify_frame(left_luma, left_map)
    right_rect = rectify_frame(right_luma, right_map)

    bbox_rect = rectify_bbox_left(bbox_left, calib.camera_matrix_left, calib.dist_coeffs_left, calib.R1, calib.P1)
    disparity_px = roi_disparity(left_rect, right_rect, bbox_rect)
    if disparity_px is None:
        return None

    cx, cy, fx = calib.P1[0, 2], calib.P1[1, 2], calib.P1[0, 0]
    _, _, r_width, r_height = bbox_rect
    r_left, r_top = bbox_rect[0], bbox_rect[1]
    try:
        return estimate_xyz_stereo(r_left, r_top, r_width, r_height, disparity_px, cx, cy, fx, calib.baseline_m)
    except ValueError:
        return None
