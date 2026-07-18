"""Band-wide stereo depth -> per-bin obstacle distances, for MISSION_MODE=="AVOID".

Unlike src/stereo_depth.py (per-YOLO-detection, small ROI, on demand), this
module runs ONE cv2.StereoSGBM pass over the full width of a center
vertical band (config.AVOID_BIN_HEIGHT_FRACTION) of the rectified frame,
then reduces that to config.AVOID_NUM_BINS scalar readings -- bin AFTER
computing depth, not one matcher call per bin (cheaper, less noisy; SGBM's
aggregation benefits from wider context per call).

Maps RealSense's 4-stage depth post-processing filter chain (Decimation /
Spatial / Temporal / Hole-filling) onto the coarser shape this module
actually needs -- config.AVOID_NUM_BINS (5) scalars per cycle, not a full
depth image:
  - Decimation: downsample the band crop by config.AVOID_DECIMATION_FACTOR
    before matching (bounds CPU cost). See _band_disparity()'s decimation
    math -- disparity is rescaled back by the same factor, NOT
    config.FOCAL_LENGTH_PX, before the depth formula.
  - Spatial: edge-preserving smoothing over the depth array before binning,
    via cv2.bilateralFilter. cv2.ximgproc's WLS disparity filter (the
    closer RealSense analog) needs a second right-based matcher pass and
    isn't part of this project's resolved OpenCV build (confirmed: no
    cv2-contrib on-device -- same evaluate-and-reject precedent as
    CLAUDE.md's Open3D note). Bilateral is still edge-preserving (won't
    blur across a real depth discontinuity the way a box/Gaussian blur
    would) at a fraction of the complexity.
  - Hole-filling: see compute_bin_distances()'s docstring below.
  - Temporal (per-bin EMA across cycles) deliberately lives in
    src/avoidance.py instead of here -- it needs state that persists
    across calls to this module, at the bin level, downstream of this
    module's per-call output.

Edge effect: the leftmost ~config.STEREO_MAX_DISPARITY_PX columns of any
single disparity pass have no valid match -- there's nothing further left
in the right image to search against. True of any real stereo system, not
a bug, and not something decimation/filtering fixes. Handled explicitly in
compute_bin_distances() by marking that strip as no-data rather than
fabricating a reading for a zone the rig structurally can't see.

UNVERIFIED on real hardware, same caveat as src/stereo_depth.py: this is
the first time pyds.get_nvds_buf_surface() and a live cv2.StereoSGBM pass
run inside the always-on pipeline (previously only reachable via the
never-invoked on-demand path from src/geolocation.py). Benchmark in
isolation before trusting this at config.AVOID_UPDATE_INTERVAL_S -- see
CLAUDE.md "Prerequisite validation".

Only meaningful with source_id==0 as the left/reference camera, matching
src/calibration.py's convention -- same as src/stereo_depth.py.
"""
from typing import Optional

import cv2
import numpy as np

from . import config
from .calibration import StereoCalibration
from .stereo_depth import extract_luma_plane, rectify_frame

# calib.rectify_maps() recomputes cv2.initUndistortRectifyMap() on every
# call (~13ms measured on-device) -- fine for stereo_depth.py's occasional
# on-demand per-detection calls, but this module runs every
# config.AVOID_UPDATE_INTERVAL_S, so the maps are cached here the first
# time estimate_bin_distances() sees a given StereoCalibration instance
# (there is only ever one for the process's lifetime -- see src/probes.py).
_rect_maps_cache: Optional[tuple] = None
_rect_maps_cache_calib_id: Optional[int] = None


def _cached_rectify_maps(calib: StereoCalibration) -> tuple:
    global _rect_maps_cache, _rect_maps_cache_calib_id
    if _rect_maps_cache is None or _rect_maps_cache_calib_id != id(calib):
        _rect_maps_cache = calib.rectify_maps()
        _rect_maps_cache_calib_id = id(calib)
    return _rect_maps_cache


def _band_rows(image_height: int) -> tuple[int, int]:
    """Row range (row_lo, row_hi) of the center vertical band -- same
    center-band convention as the D415 reference project's
    obstacle_detection.compute_bin_distances() (keeps floor/ceiling
    clutter out of the read)."""
    band_half = int(image_height * config.AVOID_BIN_HEIGHT_FRACTION / 2)
    row_lo = image_height // 2 - band_half
    row_hi = image_height // 2 + band_half
    return row_lo, row_hi


def _band_disparity(left_band: np.ndarray, right_band: np.ndarray) -> tuple[np.ndarray, int]:
    """ONE cv2.StereoSGBM pass over the full width of the band (not
    per-bin -- see module docstring). Returns (disparity, edge_width_px):
    disparity is a float32 map, same shape as the input band, in
    FULL-RESOLUTION pixel units (already rescaled for
    config.AVOID_DECIMATION_FACTOR); edge_width_px is how many leftmost
    columns are structurally unable to produce a valid match (the search
    range), for compute_bin_distances() to treat as no-data rather than a
    hole to fill.
    """
    decimation = max(1, config.AVOID_DECIMATION_FACTOR)
    h, w = left_band.shape[:2]

    if decimation > 1:
        size = (max(1, w // decimation), max(1, h // decimation))
        left_small = cv2.resize(left_band, size, interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right_band, size, interpolation=cv2.INTER_AREA)
    else:
        left_small, right_small = left_band, right_band

    block_size = config.STEREO_BLOCK_SIZE
    num_disparities = max(16, (config.STEREO_MAX_DISPARITY_PX // decimation // 16) * 16)
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
    disparity_fixed = matcher.compute(left_small, right_small)
    disparity_small = disparity_fixed.astype(np.float32) / 16.0  # StereoSGBM returns disparity * 16

    if decimation > 1:
        # Upsample spatially (nearest -- the immediately-following spatial
        # filter stage smooths any resulting blockiness) and rescale the
        # disparity VALUES back to full-resolution pixel units. These are
        # two separate corrections: resizing the array realigns it with
        # full-resolution column indices; multiplying by `decimation`
        # converts a 1px shift in downsampled space back to
        # `decimation`px in the original image.
        disparity = cv2.resize(disparity_small, (w, h), interpolation=cv2.INTER_NEAREST) * decimation
    else:
        disparity = disparity_small

    edge_width_px = num_disparities * decimation
    return disparity, edge_width_px


def _apply_spatial_filter(depth_m: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SPATIAL stage -- see module docstring for why bilateral instead of
    WLS. No-op (returns inputs unchanged) if config.AVOID_SPATIAL_FILTER_ENABLED
    is False."""
    if not config.AVOID_SPATIAL_FILTER_ENABLED:
        return depth_m, valid

    depth_filled = np.where(valid, depth_m, 0.0).astype(np.float32)
    smoothed = cv2.bilateralFilter(depth_filled, d=9, sigmaColor=0.5, sigmaSpace=9)
    # bilateralFilter has no concept of "invalid" pixels -- re-apply the
    # original mask afterward so a hole doesn't get smeared into a
    # fabricated small-but-nonzero value.
    smoothed[~valid] = 0.0
    return smoothed, valid


def _bin_min_distances(depth_m: np.ndarray, valid: np.ndarray, image_width: int) -> tuple[list, list]:
    """Splits depth_m/valid (band_height x image_width) into
    config.AVOID_NUM_BINS columns, left to right, and returns
    (raw_distances, valid_fractions) -- raw_distances[i] is the minimum
    valid depth (meters) in that bin's columns, or None if the bin has no
    valid pixels at all; valid_fractions[i] is the fraction of that bin's
    pixels that were valid (input to the hole-filling decision below)."""
    bin_edges = np.linspace(0, image_width, config.AVOID_NUM_BINS + 1, dtype=int)
    raw_distances: list = []
    valid_fractions: list = []
    for i in range(config.AVOID_NUM_BINS):
        col_lo, col_hi = bin_edges[i], bin_edges[i + 1]
        bin_valid = valid[:, col_lo:col_hi]
        bin_depth = depth_m[:, col_lo:col_hi]
        fraction = float(np.mean(bin_valid)) if bin_valid.size else 0.0
        valid_fractions.append(fraction)
        raw_distances.append(float(bin_depth[bin_valid].min()) if fraction > 0 else None)
    return raw_distances, valid_fractions


def compute_bin_distances(raw_distances: list, valid_fractions: list,
                           edge_width_px: int, image_width: int) -> tuple[list, list]:
    """HOLE-FILLING stage: turns per-bin raw readings into
    (bin_distances_m, bin_valid_mask), both length config.AVOID_NUM_BINS.

    A bin is trusted as-is when its valid-pixel fraction is >=
    config.AVOID_BIN_MIN_VALID_FRACTION. Otherwise:
      - if the bin falls (even partially) within the deterministic
        left-edge search-range strip (edge_width_px, see _band_disparity),
        it's marked no-data (bin_valid_mask[i]=False) rather than filled --
        that gap is structural, not noise, and filling it would fabricate
        a reading for a zone the rig can't actually see.
      - otherwise (occlusion, low texture elsewhere in the band -- a
        transient/sparse gap), it's filled from the average of its
        trusted immediate neighbors, if any; marked no-data if neither
        neighbor is trusted either.

    bin_distances_m[i] is only meaningful where bin_valid_mask[i] is True --
    callers (src/avoidance.py) must encode the rest as "no data"
    (MAVLink's UINT16_MAX convention), not a fabricated distance.
    """
    num_bins = len(raw_distances)
    bin_edges = np.linspace(0, image_width, num_bins + 1, dtype=int)
    trusted = [f >= config.AVOID_BIN_MIN_VALID_FRACTION for f in valid_fractions]
    is_edge_bin = [bin_edges[i] < edge_width_px for i in range(num_bins)]

    bin_distances_m = [0.0] * num_bins
    bin_valid_mask = [False] * num_bins

    for i in range(num_bins):
        if trusted[i]:
            bin_distances_m[i] = raw_distances[i]
            bin_valid_mask[i] = True
            continue
        if is_edge_bin[i]:
            continue  # structurally uncovered -- no-data, not filled
        neighbor_values = [
            raw_distances[j] for j in (i - 1, i + 1)
            if 0 <= j < num_bins and trusted[j]
        ]
        if neighbor_values:
            bin_distances_m[i] = sum(neighbor_values) / len(neighbor_values)
            bin_valid_mask[i] = True
        # else: no trusted neighbor either -- leave as no-data.

    return bin_distances_m, bin_valid_mask


def estimate_bin_distances(left_nv12: np.ndarray, right_nv12: np.ndarray,
                            calib: StereoCalibration) -> tuple[list, list]:
    """End-to-end: both cameras' raw NV12 buffers -> (bin_distances_m,
    bin_valid_mask), each length config.AVOID_NUM_BINS. Entry point for
    src/probes.py.

    Returns all-invalid ([0.0]*N, [False]*N) if buffer extraction fails
    outright (e.g. an unexpected shape) -- callers should skip sending an
    OBSTACLE_DISTANCE update that cycle, not treat it as fatal (same
    "expected, not a bug" posture as src/stereo_depth.py's per-detection
    failures).
    """
    num_bins = config.AVOID_NUM_BINS
    try:
        left_luma = extract_luma_plane(left_nv12, calib.image_width, calib.image_height)
        right_luma = extract_luma_plane(right_nv12, calib.image_width, calib.image_height)
    except ValueError:
        return [0.0] * num_bins, [False] * num_bins

    left_map, right_map = _cached_rectify_maps(calib)
    left_rect = rectify_frame(left_luma, left_map)
    right_rect = rectify_frame(right_luma, right_map)

    row_lo, row_hi = _band_rows(calib.image_height)
    left_band = left_rect[row_lo:row_hi, :]
    right_band = right_rect[row_lo:row_hi, :]

    disparity, edge_width_px = _band_disparity(left_band, right_band)
    valid = disparity > 0

    fx = calib.P1[0, 0]
    depth_m = np.zeros_like(disparity, dtype=np.float32)
    depth_m[valid] = (fx * calib.baseline_m) / disparity[valid]
    valid = valid & (depth_m >= config.AVOID_MIN_VALID_DEPTH_M) & (depth_m <= config.AVOID_MAX_VALID_DEPTH_M)

    depth_m, valid = _apply_spatial_filter(depth_m, valid)

    raw_distances, valid_fractions = _bin_min_distances(depth_m, valid, calib.image_width)
    return compute_bin_distances(raw_distances, valid_fractions, edge_width_px, calib.image_width)
