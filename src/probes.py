"""Metadata probe attached to the PGIE (nvinfer) src pad.

Runs BEFORE the tiler, so bbox coordinates are still in each camera's own
frame space -- required for the per-source X/Y/Z estimate below. The tiler
remaps obj_meta.rect_params / text_params into composite-tile coordinates
automatically downstream, so anything set here still renders correctly on
the tiled + OSD output.

This is also the "headless" output path: on_detection() is the extension
point for wiring detections into MAVLink / geolocation telemetry later.
Other code (e.g. the --debug 3D plot in src/debug_plot.py, src/mission.py)
can subscribe to every Detection via register_detection_listener() below,
without touching this file. Per-detection X/Y/Z is intentionally NOT
printed to stdout here -- it's already visible on the RTSP/--debug video
overlay and the --debug 3D plot; see src/mission.py's status log for the
MAVLink/mission-focused console output instead.

register_follow_active_query() lets src/mission.py report whether FOLLOW
is currently locked onto a target, so the object matching
config.FOLLOW_TARGET_CLASS is drawn red (locked) instead of green here.

The on-screen X/Y/Z label is smoothed (SmoothedDetection, one per
(source_id, class_id) -- see distance.py) so it only changes once a
second instead of flickering every frame; on_detection()/FOLLOW/the
debug plot still get the raw, un-smoothed per-frame estimate.

osd_sink_pad_status_probe() is a separate probe, attached to nvdsosd's
SINK pad (after tiling, so there's exactly one composited frame to draw
on) -- it draws a persistent one-line HUD (MAVLink link health, mission
mode, live flight mode, follow state) via register_frame_status_provider(),
visible in both the RTSP stream and the --debug local display. Optional:
a no-op until something calls register_frame_status_provider() (see
src/mission.py via main.py).

When MISSION_MODE=="AVOID", pgie_src_pad_buffer_probe() also collects both
sources' raw NV12 buffers (source_id 0=left, 1=right) within the SAME
buffer/probe call -- nvstreammux batches both cameras into one Gst.Buffer
per cycle, so both are visited within one pass of the frame_meta loop
below -- and, throttled to config.AVOID_UPDATE_INTERVAL_S (NOT full frame
rate), calls obstacle_depth.estimate_bin_distances() and fires
on_obstacle_reading() listeners (register_obstacle_listener(), same
pattern as on_detection()). This is entirely skipped (zero cost) in any
other mode. See src/obstacle_depth.py for why this must complete before
the probe returns rather than being deferred across calls, and CLAUDE.md
"Prerequisite validation" -- this is the first live use of
pyds.get_nvds_buf_surface() in this project's always-on pipeline.
"""
import time
from typing import Callable, Optional

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst
import pyds

from . import config
from .distance import Detection, SmoothedDetection, estimate_xyz

_listeners: list[Callable[[Detection], None]] = []
_frame_status_provider: Optional[Callable[[], Optional[str]]] = None
_follow_active_query: Optional[Callable[[], bool]] = None
_smoothers: dict = {}   # (source_id, class_id) -> SmoothedDetection

_obstacle_listeners: list[Callable[[list, list], None]] = []
_last_obstacle_update_time = 0.0
_obstacle_calib = None   # lazy StereoCalibration singleton, AVOID mode only


def register_frame_status_provider(callback: Callable[[], Optional[str]]) -> None:
    """Register a zero-argument callback returning the current HUD status
    line (or None to show nothing), drawn on-screen every frame by
    osd_sink_pad_status_probe(). Called on the streaming thread -- keep it
    cheap (read cached state, no I/O). Only one provider at a time; the
    last registration wins."""
    global _frame_status_provider
    _frame_status_provider = callback


def register_follow_active_query(callback: Callable[[], bool]) -> None:
    """Register a zero-argument callback returning True while FOLLOW is
    actively locked onto a target (e.g. mission.follow.active). While
    True, any detection matching config.FOLLOW_TARGET_CLASS is drawn red
    with a center marker instead of the normal green box. Called on the
    streaming thread -- keep it cheap."""
    global _follow_active_query
    _follow_active_query = callback


def register_detection_listener(callback: Callable[[Detection], None]) -> None:
    """Subscribe to receive every Detection (full rate).

    Called from a GStreamer streaming thread, not the main/GLib thread --
    keep callbacks cheap (e.g. append to a buffer) and thread-safe. Used by
    the --debug 3D plot and src/mission.py.
    """
    _listeners.append(callback)


def on_detection(detection: Detection) -> None:
    """Extension point: wire this to MAVLink / geolocation telemetry.

    Every registered listener (see register_detection_listener()) gets
    every detection at full rate.
    """
    for listener in _listeners:
        listener(detection)


def register_obstacle_listener(callback: Callable[[list, list], None]) -> None:
    """Subscribe to receive (bin_distances_m, bin_valid_mask) whenever a
    new obstacle reading is computed -- throttled to
    config.AVOID_UPDATE_INTERVAL_S inside pgie_src_pad_buffer_probe, NOT
    full frame rate (see module docstring). Same threading contract as
    register_detection_listener(): called from the GStreamer streaming
    thread, keep callbacks cheap/thread-safe. Used by src/mission.py."""
    _obstacle_listeners.append(callback)


def on_obstacle_reading(bin_distances_m: list, bin_valid_mask: list) -> None:
    for listener in _obstacle_listeners:
        listener(bin_distances_m, bin_valid_mask)


def _run_obstacle_depth(source_buffers: dict) -> None:
    """Called once per buffer from pgie_src_pad_buffer_probe, only after
    MISSION_MODE/listener/throttle gating already passed and both
    source_id 0 (left) and 1 (right) raw buffers were collected this
    cycle. Everything here (buffer surface extraction, rectification,
    cv2.StereoSGBM) is genuinely new, UNVERIFIED live-pipeline code -- see
    src/obstacle_depth.py's docstring before trusting its output."""
    global _obstacle_calib
    if 0 not in source_buffers or 1 not in source_buffers:
        return

    # Lazy imports -- cv2/numpy/calibration/obstacle_depth are only ever
    # touched when AVOID mode is actually configured, zero cost/impact
    # otherwise (same convention as main.py's MISSION_MODE-gated imports).
    from . import obstacle_depth
    from .calibration import load as load_calibration

    if _obstacle_calib is None:
        _obstacle_calib = load_calibration()

    bin_distances_m, bin_valid_mask = obstacle_depth.estimate_bin_distances(
        source_buffers[0], source_buffers[1], _obstacle_calib,
    )
    on_obstacle_reading(bin_distances_m, bin_valid_mask)


def pgie_src_pad_buffer_probe(pad, info, u_data):
    global _last_obstacle_update_time

    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    # Throttled BEFORE any buffer-surface extraction is attempted --
    # get_nvds_buf_surface() may not be a free call (possible
    # device-to-host copy depending on nvstreammux's memory type), so this
    # gate must run first, not just the SGBM pass downstream of it. Zero
    # cost/risk in any mode other than AVOID.
    now = time.monotonic()
    collect_obstacle_buffers = (
        config.MISSION_MODE == "AVOID"
        and bool(_obstacle_listeners)
        and (now - _last_obstacle_update_time) >= config.AVOID_UPDATE_INTERVAL_S
    )
    source_buffers = {} if collect_obstacle_buffers else None
    if collect_obstacle_buffers:
        _last_obstacle_update_time = now

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_w = frame_meta.source_frame_width or config.CAPTURE_WIDTH
        frame_h = frame_meta.source_frame_height or config.CAPTURE_HEIGHT

        if collect_obstacle_buffers and frame_meta.source_id in (0, 1):
            try:
                source_buffers[frame_meta.source_id] = pyds.get_nvds_buf_surface(
                    hash(buf), frame_meta.batch_id
                )
            except Exception as exc:  # pragma: no cover -- see obstacle_depth.py "UNVERIFIED"
                print(f"[avoid] get_nvds_buf_surface failed for source_id={frame_meta.source_id}: {exc}")

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            l_next = l_obj.next

            if obj_meta.class_id not in config.TARGET_CLASSES:
                pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
                l_obj = l_next
                continue

            rect = obj_meta.rect_params
            x, y, z = estimate_xyz(
                obj_meta.class_id, rect.left, rect.top, rect.width, rect.height,
                frame_w, frame_h,
            )

            detection = Detection(
                source_id=frame_meta.source_id,
                class_id=obj_meta.class_id,
                label=obj_meta.obj_label,
                confidence=obj_meta.confidence,
                left=rect.left, top=rect.top, width=rect.width, height=rect.height,
                x_m=x, y_m=y, z_m=z,
            )
            on_detection(detection)

            # Smoothed purely for the on-screen label -- on_detection() above
            # (and therefore FOLLOW/the debug plot) already got the raw,
            # un-smoothed x/y/z. One smoother per (source_id, class_id);
            # same "identity by class, not by object" limitation as FOLLOW's
            # target selection, since there's no tracker yet.
            smoother_key = (frame_meta.source_id, obj_meta.class_id)
            smoother = _smoothers.setdefault(smoother_key, SmoothedDetection())
            sx, sy, sz = smoother.update(x, y, z)

            # A detection is drawn as the "locked" target -- red box, center
            # marker, distance called out -- while FOLLOW is actively locked
            # on AND it matches the class being followed. There's no object
            # tracker yet (see CLAUDE.md/README roadmap), so with more than
            # one matching object in frame, all of them are marked; only one
            # actually drives ObjectFollowController (src/pid.py).
            is_locked_target = (
                _follow_active_query is not None
                and _follow_active_query()
                and obj_meta.class_id == config.FOLLOW_TARGET_CLASS
            )

            rect.border_width = 3 if is_locked_target else 2
            if is_locked_target:
                rect.border_color.set(1.0, 0.0, 0.0, 1.0)
            else:
                rect.border_color.set(0.0, 1.0, 0.0, 1.0)
            rect.has_bg_color = 0

            txt = obj_meta.text_params
            if is_locked_target:
                txt.display_text = f"TARGET LOCKED | {detection.label} | Dist: {sz:.1f}m"
            else:
                txt.display_text = (
                    f"{detection.label} {detection.confidence:.2f} | "
                    f"X:{sx:.1f} Y:{sy:.1f} Z:{sz:.1f}m"
                )
            txt.font_params.font_name = "Sans"
            txt.font_params.font_size = 11
            txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt.set_bg_clr = 1
            if is_locked_target:
                txt.text_bg_clr.set(0.6, 0.0, 0.0, 0.7)
            else:
                txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

            if is_locked_target:
                cx = rect.left + rect.width / 2.0
                cy = rect.top + rect.height / 2.0
                marker_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                marker_meta.num_circles = 1
                circle = marker_meta.circle_params[0]
                circle.xc = int(cx)
                circle.yc = int(cy)
                circle.radius = 6
                circle.circle_color.set(1.0, 0.0, 0.0, 1.0)
                circle.has_bg_color = 1
                circle.bg_color.set(1.0, 0.0, 0.0, 1.0)
                pyds.nvds_add_display_meta_to_frame(frame_meta, marker_meta)

            l_obj = l_next

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    if collect_obstacle_buffers:
        _run_obstacle_depth(source_buffers)

    return Gst.PadProbeReturn.OK


def osd_sink_pad_status_probe(pad, info, u_data):
    """Draws a persistent one-line HUD in the top-left corner of the
    composited frame (MAVLink link health / mission mode / flight mode /
    follow state), if register_frame_status_provider() has been called.
    Attached to nvdsosd's SINK pad -- after tiling, so there's exactly one
    composited frame per buffer to draw on (unlike pgie_src_pad_buffer_probe
    above, which runs per-camera before tiling)."""
    if _frame_status_provider is None:
        return Gst.PadProbeReturn.OK

    status_text = _frame_status_provider()
    if not status_text:
        return Gst.PadProbeReturn.OK

    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    if l_frame is None:
        return Gst.PadProbeReturn.OK
    try:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
    except StopIteration:
        return Gst.PadProbeReturn.OK

    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    display_meta.num_labels = 1
    text = display_meta.text_params[0]
    text.display_text = status_text
    text.x_offset = 10
    text.y_offset = 10
    text.font_params.font_name = "Sans"
    text.font_params.font_size = 12
    text.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
    text.set_bg_clr = 1
    text.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    return Gst.PadProbeReturn.OK
