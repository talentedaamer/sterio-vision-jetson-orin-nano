"""Metadata probe attached to the PGIE (nvinfer) src pad.

Runs BEFORE the tiler, so bbox coordinates are still in each camera's own
frame space -- required for the per-source X/Y/Z estimate below. The tiler
remaps obj_meta.rect_params / text_params into composite-tile coordinates
automatically downstream, so anything set here still renders correctly on
the tiled + OSD output.

This is also the "headless" output path: on_detection() is the extension
point for wiring detections into MAVLink / geolocation telemetry later.
Other code (e.g. the --debug 3D plot in src/debug_plot.py) can subscribe
to every Detection via register_detection_listener() below, without
touching this file.

osd_sink_pad_status_probe() is a separate probe, attached to nvdsosd's
SINK pad (after tiling, so there's exactly one composited frame to draw
on) -- it draws a persistent one-line HUD (MAVLink link health, mission
mode, live flight mode, follow state) via register_frame_status_provider(),
visible in both the RTSP stream and the --debug local display. Optional:
a no-op until something calls register_frame_status_provider() (see
src/mission.py via main.py).
"""
import time
from typing import Callable, Optional

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst
import pyds

from . import config
from .distance import Detection, estimate_xyz

# Printing every detection at full pipeline framerate (up to 60fps x 2
# cameras) measurably costs CPU on-device -- an SSH-terminal print() is a
# blocking syscall, not free. This was only ever a stdout placeholder ahead
# of real telemetry, so throttle it; on_detection() itself is still called
# for every detection so a future MAVLink hook gets full-rate data.
_LOG_INTERVAL_S = 0.5
_last_log_time = 0.0

_listeners: list[Callable[[Detection], None]] = []
_frame_status_provider: Optional[Callable[[], Optional[str]]] = None


def register_frame_status_provider(callback: Callable[[], Optional[str]]) -> None:
    """Register a zero-argument callback returning the current HUD status
    line (or None to show nothing), drawn on-screen every frame by
    osd_sink_pad_status_probe(). Called on the streaming thread -- keep it
    cheap (read cached state, no I/O). Only one provider at a time; the
    last registration wins."""
    global _frame_status_provider
    _frame_status_provider = callback


def register_detection_listener(callback: Callable[[Detection], None]) -> None:
    """Subscribe to receive every Detection (full rate, not throttled).

    Called from a GStreamer streaming thread, not the main/GLib thread --
    keep callbacks cheap (e.g. append to a buffer) and thread-safe. Used by
    the --debug 3D plot; a future MAVLink/geolocation hook can subscribe
    the same way instead of editing on_detection() directly.
    """
    _listeners.append(callback)


def on_detection(detection: Detection) -> None:
    """Extension point: wire this to MAVLink / geolocation telemetry.

    Every registered listener (see register_detection_listener()) gets
    every detection at full rate. The default stdout log below is
    throttled -- see module docstring.
    """
    for listener in _listeners:
        listener(detection)

    global _last_log_time
    now = time.monotonic()
    if now - _last_log_time < _LOG_INTERVAL_S:
        return
    _last_log_time = now
    print(
        f"[cam{detection.source_id}] {detection.label} "
        f"conf={detection.confidence:.2f} "
        f"X={detection.x_m:.2f} Y={detection.y_m:.2f} Z={detection.z_m:.2f}m"
    )


def pgie_src_pad_buffer_probe(pad, info, u_data):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_w = frame_meta.source_frame_width or config.CAPTURE_WIDTH
        frame_h = frame_meta.source_frame_height or config.CAPTURE_HEIGHT

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

            rect.border_width = 2
            rect.border_color.set(0.0, 1.0, 0.0, 1.0)
            rect.has_bg_color = 0

            txt = obj_meta.text_params
            txt.display_text = (
                f"{detection.label} {detection.confidence:.2f} | "
                f"X:{x:.1f} Y:{y:.1f} Z:{z:.1f}m"
            )
            txt.font_params.font_name = "Sans"
            txt.font_params.font_size = 11
            txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt.set_bg_clr = 1
            txt.text_bg_clr.set(0.0, 0.0, 0.0, 0.6)

            l_obj = l_next

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

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
