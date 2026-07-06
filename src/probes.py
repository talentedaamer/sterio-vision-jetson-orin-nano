"""Metadata probe attached to the PGIE (nvinfer) src pad.

Runs BEFORE the tiler, so bbox coordinates are still in each camera's own
frame space -- required for the per-source X/Y/Z estimate below. The tiler
remaps obj_meta.rect_params / text_params into composite-tile coordinates
automatically downstream, so anything set here still renders correctly on
the tiled + OSD output.

This is also the "headless" output path: on_detection() is the extension
point for wiring detections into MAVLink / geolocation telemetry later.
"""
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst
import pyds

from . import config
from .distance import Detection, estimate_xyz


def on_detection(detection: Detection) -> None:
    """Extension point: wire this to MAVLink / geolocation telemetry.

    Currently just logs to stdout.
    """
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
