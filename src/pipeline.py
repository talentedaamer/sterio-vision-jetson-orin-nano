"""Builds and wires the DeepStream GStreamer pipeline for dual-camera YOLOv8n
inference + on-screen distance overlay + RTSP output (+ optional local
bench-test display).

Pipeline topology (everything from nvarguscamerasrc through nvdsosd runs on
GPU/CUDA cores or dedicated hardware blocks -- see CLAUDE.md "GPU Utilization
Model" for exactly which engine does what and where the few unavoidable
CPU-touching bytes are):

    nvarguscamerasrc(0) -\
                           >- nvstreammux -> queue -> nvinfer(YOLOv8n) -> nvmultistreamtiler
    nvarguscamerasrc(1) -/                                                     |
                                                                                v
                                                     nvvideoconvert -> capsfilter(RGBA) -> nvdsosd -> tee
                                                                                                        |
                                              +-----------------------------------+---------------------+
                                              |                                                         |
        queue -> nvvideoconvert -> I420 -> nvjpegenc -> rtpjpegpay -> udpsink              queue -> nveglglessink
        (RTSP out via GstRtspServer, always on)                                            (debug/bench only)

RTSP encode uses `nvjpegenc` (hardware MJPEG via the NVJPG engine), NOT
`nvv4l2h264enc` -- this board is a Jetson Orin Nano module (confirmed via
`jetson_release` P-Number p3767-0003), which has NVENC (H.264/H.265/AV1
hardware encode) fused off. NVDEC and NVJPEG are still present. MJPEG-over-
RTP keeps encoding on dedicated hardware instead of falling back to a
CPU-bound software encoder (e.g. x264enc). See CLAUDE.md "Hardware video
encode" for the full story.

A probe on nvinfer's src pad (before tiling, so per-source bbox coordinates
are still meaningful) extracts detections, computes X/Y/Z, and sets the OSD
label text -- see probes.py. A second probe on nvdsosd's SINK pad (after
tiling, one composited frame per buffer) draws an optional one-line HUD
status overlay (MAVLink/mission state) when main.py has registered one via
probes.register_frame_status_provider() -- see probes.py.
"""
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer

from . import config
from .probes import osd_sink_pad_status_probe, pgie_src_pad_buffer_probe


def _make(factory: str, name: str) -> Gst.Element:
    elem = Gst.ElementFactory.make(factory, name)
    if not elem:
        raise RuntimeError(f"Failed to create GStreamer element '{factory}' ({name})")
    return elem


def _camera_bin(sensor_id: int) -> Gst.Bin:
    """nvarguscamerasrc -> NVMM NV12 capsfilter, wrapped as a Gst.Bin with a
    single ghosted 'src' pad so it can be linked straight into the streammux."""
    bin_ = Gst.Bin.new(f"cam_bin_{sensor_id}")

    src = _make("nvarguscamerasrc", f"argus_src_{sensor_id}")
    src.set_property("sensor-id", sensor_id)
    if config.SENSOR_MODE >= 0:
        src.set_property("sensor-mode", config.SENSOR_MODE)

    caps = _make("capsfilter", f"argus_caps_{sensor_id}")
    caps.set_property(
        "caps",
        Gst.Caps.from_string(
            f"video/x-raw(memory:NVMM), width={config.CAPTURE_WIDTH}, "
            f"height={config.CAPTURE_HEIGHT}, format=NV12, "
            f"framerate={config.FRAMERATE}/1"
        ),
    )

    bin_.add(src)
    bin_.add(caps)
    src.link(caps)

    ghost = Gst.GhostPad.new("src", caps.get_static_pad("src"))
    bin_.add_pad(ghost)
    return bin_


def _link_chain(pipeline: Gst.Pipeline, *elements: Gst.Element) -> None:
    for elem in elements:
        pipeline.add(elem)
    for a, b in zip(elements, elements[1:]):
        if not a.link(b):
            raise RuntimeError(f"Failed to link {a.get_name()} -> {b.get_name()}")


def _add_tee_branch(tee: Gst.Element, first_element: Gst.Element) -> None:
    tee_pad = tee.get_request_pad("src_%u")
    sink_pad = first_element.get_static_pad("sink")
    if tee_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
        raise RuntimeError(f"Failed to link tee -> {first_element.get_name()}")


def start_rtsp_server() -> GstRtspServer.RTSPServer:
    server = GstRtspServer.RTSPServer.new()
    server.set_service(str(config.RTSP_PORT))

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        f"( udpsrc port={config.RTSP_UDP_PORT} "
        f'caps="application/x-rtp, media=video, encoding-name=JPEG, payload=26" '
        f"! rtpjpegdepay ! rtpjpegpay name=pay0 pt=26 )"
    )
    factory.set_shared(True)

    mounts = server.get_mount_points()
    mounts.add_factory(config.RTSP_MOUNT_POINT, factory)
    server.attach(None)

    print(f"[rtsp] streaming at rtsp://<jetson-ip>:{config.RTSP_PORT}{config.RTSP_MOUNT_POINT}")
    return server


def build_pipeline(debug: bool = False) -> Gst.Pipeline:
    pipeline = Gst.Pipeline.new("dual-cam-yolov8n-stereo")

    # --- sources -> streammux -------------------------------------------------
    streammux = _make("nvstreammux", "streammux")
    streammux.set_property("batch-size", config.NUM_SOURCES)
    streammux.set_property("width", config.CAPTURE_WIDTH)
    streammux.set_property("height", config.CAPTURE_HEIGHT)
    streammux.set_property("live-source", 1)
    streammux.set_property("batched-push-timeout", config.MUX_BATCHED_PUSH_TIMEOUT_US)
    streammux.set_property("nvbuf-memory-type", 0)
    pipeline.add(streammux)

    for idx, sensor_id in enumerate(config.SENSOR_IDS):
        cam_bin = _camera_bin(sensor_id)
        pipeline.add(cam_bin)
        sink_pad = streammux.get_request_pad(f"sink_{idx}")
        if cam_bin.get_static_pad("src").link(sink_pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"Failed to link camera sensor-id={sensor_id} into streammux sink_{idx}")

    # --- inference -------------------------------------------------------------
    mux_queue = _make("queue", "mux_queue")

    pgie = _make("nvinfer", "pgie")
    pgie.set_property("config-file-path", config.PGIE_CONFIG_PATH)
    pgie.set_property("batch-size", 1)  # engine max batch = 1, see config.py note

    tiler = _make("nvmultistreamtiler", "tiler")
    tiler.set_property("rows", config.TILER_ROWS)
    tiler.set_property("columns", config.TILER_COLS)
    tiler.set_property("width", config.TILER_WIDTH)
    tiler.set_property("height", config.TILER_HEIGHT)

    conv_pre_osd = _make("nvvideoconvert", "conv_pre_osd")
    osd_caps = _make("capsfilter", "osd_caps")
    osd_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    osd = _make("nvdsosd", "osd")
    osd.set_property("process-mode", 1)  # GPU

    tee = _make("tee", "tee")

    _link_chain(pipeline, streammux, mux_queue, pgie, tiler, conv_pre_osd, osd_caps, osd, tee)

    pgie_src_pad = pgie.get_static_pad("src")
    pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_pad_buffer_probe, None)

    osd_sink_pad = osd.get_static_pad("sink")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_status_probe, None)

    # --- RTSP branch (always on) ------------------------------------------------
    # nvjpegenc (NVJPG hardware engine) instead of nvv4l2h264enc (NVENC) --
    # this Orin Nano module has no hardware H.264/H.265/AV1 encoder. See the
    # module docstring above and CLAUDE.md for why.
    rtsp_queue = _make("queue", "rtsp_queue")
    rtsp_conv = _make("nvvideoconvert", "rtsp_conv")
    rtsp_caps = _make("capsfilter", "rtsp_caps")
    rtsp_caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
    rtsp_enc = _make("nvjpegenc", "rtsp_enc")
    rtsp_enc.set_property("quality", config.RTSP_JPEG_QUALITY)
    rtsp_pay = _make("rtpjpegpay", "rtsp_pay")
    rtsp_pay.set_property("pt", 26)
    rtsp_sink = _make("udpsink", "rtsp_sink")
    rtsp_sink.set_property("host", "127.0.0.1")
    rtsp_sink.set_property("port", config.RTSP_UDP_PORT)
    rtsp_sink.set_property("sync", False)
    rtsp_sink.set_property("async", False)

    _link_chain(pipeline, rtsp_queue, rtsp_conv, rtsp_caps, rtsp_enc, rtsp_pay, rtsp_sink)
    _add_tee_branch(tee, rtsp_queue)

    # --- debug/bench branch (only when --debug) ---------------------------------
    if debug:
        # nvdsosd -> nveglglessink directly (no conversion) caused a caps
        # negotiation failure on-device ("not-negotiated (-4)") as soon as
        # this branch actually got exercised. NVIDIA's own reference
        # DeepStream apps always insert an nvvideoconvert before the final
        # display sink even when it looks redundant -- do the same here.
        debug_queue = _make("queue", "debug_queue")
        debug_conv = _make("nvvideoconvert", "debug_conv")
        debug_sink = _make("nveglglessink", "debug_sink")
        debug_sink.set_property("sync", False)

        _link_chain(pipeline, debug_queue, debug_conv, debug_sink)
        _add_tee_branch(tee, debug_queue)

    return pipeline
