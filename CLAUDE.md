# sterio-vision-jetson-orin-nano — Jetson Orin Nano Dual-Camera DeepStream Pipeline

UAV payload: real-time YOLOv8n detection + monocular distance estimation over
dual IMX296 global-shutter CSI cameras, running 100% on GPU/dedicated hardware
blocks via NVIDIA DeepStream. Stereo calibration and disparity-based depth,
localization, and MAVLink telemetry are future milestones (see "Next Steps").

## Target Hardware / Software (do not assume anything different)

**Correction (confirmed on-device via `jetson_release`, 2026-07-06):** this
board's SOM is a **Jetson Orin Nano 8GB (P-Number p3767-0003)**, not an Orin
NX as originally assumed. The carrier board's device-tree reports its model
string as "Orin NX Engineering Reference Developer Kit" because that carrier
is a shared reference design used by both Orin Nano and Orin NX SOMs — it
does not identify which SOM is actually plugged in. `jetson_release`'s
P-Number is the authoritative source; trust that over the device-tree model
string. This matters because **Orin Nano has no hardware video encoder
(NVENC)** — see "Hardware video encode" below.

| Component | Version |
|---|---|
| Hardware | NVIDIA Jetson Orin Nano 8GB module (P-Number p3767-0003), on an Orin NX/Nano Engineering Reference Developer Kit carrier board |
| OS | Ubuntu 22.04.5 LTS (Jammy) |
| Kernel | Linux 5.15.148-tegra (aarch64) |
| JetPack / L4T | L4T R36.4.7 (`jetson_release` couldn't map this exact patch level to a JetPack version string — component versions below line up with JetPack 6.1/6.2) |
| CUDA | 12.6 (V12.6.68) |
| cuDNN | 9.3.0.75 |
| TensorRT | 10.3.0.30 |
| VPI | 3.2.4 |
| DeepStream SDK | 7.1 (already installed at `/opt/nvidia/deepstream/deepstream-7.1` — do not reinstall). Python bindings (`pyds`) are NOT bundled on-device; installed from the [deepstream_python_apps v1.2.0](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/tag/v1.2.0) release wheel — see setup below |
| Python | 3.10 (uv-managed venv, `--system-site-packages`) |
| Hardware video codecs | **NVDEC (decode): yes. NVJPEG (JPEG en/decode): yes. NVENC (H.264/H.265/AV1 encode): NO — fused off on this SKU.** |
| Cameras | 2x IMX296 global shutter, CSI0 -> `/dev/video0`, CSI1 -> `/dev/video1` |

Not chased, and deliberately out of scope for this pipeline: the `nvidia-jetpack`
meta-package isn't installed (cosmetic — only affects `jetson_release`'s
version-string lookup, not functionality; don't add a new apt source to fix
it on a working board), and system OpenCV is built without CUDA (irrelevant
here — this pipeline never calls `cv2`, DeepStream/TensorRT do all the work).

Project deps (`pyproject.toml`) are managed with `uv`. `torch`/`torchvision`/
`ultralytics`/`onnx` exist only to support the (already-working, do-not-touch)
`.pt -> .engine` export script — the DeepStream runtime app never imports
them; it loads the prebuilt TensorRT engine directly through `gst-nvinfer`.

One-time environment setup (already-installed DeepStream SDK, not repeated
here as install steps since it's present on-device):
```bash
# If `uv run` was ever invoked before this, it will have already created a
# wrong .venv (uv-downloaded interpreter, no system-site-packages) -- remove it:
rm -rf .venv

# Must be the SYSTEM Python 3.10, with --system-site-packages, so the
# OS-provided `gi`/PyGObject (apt python3-gi, gir1.2-gst-1.0, etc.) is
# importable. Do NOT let uv pick its own downloaded interpreter (e.g. 3.12) --
# extension modules built for system 3.10 won't load under a different ABI.
uv venv --system-site-packages --python /usr/bin/python3.10
uv sync

# pyds is not on PyPI, and this DS 7.1 install doesn't bundle the wheel
# on-device either -- it's published on the deepstream_python_apps GitHub
# releases page (https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/tag/v1.2.0).
# v1.2.0 = DeepStream 7.1; grab the cp310/aarch64 wheel for this Jetson:
uv pip install https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_aarch64.whl

# Sanity check before running main.py:
uv run python -c "import gi; gi.require_version('Gst','1.0'); \
    gi.require_version('GstRtspServer','1.0'); \
    from gi.repository import Gst, GstRtspServer; import pyds; print('ok')"
```
Do **not** add `pygobject`/`pycairo` as a pip/uv dependency — that tries to
compile pycairo from source against `libcairo2-dev` (usually missing) and,
even if it built, would be a second ABI-mismatched copy shadowing the
working system one. If the `GstRtspServer` import above fails, install the
OS package instead: `sudo apt install gir1.2-gst-rtsp-server-1.0`.

## Model export (existing, untouched)

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', device='0', half=True, workspace=4)
```

Consequences this project's DeepStream config is built around:
- **FP16**, default **640x640** input, **COCO-80** classes.
- No `nms=True`/`end2end` export flag → the engine's output head is raw:
  shape `[1, 84, 8400]` (4 box channels + 80 class scores, sigmoid already
  applied, no NMS baked in). This is why a **custom bbox parser**
  (`nvdsinfer_custom_impl_yolov8/`) is required — DeepStream's built-in
  detector parser can't decode this layout.
- No `dynamic=True` → **the engine's max batch size is fixed at 1.**
  `nvinfer`'s `batch-size` in `configs/pgie_yolov8n_config.txt` is therefore
  set to `1`, and it runs one inference call per camera per muxer cycle
  instead of one fused batch-of-2 call. Both calls still run entirely on
  TensorRT/GPU — this only affects *fusion*, not GPU-only-ness. If fused
  dual-camera throughput becomes the bottleneck later, re-export with
  `dynamic=True` and rebuild with an explicit batch=2 profile.

## Architecture

```
nvarguscamerasrc(0) -\
                       >- nvstreammux -> queue -> nvinfer(YOLOv8n) -> nvmultistreamtiler
nvarguscamerasrc(1) -/                                                     |
                                                                            v
                                                nvvideoconvert -> RGBA -> nvdsosd -> tee
                                                                                      |
                                    +-----------------------------------+-------------+
                                    |                                                 |
    queue -> nvvideoconvert -> I420 -> nvjpegenc -> rtpjpegpay -> udpsink     queue -> nveglglessink
    (RTSP out via GstRtspServer, ALWAYS on)                                  (bench display, --debug only)
```

- A pad probe on `nvinfer`'s **src pad** (before tiling, while bbox coords
  are still per-camera) computes X/Y/Z per detection and sets the OSD label.
  This is also the **headless output path** — see `on_detection()` in
  `src/probes.py`, the extension point for MAVLink/geolocation telemetry.
- The tiler composites both camera views into one 1280x720 side-by-side
  frame and remaps object metadata (bbox + text) into tile coordinates
  automatically, so OSD/RTSP show both cameras with correct overlays.
- RTSP (ground station) and headless detection output are **always active**.
  The local display branch (`nveglglessink`) is added **only** when
  `config.DEBUG` / `--debug` is set — that's the "bench testing" toggle.

### Hardware video encode: MJPEG (nvjpegenc), not H.264

The pipeline originally used `nvv4l2h264enc` for the RTSP branch, which
failed with `Failed to create GStreamer element 'nvv4l2h264enc'`. Root
cause: **this Orin Nano SOM has no NVENC hardware encoder at all** (fused
off on this SKU; only Orin NX/AGX Orin keep it) — confirmed by
`nvidia-l4t-gstreamer` already being installed (so it wasn't a missing
package), and independently by `jetson_release`'s P-Number identifying the
module as Orin Nano. NVIDIA's own developer guide lists software (CPU)
x264 as the documented default H.264 path on Orin Nano for exactly this
reason.

Rather than accept a CPU-bound software encoder — which would violate the
"100% GPU/hardware, no CPU" requirement this project is built around — the
RTSP branch instead uses **`nvjpegenc`**, which drives the **NVJPG**
engine: a separate, dedicated hardware block that Orin Nano *does* keep
(only the video-codec NVENC/NVDEC-encode side was removed; JPEG en/decode
and general video decode via NVDEC are both intact). Trade-offs:
- MJPEG has no inter-frame compression, so it's larger per frame than
  H.264 at the same visual quality — tune `config.RTSP_JPEG_QUALITY`
  (0-100) and/or `config.TILER_WIDTH/HEIGHT` if ground-station bandwidth is
  tight.
- Every MJPEG frame is independently decodable (no GOP/keyframe
  structure), which is arguably nicer for a live low-latency monitoring
  view — no multi-frame decode delay, no visible artifacting from a
  dropped P-frame.
- `nvjpegenc` still needs an NV12/I420 NVMM input, hence
  `nvvideoconvert` before it in the RTSP branch (unchanged from the
  original design, only the encoder+payloader swapped: `nvjpegenc` +
  `rtpjpegpay` instead of `nvv4l2h264enc` + `h264parse` + `rtph264pay`).

### GPU utilization model — what actually touches which engine

| Stage | Hardware |
|---|---|
| Camera capture (Argus/ISP) | Dedicated ISP, NVMM buffers, zero-copy |
| `nvstreammux` batching | GPU/unified memory, no copy |
| `nvinfer` (YOLOv8n) | TensorRT on GPU (CUDA cores) |
| `nvvideoconvert` (NV12↔RGBA↔I420, scale/tile) | VIC (dedicated video/image compositor block) by default on Jetson — deliberately **not** forced to `compute-hw=GPU`, so it runs in parallel with TensorRT instead of contending for CUDA cores |
| `nvdsosd` (box/text draw) | GPU (`process-mode=1`) |
| `nvjpegenc` | NVJPG (dedicated hardware JPEG codec engine — NOT NVENC, which this SKU lacks) |
| RTP packetization / `udpsink` / GLib mainloop / distance math in the probe | ARM CPU — trivial control-plane work only (packet framing, tens of floats per frame), never touches pixels or tensors |

No real video pipeline is *literally* 0% CPU (mainloop scheduling, RTP
framing, and the Python distance-math probe all run on the ARM cores), but
every pixel- and tensor-heavy stage — capture, batching, inference,
colorspace/scale conversion, tiling, OSD compositing, JPEG encode — runs
on GPU or a dedicated hardware block. That's what determines whether the
pipeline lags, and none of it touches the CPU.

## File map

```
CLAUDE.md                                  this file
pyproject.toml                             uv project + deps
main.py                                    entry point, arg parsing, GLib mainloop, bus/error handling
src/config.py                              all tunables (camera, model, classes, RTSP, tiler)
src/pipeline.py                            GStreamer/DeepStream pipeline construction
src/probes.py                              per-frame metadata probe: class filter, distance calc, OSD text
src/distance.py                            monocular X/Y/Z estimator (+ stereo placeholder)
configs/pgie_yolov8n_config.txt            nvinfer config for the YOLOv8n engine
configs/labels_coco.txt                    COCO-80 class names, index-matched to the engine's output
nvdsinfer_custom_impl_yolov8/*.cpp/Makefile   custom bbox parser for the raw (no-NMS) YOLOv8 output head
models/yolov8n.engine                      <- place your exported engine here (not committed)
```

## Build & run (on the Jetson)

```bash
# 1. Place the exported engine
cp /path/to/yolov8n.engine models/yolov8n.engine

# 2. Build the custom TensorRT-output parser (must be built on-device, aarch64)
cd nvdsinfer_custom_impl_yolov8 && make && cd ..

# 3. Run — headless + RTSP only (default, for flight)
uv run main.py

# 3b. Run — headless + RTSP + local bench display (dev bench, monitor attached)
uv run main.py --debug
# or: DS_DEBUG=1 uv run main.py
```

View the ground-station stream from a laptop on the same network:
```bash
ffplay rtsp://<jetson-ip>:8554/ds-stereo
# or: gst-launch-1.0 rtspsrc location=rtsp://<jetson-ip>:8554/ds-stereo ! decodebin ! autovideosink
```

## Known limitations / Next steps

- **Distance estimation is monocular** (known-height + focal-length,
  per-camera, ported from the original prototype). `src/distance.py` has a
  stub `estimate_xyz_stereo()` — once both IMX296s are calibrated
  (intrinsics + baseline via `cv2.stereoCalibrate` or similar), replace the
  call in `src/probes.py` with a disparity-based depth lookup.
- **No object tracker yet** (no `nvtracker` in the pipeline) — detections
  aren't assigned persistent IDs across frames. Add `nvtracker` between
  `pgie` and `tiler` when ID persistence is needed (e.g. for stable
  localization output).
- **No MAVLink/geolocation hookup yet** — `on_detection()` in
  `src/probes.py` is the single extension point; wire telemetry output
  there when that milestone starts.
- **Engine batch=1** (see export notes above) — revisit if fused-batch
  throughput becomes necessary once both cameras are running at full load.
- RTSP branch currently sends **one composited (tiled) MJPEG stream**, not
  two independent per-camera streams — simplest for a ground station view;
  revisit if the ground station needs full-resolution per-camera feeds
  instead of the 1280x720 tiled composite.
- **RTSP is MJPEG (`nvjpegenc`), not H.264** — this Orin Nano SOM has no
  hardware H.264 encoder (see "Hardware video encode" above). Uses more
  bandwidth than H.264 would have at the same quality; tune
  `RTSP_JPEG_QUALITY`/tiler resolution if the ground-station link is
  constrained. If this project ever moves to an Orin NX/AGX Orin board
  (which do have NVENC), swap `nvjpegenc`/`rtpjpegpay` back for
  `nvv4l2h264enc`/`h264parse`/`rtph264pay` in `src/pipeline.py` for
  better compression.
