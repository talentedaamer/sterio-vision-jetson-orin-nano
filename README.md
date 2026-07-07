# sterio-vision-jetson-orin-nano

Real-time dual-camera object detection and monocular distance estimation for
a UAV payload, running on an NVIDIA Jetson Orin Nano via NVIDIA DeepStream.
Detection uses a YOLOv8n TensorRT engine; all pixel- and tensor-heavy work
(camera capture, inference, colorspace/scale conversion, on-screen box
drawing, video encode) runs on the GPU or a dedicated hardware block, not
the CPU. Stereo camera calibration, disparity-based depth, and flight
controller (MAVLink) integration for absolute geolocation are the next
milestones.

For the full technical reference (exact hardware/software versions, every
debugging gotcha hit while bringing this up, and the reasoning behind each
design decision), see [CLAUDE.md](CLAUDE.md). This README is the
project-level overview; CLAUDE.md is the deep-dive.

![Live dual-camera detection output](images/screenshot-1.png)

*The tiled RTSP ground-station view: both IMX296 cameras side by side, each
independently detecting the same person with a bounding box and the
per-camera monocular X/Y/Z distance overlay (`person 0.89 | X:0.1 Y:-0.2
Z:3.5m` on the left/cam0, `person 0.84 | X:-0.0 Y:-0.3 Z:3.5m` on the
right/cam1) — this is exactly what `ffplay rtsp://<jetson-ip>:8554/ds-stereo`
shows live.*

## Status: what's working today

- ✅ Both IMX296 CSI cameras streaming live via `nvarguscamerasrc`
- ✅ YOLOv8n TensorRT engine running inference on both streams via
  DeepStream's `nvinfer`, with a custom bbox parser for the engine's raw
  (no-NMS) output head
- ✅ Per-detection monocular X/Y/Z distance estimate, overlaid on-screen and
  logged
- ✅ Composited (tiled) dual-camera view streamed live over RTSP for a
  ground-station viewer
- ✅ Optional local bench display for testing with a monitor attached
- ✅ systemd unit for unattended boot-time startup
- ⏳ Not yet done: stereo calibration / disparity depth, object tracking
  (persistent IDs), MAVLink/geolocation output — see
  [Roadmap](#roadmap--known-limitations)

## Hardware

| Component | Detail |
|---|---|
| **Compute** | NVIDIA Jetson Orin Nano 8GB module (P-Number `p3767-0003`), on an Orin NX/Nano Engineering Reference Developer Kit carrier board |
| **Cameras** | 2x IMX296 global-shutter CSI cameras — CSI0 → `/dev/video0` (left), CSI1 → `/dev/video1` (right) |
| **Camera capture mode** | 1456x1088 @ 60fps, NV12, via `nvarguscamerasrc` |
| **Hardware video codecs** | NVDEC (decode): yes. NVJPEG (JPEG en/decode): yes. **NVENC (H.264/H.265/AV1 encode): no** — fused off on this SOM (only Orin NX/AGX Orin keep it). This is why RTSP output uses hardware MJPEG (`nvjpegenc`) instead of H.264 — see [CLAUDE.md § Hardware video encode](CLAUDE.md#hardware-video-encode-mjpeg-nvjpegenc-not-h264) |

> The carrier board's device-tree reports its model string as "Orin NX
> Engineering Reference Developer Kit" — that's the *carrier's* name, not
> the SOM's. It's a shared reference design used by both Orin Nano and
> Orin NX modules. `jetson_release`'s P-Number is the authoritative source
> for which module is actually installed.

## Software / library versions

| Component | Version |
|---|---|
| OS | Ubuntu 22.04.5 LTS (Jammy) |
| Kernel | Linux 5.15.148-tegra (aarch64) |
| L4T | R36.4.7 (JetPack 6.1/6.2 generation) |
| CUDA | 12.6 (V12.6.68) |
| cuDNN | 9.3.0.75 |
| TensorRT | 10.3.0.30 |
| cuDSS | 0.8.0 (installed as a **system package** via NVIDIA's `.deb` installer — required by the Jetson-native PyTorch build; the pip `nvidia-cudss-cu12` package is deliberately *not* used, see below) |
| VPI | 3.2.4 |
| DeepStream SDK | 7.1 |
| DeepStream Python bindings (`pyds`) | 1.2.0, from the [deepstream_python_apps v1.2.0](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/tag/v1.2.0) release wheel (not bundled with the SDK install, not on PyPI) |
| Python | 3.10 (system interpreter, uv-managed venv with `--system-site-packages`) |
| PyTorch | 2.11.0 — **Jetson-native build** from the [Jetson AI Lab pip index](https://pypi.jetson-ai-lab.io/jp6/cu126), not standard PyPI (see [Gotchas](#notable-gotchas-solved-during-bring-up)) |
| torchvision | 0.26.0, same Jetson AI Lab index |
| Ultralytics | ≥8.4.87 (export-only; not used by the runtime pipeline) |
| ONNX | ≥1.22.0 (export-only) |
| Package manager | [uv](https://docs.astral.sh/uv/) |

Package management notes:
- `torch`/`torchvision`/`ultralytics`/`onnx` are only needed to run
  [`export_engine.py`](export_engine.py) (the `.pt → .engine` conversion).
  The runtime pipeline (`main.py`) never imports them — it loads the
  prebuilt TensorRT engine directly through `gst-nvinfer`.
- `PyGObject`/`pycairo` are intentionally **not** pip dependencies — the
  OS already provides working, correctly-built `gi` bindings
  (`python3-gi`, `gir1.2-gst-1.0`, `gir1.2-gst-rtsp-server-1.0`), and a pip
  install would shadow them with an ABI-mismatched copy.

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

1. **Capture** — both IMX296 cameras feed `nvarguscamerasrc`, producing
   NV12 frames directly in NVMM (GPU-accessible) memory.
2. **Batching** — `nvstreammux` combines both camera streams into one
   batch per cycle for the inference stage.
3. **Inference** — `nvinfer` runs the YOLOv8n TensorRT engine on each
   frame (see [Model](#model) below for why a custom bbox parser is
   needed).
4. **Distance estimation** — a pad probe on `nvinfer`'s output (before
   tiling, so bbox coordinates are still per-camera) computes a monocular
   X/Y/Z estimate per detection and writes it into the on-screen label.
   This same probe is the extension point for future MAVLink/geolocation
   output (see [Roadmap](#roadmap--known-limitations)).
5. **Composite view** — `nvmultistreamtiler` combines both camera views
   into one side-by-side 1280x720 frame; `nvdsosd` draws the detection
   boxes and labels onto it.
6. **Output** — the composited, annotated video is always streamed out
   over RTSP (MJPEG) for a ground-station viewer, and detection results
   are always logged. A local display branch is added only in `--debug`
   mode for bench testing with a monitor attached.

### Model

`models/yolov8n.engine` is a YOLOv8n TensorRT engine, exported via:
```python
from ultralytics import YOLO
YOLO('yolov8n.pt').export(format='engine', device='0', half=True, workspace=4)
```
This is a raw (no built-in NMS) FP16 engine with a `[1, 84, 8400]` output
head (4 box channels + 80 COCO class scores, sigmoid already applied).
DeepStream's built-in detector parser can't decode this layout, so
[`nvdsinfer_custom_impl_yolov8/`](nvdsinfer_custom_impl_yolov8) provides a
custom one; NMS itself is handled by `nvinfer`'s own clustering
(`cluster-mode=2` in [`configs/pgie_yolov8n_config.txt`](configs/pgie_yolov8n_config.txt)).

**TensorRT engines are hardware- and version-locked** — a `.engine` file
built on one machine will not load on another unless the TensorRT version
and GPU architecture match exactly. This one must be (re-)exported
directly on this Jetson whenever the model changes; see
[Building & running](#building--running).

## Repository layout

| Path | Purpose |
|---|---|
| [`main.py`](main.py) | Entry point. Parses `--debug`, builds the pipeline, runs the GLib mainloop, handles GStreamer bus errors and clean shutdown. |
| [`export_engine.py`](export_engine.py) | Runs the (unmodified) `.pt → .engine` export directly on-device, with diagnostics (torch/CUDA/TensorRT versions, free GPU memory, clear failure messages) and automatic install into `models/`. |
| [`src/config.py`](src/config.py) | All tunables in one place: camera capture settings, model/engine paths, target detection classes, distance-estimation constants, tiler/RTSP settings. |
| [`src/pipeline.py`](src/pipeline.py) | Builds and links the full GStreamer/DeepStream pipeline (cameras → mux → inference → tiler → OSD → RTSP/debug branches) and starts the RTSP server. |
| [`src/probes.py`](src/probes.py) | The per-frame metadata probe: filters to target classes, computes distance per detection, sets on-screen label text. `register_detection_listener()` lets other code subscribe to every detection (full rate) without editing this file — used by `src/debug_plot.py` and the intended hook point for future MAVLink/telemetry output. |
| [`src/distance.py`](src/distance.py) | Monocular X/Y/Z estimator (known object height + focal length), plus a placeholder for the future stereo-disparity estimator. |
| [`src/debug_plot.py`](src/debug_plot.py) | `--debug`-only: live 3D scatter plot of detection X/Y/Z via matplotlib, colored by camera. Needs a display attached to the Jetson and a working GUI backend (Tk/Qt/GTK) — same physical requirement as the `nveglglessink` bench-display branch; degrades to a harmless no-op with a warning if unavailable. |
| [`configs/pgie_yolov8n_config.txt`](configs/pgie_yolov8n_config.txt) | `nvinfer` configuration for the YOLOv8n engine (batch size, NMS thresholds, custom parser wiring). |
| [`configs/labels_coco.txt`](configs/labels_coco.txt) | COCO-80 class names, index-matched to the engine's output classes. |
| [`nvdsinfer_custom_impl_yolov8/`](nvdsinfer_custom_impl_yolov8) | C++ custom bbox parser (`.cpp` + `Makefile`) for the engine's raw output head — must be compiled on-device. |
| [`deploy/sterio-vision.service`](deploy/sterio-vision.service) | systemd unit for unattended startup on boot. |
| `models/yolov8n.engine` | The exported TensorRT engine (not committed — built per-device, see below). |
| [`pyproject.toml`](pyproject.toml) | `uv`-managed project dependencies and environment notes. |

## Setup (one-time, on the Jetson)

```bash
# venv must be the system Python 3.10, with access to OS-provided GStreamer
# bindings (gi/PyGObject), since those aren't pip packages here
uv venv --system-site-packages --python /usr/bin/python3.10
uv sync

# pyds isn't on PyPI or bundled with the DeepStream SDK install
uv pip install https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_aarch64.whl

# sanity checks
uv run python -c "import gi; gi.require_version('Gst','1.0'); gi.require_version('GstRtspServer','1.0'); from gi.repository import Gst, GstRtspServer; import pyds; print('ok')"
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```

If `torch.cuda.is_available()` is `False`, or import fails with a missing
`libcudss.so`/`libcusparseLt.so` error, see
[CLAUDE.md § Notable gotchas](CLAUDE.md) — both are known, solved issues
with the Jetson-native PyTorch build.

## Building & running

```bash
# 1. Export the TensorRT engine directly on this device (must be built here,
#    not copied from another machine — see "Model" above)
uv run python export_engine.py

# 2. Build the custom bbox parser (must be compiled on-device, aarch64)
cd nvdsinfer_custom_impl_yolov8 && make && cd ..

# 3. Run headless + RTSP (default — for actual flight)
uv run main.py

# 3b. Run headless + RTSP + local bench display (needs a monitor physically
#     attached to the Jetson via HDMI/DP — does nothing over plain SSH)
uv run main.py --debug
```

View the live annotated stream from any machine on the same network:
```bash
ffplay rtsp://<jetson-ip>:8554/ds-stereo
```

### Production deployment

[`deploy/sterio-vision.service`](deploy/sterio-vision.service) starts the
pipeline headlessly on boot and restarts it on crash. Edit its
`User=`/`WorkingDirectory=` to match your setup, then:
```bash
sudo cp deploy/sterio-vision.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sterio-vision.service
journalctl -u sterio-vision -f   # tail logs
```
The engine export and custom-parser build are one-time (or per-model-change)
manual steps — the service only runs `main.py`.

## Verifying GPU utilization

```bash
tegrastats --interval 500
```
`GR3D_FREQ` reflects real GPU load from inference/hardware processing.
Note: some CPU usage is normal and expected for a DeepStream pipeline of
this shape (GStreamer thread/metadata orchestration across ~15 elements
per camera path, plus `nvdsosd`'s text-label rendering, which is
CPU-bound even in GPU OSD mode) — see
[CLAUDE.md § GPU utilization model](CLAUDE.md) for the full breakdown and
actual measured numbers from this device.

## Roadmap / known limitations

- **Distance is monocular**, not stereo — an interim known-height +
  focal-length estimate per camera, not yet using both cameras together.
  `src/distance.py` has a stub for a future disparity-based estimator once
  the two IMX296s are calibrated together (`cv2.stereoCalibrate` or
  similar).
- **No object tracker** — detections don't have persistent IDs across
  frames yet (`nvtracker` isn't in the pipeline). Needed before reliable
  multi-frame localization.
- **No MAVLink/geolocation output yet** — `on_detection()` in
  `src/probes.py` is the wired extension point; combining flight
  controller attitude/GPS with a per-detection camera-relative X/Y/Z would
  turn it into an absolute geolocated position. Needs flight controller
  connection details (ArduPilot/PX4, link type, camera mounting offset) to
  implement.
- **Engine batch size is 1** — the exported engine doesn't use
  `dynamic=True`, so `nvinfer` runs one inference call per camera per
  cycle rather than a single fused batch-of-2 call. Both still run
  entirely on TensorRT/GPU; this only affects fusion efficiency, not
  correctness or GPU-only-ness.
- **RTSP is MJPEG, not H.264** — this Orin Nano SOM has no hardware H.264
  encoder (NVENC is fused off on this SKU). MJPEG uses more bandwidth than
  H.264 would at the same quality; tune `RTSP_JPEG_QUALITY`/tiler
  resolution in `src/config.py` if the ground-station link is bandwidth
  constrained.
- **RTSP sends one composited (tiled) stream**, not two independent
  per-camera feeds — simplest for a single ground-station view; revisit
  if full-resolution independent per-camera streams are needed later.

## Notable gotchas solved during bring-up

Kept here briefly for quick reference; full detail and exact commands are
in [CLAUDE.md](CLAUDE.md):

- **`nvv4l2h264enc` doesn't exist on this board** — it's an Orin Nano, not
  Orin NX as originally assumed; Orin Nano has no hardware video encoder.
  Fixed by using `nvjpegenc` (hardware JPEG via the NVJPG engine) instead.
- **Standard PyPI `torch` doesn't work on Jetson** — it's built for
  desktop/server CUDA, not Tegra. Fixed by pinning `torch`/`torchvision` to
  the Jetson AI Lab index in `pyproject.toml`.
- **`libcudss.so.0` / cuBLAS conflicts** — the Jetson-native PyTorch build
  needs cuDSS installed as a system package (not pip), and the pip
  `nvidia-cudss-cu12` package must be avoided entirely since it pulls in
  an incompatible pip `nvidia-cublas-cu12` that silently shadows the
  correct system library.
- **`.engine` files aren't a bare TensorRT plan** — Ultralytics prepends a
  JSON metadata header that DeepStream doesn't know to skip, producing a
  deserialization error that looks identical to a genuine TensorRT
  version mismatch. `export_engine.py` strips it automatically.
- **TensorRT engines aren't portable across machines** — must be exported
  directly on the target Jetson.
- **`--debug`'s 3D plot needs a matplotlib workaround** — the OS ships an
  old `python3-matplotlib` (3.5.1) whose `mpl_toolkits` uses the legacy
  `pkg_resources` namespace style (a real `__init__.py`), which always
  wins as a "regular package" over the venv's own newer `mpl_toolkits`,
  regardless of `sys.path` order — breaking `Axes3D` with an
  `ImportError: cannot import name 'docstring'`. `src/debug_plot.py`
  works around it by hiding the system site-packages path from `sys.path`
  only while importing matplotlib.
