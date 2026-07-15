# sterio-vision-jetson-orin-nano

Real-time dual-camera object detection and monocular distance estimation for
a UAV payload, running on an NVIDIA Jetson Orin Nano via NVIDIA DeepStream.
Detection uses a YOLO26n TensorRT engine; all pixel- and tensor-heavy work
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
- ✅ YOLO26n TensorRT engine running inference on both streams via
  DeepStream's `nvinfer`, with a custom bbox parser for the engine's
  already-NMS'd (end-to-end) output head
- ✅ Per-detection monocular X/Y/Z distance estimate, overlaid on-screen and
  logged
- ✅ Composited (tiled) dual-camera view streamed live over RTSP for a
  ground-station viewer
- ✅ Optional local bench display for testing with a monitor attached
- ✅ systemd unit for unattended boot-time startup
- ✅ MAVLink telemetry link (IMU/GPS/compass) + PID-based object-follow
  mission, gated behind config + live flight mode — **implemented but
  untested against real flight hardware, defaults to a safe dry-run mode**,
  see [MAVLink / Mission](#mavlink--mission)
- ✅ Real chessboard stereo calibration done (`configs/stereo_calibration.yaml`,
  `src/calibration.py`) — `config.FOCAL_LENGTH_PX`/`STEREO_BASELINE_M` now use
  it, improving the existing monocular estimate. Full detail in
  [CLAUDE.md § Stereo calibration](CLAUDE.md)
- ✅ Camera detection → absolute lat/lon/altitude pipeline
  (`src/geolocation.py camera_to_latlon()`, real on-demand stereo depth via
  `src/stereo_depth.py`, camera→body mounting transform via
  `src/extrinsics.py`) — **written but unverified on real hardware, and
  the mounting transform is still placeholder/unmeasured values**, see
  [CLAUDE.md § Geolocation](CLAUDE.md)
- ⏳ Not yet done: object tracking (persistent IDs), ISR data-logging
  mission mode, multi-frame fusion for geolocation — see
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
| pymavlink | ≥2.4.40 (flight controller telemetry/command link, see [MAVLink / Mission](#mavlink--mission)) |
| scipy | ≥1.11 (camera→body→NED rotation chain, see [CLAUDE.md § Geolocation](CLAUDE.md)) |
| pymap3d | ≥3.0 (NED→geodetic lat/lon/alt, pure Python) |
| pyyaml | ≥6.0 (camera-body mounting transform config) |
| Package manager | [uv](https://docs.astral.sh/uv/) |

See [CLAUDE.md § Python package dependencies](CLAUDE.md) for the full,
kept-current table (exact resolved versions, aarch64/Jetson caveats per
package, and what's been evaluated and rejected — e.g. Open3D) — update
that table whenever a package is added via `uv`.

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
                       >- nvstreammux -> queue -> nvinfer(YOLO26n) -> nvmultistreamtiler
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
3. **Inference** — `nvinfer` runs the YOLO26n TensorRT engine on each
   frame (see [Model](#model) below for why a custom bbox parser is
   needed).
4. **Distance estimation** — a pad probe on `nvinfer`'s output (before
   tiling, so bbox coordinates are still per-camera) computes a monocular
   X/Y/Z estimate per detection and writes it into the on-screen label.
   This same probe is the extension point for future MAVLink/geolocation
   output (see [Roadmap](#roadmap--known-limitations)).
5. **Composite view** — `nvmultistreamtiler` combines both camera views
   into one side-by-side frame, sized to keep each tile's aspect ratio
   matching the camera's actual capture resolution (`config.TILER_SCALE`
   controls output size, not aspect ratio — see `src/config.py`);
   `nvdsosd` draws the detection boxes and labels onto it.
6. **Output** — the composited, annotated video is always streamed out
   over RTSP (MJPEG) for a ground-station viewer, and detection results
   are always logged. A local display branch is added only in `--debug`
   mode for bench testing with a monitor attached.

### Model

`models/yolo26n.engine` is a YOLO26n TensorRT engine (switched from
YOLOv8n — see "Notable gotchas" below), exported via:
```python
from ultralytics import YOLO
YOLO('yolo26n.pt').export(format='engine', device='0', half=True, workspace=4)
```
YOLO26 is natively NMS-free (`end2end=True` is its default export mode):
the output head is already final, post-NMS detections, shape `[1, 300, 6]`
(`[x1, y1, x2, y2, confidence, class_id]` per row, up to 300 detections).
DeepStream's built-in detector parser can't decode this layout either, so
[`nvdsinfer_custom_impl_yolo26/`](nvdsinfer_custom_impl_yolo26) provides a
custom one — it only applies a confidence threshold, no decode or NMS
needed. `cluster-mode=4` ("None") in
[`configs/pgie_yolo26n_config.txt`](configs/pgie_yolo26n_config.txt) is
deliberate: nvinfer must not re-cluster detections the model already
finalized.

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
| [`src/probes.py`](src/probes.py) | The per-frame metadata probe: filters to target classes, computes distance per detection, sets on-screen label text. `register_detection_listener()` lets other code subscribe to every detection (full rate) without editing this file — used by `src/debug_plot.py` and `src/mission.py`. `register_frame_status_provider()` draws a one-line on-screen HUD (MAVLink/mission status) in both the RTSP stream and `--debug`'s local display. |
| [`src/distance.py`](src/distance.py) | Monocular X/Y/Z estimator (known object height + focal length, always-on), plus `estimate_xyz_stereo()` — the pure-math half of real stereo depth, given a disparity value. |
| [`src/calibration.py`](src/calibration.py) | Loads `configs/stereo_calibration.yaml`; rectification maps for stereo depth. |
| [`src/stereo_depth.py`](src/stereo_depth.py) | On-demand (not per-frame) real stereo disparity for one detection at a time. See [CLAUDE.md § Geolocation](CLAUDE.md). |
| [`src/extrinsics.py`](src/extrinsics.py) | Loads `configs/camera_body_extrinsics.yaml`, the camera→body mounting transform. |
| [`src/geolocation.py`](src/geolocation.py) | `camera_to_latlon()` — the full camera → body → NED → lat/lon/alt chain. |
| [`src/debug_plot.py`](src/debug_plot.py) | `--debug`-only: live 3D scatter plot of detection X/Y/Z via matplotlib, colored as a depth heatmap (near=red, far=blue) with marker shape indicating camera. Needs a display attached to the Jetson and a working GUI backend (Tk/Qt/GTK) — same physical requirement as the `nveglglessink` bench-display branch; degrades to a harmless no-op with a warning if unavailable. |
| [`src/mavlink_link.py`](src/mavlink_link.py) | `MavlinkLink`: UART connection to the flight controller, background telemetry reader, IMU/GPS/compass getters, `send_velocity_setpoint()`. See [MAVLink / Mission](#mavlink--mission). |
| [`src/pid.py`](src/pid.py) | `PIDController` (generic) + `ObjectFollowController` (the drone-follow control loop). |
| [`src/mission.py`](src/mission.py) | `Mission`: gates FOLLOW/ISR behind `config.MISSION_MODE` + the flight controller's live flight mode. ISR is scaffolded, not yet implemented. |
| [`configs/pgie_yolo26n_config.txt`](configs/pgie_yolo26n_config.txt) | `nvinfer` configuration for the YOLO26n engine (batch size, confidence threshold, custom parser wiring, `cluster-mode=4` since the model's output is already NMS'd). |
| [`configs/labels_coco.txt`](configs/labels_coco.txt) | COCO-80 class names, index-matched to the engine's output classes. |
| [`nvdsinfer_custom_impl_yolo26/`](nvdsinfer_custom_impl_yolo26) | C++ custom bbox parser (`.cpp` + `Makefile`) for the engine's already-final (NMS-free/end-to-end) output head — must be compiled on-device. |
| [`deploy/sterio-vision.service`](deploy/sterio-vision.service) | systemd unit for unattended startup on boot. |
| `models/yolo26n.engine` | The exported TensorRT engine (not committed — built per-device, see below). |
| [`pyproject.toml`](pyproject.toml) | `uv`-managed project dependencies and environment notes. |

## Setup (one-time, on the Jetson)

```bash
# venv must be the system Python 3.10, with access to OS-provided GStreamer
# bindings (gi/PyGObject), since those aren't pip packages here
uv venv --system-site-packages --python /usr/bin/python3.10
uv sync   # also installs pyds (declared as a direct wheel URL in pyproject.toml,
          # not on PyPI -- do not install it separately, see CLAUDE.md)

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
cd nvdsinfer_custom_impl_yolo26 && make && cd ..

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

## MAVLink / Mission

A companion-computer link to the flight controller over UART
(`/dev/ttyTHS1`), plus a PID-based mission that can make the drone follow
a detected object. Entirely opt-in via `config.MISSION_MODE` (default
`"NONE"` — `main.py` doesn't even open the MAVLink connection unless a
mission mode is set).

- **`MavlinkLink`** ([src/mavlink_link.py](src/mavlink_link.py)) — three
  telemetry methods: `get_imu_telemetry()` (roll/pitch/yaw, gyro rate, xyz
  acceleration, groundspeed, relative altitude — no GPS needed),
  `get_gps_compass()` (position + compass heading, valid only when
  `.has_fix`), and `get_telemetry()` (merges both when a GPS fix exists,
  falls back to IMU-only otherwise — this is the main method to call).
- **`PIDController` + `ObjectFollowController`** ([src/pid.py](src/pid.py))
  — a generic PID and a follow controller running three of them to hold a
  configured standoff distance from the target, centered in frame.
- **`Mission`** ([src/mission.py](src/mission.py)) — the gate. FOLLOW only
  runs while the flight controller reports
  `config.FOLLOW_TRIGGER_FLIGHT_MODE` (default `GUIDED`); ISR is
  scaffolded (detects its trigger condition) but data logging itself is
  not yet implemented — that's the next milestone.

> [!WARNING]
> This drives real vehicle motion and **has not been validated against
> real flight hardware**. `config.FOLLOW_DRY_RUN` defaults to `True` —
> every setpoint is computed and logged, never sent to the flight
> controller. The PID gains and axis sign conventions are documented
> starting points, not tuned values. Before ever setting
> `FOLLOW_DRY_RUN=0`: validate telemetry reads first, then bench-test with
> props off, then a supervised low-altitude tethered test — never go
> straight to free flight. Full detail in
> [CLAUDE.md § MAVLink / Mission](CLAUDE.md).

## Roadmap / known limitations

- **The always-on per-frame path is still monocular** — real stereo depth
  now exists (`src/stereo_depth.py`) but only as an on-demand call for one
  detection at a time, not wired into the 60fps overlay/FOLLOW loop (the
  CPU cost of full block matching at that rate hasn't been measured).
  FOLLOW still holds station on the monocular Z estimate in the meantime.
- **Geolocation (`src/geolocation.py`) is unverified on real hardware** —
  implemented without device access. The main risk is
  `pyds.get_nvds_buf_surface()` against this pipeline's NV12 buffers
  (official DeepStream samples all use it against RGBA instead); the
  camera→body mounting transform in
  `configs/camera_body_extrinsics.yaml` is also still placeholder
  (unmeasured) values. See [CLAUDE.md § Geolocation](CLAUDE.md) before
  trusting any lat/lon this produces.
- **ISR mission mode is scaffolded but not implemented** — `Mission.
  _update_isr()` detects its trigger condition (flight mode + altitude)
  but doesn't log anything yet. Next milestone.
- **No object tracker** — detections don't have persistent IDs across
  frames yet (`nvtracker` isn't in the pipeline). Needed before reliable
  FOLLOW target selection with multiple similar objects in frame, or
  stable multi-frame localization.
- **FOLLOW is implemented but flight-untested** — see the warning above;
  defaults to a dry-run mode that never sends commands to the flight
  controller.
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

- **Switching YOLOv8n → YOLO26n was low-effort, contained to two files** —
  YOLO26 (Ultralytics, Jan 2026) is natively NMS-free: exported output is
  already-final detections `[1, 300, 6]` (`x1,y1,x2,y2,confidence,
  class_id]`), not YOLOv8's raw `[1, 84, 8400]` needing decode + NMS. The
  custom bbox parser got *simpler* (confidence-filter only, no per-anchor
  argmax loop), `cluster-mode` changed `2` → `4` ("None" — don't
  re-cluster already-final detections), and the export/pipeline/probe
  code needed zero changes since they all consume the same
  `NvDsObjectMeta` structure regardless of which model produced it.
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
