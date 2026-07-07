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

Run via `uv run python export_engine.py` (project root) rather than typing
this inline — same unchanged export call, wrapped with diagnostics (torch/
CUDA/TensorRT versions, free GPU memory, clear failure messages) and it
copies the result into `models/yolov8n.engine` for you. **Must be run
directly on this exact Jetson** — see the TensorRT portability note below.

**Ultralytics' `.engine` files are not a bare TensorRT plan.** They're
prefixed with a length-tagged JSON metadata blob (4-byte little-endian
length, then that many bytes of JSON: model description, stride, names,
imgsz, batch, ...) that Ultralytics' own `YOLO(...)` loader knows to skip.
`nvinfer` (and TensorRT's raw `deserialize_cuda_engine()`) don't know about
this wrapper and choke on the JSON text where they expect the plan's magic
tag — producing an `IRuntime::deserializeCudaEngine` /
"incompatible serialization version" error that looks *identical* to a
genuine TensorRT version or GPU-architecture mismatch (this cost real
debugging time before a hex dump of the file exposed it:
`2707 0000 7b22 6465 7363 7269 7074 696f 6e22...` = length `0x727`, then
`{"description"...`). `export_engine.py`'s `install_engine()` strips this
header automatically before writing to `models/yolov8n.engine` — if you
ever export by hand instead of via that script, either use `YOLO(...)` to
load it (which handles the wrapper) or strip it yourself:
```python
import struct
data = open("yolov8n.engine", "rb").read()
n = struct.unpack("<I", data[:4])[0]
open("models/yolov8n.engine", "wb").write(data[4 + n:])
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
| `nvdsosd` box/line drawing | GPU (`process-mode=1`) |
| `nvdsosd` **text** drawing (on-screen labels) | **CPU** — Pango/Cairo glyph rasterization; `process-mode` does not accelerate this in DeepStream's OSD plugin, this is a real, non-eliminable-while-keeping-readable-labels CPU cost |
| `nvjpegenc` | NVJPG (dedicated hardware JPEG codec engine — NOT NVENC, which this SKU lacks) |
| RTP packetization / `udpsink` / GLib mainloop / distance math in the probe | ARM CPU — small per-frame work |
| GStreamer pipeline orchestration (buffer/metadata passing, thread scheduling across ~15 elements x 2 camera paths) | ARM CPU |

**Measured on-device (2026-07-07, `tegrastats`, both cameras live, full
pipeline running):** `GR3D_FREQ` 65–87% (GPU genuinely busy — confirms
inference/compute is real), CPU **all 6 cores at 45–70%** each. That CPU
number is real and higher than this doc originally claimed ("trivial
control-plane only") — corrected here rather than left wrong. Two
contributors identified: (1) `on_detection()` in `src/probes.py` was
`print()`-ing every detection at full pipeline framerate (up to 60fps x 2
cameras) — fixed, now throttled to 2 Hz, since that was only ever a stdout
placeholder ahead of real telemetry, not core functionality; (2) `nvdsosd`
text-label rasterization (see table row above) plus normal GStreamer
thread/metadata orchestration overhead across this many elements is a
genuine, expected CPU floor for a DeepStream pipeline of this shape — not
a bug, and not something eliminable via a config flag while keeping
human-readable on-screen labels. The pipeline is not CPU-*bottlenecked*
(cores aren't pegged at 100%, detection keeps up in real time), but "100%
GPU / ~0% CPU" was an overstatement — all pixel- and tensor-heavy work
(capture, inference, colorspace/scale conversion, box/line OSD draw, JPEG
encode) does run on GPU or a dedicated hardware block, which is what
actually determines whether the pipeline lags; a real CPU floor from
orchestration + text rendering remains on top of that.

## File map

```
CLAUDE.md                                  this file
pyproject.toml                             uv project + deps
main.py                                    entry point, arg parsing, GLib mainloop, bus/error handling
src/config.py                              all tunables (camera, model, classes, RTSP, tiler)
src/pipeline.py                            GStreamer/DeepStream pipeline construction
src/probes.py                              per-frame metadata probe: class filter, distance calc, OSD text; register_detection_listener() is the subscribe point for every Detection (full rate)
src/distance.py                            monocular X/Y/Z estimator (+ stereo placeholder)
src/debug_plot.py                          --debug-only live 3D scatter plot of detection X/Y/Z (matplotlib); needs a display + GUI backend, same caveat as nveglglessink
src/mavlink_link.py                        MavlinkLink: UART connection to the flight controller, IMU/GPS/compass telemetry getters, send_velocity_setpoint()
src/pid.py                                 PIDController (generic) + ObjectFollowController (drone-follow control loop built on it) -- see "MAVLink / Mission" below
src/mission.py                             Mission: gates FOLLOW/ISR behind config.MISSION_MODE + the FC's live flight mode; ISR is scaffolded, not yet implemented
configs/pgie_yolov8n_config.txt            nvinfer config for the YOLOv8n engine
configs/labels_coco.txt                    COCO-80 class names, index-matched to the engine's output
nvdsinfer_custom_impl_yolov8/*.cpp/Makefile   custom bbox parser for the raw (no-NMS) YOLOv8 output head
export_engine.py                           runs the (unchanged) .pt -> .engine export on-device with diagnostics, copies result into models/
models/yolov8n.engine                      <- exported engine lives here (not committed)
```

## Build & run (on the Jetson)

```bash
# 1. Export the engine directly on this device (TensorRT engines are not
#    portable across machines -- see "Model export" above). Copies the
#    result into models/yolov8n.engine automatically.
uv run python export_engine.py

# 2. Build the custom TensorRT-output parser (must be built on-device, aarch64)
cd nvdsinfer_custom_impl_yolov8 && make && cd ..

# 3. Run — headless + RTSP only (default, for flight)
uv run main.py

# 3b. Run — headless + RTSP + local bench display (dev bench, monitor attached)
uv run main.py --debug
# or: DS_DEBUG=1 uv run main.py
```

Note: `--debug`'s local display (`nveglglessink`) needs an actual monitor
physically connected to the Jetson (HDMI/DP) with a desktop session —
it does nothing useful over a plain SSH session (no `DISPLAY`/compositor to
attach to). Over SSH, just view the always-on RTSP stream from your own
machine instead (`ffplay rtsp://<jetson-ip>:8554/ds-stereo`) — same tiled
dual-camera view with detection overlays, no Jetson-side display needed.

### Production deployment (systemd)

`deploy/sterio-vision.service` runs the pipeline headless on boot,
restarts on crash, and waits on `nvargus-daemon` (camera ISP) first. It
calls the venv's `python` directly rather than `uv run`, so boot doesn't
depend on `uv` re-checking the lockfile against the network — there may be
no internet on this device in the field. Edit `User=`/`WorkingDirectory=`
in the file if they don't match your actual username/path, then:
```bash
sudo cp deploy/sterio-vision.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sterio-vision.service
journalctl -u sterio-vision -f   # tail logs
```
Re-run `export_engine.py` and the `nvdsinfer_custom_impl_yolov8` build
manually first if not already done — the service does not build/export
anything itself, it only runs `main.py`.

View the ground-station stream from a laptop on the same network:
```bash
ffplay rtsp://<jetson-ip>:8554/ds-stereo
# or: gst-launch-1.0 rtspsrc location=rtsp://<jetson-ip>:8554/ds-stereo ! decodebin ! autovideosink
```

## MAVLink / Mission (FOLLOW implemented, ISR scaffolded)

Companion-computer link to the flight controller over UART
(`config.MAVLINK_DEVICE = /dev/ttyTHS1`, requires `SERIALx_PROTOCOL=2` and
a matching `SERIALx_BAUD` on that port). Entirely opt-in:
`config.MISSION_MODE` defaults to `"NONE"`, in which case `main.py` never
even opens the MAVLink connection — zero impact on the camera/detection
pipeline for anyone not using this.

- **`src/mavlink_link.py` — `MavlinkLink`**: opens the connection, runs a
  background reader thread caching the latest `ATTITUDE`/`RAW_IMU`/
  `VFR_HUD`/`GPS_RAW_INT`/`GLOBAL_POSITION_INT`/`HEARTBEAT` messages.
  Three telemetry methods, matching the intended usage: `get_imu_telemetry()`
  (IMU-only: roll/pitch/yaw, gyro rate, xyz accel, groundspeed, relative
  altitude — never needs GPS), `get_gps_compass()` (position + compass
  heading, only trust the fields when `.has_fix` is True), and
  `get_telemetry()` (the main entry point: merges IMU + GPS/compass when a
  fix is available, falls back to IMU-only otherwise). Also
  `get_flight_mode()` (drives the mission gating below) and
  `send_velocity_setpoint(vx, vy, vz)` (body-frame m/s via
  `SET_POSITION_TARGET_LOCAL_NED`).
- **`src/pid.py` — `PIDController` + `ObjectFollowController`**: a generic
  clamped/anti-windup PID, and a follow controller that runs three of them
  (lateral/vertical/forward) against a detection's camera-relative X/Y/Z to
  hold `config.FOLLOW_TARGET_DISTANCE_M` from `config.FOLLOW_TARGET_CLASS`,
  centered in frame.
- **`src/mission.py` — `Mission`**: the gate. `on_detection()` feeds the
  follow controller (wire into `probes.register_detection_listener()`);
  `update()` (driven by `GLib.timeout_add` from `main.py`, main thread
  only) starts/stops FOLLOW based on whether the flight controller is
  *currently* in `config.FOLLOW_TRIGGER_FLIGHT_MODE` — the mission never
  starts just because the process is running. ISR is scaffolded
  (`_update_isr()` checks `config.ISR_TRIGGER_FLIGHT_MODE` +
  `ISR_TRIGGER_ALTITUDE_M` and prints when triggered) but **CSV/JSON
  logging itself is not yet implemented** — next milestone.

**Safety, read before ever touching `FOLLOW_DRY_RUN`:** this drives real
vehicle motion and has not been validated against real flight hardware.
`config.FOLLOW_DRY_RUN` defaults to `True` — every setpoint is computed
and logged (`[follow] DRY RUN setpoint ...`) but never sent to the flight
controller. The gains in `config.FOLLOW_PID_*` and the axis sign
conventions in `ObjectFollowController.update()` are documented starting
points, not tuned/validated values. Before ever setting
`FOLLOW_DRY_RUN=0`: validate telemetry reads first (safe, no motion
involved), then bench-test with props off, then a supervised low-altitude
tethered GUIDED-mode test — do not go straight to free flight.

No object tracker exists yet (see below), so `ObjectFollowController` just
follows the latest detection matching `FOLLOW_TARGET_CLASS` each cycle —
with more than one matching object in frame, which one gets followed can
change frame to frame.

## Known limitations / Next steps

- **Distance estimation is monocular** (known-height + focal-length,
  per-camera, ported from the original prototype). `src/distance.py` has a
  stub `estimate_xyz_stereo()` — once both IMX296s are calibrated
  (intrinsics + baseline via `cv2.stereoCalibrate` or similar), replace the
  call in `src/probes.py` with a disparity-based depth lookup. This also
  directly limits FOLLOW's accuracy, since it holds station based on this
  same monocular Z estimate.
- **No object tracker yet** (no `nvtracker` in the pipeline) — detections
  aren't assigned persistent IDs across frames. Add `nvtracker` between
  `pgie` and `tiler` when ID persistence is needed (e.g. for stable
  FOLLOW target selection or localization output).
- **ISR mission mode is scaffolded but not implemented** — `Mission.
  _update_isr()` detects the trigger condition (flight mode + altitude)
  but does not yet log anything. Next milestone per project staging.
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
