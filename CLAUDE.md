# sterio-vision-jetson-orin-nano — Jetson Orin Nano Dual-Camera DeepStream Pipeline

UAV payload: real-time YOLO26n detection + monocular distance estimation over
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

### Python package dependencies (`pyproject.toml` — keep this table in sync)

Whenever a package is added via `uv`, add it here too, with the version
actually resolved on-device and any aarch64/Jetson-specific caveat —
this table is the fast way to check compatibility before adding another
one, instead of re-discovering the same class of problem from scratch.

| Package | Version (resolved on-device) | Notes |
|---|---|---|
| `torch` | 2.11.0 | **Jetson-native build** from the [Jetson AI Lab index](https://pypi.jetson-ai-lab.io/jp6/cu126) (`[tool.uv.sources]` in `pyproject.toml`), NOT default PyPI — see "Model export" below for why |
| `torchvision` | 0.26.0 | Same Jetson AI Lab index as `torch` |
| `ultralytics` | ≥8.4.87 | Export-only, default PyPI, no aarch64 issues (pure Python + already-solved deps) |
| `onnx` | ≥1.22.0 | Export-only, default PyPI, no aarch64 issues |
| `pymavlink` | ≥2.4.40 | Default PyPI, pure Python + a small C extension — no aarch64/Jetson-specific issues encountered |
| `pyds` | 1.2.0 | **Not on PyPI.** Declared as a direct wheel URL (`pyds @ https://...`) from the [deepstream_python_apps v1.2.0](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/tag/v1.2.0) release, matching DS 7.1 + cp310 + aarch64. Must be declared this way, not installed via a separate manual `uv pip install <url>` — anything not in `dependencies` is untracked and gets removed by the next `uv sync` (happened once). Also: because this wheel is `cp310`/`linux_aarch64`-only, adding it as a real dependency broke `uv sync` outright once (`requires-python = ">=3.10"` alone lets uv's universal resolver try — and fail — to also solve for e.g. Python 3.14 on Windows, where no `pyds` wheel exists at all, even though nothing runs this project there). Fixed by pinning `requires-python = ">=3.10,<3.11"` and adding `[tool.uv] environments = ["sys_platform == 'linux' and platform_machine == 'aarch64'"]` — both narrow uv's resolution to what this project actually is. |
| `matplotlib` | 3.10.9 (transitive, via `ultralytics`) | Works, but needed a runtime workaround for `mpl_toolkits`/`Axes3D` — see `src/debug_plot.py` and README.md's "Notable gotchas" section |
| `opencv-python` | ≥4.8.0 | Already installed transitively via `ultralytics` (confirmed working on this device), now also declared explicitly since `src/calibration.py` genuinely depends on it — no new aarch64 risk |
| `scipy` | ≥1.11 | For `src/geolocation.py`'s camera→body→NED rotation chain (`scipy.spatial.transform.Rotation`). Mainstream scientific-Python package with long-standing manylinux aarch64 wheels — low risk by this project's established pattern (see note below the table), but **added for the geolocation feature and not yet actually run on-device** — confirm with `uv sync` + the sanity check below before trusting |
| `pymap3d` | ≥3.0 | For `src/geolocation.py`'s NED→geodetic step (`pymap3d.ned2geodetic`). Pure Python, numpy-only, no compiled extensions — no platform-specific wheel risk at all, unlike `scipy`/`opencv-python`/`torch`. Also unverified on-device yet, but that risk is essentially nil here given it's pure Python |
| `pyyaml` | ≥6.0 | For `src/extrinsics.py`'s `configs/camera_body_extrinsics.yaml` loader. Almost certainly already present transitively (`ultralytics` uses YAML internally) — declared explicitly since `extrinsics.py` genuinely depends on it, same reasoning as `opencv-python` |
| **`open3d` — evaluated, rejected** | — | **Do not re-add without reading this first.** Official Open3D on PyPI has never published a Linux aarch64 wheel for Python 3.10 (checked the full release history via PyPI's API — 0.16+ added cp310 but only for x86_64/macOS/Windows; earlier versions with `manylinux2014_aarch64` wheels topped out at Python 3.9). A third-party `open3d-unofficial-arm` wheel does exist for this exact platform/Python combo, and building from source is the officially-documented Jetson path — both were rejected as disproportionate for a debug-only visualization (unaudited binary vs. a 1-3+ hour build with known Jetson-specific build issues). The depth-heatmap feature was folded into the existing `src/debug_plot.py` (matplotlib) instead — see below. |

Every package above other than the Jetson-native `torch`/`torchvision` pair
resolves fine from default PyPI with no special handling — the aarch64
wheel problems in this project have consistently been about *GPU/native
binary* packages (torch, its transitive CUDA libs, the system-vs-pip
matplotlib conflict), not pure-Python ones. Keep that pattern in mind
before assuming a new package will be simple *or* assuming it won't.

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
uv sync   # also installs pyds -- declared as a direct wheel URL in pyproject.toml,
          # NOT a separate manual `uv pip install` step (that gets removed by the
          # next `uv sync` since it wouldn't be a declared dependency -- happened once)

# Sanity check before running main.py:
uv run python -c "import gi; gi.require_version('Gst','1.0'); \
    gi.require_version('GstRtspServer','1.0'); \
    from gi.repository import Gst, GstRtspServer; import pyds; print('ok')"

# Sanity check for the geolocation feature's new dependencies (not yet
# run on this device -- see the package table above):
uv run python -c "import scipy, pymap3d, yaml; print('ok')"
```
Do **not** add `pygobject`/`pycairo` as a pip/uv dependency — that tries to
compile pycairo from source against `libcairo2-dev` (usually missing) and,
even if it built, would be a second ABI-mismatched copy shadowing the
working system one. If the `GstRtspServer` import above fails, install the
OS package instead: `sudo apt install gir1.2-gst-rtsp-server-1.0`.

## Model export

**Switched from YOLOv8n to YOLO26n** (Ultralytics, released January 2026)
— YOLO26 is natively NMS-free (end-to-end one-to-one head), which changed
the custom bbox parser and PGIE config; everything else in this project
was unaffected. See README.md's "Notable gotchas" section for the
before/after and why the swap was low-effort. `yolo26n.pt` isn't bundled
with this repo — get
it onto the device yourself (e.g. `YOLO('yolo26n.pt')` in a throwaway
script lets Ultralytics auto-download it) before running the export.

```python
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
model.export(format='engine', device='0', half=True, workspace=4)
```

Run via `uv run python export_engine.py` (project root) rather than typing
this inline — same unchanged export call, wrapped with diagnostics (torch/
CUDA/TensorRT versions, free GPU memory, clear failure messages) and it
copies the result into `models/yolo26n.engine` for you. **Must be run
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
header automatically before writing to `models/yolo26n.engine` — if you
ever export by hand instead of via that script, either use `YOLO(...)` to
load it (which handles the wrapper) or strip it yourself:
```python
import struct
data = open("yolo26n.engine", "rb").read()
n = struct.unpack("<I", data[:4])[0]
open("models/yolo26n.engine", "wb").write(data[4 + n:])
```

Consequences this project's DeepStream config is built around:
- **FP16**, default **640x640** input, **COCO-80** classes.
- YOLO26's `end2end=True` export is its **default** (no extra export
  flags needed) → the engine's output head is already final, post-NMS
  detections: shape `[1, 300, 6]` (`[x1, y1, x2, y2, confidence,
  class_id]` per row, up to 300 detections). This is why the **custom
  bbox parser** (`nvdsinfer_custom_impl_yolo26/`) only applies a
  confidence threshold — there's no decode or NMS left to do, unlike
  YOLOv8's raw `[1, 84, 8400]` head this project used previously.
  `cluster-mode=4` ("None") in the PGIE config is deliberate — nvinfer
  must not re-cluster detections the model already finalized.
- No `dynamic=True` → **the engine's max batch size is fixed at 1.**
  `nvinfer`'s `batch-size` in `configs/pgie_yolo26n_config.txt` is therefore
  set to `1`, and it runs one inference call per camera per muxer cycle
  instead of one fused batch-of-2 call. Both calls still run entirely on
  TensorRT/GPU — this only affects *fusion*, not GPU-only-ness. If fused
  dual-camera throughput becomes the bottleneck later, re-export with
  `dynamic=True` and rebuild with an explicit batch=2 profile.

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

- A pad probe on `nvinfer`'s **src pad** (before tiling, while bbox coords
  are still per-camera) computes X/Y/Z per detection and sets the OSD label.
  This is also the **headless output path** — see `on_detection()` in
  `src/probes.py`, the extension point for MAVLink/geolocation telemetry.
- The tiler composites both camera views into one side-by-side frame
  (`config.TILER_WIDTH/HEIGHT`, derived from `CAPTURE_WIDTH/HEIGHT` and
  `TILER_SCALE` so each tile always keeps the camera's actual aspect
  ratio — see `src/config.py`) and remaps object metadata (bbox + text)
  into tile coordinates automatically, so OSD/RTSP show both cameras with
  correct overlays.
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
| `nvinfer` (YOLO26n) | TensorRT on GPU (CUDA cores) |
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
src/probes.py                              per-frame metadata probe: class filter, distance calc, OSD text; register_detection_listener() is the subscribe point for every Detection (full rate); register_frame_status_provider() draws an on-screen HUD line (MAVLink/mission status); register_obstacle_listener() is the subscribe point for (bin_distances_m, bin_valid_mask) readings, throttled to config.AVOID_UPDATE_INTERVAL_S -- see "Obstacle avoidance" below
src/distance.py                            monocular X/Y/Z estimator (always-on) + estimate_xyz_stereo() (pure math, given a disparity value); SmoothedDetection averages the on-screen label over config.DISPLAY_AVERAGE_WINDOW_S, presentational only -- FOLLOW/debug plot still get the raw per-frame estimate
src/calibration.py                         loads configs/stereo_calibration.yaml (cv2.FileStorage); StereoCalibration dataclass + lazy rectify_maps() -- see "Stereo calibration" below
src/stereo_depth.py                        on-demand (NOT per-frame) real stereo disparity for one detection at a time -- pyds buffer -> numpy, rectification, cv2.StereoSGBM on an ROI; see "Geolocation" below
src/obstacle_depth.py                      band-wide (NOT per-detection) stereo disparity -> per-bin depth for MISSION_MODE=="AVOID" -- one cv2.StereoSGBM pass over a center vertical band, RealSense-style decimation/spatial/hole-filling stages, reduced to config.AVOID_NUM_BINS scalars; see "Obstacle avoidance" below
src/avoidance.py                           NOT a steering controller -- BinDistanceSmoother (temporal EMA stage) + build_obstacle_distance() (pure) + ObstacleAvoidance (streams MAVLink OBSTACLE_DISTANCE to ArduPilot's own OA_TYPE); see "Obstacle avoidance" below
src/extrinsics.py                          loads configs/camera_body_extrinsics.yaml -- fixed camera->body mounting transform (rotation + lever-arm); see "Geolocation" below
src/geolocation.py                         camera_to_latlon(): the full camera -> body -> NED -> lat/lon/alt chain; see "Geolocation" below
src/debug_plot.py                          --debug-only live 3D scatter plot of detection X/Y/Z (matplotlib), colored as a depth heatmap (near=red, far=blue), marker shape = camera; needs a display + GUI backend, same caveat as nveglglessink
src/mavlink_link.py                        MavlinkLink: UART connection to the flight controller, IMU/GPS/compass telemetry getters, get_interpolated_attitude() (rolling ATTITUDE buffer, see "Geolocation" below), send_velocity_setpoint() (FOLLOW), send_obstacle_distance() (AVOID)
src/pid.py                                 PIDController (generic) + ObjectFollowController (drone-follow control loop built on it) -- see "MAVLink / Mission" below
src/mission.py                             Mission: gates FOLLOW/ISR/AVOID behind config.MISSION_MODE (FOLLOW/ISR additionally gated on the FC's live flight mode; AVOID is not, see "Obstacle avoidance" below); ISR is scaffolded, not yet implemented
configs/pgie_yolo26n_config.txt            nvinfer config for the YOLO26n engine
configs/labels_coco.txt                    COCO-80 class names, index-matched to the engine's output
configs/stereo_calibration.yaml            compact chessboard stereo calibration (cv2.stereoCalibrate/stereoRectify output, no baked-in remap tables) -- see "Stereo calibration" below
configs/camera_body_extrinsics.yaml        camera->body mounting transform (placeholder/unmeasured) -- see "Geolocation" below
nvdsinfer_custom_impl_yolo26/*.cpp/Makefile   custom bbox parser for YOLO26's NMS-free (already-final) output head -- confidence-filter only, no decode/NMS
export_engine.py                           runs the (unchanged) .pt -> .engine export on-device with diagnostics, copies result into models/
models/yolo26n.engine                      <- exported engine lives here (not committed)
```

## Build & run (on the Jetson)

```bash
# 1. Export the engine directly on this device (TensorRT engines are not
#    portable across machines -- see "Model export" above). Copies the
#    result into models/yolo26n.engine automatically.
uv run python export_engine.py

# 2. Build the custom TensorRT-output parser (must be built on-device, aarch64)
cd nvdsinfer_custom_impl_yolo26 && make && cd ..

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
Re-run `export_engine.py` and the `nvdsinfer_custom_impl_yolo26` build
manually first if not already done — the service does not build/export
anything itself, it only runs `main.py`.

View the ground-station stream from a laptop on the same network:
```bash
ffplay rtsp://<jetson-ip>:8554/ds-stereo
# or: gst-launch-1.0 rtspsrc location=rtsp://<jetson-ip>:8554/ds-stereo ! decodebin ! autovideosink
```

## Stereo calibration (done — real chessboard data, not a ruler measurement)

`configs/stereo_calibration.yaml` holds the output of a real chessboard
stereo calibration (`cv2.stereoCalibrate` + `cv2.stereoRectify`) for this
exact camera rig: both cameras' intrinsics (`camera_matrix_left/right`) +
distortion (`dist_coeffs_left/right`), stereo extrinsics (`R`, `T`, `E`,
`F`), rectification (`R1`, `R2`, `P1`, `P2`, `Q`), and:

| Parameter | Value |
|---|---|
| Image size | 1456×1088 |
| Chessboard | 8×5 squares, 30mm each |
| Baseline | 0.094319m (94.3mm) — supersedes the earlier ruler measurement (0.094m) |
| Focal length | 921.5871px (post-rectification, shared by both cameras via `P1`/`P2`) |
| Stereo RMS reprojection error | 1.42px |

`config.FOCAL_LENGTH_PX`/`STEREO_BASELINE_M` were updated to these values
— this directly improves the existing monocular estimator in
`src/distance.py` even before real stereo disparity is implemented, since
it was previously using a rough 800.0px guess.

**Only the compact parameters are committed**, not the full calibration
output. The original file (wherever the calibration script wrote it, kept
local, gitignored via `/stereo_calibration.yaml`) is ~512KB/739K lines
because it also bakes in the precomputed undistort/rectify remap tables
(`map_left_1/2`, `map_right_1/2`) — those are 100% derivable at runtime
from the compact parameters via `cv2.initUndistortRectifyMap()`
(`src/calibration.py`'s `StereoCalibration.rectify_maps()` does exactly
this, lazily, cheap at ms-scale), so persisting ~500KB of precomputed
lookup tables in git would be pure bloat with zero information gain.

`src/calibration.py`'s `load()` reads the compact YAML via
`cv2.FileStorage` (required for OpenCV's `!!opencv-matrix` YAML tags —
plain `yaml.safe_load()` can't parse this format) and returns a
`StereoCalibration` dataclass with every matrix above as a numpy array,
ready for whatever consumes it next (real stereo disparity in
`src/distance.py`, or an ORB-SLAM3 settings file). Nothing in the runtime
pipeline (`main.py`) loads this yet — it's data + a loader, not wired into
`src/probes.py` in this pass. `opencv-python` was added as an explicit
dependency for this (it was already installed transitively via
`ultralytics`, so no new aarch64 risk — see the package table above).

This directly unblocks two roadmap items that were both waiting on it:
real stereo depth (replacing monocular `estimate_xyz()` with a disparity-
based `estimate_xyz_stereo()`), and step 1 of integrating ORB-SLAM3 or
similar visual(-inertial) SLAM, which needs exactly this kind of
calibration data for its stereo mode. Neither is built yet — see "Known
limitations / Next steps" below for what's still required for each.

## Geolocation (camera detection -> absolute lat/lon/altitude)

`src/geolocation.py`'s `camera_to_latlon(detection, telemetry, extrinsics,
xyz_cam_override=None, attitude_override=None) -> Optional[GeoPosition]`
chains: camera frame -> body frame -> local NED -> geodetic lat/lon/alt.
Not wired into `main.py`/`src/probes.py`'s always-on per-frame path — it's
callable infrastructure, meant to be invoked on demand (e.g. once per
FOLLOW-locked target, or per ISR log tick once that's implemented), for
two reasons: real stereo depth (below) is genuine CPU work this project
has otherwise avoided at 60fps, and geolocation itself doesn't need to run
at video framerate.

**The chain, in order:**

1. **Camera -> body** (`src/extrinsics.py`, `configs/camera_body_extrinsics.yaml`):
   a fixed rotation (`mount_roll/pitch/yaw_deg`, aerospace ZYX) + lever-arm
   translation (`lever_arm_x/y/z_m`) describing how the camera is
   physically mounted relative to the flight controller's body frame (x
   forward, y right, z down). **Currently placeholder/unmeasured values
   (identity mount, zero lever-arm)** — every lat/lon this pipeline
   produces is silently biased by whatever the true mounting angle is
   until this is measured and validated (point the rig at a known-bearing/
   known-range ground landmark with the drone stationary and GPS-fixed,
   compare, adjust). Do this before trusting FOLLOW/ISR logging output
   downstream of this chain.
2. **Real stereo depth** (`src/stereo_depth.py`), optional — pass its
   result as `camera_to_latlon`'s `xyz_cam_override`; omitting it falls
   back to the detection's own monocular `x_m/y_m/z_m`
   (`src/distance.py estimate_xyz()`, already used everywhere else in the
   pipeline). Deliberately an **ROI-based, on-demand** disparity search
   for one detection at a time, not a full-frame dense disparity map:
   - `extract_luma_plane()` pulls one camera's raw pixels out of the
     DeepStream NVMM buffer into a CPU-accessible numpy array via
     `pyds.get_nvds_buf_surface()`. **Unverified on real hardware** — every
     official DeepStream Python sample uses this against RGBA buffers
     (after an explicit `nvvideoconvert`); this project's PGIE src-pad
     probe runs on raw NV12 buffers instead (see `src/pipeline.py`), and
     whether `get_nvds_buf_surface()` hands back a usable array in that
     format, and at what shape/stride, has not been confirmed on-device.
     If it doesn't, the fix is either converting to RGBA first (cheap on
     GPU, needs a spare tee branch) or reading the NV12 buffer at a
     different assumed layout.
   - `rectify_bbox_left()` maps a detection's bbox from the original
     (distorted) image into rectified-image coordinates via
     `cv2.undistortPoints(..., R=R1, P=P1)` — not a plain coordinate
     scale, since undistortion is a nonlinear per-pixel warp.
   - `roi_disparity()` crops just wide enough to cover the bbox plus
     `config.STEREO_MAX_DISPARITY_PX` (must stay a multiple of 16) and
     runs `cv2.StereoSGBM` on that crop only, returning the median
     disparity over the bbox's own pixels (robust to a few bad/occluded
     matches). Returns `None` on failure (object closer than the
     configured max-disparity's minimum depth, near a frame edge, no
     texture) — callers fall back to monocular, this is expected, not a
     bug.
   - Only meaningful for `source_id==0` (left camera, matching
     `src/calibration.py`'s left-as-reference convention) — right-camera
     detections stay monocular.
   - `estimate_stereo_xyz()` ties the above together end to end; the pure
     math (disparity -> X/Y/Z given the rectified principal point/focal
     length) is `src/distance.py`'s `estimate_xyz_stereo()`.
3. **Body -> NED**: `scipy.spatial.transform.Rotation.from_euler('ZYX',
   [yaw, pitch, roll], degrees=True)` applied to the body-frame vector,
   using the drone's own attitude — either `telemetry.imu` (whatever
   `MavlinkLink.get_telemetry()` last cached) or, better,
   `attitude_override` from `MavlinkLink.get_interpolated_attitude(
   frame_capture_time)` (below).
4. **NED -> geodetic**: `pymap3d.ned2geodetic(n, e, d, lat0, lon0, h0)` — a
   flat-earth/local-tangent-plane approximation anchored at the drone's
   *current* GPS fix (`telemetry.gps.latitude_deg/longitude_deg/
   altitude_msl_m`). Fine at FOLLOW/ISR ranges; would need a proper
   ECEF-based approach if this pipeline is ever used at ranges where
   earth curvature matters.
5. **Fix-gating**: `camera_to_latlon()` returns `None` outright if
   `telemetry.gps is None` or `not telemetry.gps.has_fix` — this reuses
   `MavlinkLink.get_telemetry()`'s existing IMU-only-fallback logic, no
   new gating code was needed here.
6. **Timestamp buffering** (`src/mavlink_link.py`): unlike every other
   `get_*()` method on `MavlinkLink` (which only ever track the single
   latest message of each type), ATTITUDE messages are additionally kept
   in a short rolling buffer (`config.MAVLINK_ATTITUDE_BUFFER_S`, default
   2.0s). `get_interpolated_attitude(timestamp)` linearly interpolates
   roll/pitch/yaw to a specific timestamp (e.g. a camera frame's capture
   time) instead of returning whatever ATTITUDE message arrived most
   recently relative to whenever it's called — handles yaw's ±180°
   wraparound correctly (shortest-path interpolation, not a naive
   subtraction). GPS position deliberately does **not** get this
   treatment — it changes slowly enough that nearest/latest
   (`get_gps_compass()`) is an accepted simplification, not an oversight.

**Not yet done** (explicitly deferred, per the project's own build order):
multi-frame fusion across repeated sightings of the same object — that
needs an object tracker (`nvtracker`, see "Known limitations" below) to
even define "the same object" across frames, and neither exists yet.

**New dependencies**: `scipy`, `pymap3d`, `pyyaml` — see the package
table above. Genuinely new/unverified on this exact device (unlike most
of this project's dependencies, which were already confirmed working
before being declared) — run the sanity check in that section and `uv
sync` before relying on any of this.

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
  `status_text()` returns a one-line summary (`MAVLINK:CONNECTED
  MODE:FOLLOW FC:GUIDED MISSION:ACTIVE`) — `update()` prints it to the
  console every `_STATUS_LOG_INTERVAL_S` (1s), and `main.py` also wires it
  into `probes.register_frame_status_provider()`, which draws the same
  line as a persistent on-screen HUD in the top-left corner of the video
  (visible in both the RTSP stream and `--debug`'s local display) via a
  second probe on `nvdsosd`'s sink pad (`osd_sink_pad_status_probe()` in
  `src/probes.py` — runs after tiling, so there's exactly one composited
  frame to draw the HUD on, unlike the per-object probe on `nvinfer`'s src
  pad which runs before tiling). `update()` also prints immediately on a
  link-health or flight-mode *change* (`[mavlink] heartbeat LOST...`,
  `[mavlink] flight mode changed: STABILIZE -> GUIDED`), on top of the
  periodic line, since those are the events worth seeing right away.
- **On-screen target lock** — while FOLLOW is active, any detection
  matching `config.FOLLOW_TARGET_CLASS` is drawn with a **red** box (vs.
  the normal green), a small red center-dot marker, and its label switches
  to `TARGET LOCKED | <class> | Dist: X.Xm` (`pgie_src_pad_buffer_probe()`
  in `src/probes.py`, gated by `register_follow_active_query()`). No
  tracker exists yet, so with more than one matching object in frame, all
  of them get marked — only one actually drives
  `ObjectFollowController`. The center-dot is added as frame-level display
  meta *before* the tiler (same as the box/text); this hasn't yet been
  visually confirmed to survive the tiler's coordinate remap the same way
  `rect_params`/`text_params` reliably do — check this on-device.
- Per-detection X/Y/Z is **no longer printed to stdout** on every
  detection (it was redundant with the video overlay and the `--debug` 3D
  plot, and was the CPU-cost fix from earlier in this project's history) —
  `src/mission.py`'s status log is the console output now.

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

## Obstacle avoidance (`MISSION_MODE="AVOID"`)

Unlike FOLLOW, this mode does **not** compute a steering command on this
companion computer. It streams MAVLink `OBSTACLE_DISTANCE` (built from a
low-rate stereo depth read) to ArduPilot's own proximity/object-avoidance
system, which decides whether/how to bend the flight path
(`OA_TYPE`=BendyRuler or Dijkstra's). This mirrors how a RealSense-class
depth camera normally integrates with ArduPilot — the companion computer's
job is producing a clean, correctly-shaped depth reading, not doing the
avoidance itself.

**FC-side setup (outside this repo)**: `PRX1_TYPE` = MAVLink,
`OA_TYPE` = BendyRuler or Dijkstra's, `OA_DB_SIZE`/`OA_DB_EXPIRE` as
needed — analogous to the `SERIALx_PROTOCOL`/`SERIALx_BAUD` prerequisite
already documented above for this project's MAVLink link.

**Depth pipeline (`src/obstacle_depth.py`)**: unlike `src/stereo_depth.py`
(small ROI, one YOLO detection at a time, on demand), this runs ONE
`cv2.StereoSGBM` pass over the full width of a center vertical band
(`config.AVOID_BIN_HEIGHT_FRACTION`) of the rectified frame, then reduces
that to `config.AVOID_NUM_BINS` (5) scalar readings — bin *after*
computing depth, not one matcher call per bin. Maps RealSense's 4-stage
post-processing filter chain onto this coarser shape:
- **Decimation** — downsample the band crop by `config.AVOID_DECIMATION_FACTOR`
  before matching (disparity rescaled back afterward — see the module's
  decimation-math comment; easy to get backwards silently).
- **Spatial** — edge-preserving smoothing via `cv2.bilateralFilter`.
  `cv2.ximgproc`'s WLS disparity filter (the closer RealSense analog) was
  considered and skipped — this project's resolved OpenCV build has no
  `cv2-contrib` (confirmed absent on-device), same evaluate-and-reject
  precedent as the Open3D note in the package table above.
- **Temporal** — a per-bin EMA (`config.AVOID_TEMPORAL_EMA_ALPHA`) applied
  *after* binning, across cycles, in `src/avoidance.py`'s
  `BinDistanceSmoother` — the only stage needing state across calls.
- **Hole-filling** — a bin below `config.AVOID_BIN_MIN_VALID_FRACTION`
  valid pixels is filled from trusted neighboring bins (sparse/noisy
  gaps), **except** the deterministic leftmost
  `~config.STEREO_MAX_DISPARITY_PX`-wide strip (no valid match is possible
  there regardless of scene content — nothing further left in the right
  image to search against, true of any stereo system), which is marked
  no-data rather than fabricated. `src/avoidance.py build_obstacle_distance()`
  encodes no-data bins as MAVLink's `65535` sentinel, not a guessed value.

**Measured on-device** (this Jetson, synthetic stereo pairs at this rig's
real 1456×1088 / `AVOID_BIN_HEIGHT_FRACTION`=0.4 band size, real chessboard
calibration): full `estimate_bin_distances()` (rectify+band+SGBM+spatial-
filter+binning) averaged ~145ms at `AVOID_DECIMATION_FACTOR=2`, ~90ms at
`=3`, ~77ms at `=4` — against a 200ms/5Hz budget
(`AVOID_UPDATE_INTERVAL_S`) and this project's own measured YOLO+OSD
baseline of 45-70% CPU/core (see "GPU utilization model" above).
`AVOID_DECIMATION_FACTOR` defaults to `4` for this reason — 2 leaves too
little headroom once the still-unmeasured buffer-extraction cost (below)
is added. This does NOT include `pyds.get_nvds_buf_surface()`'s cost,
which remains genuinely unverified on real hardware (see "Known
limitations" below) — this benchmark used synthetic NumPy arrays in place
of that call.

**Threading**: `src/probes.py`'s `pgie_src_pad_buffer_probe()` collects
both cameras' raw NV12 buffers within a SINGLE buffer/probe invocation
(`nvstreammux` batches both sources into one `Gst.Buffer` per cycle) and
completes the depth computation before returning — never stashed across
calls, which would risk consuming a `get_nvds_buf_surface()` array after
its surface is recycled. The throttle (`config.AVOID_UPDATE_INTERVAL_S`)
gates buffer extraction itself, not just the SGBM pass, since
`get_nvds_buf_surface()` may not be a free call. `src/mission.py`'s
`Mission.on_obstacle_reading()` (the streaming-thread producer) and
`ObstacleAvoidance.update()` (main-thread consumer, called from
`main.py`'s `GLib.timeout_add`) follow the same producer/consumer split
already established by `ObjectFollowController`.

**No flight-mode gate, unlike FOLLOW/ISR** — `Mission.update()` streams
`OBSTACLE_DISTANCE` continuously whenever the mavlink link is healthy,
since this is sensor telemetry ArduPilot's own OA logic can use across
GUIDED/AUTO/RTL, not a vehicle-motion command tied to one mode.

**Safety, read before ever touching `AVOID_DRY_RUN`:** bad/wrongly-shaped
distance data can still make ArduPilot's OA swerve incorrectly or refuse
to proceed, even though this companion computer never sends a velocity
setpoint for this mode. `config.AVOID_DRY_RUN` defaults `True` — every
message is computed and logged (`[avoid] DRY RUN obstacle_distance ...`)
but never sent. Validate bin-distance sanity and message contents on the
bench first; then, with a GCS connected to the FC, confirm
`OBSTACLE_DISTANCE` actually arrives (MAVLink Inspector/Proximity view)
with `AVOID_DRY_RUN=False` in a supervised, props-off bench test — before
ever trusting `OA_TYPE` to act on it in flight.

**Horizontal FOV / angle mapping** (`src/avoidance.py build_obstacle_distance()`):
derived from the rig's real rectified calibration
(`2*atan(image_width / (2*fx))`), not hardcoded — same preference this
project already applied when it sourced `FOCAL_LENGTH_PX`/
`STEREO_BASELINE_M` from real chessboard calibration instead of a guess.
Assumes the camera's optical axis is boresight-aligned with the vehicle's
forward direction — unlike `src/extrinsics.py`'s geolocation chain, no
camera→body mounting-angle correction is applied here yet (see "Known
limitations" below).

**Mutually exclusive with FOLLOW/ISR by `MISSION_MODE`'s design** — not a
technical necessity. AVOID only streams sensor data rather than sending
its own velocity setpoints, so it doesn't actually compete with FOLLOW's
`SET_POSITION_TARGET_LOCAL_NED` commands for control authority (a real
depth-camera + BendyRuler setup commonly runs proximity streaming *and* a
guided-mode command source together) — see "Known limitations" below.

## Known limitations / Next steps

- **`src/probes.py`'s always-on per-frame path is still monocular** —
  real stereo depth exists now (`src/stereo_depth.py`, see "Geolocation"
  above) but deliberately only as an on-demand call for one detection at
  a time, not wired into the 60fps per-frame overlay/FOLLOW loop (full
  block matching at that rate would be real, currently-unmeasured CPU
  cost). FOLLOW therefore still holds station on the monocular Z estimate
  in the meantime. Wiring stereo into FOLLOW's control loop specifically
  (lower rate than 60fps is fine there — it already runs at
  `config.FOLLOW_UPDATE_INTERVAL_S`, 5Hz) would be the natural next step
  once `src/stereo_depth.py` is validated on real hardware.
- **`src/stereo_depth.py` and `src/geolocation.py` are unverified on real
  hardware** — written and reasoned through carefully but never run on
  the Jetson (this was implemented without device access). The biggest
  risk is `pyds.get_nvds_buf_surface()` against this pipeline's NV12
  buffers (every official DeepStream sample uses it against RGBA
  instead) — see "Geolocation" above for the fallback if that doesn't
  work as assumed. Second: `configs/camera_body_extrinsics.yaml`'s
  mounting angles are still placeholder (identity/zero) values and need
  measuring against a real known-bearing target before the lat/lon output
  can be trusted.
- **No object tracker yet** (no `nvtracker` in the pipeline) — detections
  aren't assigned persistent IDs across frames. Add `nvtracker` between
  `pgie` and `tiler` when ID persistence is needed (e.g. for stable
  FOLLOW target selection or localization output).
- **ISR mission mode is scaffolded but not implemented** — `Mission.
  _update_isr()` detects the trigger condition (flight mode + altitude)
  but does not yet log anything. Next milestone per project staging.
- **AVOID's `pyds.get_nvds_buf_surface()` call is unverified on real
  hardware** — same risk `src/stereo_depth.py` already flagged, now
  actually exercised live for the first time (`src/probes.py`
  `pgie_src_pad_buffer_probe`). Confirm it returns the expected shape
  against this pipeline's NV12 buffers, and doesn't trigger a hidden
  device-to-host sync copy per call (depends on `nvstreammux`'s memory
  type) — the ~77-145ms benchmark in "Obstacle avoidance" above used
  synthetic arrays in place of this call, not the real thing.
- **AVOID's control-authority split from FOLLOW is unvalidated** —
  `MISSION_MODE` gates them as mutually exclusive today (see "Obstacle
  avoidance" above for why that's a config choice, not a hard technical
  requirement). Running both concurrently (FOLLOW's velocity setpoints +
  AVOID's `OBSTACLE_DISTANCE` streaming, letting ArduPilot's OA arbitrate)
  is the natural real-world combination but hasn't been built or tested.
- **AVOID assumes boresight-aligned camera mounting** — no camera→body
  yaw correction is applied to `build_obstacle_distance()`'s angle
  mapping, unlike the geolocation chain's (still placeholder)
  `configs/camera_body_extrinsics.yaml`. A real mounting yaw offset would
  bias which angular zone ArduPilot thinks each reading came from.
- **Engine batch=1** (see export notes above) — revisit if fused-batch
  throughput becomes necessary once both cameras are running at full load.
- RTSP branch currently sends **one composited (tiled) MJPEG stream**, not
  two independent per-camera streams — simplest for a ground station view;
  revisit if the ground station needs full-resolution per-camera feeds
  instead of the tiled composite (`config.TILER_SCALE`, default 0.5 —
  half native resolution per tile, aspect-ratio-preserving).
- **RTSP is MJPEG (`nvjpegenc`), not H.264** — this Orin Nano SOM has no
  hardware H.264 encoder (see "Hardware video encode" above). Uses more
  bandwidth than H.264 would have at the same quality; tune
  `RTSP_JPEG_QUALITY`/tiler resolution if the ground-station link is
  constrained. If this project ever moves to an Orin NX/AGX Orin board
  (which do have NVENC), swap `nvjpegenc`/`rtpjpegpay` back for
  `nvv4l2h264enc`/`h264parse`/`rtph264pay` in `src/pipeline.py` for
  better compression.