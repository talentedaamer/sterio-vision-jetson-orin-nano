# object-detection-win — Jetson Orin NX Dual-Camera DeepStream Pipeline

UAV payload: real-time YOLOv8n detection + monocular distance estimation over
dual IMX296 global-shutter CSI cameras, running 100% on GPU/dedicated hardware
blocks via NVIDIA DeepStream. Stereo calibration and disparity-based depth,
localization, and MAVLink telemetry are future milestones (see "Next Steps").

## Target Hardware / Software (do not assume anything different)

| Component | Version |
|---|---|
| Hardware | NVIDIA Jetson Orin NX Engineering Reference Developer Kit |
| OS | Ubuntu 22.04.5 LTS (Jammy) |
| Kernel | Linux 5.15.148-tegra (aarch64) |
| JetPack / L4T | JetPack 6.2.1 (L4T R36.4.4) |
| CUDA | 12.6 (V12.6.68) |
| TensorRT | 10.3.0.30 (Python package `tensorrt==10.3.0`) |
| DeepStream SDK | 7.1 (already installed at `/opt/nvidia/deepstream/deepstream-7.1`, matches JetPack 6.2.1 — do not reinstall) |
| Python | 3.10 (uv-managed venv, `--system-site-packages`) |
| Cameras | 2x IMX296 global shutter, CSI0 -> `/dev/video0`, CSI1 -> `/dev/video1` |

Project deps (`pyproject.toml`) are managed with `uv`. `torch`/`torchvision`/
`ultralytics`/`onnx` exist only to support the (already-working, do-not-touch)
`.pt -> .engine` export script — the DeepStream runtime app never imports
them; it loads the prebuilt TensorRT engine directly through `gst-nvinfer`.

One-time environment setup (already-installed DeepStream SDK, not repeated
here as install steps since it's present on-device):
```bash
uv venv --system-site-packages   # required so `gi`/PyGObject (OS-provided) is importable
uv sync
uv pip install /opt/nvidia/deepstream/deepstream-7.1/lib/pyds-*.whl  # pyds is not on PyPI
```

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
   queue -> nvvideoconvert -> I420 -> nvv4l2h264enc -> h264parse -> rtph264pay -> udpsink   queue -> nveglglessink
   (RTSP out via GstRtspServer, ALWAYS on)                                          (bench display, --debug only)
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

### GPU utilization model — what actually touches which engine

| Stage | Hardware |
|---|---|
| Camera capture (Argus/ISP) | Dedicated ISP, NVMM buffers, zero-copy |
| `nvstreammux` batching | GPU/unified memory, no copy |
| `nvinfer` (YOLOv8n) | TensorRT on GPU (CUDA cores) |
| `nvvideoconvert` (NV12↔RGBA↔I420, scale/tile) | VIC (dedicated video/image compositor block) by default on Jetson — deliberately **not** forced to `compute-hw=GPU`, so it runs in parallel with TensorRT instead of contending for CUDA cores |
| `nvdsosd` (box/text draw) | GPU (`process-mode=1`) |
| `nvv4l2h264enc` | NVENC (dedicated hardware encoder) |
| RTP packetization / `udpsink` / GLib mainloop / distance math in the probe | ARM CPU — trivial control-plane work only (packet framing, tens of floats per frame), never touches pixels or tensors |

No real video pipeline is *literally* 0% CPU (mainloop scheduling, RTP
framing, and the Python distance-math probe all run on the ARM cores), but
every pixel- and tensor-heavy stage — capture, batching, inference,
colorspace/scale conversion, tiling, OSD compositing, H.264 encode — runs
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
- RTSP branch currently sends **one composited (tiled) stream**, not two
  independent per-camera streams — simplest/lowest-bandwidth for a ground
  station view; revisit if the ground station needs full-resolution
  per-camera feeds instead of the 1280x720 tiled composite.
