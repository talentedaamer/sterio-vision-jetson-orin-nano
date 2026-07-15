"""Central configuration for the dual-camera DeepStream detection pipeline.

Tune camera capture, model paths, target classes, and output behavior here.
Nothing in pipeline.py or probes.py should need touching for these changes.
"""
import os

# ---------------------------------------------------------------------------
# Runtime mode
# ---------------------------------------------------------------------------
# When True, an extra local-display branch (nveglglessink) is added for bench
# testing on an attached HDMI/DP monitor. Headless detection output (stdout /
# on_detection hook) and the RTSP ground-station stream are ALWAYS active
# regardless of this flag -- it only adds the bench-test branch on top.
DEBUG = os.environ.get("DS_DEBUG", "0") == "1"

# ---------------------------------------------------------------------------
# Camera capture -- dual IMX296 global-shutter, CSI0 + CSI1
# ---------------------------------------------------------------------------
SENSOR_IDS = [0, 1]   # nvarguscamerasrc sensor-id -> /dev/video0 (left), /dev/video1 (right)
NUM_SOURCES = len(SENSOR_IDS)
CAPTURE_WIDTH = 1456
CAPTURE_HEIGHT = 1088
FRAMERATE = 60
SENSOR_MODE = -1     # -1 = let nvarguscamerasrc auto-select a mode matching the caps below

# ---------------------------------------------------------------------------
# Inference -- YOLO26n TensorRT engine, exported via ultralytics (switched
# from YOLOv8n -- YOLO26 is natively NMS-free/end-to-end, see
# nvdsinfer_custom_impl_yolo26/ and CLAUDE.md "Model export")
# ---------------------------------------------------------------------------
# Exported with: model.export(format='engine', device='0', half=True, workspace=4)
# No dynamic=True was used, so the engine's max batch size is fixed at 1.
# nvinfer's `batch-size` (configs/pgie_yolo26n_config.txt) MUST stay 1 to
# match -- it will run one inference call per source per muxer batch rather
# than a single fused batch=2 call. Still 100% GPU/TensorRT, just not fused.
# Re-export with dynamic=True (batch=2) later if fused throughput matters.
ENGINE_PATH = "models/yolo26n.engine"
PGIE_CONFIG_PATH = "configs/pgie_yolo26n_config.txt"
TARGET_CLASSES = {0, 2, 3}   # COCO ids: 0=person, 2=car, 3=motorcycle

# ---------------------------------------------------------------------------
# Monocular distance estimation (interim -- replaced by stereo disparity once
# estimate_xyz_stereo() in distance.py is implemented; see src/calibration.py)
# ---------------------------------------------------------------------------
KNOWN_HEIGHTS_M = {
    0: 1.7,   # person
    2: 1.5,   # car
    3: 1.1,   # motorcycle
}
# FOCAL_LENGTH_PX and STEREO_BASELINE_M below are from real chessboard
# stereo calibration (cv2.stereoCalibrate + cv2.stereoRectify), not a
# placeholder/ruler measurement anymore -- see configs/stereo_calibration.yaml
# and src/calibration.py (the full loader: both cameras' intrinsics +
# distortion, R/T/R1/R2/P1/P2/Q). Calibration RMS reprojection error:
# 1.42px. This directly improves the existing monocular estimate below
# even before stereo disparity is implemented, since it was previously
# using a rough 800.0px guess.
FOCAL_LENGTH_PX = 921.5871   # P1/P2's fx, post-rectification (both cameras share it)
STEREO_BASELINE_M = 0.094319  # from calibration's T vector, not the earlier ruler measurement

# ---------------------------------------------------------------------------
# Real stereo depth (src/stereo_depth.py, src/distance.py estimate_xyz_stereo)
# -- ROI-based disparity around a single detection's bbox, NOT a full-frame
# dense disparity map (far too expensive on this ARM CPU at full resolution/
# framerate; see src/stereo_depth.py docstring). UNVERIFIED on real
# hardware -- this is genuine CPU work this project has otherwise avoided,
# so it is NOT wired into the always-on per-frame probe by default; it's
# called on demand (e.g. from src/geolocation.py) for one detection at a
# time. Tune STEREO_MAX_DISPARITY_PX down (must stay a multiple of 16) if
# it costs too much CPU per call once measured on-device.
# ---------------------------------------------------------------------------
# max_disparity = focal_length_px * baseline_m / min_expected_depth_m.
# 192px -> ~0.45m minimum depth at the calibrated focal length/baseline
# above; closer objects than that will fail to find a valid disparity and
# fall back to the monocular estimate (see estimate_xyz_stereo's caller).
STEREO_MAX_DISPARITY_PX = 192
STEREO_BLOCK_SIZE = 7

# On-screen X/Y/Z label smoothing (src/distance.py SmoothedDetection) --
# presentational only, does NOT affect FOLLOW/debug-plot, which still use
# the raw per-frame estimate directly (see src/probes.py).
DISPLAY_AVERAGE_WINDOW_S = 1.0     # how much recent history to average over
DISPLAY_UPDATE_INTERVAL_S = 1.0    # how often the displayed number changes

# ---------------------------------------------------------------------------
# Depth "heatmap" coloring for the --debug 3D plot (src/debug_plot.py).
# Colors each detected object by its existing monocular Z estimate (near =
# hot/red, far = cool/blue). Fixed range, not the current buffer's min/max,
# so colors stay comparable frame to frame.
# (Open3D was evaluated for this and rejected -- no aarch64/Python 3.10
# wheel exists; see CLAUDE.md "Python package dependencies" before
# reconsidering it.)
# ---------------------------------------------------------------------------
PLOT_DEPTH_MIN_M = 0.5
PLOT_DEPTH_MAX_M = 25.0

# ---------------------------------------------------------------------------
# Tiled composite output (both camera views side-by-side) + RTSP
# ---------------------------------------------------------------------------
TILER_ROWS = 1
TILER_COLS = 2
# Derived from CAPTURE_WIDTH/HEIGHT (not independent hardcoded numbers) so
# each tile always keeps the camera's actual aspect ratio -- previously
# TILER_WIDTH/HEIGHT were set independently (1456x1088 total = 728x1088 per
# tile, vs. the source's actual 1456x1088/tile), squeezing every frame's
# width down to roughly half without adjusting height, which is exactly
# what produced the vertically-squeezed/stretched picture in --debug and
# on RTSP. TILER_SCALE controls output size/bandwidth; it does NOT affect
# aspect ratio, which is always preserved by construction.
TILER_SCALE = 0.5
TILER_TILE_WIDTH = int(CAPTURE_WIDTH * TILER_SCALE)
TILER_TILE_HEIGHT = int(CAPTURE_HEIGHT * TILER_SCALE)
TILER_WIDTH = TILER_TILE_WIDTH * TILER_COLS     # composite output width (both tiles combined)
TILER_HEIGHT = TILER_TILE_HEIGHT * TILER_ROWS

RTSP_PORT = 8554
RTSP_MOUNT_POINT = "/ds-stereo"
RTSP_UDP_PORT = 5400
# MJPEG via nvjpegenc (hardware NVJPG engine), not H.264 -- this Orin Nano
# module has no hardware video encoder (NVENC fused off). Quality 0-100;
# lower it (and/or shrink TILER_WIDTH/HEIGHT) if ground-station bandwidth is
# tight, since MJPEG has no inter-frame compression like H.264 would have.
RTSP_JPEG_QUALITY = 80

# ---------------------------------------------------------------------------
# Streammux
# ---------------------------------------------------------------------------
MUX_BATCHED_PUSH_TIMEOUT_US = 33000

# ---------------------------------------------------------------------------
# MAVLink -- flight controller telemetry + guided-mode command link
# ---------------------------------------------------------------------------
MAVLINK_DEVICE = "/dev/ttyTHS1"    # Jetson Orin Nano UART wired to the FC's telemetry port
MAVLINK_BAUD = 57600               # must match that serial port's SERIALx_BAUD param on the FC
MAVLINK_HEARTBEAT_TIMEOUT_S = 5.0  # link considered down if no HEARTBEAT for this long
# How much recent ATTITUDE history MavlinkLink.get_interpolated_attitude()
# (src/mavlink_link.py) keeps, to align a camera frame's capture time with
# the nearest-bracketing attitude samples instead of whatever ATTITUDE
# message happened to arrive last. GPS position is NOT buffered this way
# (nearest/latest value only) -- it changes slowly enough that this isn't
# worth the extra complexity; see src/geolocation.py.
MAVLINK_ATTITUDE_BUFFER_S = 2.0

# ---------------------------------------------------------------------------
# Mission mode -- gates FOLLOW / ISR behavior (see src/mission.py). Neither
# mission ever starts just because this process is running: FOLLOW also
# requires the flight controller to actually be in FOLLOW_TRIGGER_FLIGHT_MODE,
# and ISR requires ISR_TRIGGER_FLIGHT_MODE + reaching ISR_TRIGGER_ALTITUDE_M.
# ---------------------------------------------------------------------------
# "NONE"   -- no MAVLink connection is even opened; camera/detection
#             pipeline behaves exactly as before this feature existed.
# "FOLLOW" -- PID-drive the drone to hold station on FOLLOW_TARGET_CLASS.
# "ISR"    -- NOT YET IMPLEMENTED (next milestone). Will log detected-object
#             + IMU + GPS data to CSV/JSON once triggered.
MISSION_MODE = os.environ.get("MISSION_MODE", "NONE").upper()

# --- FOLLOW ---
FOLLOW_TARGET_CLASS = 0           # COCO id to follow -- 0 = person
FOLLOW_TRIGGER_FLIGHT_MODE = "GUIDED"
FOLLOW_TARGET_DISTANCE_M = 3.0    # standoff distance to hold from the target (Z axis)
FOLLOW_MAX_VELOCITY_MPS = 1.5     # hard clamp on every axis -- keep conservative until flight-tested
FOLLOW_UPDATE_INTERVAL_S = 0.2    # 5 Hz control loop; plenty for velocity setpoints
# (kp, ki, kd) -- untuned starting points, NOT validated gains. Must be
# tuned incrementally via real flight testing, see src/pid.py docstring.
FOLLOW_PID_LATERAL = (0.6, 0.0, 0.15)    # drives X (lateral, m) -> vy (right)
FOLLOW_PID_VERTICAL = (0.6, 0.0, 0.15)   # drives Y (vertical, m) -> vz (down)
FOLLOW_PID_FORWARD = (0.4, 0.0, 0.1)     # drives (Z - target dist, m) -> vx (forward)
# SAFETY: while True, setpoints are computed and logged but never sent to
# the flight controller. Only set to False after validating telemetry
# reads, sign conventions, and gains on the bench / in a supervised,
# low-altitude tethered test.
FOLLOW_DRY_RUN = os.environ.get("FOLLOW_DRY_RUN", "1") == "1"

# --- ISR (not yet implemented) ---
ISR_TRIGGER_FLIGHT_MODE = "AUTO"
ISR_TRIGGER_ALTITUDE_M = 30.0
ISR_LOG_FORMAT = "csv"  # or "json" -- see src/mission.py
