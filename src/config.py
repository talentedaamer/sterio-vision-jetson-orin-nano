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
# Inference -- YOLOv8n TensorRT engine, exported via ultralytics
# ---------------------------------------------------------------------------
# Exported with: model.export(format='engine', device='0', half=True, workspace=4)
# No dynamic=True was used, so the engine's max batch size is fixed at 1.
# nvinfer's `batch-size` (configs/pgie_yolov8n_config.txt) MUST stay 1 to
# match -- it will run one inference call per source per muxer batch rather
# than a single fused batch=2 call. Still 100% GPU/TensorRT, just not fused.
# Re-export with dynamic=True (batch=2) later if fused throughput matters.
ENGINE_PATH = "models/yolov8n.engine"
PGIE_CONFIG_PATH = "configs/pgie_yolov8n_config.txt"
TARGET_CLASSES = {0, 2, 3}   # COCO ids: 0=person, 2=car, 3=motorcycle

# ---------------------------------------------------------------------------
# Monocular distance estimation (interim -- replaced by stereo disparity once
# the two cameras are calibrated; see distance.py)
# ---------------------------------------------------------------------------
KNOWN_HEIGHTS_M = {
    0: 1.7,   # person
    2: 1.5,   # car
    3: 1.1,   # motorcycle
}
FOCAL_LENGTH_PX = 800.0   # calibrate per-camera; same lens assumed on both IMX296s for now

# Physical lens-center-to-lens-center distance between the two IMX296s,
# measured directly (ruler/calipers) -- NOT a substitute for photographic
# stereo calibration (cv2.stereoCalibrate), which still needs to happen
# before estimate_xyz_stereo() in distance.py can be implemented: a ruler
# measurement gives the baseline but not rotational misalignment between
# the two cameras or lens distortion, both of which stereoCalibrate solves
# for using checkerboard captures from this exact rig.
STEREO_BASELINE_M = 0.094   # 9.4cm

# ---------------------------------------------------------------------------
# Open3D depth "heatmap" view (--debug only) -- see src/debug_depth_view.py.
# Colors each detected object by its existing monocular Z estimate (near =
# hot/red, far = cool/blue). Fixed range, not the current buffer's min/max,
# so colors stay comparable frame to frame.
# ---------------------------------------------------------------------------
DEPTH_VIEW_MIN_M = 0.5
DEPTH_VIEW_MAX_M = 25.0

# ---------------------------------------------------------------------------
# Tiled composite output (both camera views side-by-side) + RTSP
# ---------------------------------------------------------------------------
TILER_ROWS = 1
TILER_COLS = 2
TILER_WIDTH = 1456    # composite output width (both tiles combined)
TILER_HEIGHT = 1088

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
