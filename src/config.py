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

# ---------------------------------------------------------------------------
# Tiled composite output (both camera views side-by-side) + RTSP
# ---------------------------------------------------------------------------
TILER_ROWS = 1
TILER_COLS = 2
TILER_WIDTH = 1280    # composite output width (both tiles combined)
TILER_HEIGHT = 720

RTSP_PORT = 8554
RTSP_MOUNT_POINT = "/ds-stereo"
RTSP_UDP_PORT = 5400
RTSP_BITRATE = 4_000_000

# ---------------------------------------------------------------------------
# Streammux
# ---------------------------------------------------------------------------
MUX_BATCHED_PUSH_TIMEOUT_US = 33000
