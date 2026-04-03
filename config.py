# ══════════════════════════════════════════════════════
#  Unified Config — Fall Detection + Vitals Monitoring
#  Works on Mac (built-in cam) and Raspberry Pi Camera
# ══════════════════════════════════════════════════════

# ── Camera ────────────────────────────────────────────
VIDEO_PATH        = ""      # "" = live camera; "/path/file.mp4" = file
CAMERA_INDEX      = 0       # 0 = default camera (ignored when USE_PICAMERA2=True)
LOOP_VIDEO        = False
PLAYBACK_FPS      = 30

# Set True on Raspberry Pi with Camera Module (uses libcamera via picamera2).
# Leave False on Mac / USB webcams (uses OpenCV VideoCapture / V4L2).
USE_PICAMERA2     = False

# Force fixed exposure & white balance for accurate rPPG
# (Auto-exposure flicker is the #1 cause of bad vitals readings)
CAMERA_FIXED_EXPOSURE    = True
CAMERA_EXPOSURE_VALUE    = -6     # OpenCV units; tune per your camera/room
CAMERA_FIXED_WB          = True

# ── Fall Detection Model ──────────────────────────────
MODEL_NAME        = "yolov8n-pose.pt"
INFERENCE_SIZE    = 320
INFERENCE_EVERY_N = 2       # run pose inference every N frames
DEVICE            = "cpu"

# ── Fall Scoring Weights (must sum to 1.0) ────────────
WEIGHT_ASPECT_RATIO        = 0.35
WEIGHT_TORSO_ANGLE         = 0.35
WEIGHT_HIP_VELOCITY        = 0.20
WEIGHT_KEYPOINT_CONFIDENCE = 0.10

# ── Fall Thresholds ───────────────────────────────────
SCORE_NORMAL          = 0.50
SCORE_AT_RISK         = 0.70
FALL_CONFIRM_FRAMES   = 5

# ── Fall Heuristic Tuning ─────────────────────────────
ASPECT_RATIO_FALLEN    = 1.0
ASPECT_RATIO_STANDING  = 2.5
TORSO_ANGLE_STANDING   = 20.0   # degrees
TORSO_ANGLE_FALLEN     = 70.0   # degrees
HIP_VELOCITY_THRESHOLD = 0.04
MIN_KEYPOINT_CONF      = 0.40

# ── Vitals (rPPG) Settings ────────────────────────────
# Face detection/mesh
FACE_DETECTION_CONFIDENCE  = 0.6
FACE_MESH_CONFIDENCE       = 0.6
VITALS_EVERY_N_FRAMES      = 1    # process face ROI every frame (light operation)

# Signal buffer
RPPG_BUFFER_SECONDS  = 20    # seconds of signal to accumulate before first reading
RPPG_SAMPLE_RATE     = 30    # expected camera FPS (used for frequency calculations)

# Bandpass filter for heart rate (Hz)
HR_FREQ_LOW   = 0.7    # 42 bpm
HR_FREQ_HIGH  = 3.0    # 180 bpm

# Bandpass for respiratory rate (Hz)
RR_FREQ_LOW   = 0.1    # 6 breaths/min
RR_FREQ_HIGH  = 0.5    # 30 breaths/min

# Normal ranges (used to colour-code the display)
HR_NORMAL_LOW    = 60
HR_NORMAL_HIGH   = 100

RR_NORMAL_LOW    = 12
RR_NORMAL_HIGH   = 20

SPO2_NORMAL_LOW  = 95     # below this → warning

HRV_LOW_STRESS   = 40     # RMSSD (ms): above = relaxed; below = stressed

# Vitals are recomputed every this many seconds
VITALS_UPDATE_INTERVAL_SEC = 5

# ── Display ───────────────────────────────────────────
WINDOW_NAME   = "Health Monitor — Fall + Vitals"
SHOW_SKELETON = True
SHOW_BBOX     = True
FONT_SCALE    = 0.60
BAR_HEIGHT    = 24
PANEL_WIDTH   = 260    # width of vitals sidebar (pixels)

# ── Alerts ───────────────────────────────────────────
ALERT_LOG_FILE        = "alerts.log"
ALERT_BEEP            = True
ALERT_COOLDOWN_SEC    = 10

# Vitals alert thresholds
ALERT_HR_LOW      = 50
ALERT_HR_HIGH     = 110
ALERT_RR_HIGH     = 25
ALERT_SPO2_LOW    = 92
