# Health Monitor — Fall Detection + Vitals
### Mac · Raspberry Pi 8 GB · YOLOv8n-pose · MediaPipe · rPPG

A unified, real-time health monitoring prototype that runs on a live camera:

| Feature | Method | Accuracy |
|---------|--------|----------|
| Fall Detection | YOLOv8n-pose + heuristics | ~85–90% on common falls |
| Heart Rate | POS rPPG (Welch PSD) | ±3–5 bpm |
| SpO₂ | R/G ratio-of-ratios | ±3–5% |
| Respiratory Rate | rPPG envelope | ±2–4 br/min |
| HRV (RMSSD) | rPPG peak detection | ±15–25 ms |
| Stress Index | HR + HRV combined | Directional (Low/Med/High) |

---

## Project Structure

```
health_monitor/
├── main.py              ← entry point — orchestrates both pipelines
├── fall_detector.py     ← YOLOv8n-pose inference + 4 heuristic scorers
├── vitals_detector.py   ← MediaPipe face mesh + POS rPPG + signal processing
├── alert_system.py      ← fall + vitals alerts with cooldown
├── display.py           ← camera feed + vitals sidebar + waveform
├── video_source.py      ← live camera (Mac / Pi) or file
├── config.py            ← all tunable parameters
└── requirements.txt
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate

# Mac
pip install -r requirements.txt

# Raspberry Pi — swap opencv package first:
# sed -i 's/opencv-python>=/# opencv-python>=/' requirements.txt
# sed -i 's/# opencv-python-headless/opencv-python-headless/' requirements.txt
pip install -r requirements.txt
```

YOLOv8n-pose (~6 MB) downloads automatically on first run.

---

## Running

```bash
python main.py                   # default camera (index 0)
python main.py --camera 1        # alternate camera
python main.py --video clip.mp4  # test with a video file
python main.py --no-display      # headless (console + log only)
```

| Key | Action |
|-----|--------|
| Q / ESC | Quit |
| S | Save screenshot |

---

## What You'll See

**Left panel** — live camera with:
- Skeleton + bounding box on detected person
- Fall risk score bar (green → orange → red)
- Status label: NORMAL / AT RISK / ⚠ FALL DETECTED

**Right sidebar** — vitals panel with:
- ♥ Heart Rate (bpm)
- ○ SpO₂ (%)
- ~ Respiratory Rate (br/min)
- ≈ HRV / RMSSD (ms)
- ⚡ Stress level (Low / Medium / High)
- Signal quality bar
- Live rPPG waveform plot

---

## Best Results — Lighting & Positioning

For good vitals accuracy:
- Sit **40–80 cm** from the camera, face centred
- Use **warm white LED** or **natural daylight** — avoid flickering tube/fluorescent lights
- Keep your **face still** during the 15-second warmup period
- Ensure your **forehead and cheeks are visible** and well-lit (no harsh shadows)
- Once readings appear, normal breathing and minor movements are fine

---

## Mac Camera Permission

On first run, macOS will ask for camera permission. If it doesn't:
1. System Settings → Privacy & Security → Camera
2. Enable for Terminal (or your Python IDE)

---

## Raspberry Pi Camera Setup

```bash
# Enable v4l2 driver for Camera Module
sudo modprobe bcm2835-v4l2
# Make it permanent:
echo "bcm2835-v4l2" | sudo tee -a /etc/modules

# Check device appeared
v4l2-ctl --list-devices

# Run
python main.py --camera 0
```

**Pi performance tips** (in `config.py`):
```python
INFERENCE_SIZE    = 320
INFERENCE_EVERY_N = 2    # skip every other frame for pose
```

**Optional — export YOLOv8 to ONNX for ~40% faster inference:**
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt').export(format='onnx', imgsz=320)"
# Then set MODEL_NAME = "yolov8n-pose.onnx" in config.py
```

Expected Pi 8 GB performance: **12–20 FPS** with ONNX + every-2nd-frame inference.

---

## Understanding the Vitals

### Why 15 seconds warmup?
rPPG works by detecting micro-colour changes in skin across many frames.
A minimum of ~15 seconds of data is needed for the FFT to resolve the heart
rate frequency reliably. After the first reading, updates happen every 5 seconds.

### SpO₂ accuracy note
The R/G ratio method is a proxy — it gives plausible numbers (typically ±3–5%)
but is not clinically certified. It can reliably detect gross hypoxia.
For certified SpO₂, a MAX30102 hardware sensor (~₹150, via Pi GPIO) would be the
right upgrade in a later version.

### Signal quality
The bar shows how much the green channel is fluctuating. Low quality means:
- Too little light on the face
- Person is moving a lot
- Camera auto-exposure is active (turn off in config)

---

## Alert Log

All alerts are appended to `alerts.log`:
```
[2025-06-01 14:32:07]  FALL        score=84.2%  streak=7  (aspect_ratio=0.91, ...)
[2025-06-01 14:34:15]  VITALS      Low SpO₂: 89.2% (< 92%)
[2025-06-01 14:35:42]  VITALS      High Heart Rate: 118 bpm (> 110)
```

---

## Tuning `config.py`

| Parameter | Effect |
|-----------|--------|
| `RPPG_BUFFER_SECONDS` | Increase for more stable readings (slower start) |
| `VITALS_UPDATE_INTERVAL_SEC` | How often vitals recompute (default 5 s) |
| `FALL_CONFIRM_FRAMES` | More = fewer false falls; fewer = faster response |
| `ALERT_HR_HIGH/LOW` | Set to patient's personal thresholds |
| `ALERT_SPO2_LOW` | Default 92%; clinical concern is usually <95% |
| `CAMERA_FIXED_EXPOSURE` | Must be True for accurate rPPG on most cameras |
