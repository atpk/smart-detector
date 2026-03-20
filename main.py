"""
main.py
───────
Unified Health Monitor — Fall Detection + Vitals (rPPG)

Runs two parallel pipelines on every camera frame:
  1. YOLOv8n-pose  → fall risk score + alert
  2. MediaPipe + POS rPPG → HR, HRV, RR, SpO₂, Stress

Usage
─────
  python main.py                  # live camera (default)
  python main.py --camera 1       # pick camera index
  python main.py --video clip.mp4 # use a video file
  python main.py --no-display     # headless (log only)

Press  Q / ESC  to quit.
Press  S        to save a screenshot.
"""

import sys
import time
import argparse
import datetime

import cv2

import config
from video_source    import VideoSource
from fall_detector   import FallDetector
from vitals_detector import VitalsDetector
from alert_system    import AlertSystem
from display         import Display


def _parse_args():
    p = argparse.ArgumentParser(description="Health Monitor — Fall + Vitals")
    p.add_argument("--camera",     type=int,  default=None)
    p.add_argument("--video",      type=str,  default=None)
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.video:
        config.VIDEO_PATH = args.video
    if args.camera is not None:
        config.CAMERA_INDEX = args.camera

    headless = args.no_display

    print("\n" + "═" * 56)
    print("  Health Monitor  —  Fall Detection + Vitals (rPPG)")
    print("═" * 56)

    # ── Initialise all components ──────────────────────────────────────────
    fall_detector   = FallDetector()
    vitals_detector = VitalsDetector()
    alerts          = AlertSystem()
    display         = Display() if not headless else None
    source          = VideoSource()

    mode = f"file: {config.VIDEO_PATH}" if config.VIDEO_PATH else f"camera index {config.CAMERA_INDEX}"
    print(f"\n[Main] Source: {mode}")
    print(f"       Vitals readings start after {config.RPPG_BUFFER_SECONDS}s of face data")
    print("       Press  Q / ESC  to quit   |   S  to screenshot\n")

    # ── Performance counters ───────────────────────────────────────────────
    fps_counter   = 0
    fps_timer     = time.time()
    current_fps   = 0.0

    vitals_result  = None
    from vitals_detector import VitalsResult
    vitals_result  = VitalsResult()

    # ── Main loop ──────────────────────────────────────────────────────────
    try:
        for frame_idx, frame in source.frames():
            now = time.time()

            # ── Fall detection ───────────────────────────────────────────
            fall_result = fall_detector.process(frame, frame_idx)

            # ── Vitals (rPPG + face mesh) ────────────────────────────────
            # Runs every frame — it's lightweight (just ROI averaging + buffer append)
            # FFT recomputation happens every VITALS_UPDATE_INTERVAL_SEC seconds
            vitals_result = vitals_detector.process(frame, now)

            # ── Alerts ───────────────────────────────────────────────────
            fired = alerts.check(fall_result, vitals_result)
            alert_fired = bool(fired.get("fall"))

            # ── Display ──────────────────────────────────────────────────
            if display is not None:
                canvas = display.render(
                    frame, fall_result, vitals_result,
                    alert_fired, fps=current_fps,
                )
                display.show(canvas)

                key = display.wait_key(1)
                if key in (ord("q"), ord("Q"), 27):
                    print("\n[Main] Quit.")
                    break
                elif key in (ord("s"), ord("S")):
                    _screenshot(canvas)

            # ── FPS ───────────────────────────────────────────────────────
            fps_counter += 1
            if now - fps_timer >= 1.0:
                current_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer   = now
                _console_status(fall_result, vitals_result, current_fps, alerts.alert_count)

    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    finally:
        source.release()
        vitals_detector.release()
        if display is not None:
            display.destroy()
        _summary(alerts)


# ── Helpers ────────────────────────────────────────────────────────────────

def _console_status(fr, vr, fps, alert_count):
    fall_str = f"{fr.status:<8} {fr.score*100:.0f}%"
    if vr.ready:
        hr   = f"{vr.heart_rate:.0f}" if vr.heart_rate else "—"
        spo2 = f"{vr.spo2:.1f}" if vr.spo2 else "—"
        rr   = f"{vr.resp_rate:.0f}" if vr.resp_rate else "—"
        hrv  = f"{vr.hrv_rmssd:.0f}" if vr.hrv_rmssd else "—"
        vitals_str = f"HR={hr}bpm  SpO₂={spo2}%  RR={rr}  HRV={hrv}ms  Stress={vr.stress_label}"
    elif vr.face_detected:
        vitals_str = "collecting data..."
    else:
        vitals_str = "no face"

    print(
        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
        f"Fall: {fall_str}  |  Vitals: {vitals_str}  |  "
        f"FPS: {fps:.1f}  Alerts: {alert_count}"
    )


def _screenshot(canvas):
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"screenshot_{ts}.png"
    cv2.imwrite(path, canvas)
    print(f"[Main] Screenshot saved: {path}")


def _summary(alerts):
    print("\n" + "═" * 56)
    print(f"  Session ended.  Total alerts fired: {alerts.alert_count}")
    print(f"  Alert log: {config.ALERT_LOG_FILE}")
    print("═" * 56)


if __name__ == "__main__":
    main()
