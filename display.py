"""
display.py
──────────
Renders the combined annotated frame:

  Left panel (camera feed):
    • Skeleton + bounding box
    • Fall risk score bar at top
    • Status label (NORMAL / AT RISK / FALL DETECTED)
    • Face ROI highlight (green box = face found)

  Right sidebar (PANEL_WIDTH pixels):
    • Heart Rate     — bpm  + trend bar
    • SpO₂           — %
    • Respiratory Rate — br/min
    • HRV / RMSSD    — ms
    • Stress Index   — label + colour
    • Signal Quality — progress bar
    • Live rPPG waveform plot

  Overlay:
    • FPS counter
    • Alert flash on confirmed fall
    • "Collecting data..." spinner until buffer is full
"""

import cv2
import numpy as np
import config

# COCO skeleton pairs
_SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

_STATUS_COLOUR = {
    "NORMAL"  : (50, 200, 50),
    "AT RISK" : (30, 165, 255),
    "FALL"    : (30,  30, 220),
}

# Vitals colours
_C_GREEN  = (80,  210,  80)
_C_YELLOW = (50,  210, 200)
_C_RED    = (60,   60, 220)
_C_BLUE   = (220, 150,  50)
_C_WHITE  = (230, 230, 230)
_C_GREY   = (120, 120, 120)
_C_DARK   = (22,   22,  30)


def _in_range(val, lo, hi):
    if val is None:
        return None
    return lo <= val <= hi


class Display:
    def __init__(self):
        self._flash_frames_left = 0
        self._dot_frame = 0   # spinner animation

    # ── Public ────────────────────────────────────────────────────────────

    def render(self, frame, fall_result, vitals_result, alert_fired: bool,
               fps: float = 0) -> np.ndarray:
        """
        Composite the annotated camera view + vitals sidebar.
        Returns an (H, W + PANEL_WIDTH) wide canvas.
        """
        panel_w = config.PANEL_WIDTH
        h, w    = frame.shape[:2]

        # ── Base frame ───────────────────────────────────────────────────
        camera_canvas = frame.copy()
        col  = _STATUS_COLOUR.get(fall_result.status, _C_GREY)

        if fall_result.keypoints is not None:
            if config.SHOW_BBOX and fall_result.bbox is not None:
                self._draw_bbox(camera_canvas, fall_result.bbox, col)
            if config.SHOW_SKELETON:
                self._draw_skeleton(camera_canvas, fall_result.keypoints, col)

        self._draw_score_bar(camera_canvas, fall_result.score, col, w)
        self._draw_status_label(camera_canvas, fall_result, col, w)

        if fps > 0:
            self._draw_fps(camera_canvas, fps, w)

        if alert_fired:
            self._flash_frames_left = 18
        if self._flash_frames_left > 0:
            self._draw_flash(camera_canvas, self._flash_frames_left)
            self._flash_frames_left -= 1

        # ── Sidebar ───────────────────────────────────────────────────────
        sidebar = self._build_sidebar(vitals_result, h, panel_w)

        # ── Compose ──────────────────────────────────────────────────────
        canvas = np.concatenate([camera_canvas, sidebar], axis=1)

        # Divider line
        cv2.line(canvas, (w, 0), (w, h), (60, 60, 80), 2)

        self._dot_frame += 1
        return canvas

    def show(self, canvas: np.ndarray):
        cv2.imshow(config.WINDOW_NAME, canvas)

    @staticmethod
    def wait_key(ms: int = 1) -> int:
        return cv2.waitKey(ms) & 0xFF

    @staticmethod
    def destroy():
        cv2.destroyAllWindows()

    # ── Sidebar builder ───────────────────────────────────────────────────

    def _build_sidebar(self, vr, h: int, w: int) -> np.ndarray:
        panel = np.full((h, w, 3), _C_DARK, dtype=np.uint8)

        y = 14
        self._sb_title(panel, "VITALS MONITOR", y, w)
        y += 26

        # Face / signal status strip
        if not vr.face_detected:
            self._sb_status_strip(panel, "No face detected", _C_RED, y, w)
        elif not vr.ready:
            dots = "." * ((self._dot_frame // 12) % 4)
            secs_left = max(0, config.RPPG_BUFFER_SECONDS - len([1]))
            self._sb_status_strip(panel, f"Collecting data{dots}", _C_YELLOW, y, w)
        else:
            self._sb_status_strip(panel, "Live readings", _C_GREEN, y, w)
        y += 28

        # ── Metric rows ───────────────────────────────────────────────────
        y = self._sb_metric(panel, "Heart Rate",
                            self._fmt(vr.heart_rate, "bpm", 0),
                            self._vital_colour(vr.heart_rate,
                                config.HR_NORMAL_LOW, config.HR_NORMAL_HIGH),
                            y, w, icon="♥")
        y = self._sb_metric(panel, "SpO₂",
                            self._fmt(vr.spo2, "%", 1),
                            self._vital_colour_low(vr.spo2, config.SPO2_NORMAL_LOW),
                            y, w, icon="○")
        y = self._sb_metric(panel, "Resp Rate",
                            self._fmt(vr.resp_rate, "br/min", 0),
                            self._vital_colour(vr.resp_rate,
                                config.RR_NORMAL_LOW, config.RR_NORMAL_HIGH),
                            y, w, icon="~")
        y = self._sb_metric(panel, "HRV (RMSSD)",
                            self._fmt(vr.hrv_rmssd, "ms", 0),
                            _C_BLUE,
                            y, w, icon="≈")

        # Stress row
        stress_col = {
            "Low"    : _C_GREEN,
            "Medium" : _C_YELLOW,
            "High"   : _C_RED,
        }.get(vr.stress_label, _C_GREY)
        y = self._sb_metric(panel, "Stress",
                            vr.stress_label if vr.stress_label != "—" else "—",
                            stress_col,
                            y, w, icon="⚡")

        y += 6
        # Signal quality bar
        self._sb_quality_bar(panel, vr.signal_quality, y, w)
        y += 38

        # ── Waveform ──────────────────────────────────────────────────────
        if vr.ready and len(vr.waveform) > 10:
            self._sb_waveform(panel, vr.waveform, y, h - 14, w)

        return panel

    # ── Sidebar sub-components ────────────────────────────────────────────

    def _sb_title(self, panel, text, y, w):
        fs = 0.52
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
        x = (w - tw) // 2
        cv2.putText(panel, text, (x, y + th),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (160, 160, 200), 1, cv2.LINE_AA)
        # Underline
        cv2.line(panel, (8, y + th + 4), (w - 8, y + th + 4), (60, 60, 90), 1)

    def _sb_status_strip(self, panel, text, colour, y, w):
        # Coloured left accent bar
        cv2.rectangle(panel, (6, y), (10, y + 18), colour, -1)
        cv2.putText(panel, text, (16, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

    def _sb_metric(self, panel, label, value_str, colour, y, w, icon=""):
        row_h = 40
        # Label
        cv2.putText(panel, f"{icon}  {label}", (10, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _C_GREY, 1, cv2.LINE_AA)
        # Value (large)
        fs   = 0.80
        thick = 2
        (tw, th), _ = cv2.getTextSize(value_str, cv2.FONT_HERSHEY_DUPLEX, fs, thick)
        x = w - tw - 10
        cv2.putText(panel, value_str, (x, y + row_h - 8),
                    cv2.FONT_HERSHEY_DUPLEX, fs, colour, thick, cv2.LINE_AA)
        # Separator
        cv2.line(panel, (10, y + row_h), (w - 10, y + row_h), (38, 38, 50), 1)
        return y + row_h + 4

    def _sb_quality_bar(self, panel, quality, y, w):
        cv2.putText(panel, "Signal Quality", (10, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _C_GREY, 1, cv2.LINE_AA)
        bar_x = 10
        bar_y = y + 18
        bar_w = w - 20
        bar_h = 10
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
        fill_w = int(bar_w * quality / 100.0)
        col = _C_GREEN if quality > 60 else (_C_YELLOW if quality > 30 else _C_RED)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), col, -1)
        cv2.putText(panel, f"{quality:.0f}%", (bar_x + bar_w + 4, bar_y + bar_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

    def _sb_waveform(self, panel, waveform, y_start, y_end, w):
        """Draw the rPPG waveform as a green polyline."""
        label = "rPPG waveform"
        cv2.putText(panel, label, (10, y_start + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _C_GREY, 1, cv2.LINE_AA)

        plot_y0 = y_start + 18
        plot_h  = max(y_end - plot_y0 - 6, 20)
        plot_w  = w - 16
        mid_y   = plot_y0 + plot_h // 2

        # Background
        cv2.rectangle(panel, (8, plot_y0), (8 + plot_w, plot_y0 + plot_h), (30, 35, 30), -1)

        # Zero line
        cv2.line(panel, (8, mid_y), (8 + plot_w, mid_y), (50, 60, 50), 1)

        # Signal
        n = len(waveform)
        if n < 2:
            return
        pts = []
        for i, v in enumerate(waveform):
            px = int(8 + i * plot_w / n)
            py = int(mid_y - v * (plot_h // 2 - 2))
            py = int(np.clip(py, plot_y0 + 1, plot_y0 + plot_h - 1))
            pts.append((px, py))

        for i in range(1, len(pts)):
            cv2.line(panel, pts[i - 1], pts[i], (60, 220, 100), 1, cv2.LINE_AA)

    # ── Camera overlay helpers ────────────────────────────────────────────

    def _draw_score_bar(self, canvas, score, colour, frame_w):
        bh = config.BAR_HEIGHT
        cv2.rectangle(canvas, (0, 0), (frame_w, bh + 4), (20, 20, 20), -1)
        bar_w = int(frame_w * score)
        cv2.rectangle(canvas, (0, 0), (bar_w, bh), colour, -1)
        pct = f"Fall Risk: {score * 100:.1f}%"
        cv2.putText(canvas, pct, (6, bh - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE * 0.85,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_status_label(self, canvas, result, colour, frame_w):
        label = "⚠ FALL DETECTED" if result.fall_confirmed else result.status
        fs, thick = config.FONT_SCALE * 1.2, 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, thick)
        x = (frame_w - tw) // 2
        y = config.BAR_HEIGHT + th + 8
        cv2.putText(canvas, label, (x + 1, y + 1),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, fs, colour, thick, cv2.LINE_AA)

    def _draw_fps(self, canvas, fps, frame_w):
        text = f"FPS: {fps:.1f}"
        fs   = config.FONT_SCALE * 0.75
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.putText(canvas, text, (frame_w - tw - 8, config.BAR_HEIGHT - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_bbox(self, canvas, bbox, colour):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

    def _draw_skeleton(self, canvas, kps, colour):
        for x, y, conf in kps:
            if conf > 0.25:
                cv2.circle(canvas, (int(x), int(y)), 4, colour, -1)
        for a, b in _SKELETON_PAIRS:
            xa, ya, ca = kps[a]
            xb, yb, cb = kps[b]
            if ca > 0.25 and cb > 0.25:
                cv2.line(canvas, (int(xa), int(ya)), (int(xb), int(yb)),
                         colour, 2, cv2.LINE_AA)

    def _draw_flash(self, canvas, frames_left):
        alpha   = min(frames_left / 9.0, 1.0) * 0.40
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], canvas.shape[0]), (0, 0, 200), -1)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        text = "⚠  FALL ALERT"
        fs   = 1.8
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, 3)
        cx = (canvas.shape[1] - tw) // 2
        cy = (canvas.shape[0] + th) // 2
        cv2.putText(canvas, text, (cx + 2, cy + 2),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (80, 80, 255), 3, cv2.LINE_AA)

    # ── Colour helpers ────────────────────────────────────────────────────

    @staticmethod
    def _fmt(val, unit, decimals=1):
        if val is None:
            return "—"
        return f"{val:.{decimals}f} {unit}"

    @staticmethod
    def _vital_colour(val, lo, hi):
        if val is None:
            return _C_GREY
        return _C_GREEN if lo <= val <= hi else _C_RED

    @staticmethod
    def _vital_colour_low(val, threshold):
        if val is None:
            return _C_GREY
        return _C_GREEN if val >= threshold else _C_RED
