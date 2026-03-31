"""
display.py
──────────
Two-column dashboard layout:

  Left panel  : VITALS  — Heart Rate · SpO₂ · HRV + rPPG waveform
  Right panel : MONITORING — Fall Risk · Stress · Respiration · Signal Quality
                             + annotated camera feed at the bottom
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

# Colour palette (BGR)
_C_GREEN   = (80,  210,  80)
_C_YELLOW  = (50,  210, 200)
_C_RED     = (60,   60, 220)
_C_BLUE    = (220, 150,  50)
_C_WHITE   = (230, 230, 230)
_C_GREY    = (130, 130, 130)
_C_DARK    = (16,   16,  26)
_C_PANEL   = (22,   22,  36)
_C_DIVIDER = (50,   50,  80)

# Layout
_CANVAS_H = 660
_LEFT_W   = 350    # VITALS panel
_RIGHT_W  = 580    # MONITORING + camera panel
_CANVAS_W = _LEFT_W + _RIGHT_W   # 930

# Camera thumbnail shown at the bottom of the right panel
_THUMB_W  = 380
_THUMB_H  = 285    # 4:3 ratio


class Display:
    def __init__(self):
        self._flash_frames_left = 0
        self._dot_frame = 0

    # ── Public ────────────────────────────────────────────────────────────

    def render(self, frame, fall_result, vitals_result, alert_fired: bool,
               fps: float = 0) -> np.ndarray:
        canvas = np.full((_CANVAS_H, _CANVAS_W, 3), _C_DARK, dtype=np.uint8)

        # Left panel
        left = self._build_left_panel(vitals_result, _CANVAS_H, _LEFT_W)
        canvas[:, :_LEFT_W] = left

        # Right panel (includes camera at bottom)
        right = self._build_right_panel(vitals_result, fall_result,
                                        frame, fps, _CANVAS_H, _RIGHT_W)
        canvas[:, _LEFT_W:] = right

        # Divider
        cv2.line(canvas, (_LEFT_W, 0), (_LEFT_W, _CANVAS_H), _C_DIVIDER, 1)

        # Fall alert flash
        if alert_fired:
            self._flash_frames_left = 18
        if self._flash_frames_left > 0:
            self._draw_flash(canvas, self._flash_frames_left)
            self._flash_frames_left -= 1

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

    # ── Left panel ────────────────────────────────────────────────────────

    def _build_left_panel(self, vr, h: int, w: int) -> np.ndarray:
        panel = np.full((h, w, 3), _C_PANEL, dtype=np.uint8)
        y = 18

        self._centered(panel, "VITALS", y, w, 0.65, _C_WHITE, 1)
        y += 36
        cv2.line(panel, (16, y), (w - 16, y), _C_DIVIDER, 1)
        y += 14

        # Status indicator
        if not vr.face_detected:
            s_txt, s_col = "No face detected", _C_RED
        elif not vr.ready:
            dots  = "." * ((self._dot_frame // 12) % 4)
            s_txt, s_col = f"Collecting{dots}", _C_YELLOW
        else:
            s_txt, s_col = "Live readings", _C_GREEN
        cv2.circle(panel, (16, y + 7), 5, s_col, -1)
        cv2.putText(panel, s_txt, (28, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, s_col, 1, cv2.LINE_AA)
        y += 28

        cv2.line(panel, (16, y), (w - 16, y), _C_DIVIDER, 1)
        y += 12

        # ── Heart Rate ────────────────────────────────────────────────────
        hr_col = self._vital_colour(vr.heart_rate,
                                    config.HR_NORMAL_LOW, config.HR_NORMAL_HIGH)
        y = self._big_metric(panel, "HR", "Heart Rate",
                             vr.heart_rate, "BPM", 0, hr_col, y, w)

        # ── SpO₂ ─────────────────────────────────────────────────────────
        spo2_col = self._vital_colour_low(vr.spo2, config.SPO2_NORMAL_LOW)
        y = self._big_metric(panel, "O2", "SpO2",
                             vr.spo2, "%", 1, spo2_col, y, w)

        # ── HRV ──────────────────────────────────────────────────────────
        y = self._big_metric(panel, "HRV", "HRV / RMSSD",
                             vr.hrv_rmssd, "ms", 0, _C_BLUE, y, w)

        y += 10
        cv2.line(panel, (16, y), (w - 16, y), _C_DIVIDER, 1)
        y += 12

        # ── rPPG waveform ─────────────────────────────────────────────────
        if vr.ready and len(vr.waveform) > 10:
            cv2.putText(panel, "rPPG Signal", (16, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, _C_GREY, 1, cv2.LINE_AA)
            y += 20
            self._draw_waveform(panel, vr.waveform, y, h - 16, w)

        return panel

    def _big_metric(self, panel, abbr, label, val, unit, decimals,
                    colour, y, w) -> int:
        block_h = 95

        # Left accent strip
        cv2.rectangle(panel, (5, y + 6), (9, y + block_h - 10), colour, -1)

        # Abbreviation
        cv2.putText(panel, abbr, (18, y + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, colour, 1, cv2.LINE_AA)

        # Label
        cv2.putText(panel, label, (18, y + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, _C_GREY, 1, cv2.LINE_AA)

        # Large value
        val_str = f"{val:.{decimals}f}" if val is not None else "--"
        fs, tk = 1.35, 2
        (tw, _), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_DUPLEX, fs, tk)
        cv2.putText(panel, val_str, (18, y + 84),
                    cv2.FONT_HERSHEY_DUPLEX, fs, colour, tk, cv2.LINE_AA)

        # Unit
        if val is not None:
            cv2.putText(panel, unit, (18 + tw + 7, y + 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, _C_GREY, 1, cv2.LINE_AA)

        cv2.line(panel, (16, y + block_h - 2),
                 (w - 16, y + block_h - 2), _C_DIVIDER, 1)
        return y + block_h + 2

    # ── Right panel ───────────────────────────────────────────────────────

    def _build_right_panel(self, vr, fr, frame, fps,
                           h: int, w: int) -> np.ndarray:
        panel = np.full((h, w, 3), _C_PANEL, dtype=np.uint8)
        y = 18

        self._centered(panel, "MONITORING", y, w, 0.58, _C_WHITE, 1)
        y += 36
        cv2.line(panel, (16, y), (w - 16, y), _C_DIVIDER, 1)
        y += 16

        fall_col  = _STATUS_COLOUR.get(fr.status, _C_GREY)
        bar_w     = w - 32

        # ── Fall Risk box ─────────────────────────────────────────────────
        risk_label = "FALL DETECTED" if fr.fall_confirmed else fr.status
        cv2.rectangle(panel, (12, y), (w - 12, y + 70), fall_col, 2)
        icon = "!" if fr.score >= config.SCORE_AT_RISK else "+"
        cv2.putText(panel, icon, (24, y + 36),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, fall_col, 2, cv2.LINE_AA)
        cv2.putText(panel, "Fall Risk:", (56, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, _C_GREY, 1, cv2.LINE_AA)
        cv2.putText(panel, risk_label, (56, y + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, fall_col, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{fr.score * 100:.1f}% probability",
                    (18, y + 64), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, _C_GREY, 1, cv2.LINE_AA)
        y += 76

        # Risk score bar
        cv2.rectangle(panel, (16, y), (16 + bar_w, y + 10), (38, 38, 58), -1)
        fill = int(bar_w * fr.score)
        cv2.rectangle(panel, (16, y), (16 + fill, y + 10), fall_col, -1)
        y += 22

        if fps > 0:
            fps_txt = f"FPS: {fps:.1f}"
            (fw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(panel, fps_txt, (w - fw - 14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, _C_GREY, 1, cv2.LINE_AA)

        cv2.line(panel, (16, y + 6), (w - 16, y + 6), _C_DIVIDER, 1)
        y += 20

        # ── Stress ────────────────────────────────────────────────────────
        stress_col = {
            "Low"    : _C_GREEN,
            "Medium" : _C_YELLOW,
            "High"   : _C_RED,
        }.get(vr.stress_label, _C_GREY)
        stress_val = vr.stress_label if vr.stress_label not in ("—", "--", "") else "--"

        cv2.putText(panel, "Stress Level", (16, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _C_GREY, 1, cv2.LINE_AA)
        fs_s = 0.95
        (tw, _), _ = cv2.getTextSize(stress_val, cv2.FONT_HERSHEY_DUPLEX, fs_s, 2)
        cv2.putText(panel, stress_val, (w - tw - 18, y + 46),
                    cv2.FONT_HERSHEY_DUPLEX, fs_s, stress_col, 2, cv2.LINE_AA)
        y += 58

        cv2.line(panel, (16, y), (w - 16, y), _C_DIVIDER, 1)
        y += 16

        # ── Respiratory Rate ──────────────────────────────────────────────
        rr_col = self._vital_colour(vr.resp_rate,
                                    config.RR_NORMAL_LOW, config.RR_NORMAL_HIGH)
        cv2.putText(panel, "Respiration", (16, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _C_GREY, 1, cv2.LINE_AA)
        rr_str = f"{vr.resp_rate:.0f}" if vr.resp_rate is not None else "--"
        fs_rr = 1.15
        (tw, _), _ = cv2.getTextSize(rr_str, cv2.FONT_HERSHEY_DUPLEX, fs_rr, 2)
        cv2.putText(panel, rr_str, (16, y + 56),
                    cv2.FONT_HERSHEY_DUPLEX, fs_rr, rr_col, 2, cv2.LINE_AA)
        if vr.resp_rate is not None:
            cv2.putText(panel, "br/min", (16 + tw + 8, y + 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, _C_GREY, 1, cv2.LINE_AA)
        y += 70

        cv2.line(panel, (16, y), (w - 16, y), _C_DIVIDER, 1)
        y += 14

        # ── Signal Quality ────────────────────────────────────────────────
        cv2.putText(panel, "Signal Quality", (16, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, _C_GREY, 1, cv2.LINE_AA)
        sq_col = (_C_GREEN if vr.signal_quality > 60
                  else (_C_YELLOW if vr.signal_quality > 30 else _C_RED))
        bar_y = y + 22
        cv2.rectangle(panel, (16, bar_y), (16 + bar_w, bar_y + 12), (38, 38, 58), -1)
        fill_sq = int(bar_w * vr.signal_quality / 100.0)
        cv2.rectangle(panel, (16, bar_y), (16 + fill_sq, bar_y + 12), sq_col, -1)
        pct_sq = f"{vr.signal_quality:.0f}%"
        (ptw, _), _ = cv2.getTextSize(pct_sq, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(panel, pct_sq, (w - ptw - 14, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, sq_col, 1, cv2.LINE_AA)
        y += 46

        cv2.line(panel, (16, y), (w - 16, y), _C_DIVIDER, 1)
        y += 12

        # ── Camera feed (bottom of right panel) ───────────────────────────
        remaining = h - y - 16
        if remaining >= 80:
            cam_h = min(_THUMB_H, remaining)
            cam_w = cam_h * 4 // 3
            if cam_w > w - 32:
                cam_w = w - 32
                cam_h = cam_w * 3 // 4

            self._draw_camera_thumb(panel, frame, fr,
                                    y, cam_w, cam_h, w)

        return panel

    def _draw_camera_thumb(self, panel, frame, fall_result,
                           y_start, cam_w, cam_h, panel_w):
        # Annotate frame copy
        annotated = frame.copy()
        col = _STATUS_COLOUR.get(fall_result.status, _C_GREY)

        if fall_result.keypoints is not None:
            if config.SHOW_BBOX and fall_result.bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in fall_result.bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            if config.SHOW_SKELETON:
                for x, y, conf in fall_result.keypoints:
                    if conf > 0.25:
                        cv2.circle(annotated, (int(x), int(y)), 3, col, -1)
                for a, b in _SKELETON_PAIRS:
                    xa, ya, ca = fall_result.keypoints[a]
                    xb, yb, cb = fall_result.keypoints[b]
                    if ca > 0.25 and cb > 0.25:
                        cv2.line(annotated,
                                 (int(xa), int(ya)), (int(xb), int(yb)),
                                 col, 2, cv2.LINE_AA)

        thumb = cv2.resize(annotated, (cam_w, cam_h))

        # Center horizontally in the panel
        tx = (panel_w - cam_w) // 2
        ty = y_start

        # Border
        cv2.rectangle(panel,
                      (tx - 2, ty - 2),
                      (tx + cam_w + 2, ty + cam_h + 2),
                      col, 2)

        # LIVE badge
        cv2.rectangle(panel, (tx, ty), (tx + 52, ty + 20), (18, 18, 30), -1)
        cv2.putText(panel, "LIVE", (tx + 5, ty + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 220, 100), 1, cv2.LINE_AA)
        cv2.circle(panel, (tx + 46, ty + 10), 4, (80, 220, 100), -1)

        panel[ty:ty + cam_h, tx:tx + cam_w] = thumb

    # ── Flash overlay ─────────────────────────────────────────────────────

    def _draw_flash(self, canvas, frames_left):
        alpha = min(frames_left / 9.0, 1.0) * 0.40
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (_CANVAS_W, _CANVAS_H), (0, 0, 200), -1)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
        text = "FALL ALERT"
        fs, tk = 2.0, 3
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, tk)
        cx = (_CANVAS_W - tw) // 2
        cy = (_CANVAS_H + th) // 2
        cv2.putText(canvas, text, (cx + 2, cy + 2),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (0, 0, 0), tk + 1, cv2.LINE_AA)
        cv2.putText(canvas, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (80, 80, 255), tk, cv2.LINE_AA)

    # ── Waveform ──────────────────────────────────────────────────────────

    def _draw_waveform(self, panel, waveform, y_start, y_end, w):
        plot_h = max(y_end - y_start - 4, 30)
        plot_w = w - 32
        mid_y  = y_start + plot_h // 2

        cv2.rectangle(panel, (16, y_start), (16 + plot_w, y_start + plot_h),
                      (26, 34, 26), -1)
        cv2.line(panel, (16, mid_y), (16 + plot_w, mid_y), (50, 60, 50), 1)

        n = len(waveform)
        if n < 2:
            return
        pts = []
        for i, v in enumerate(waveform):
            px = int(16 + i * plot_w / n)
            py = int(mid_y - v * (plot_h // 2 - 3))
            py = int(np.clip(py, y_start + 1, y_start + plot_h - 1))
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(panel, pts[i - 1], pts[i], (60, 220, 100), 1, cv2.LINE_AA)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _centered(self, panel, text, y, w, fs, colour, thick):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, thick)
        x = (w - tw) // 2
        cv2.putText(panel, text, (x, y + th),
                    cv2.FONT_HERSHEY_DUPLEX, fs, colour, thick, cv2.LINE_AA)

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
