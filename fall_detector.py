"""
fall_detector.py
────────────────
Wraps YOLOv8n-pose inference and converts raw keypoints into a
fall-probability score (0.0 – 1.0) using four weighted heuristics.

COCO Keypoint Index Reference
──────────────────────────────
 0  Nose          5  L-Shoulder   6  R-Shoulder
 7  L-Elbow       8  R-Elbow      9  L-Wrist   10  R-Wrist
11  L-Hip        12  R-Hip       13  L-Knee    14  R-Knee
15  L-Ankle      16  R-Ankle
"""

import math
import numpy as np
from collections import deque
from ultralytics import YOLO

import config


# ── COCO keypoint indices used by our heuristics ──────────────────────────
IDX_L_SHOULDER = 5
IDX_R_SHOULDER = 6
IDX_L_HIP      = 11
IDX_R_HIP      = 12
IDX_L_ANKLE    = 15
IDX_R_ANKLE    = 16


class FallDetector:
    """
    Loads YOLOv8n-pose and scores each frame for fall probability.

    Usage
    -----
    detector = FallDetector()
    result   = detector.process(frame)
    print(result.score, result.status)
    """

    def __init__(self):
        print(f"[FallDetector] Loading model: {config.MODEL_NAME} …")
        self._model = YOLO(config.MODEL_NAME)
        # Warm-up pass (avoids first-frame latency spike)
        dummy = np.zeros((config.INFERENCE_SIZE, config.INFERENCE_SIZE, 3), dtype=np.uint8)
        self._model(dummy, imgsz=config.INFERENCE_SIZE, device=config.DEVICE, verbose=False)
        print("[FallDetector] Model ready.")

        # Sliding window of hip Y positions (normalised) for velocity heuristic
        self._hip_y_history  = deque(maxlen=6)
        # Counter for consecutive fall-scored frames
        self._fall_streak     = 0
        # Last scored result (returned on skipped frames)
        self._last_result     = DetectionResult()

    # ── Public ────────────────────────────────────────────────────────────

    def process(self, frame, frame_idx: int) -> "DetectionResult":
        """Run inference (or skip) and return a DetectionResult."""
        # Skip frames as configured for performance
        if frame_idx % config.INFERENCE_EVERY_N != 0:
            return self._last_result

        h, w = frame.shape[:2]
        results = self._model(
            frame,
            imgsz   = config.INFERENCE_SIZE,
            device  = config.DEVICE,
            verbose = False,
        )

        result = self._build_result(results, frame_size=(w, h))
        self._update_streak(result)
        self._last_result = result
        return result

    # ── Private – orchestration ───────────────────────────────────────────

    def _build_result(self, yolo_results, frame_size) -> "DetectionResult":
        """Parse YOLO output and compute fall score."""
        result = DetectionResult()

        # Pick the most prominent person (largest bounding box)
        person = self._dominant_person(yolo_results)
        if person is None:
            # No person detected – treat as uncertain, low score
            self._hip_y_history.clear()
            return result

        keypoints = person["keypoints"]   # shape (17, 3): x, y, conf
        bbox      = person["bbox"]        # x1, y1, x2, y2

        result.keypoints = keypoints
        result.bbox      = bbox

        w_frame, h_frame = frame_size

        # ── Four heuristic sub-scores (each 0.0–1.0) ─────────────────────
        s_aspect  = self._score_aspect_ratio(bbox)
        s_torso   = self._score_torso_angle(keypoints)
        s_vel     = self._score_hip_velocity(keypoints, h_frame)
        s_conf    = self._score_keypoint_confidence(keypoints)

        result.sub_scores = {
            "aspect_ratio" : round(s_aspect, 3),
            "torso_angle"  : round(s_torso,  3),
            "hip_velocity" : round(s_vel,    3),
            "kp_confidence": round(s_conf,   3),
        }

        # Weighted aggregate
        score = (
            config.WEIGHT_ASPECT_RATIO        * s_aspect +
            config.WEIGHT_TORSO_ANGLE         * s_torso  +
            config.WEIGHT_HIP_VELOCITY        * s_vel    +
            config.WEIGHT_KEYPOINT_CONFIDENCE * s_conf
        )
        result.score  = min(max(score, 0.0), 1.0)
        result.status = self._classify(result.score)
        return result

    def _update_streak(self, result: "DetectionResult"):
        if result.status == "FALL":
            self._fall_streak += 1
        else:
            self._fall_streak = 0
        result.fall_confirmed = (self._fall_streak >= config.FALL_CONFIRM_FRAMES)
        result.fall_streak    = self._fall_streak

    # ── Private – person selection ────────────────────────────────────────

    def _dominant_person(self, yolo_results):
        """Return keypoints + bbox of the largest detected person, or None."""
        best_area = -1
        best      = None

        for r in yolo_results:
            if r.keypoints is None or r.boxes is None:
                continue
            kps   = r.keypoints.data.cpu().numpy()   # (N, 17, 3)
            boxes = r.boxes.xyxy.cpu().numpy()       # (N, 4)

            for i, (kp, box) in enumerate(zip(kps, boxes)):
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best = {"keypoints": kp, "bbox": box}
        return best

    # ── Private – heuristics ──────────────────────────────────────────────

    def _score_aspect_ratio(self, bbox) -> float:
        """
        Compares bounding-box height to width.
        Standing: tall (h >> w) → score near 0.
        Fallen  : flat (w >= h) → score near 1.
        """
        x1, y1, x2, y2 = bbox
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        ratio = box_h / box_w   # >2.5 = standing, <1.0 = fallen

        return _clamp_linear(
            ratio,
            high_input = config.ASPECT_RATIO_STANDING,  # ratio → score 0.0
            low_input  = config.ASPECT_RATIO_FALLEN,     # ratio → score 1.0
            invert     = True,
        )

    def _score_torso_angle(self, kps) -> float:
        """
        Angle of the torso (hip-midpoint → shoulder-midpoint) from vertical.
        Upright = 0°, flat = 90°.
        """
        ls, rs = kps[IDX_L_SHOULDER], kps[IDX_R_SHOULDER]
        lh, rh = kps[IDX_L_HIP],      kps[IDX_R_HIP]

        # Use only well-detected keypoints
        shoulder_conf = min(ls[2], rs[2])
        hip_conf      = min(lh[2], rh[2])
        if shoulder_conf < 0.2 or hip_conf < 0.2:
            return 0.5  # uncertain

        sx = (ls[0] + rs[0]) / 2
        sy = (ls[1] + rs[1]) / 2
        hx = (lh[0] + rh[0]) / 2
        hy = (lh[1] + rh[1]) / 2

        dx = sx - hx
        dy = sy - hy   # positive dy = shoulders above hips (normal)
        angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

        return _clamp_linear(
            angle_from_vertical,
            high_input = config.TORSO_ANGLE_FALLEN,    # → score 1.0
            low_input  = config.TORSO_ANGLE_STANDING,  # → score 0.0
        )

    def _score_hip_velocity(self, kps, frame_h: int) -> float:
        """
        Tracks normalised Y-position of hip midpoint over recent frames.
        A sudden downward spike → high score.
        """
        lh, rh = kps[IDX_L_HIP], kps[IDX_R_HIP]
        if lh[2] < 0.2 and rh[2] < 0.2:
            return 0.0  # can't see hips

        hip_y_norm = ((lh[1] + rh[1]) / 2) / max(frame_h, 1)
        self._hip_y_history.append(hip_y_norm)

        if len(self._hip_y_history) < 2:
            return 0.0

        # Max downward velocity in recent window (positive = downward in image)
        velocities = [
            self._hip_y_history[i] - self._hip_y_history[i - 1]
            for i in range(1, len(self._hip_y_history))
        ]
        max_downward = max(velocities)   # most downward movement

        return _clamp_linear(
            max_downward,
            low_input  = 0.0,
            high_input = config.HIP_VELOCITY_THRESHOLD,
        )

    def _score_keypoint_confidence(self, kps) -> float:
        """
        Low average keypoint confidence can indicate a collapsed or
        partially occluded person (edge case support).
        """
        mean_conf = float(np.mean(kps[:, 2]))
        # Low confidence → high sub-score (invert)
        return _clamp_linear(
            mean_conf,
            high_input = config.MIN_KEYPOINT_CONF,   # below this → score 1.0
            low_input  = 0.80,                        # above this → score 0.0
            invert     = True,
        )

    # ── Private – classification ──────────────────────────────────────────

    @staticmethod
    def _classify(score: float) -> str:
        if score < config.SCORE_NORMAL:
            return "NORMAL"
        elif score < config.SCORE_AT_RISK:
            return "AT RISK"
        else:
            return "FALL"


# ── Helpers ───────────────────────────────────────────────────────────────

def _clamp_linear(value, low_input, high_input, invert=False) -> float:
    """Linearly maps value from [low_input, high_input] → [0.0, 1.0]."""
    span = high_input - low_input
    if span == 0:
        return 0.0
    t = (value - low_input) / span
    t = min(max(t, 0.0), 1.0)
    return (1.0 - t) if invert else t


# ── Data class ────────────────────────────────────────────────────────────

class DetectionResult:
    """All information about a single frame's fall analysis."""

    def __init__(self):
        self.score          : float       = 0.0
        self.status         : str         = "NORMAL"
        self.keypoints      : np.ndarray  = None   # (17, 3)
        self.bbox           : np.ndarray  = None   # (x1, y1, x2, y2)
        self.sub_scores     : dict        = {}
        self.fall_confirmed : bool        = False
        self.fall_streak    : int         = 0
