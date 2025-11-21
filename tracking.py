# tracking.py
import cv2
import mediapipe as mp
import time
from collections import deque
from typing import Deque, Tuple, List, Optional

mp_hands = mp.solutions.hands


class HandTracker:
    """
    MediaPipe Hands wrapper with fingertip smoothing, wrist tracking,
    and recent path buffer for polyline swipe detection.
    """

    def __init__(self, max_len: int = 32, detection_conf: float = 0.6, tracking_conf: float = 0.5, smooth_alpha: float = 0.35):
        self.hands = mp_hands.Hands(static_image_mode=False,
                                    max_num_hands=1,
                                    min_detection_confidence=detection_conf,
                                    min_tracking_confidence=tracking_conf)
        self.recent: Deque[Tuple[int, int, float]] = deque(maxlen=max_len)  # x_px, y_px, ts
        self.frame_size = (640, 480)
        self.last_landmarks = None
        self.smoothed_tip: Optional[Tuple[float, float]] = None
        self.smoothed_wrist: Optional[Tuple[float, float]] = None
        self.smooth_alpha = smooth_alpha

    def to_px(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        w, h = self.frame_size
        return int(norm_x * w), int(norm_y * h)

    def process(self, frame_bgr) -> dict:
        h, w = frame_bgr.shape[:2]
        self.frame_size = (w, h)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        out = {
            "hand_present": False,
            "index_tip": None,
            "wrist": None,
            "landmarks": None,
            "handedness": None,
            "detection_confidence": 0.0
        }
        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            self.last_landmarks = lm
            out["hand_present"] = True
            ix, iy = lm.landmark[8].x, lm.landmark[8].y
            wx, wy = lm.landmark[0].x, lm.landmark[0].y
            tip_px = self.to_px(ix, iy)
            wrist_px = self.to_px(wx, wy)
            out["index_tip"] = tip_px
            out["wrist"] = wrist_px
            out["landmarks"] = lm
            out["detection_confidence"] = results.multi_handedness[0].classification[0].score
            out["handedness"] = results.multi_handedness[0].classification[0].label
            ts = time.time()

            # smoothing (exponential)
            if self.smoothed_tip is None:
                self.smoothed_tip = (tip_px[0], tip_px[1])
            else:
                sx = self.smooth_alpha * tip_px[0] + (1 - self.smooth_alpha) * self.smoothed_tip[0]
                sy = self.smooth_alpha * tip_px[1] + (1 - self.smooth_alpha) * self.smoothed_tip[1]
                self.smoothed_tip = (sx, sy)

            if self.smoothed_wrist is None:
                self.smoothed_wrist = (wrist_px[0], wrist_px[1])
            else:
                sx = self.smooth_alpha * wrist_px[0] + (1 - self.smooth_alpha) * self.smoothed_wrist[0]
                sy = self.smooth_alpha * wrist_px[1] + (1 - self.smooth_alpha) * self.smoothed_wrist[1]
                self.smoothed_wrist = (sx, sy)

            # store smoothed point
            self.recent.append((int(self.smoothed_tip[0]), int(self.smoothed_tip[1]), ts))
        return out

    def get_swipe_segments(self, min_speed_px_s: float = 900.0, min_length_px: float = 30.0) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        segs = []
        pts = list(self.recent)
        if len(pts) < 2:
            return segs
        for i in range(1, len(pts)):
            x1, y1, t1 = pts[i - 1]
            x2, y2, t2 = pts[i]
            dt = max(1e-6, t2 - t1)
            dx = x2 - x1
            dy = y2 - y1
            speed = ((dx * dx + dy * dy) ** 0.5) / dt
            length = (dx * dx + dy * dy) ** 0.5
            if speed >= min_speed_px_s and length >= min_length_px:
                segs.append(((x1, y1), (x2, y2)))
        return segs

    def get_recent_polyline(self) -> List[Tuple[int, int]]:
        return [(x, y) for (x, y, _) in self.recent]

    def close(self):
        self.hands.close()
