# utils.py
import math
import json
import os
from typing import Tuple, List, Optional

HIGH_SCORE_FILE = "high_scores.json"


def line_intersects_circle(p1: Tuple[float, float], p2: Tuple[float, float],
                           center: Tuple[float, float], radius: float) -> bool:
    (x1, y1), (x2, y2) = p1, p2
    (cx, cy) = center
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(cx - x1, cy - y1) <= radius
    t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    dist = math.hypot(cx - nearest_x, cy - nearest_y)
    return dist <= radius


def line_segment_distance(p1: Tuple[float, float], p2: Tuple[float, float], c: Tuple[float, float]) -> float:
    (x1, y1), (x2, y2) = p1, p2
    (cx, cy) = c
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(cx - x1, cy - y1)
    t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    nx = x1 + t * dx
    ny = y1 + t * dy
    return math.hypot(cx - nx, cy - ny)


def polyline_intersects_circle(polyline: List[Tuple[float, float]], center: Tuple[float, float],
                               radius: float, thickness: float = 12.0) -> bool:
    if not polyline or len(polyline) < 2:
        return False
    for i in range(1, len(polyline)):
        if line_segment_distance(polyline[i - 1], polyline[i], center) <= (radius + thickness):
            return True
    return False


def save_high_score(name: str, score: int):
    data = {}
    if os.path.exists(HIGH_SCORE_FILE):
        try:
            with open(HIGH_SCORE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[name] = max(score, data.get(name, 0))
    with open(HIGH_SCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_high_scores() -> dict:
    if not os.path.exists(HIGH_SCORE_FILE):
        return {}
    try:
        with open(HIGH_SCORE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
