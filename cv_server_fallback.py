import time
from typing import Any

import cv2


frame_count = 0
groq_call_count = 0
GROQ_SESSION_CAP = None


class _Scene:
    def __init__(self):
        self.frames_since_groq = 0


scene_change_detector = _Scene()


def run_sentinel_pipeline(frame, verbose=False, waypoint_id=None):
    global frame_count
    frame_count += 1
    scene_change_detector.frames_since_groq += 1

    h, w = frame.shape[:2]
    return {
        "frame": frame_count,
        "nav_hazards": [],
        "confirmed_threats": [],
        "gesture": None,
        "rover_command": "PROCEED",
        "threat_summary": [],
        "qml_waypoints": [],
        "scene_summary": {
            "mode": "fallback",
            "note": "Full cv_server unavailable; streaming camera with fallback CV module.",
            "resolution": f"{w}x{h}",
            "waypoint_id": waypoint_id,
        },
        "pipeline_time_s": 0.0,
    }


def draw_rich_overlay(frame, output: dict[str, Any]):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        out,
        "FALLBACK CV ACTIVE",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return out