# ============================================================
# SENTINEL CV PIPELINE v6 — GOOGLE COLAB
# Military & Rescue Autonomous Reconnaissance Rover
# ============================================================
# CHANGES FROM v5:
#   [R1] SceneChangeDetector — gates Groq calls on real delta
#   [R2] ObjectTracker — IoU dedup + grid-cell memory
#   [R3] Stripped enrichment — only what Groq can't see itself
#   [R4] Raised confidence thresholds (CONF_OBJECT 0.30→0.40)
#   [R5] Waypoint-aligned Groq — fires on checkpoint transitions
#   [R6] Compressed prompt — ~60% shorter, same assessment quality
# ============================================================



# ── CELL 2: Imports ─────────────────────────────────────────
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from google.colab import files
import numpy as np
from groq import Groq
import base64
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import json, time, urllib.request, math, io
from datetime import datetime, timezone
from collections import defaultdict
from PIL import Image

print("✅ All imports successful.")

# ── CELL 3: Configuration ────────────────────────────────────
import os
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
# Model options (groq SDK + console.groq.com API key):
#   "meta-llama/llama-4-scout-17b-16e-instruct" — latest multimodal ✓  ← recommended
#   "llama-3.2-90b-vision-preview"  — smarter, multimodal ✓
#   "llama-3.2-11b-vision-preview"  — fast, multimodal ✓
# NOTE: All Groq vision models are multimodal (image + text input supported).
# Set your key: export GROQ_API_KEY="gsk_..."
GROQ_MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"

# All Groq vision models support image input.
MODEL_IS_MULTIMODAL = True

# Request timeout — prevents indefinite hang on bad model strings or network issues.
GROQ_REQUEST_TIMEOUT = 45

DANGER_THRESHOLD       = 5
GROQ_MIN_INTERVAL    = 3.0   # wall-clock seconds — kept as backstop

# [R4] Raised thresholds — reduces garbage candidates
CONF_PERSON            = 0.40
CONF_OBJECT            = 0.40  # was 0.30 — eliminates low-confidence noise
PERSISTENCE_THRESHOLD  = 2
MOTION_MIN_AREA        = 1200
MOTION_THRESHOLD       = 22
HSV_FIRE_MIN_AREA      = 1500  # was 800 — cuts small orange-patch false positives
HSV_SMOKE_MIN_COVERAGE = 0.15
YOLO_IMGSZ             = 640

# [R1] Scene-change gating parameters
SCENE_DELTA_NEW_CLASS_WEIGHT  = 1.0  # full point for a brand-new class appearing
SCENE_DELTA_BBOX_MOVE_THRESH  = 0.15 # fraction of frame width = significant move
# [FIX 1] Raised 1.0 → 2.5 — requires multiple meaningful changes before Groq fires.
# Scoring: new class=+1.0, class gone=+0.5, significant move=+0.5, special candidate=+0.8
# Examples: 3 new classes=3.0 ✓ | 2 new + fire=2.8 ✓ | 1 new + fire=1.8 ✗
SCENE_DELTA_MIN_TRIGGER       = 2.5

# [FIX 2] NEW — minimum processed frames between any two Groq calls.
# Frame-count based, not time-based — actually limits calls during video processing.
# At FRAME_SKIP=3 on 30fps: 8 processed frames ≈ 0.8s of real footage between calls.
GROQ_MIN_FRAMES_BETWEEN     = 8

# [FIX 3] NEW — hard cap on total Groq calls per session.
# Free tier: 1,500/day. 40 calls is conservative, leaves room for multiple runs.
# Set to 0 to disable.
GROQ_SESSION_CAP            = 40

# [R2] Object tracker parameters
IOU_DEDUP_THRESHOLD    = 0.50  # merge same-class boxes with IoU > this
GRID_CELLS_X           = 10    # frame divided into 10x8 grid for position quantization
GRID_CELLS_Y           = 8
TRACKER_TTL_FRAMES     = 20    # known objects expire after this many frames

# Debug flags
DEBUG_CANDIDATES       = True
DEBUG_GROQ_PROMPT    = False
DEBUG_GROQ_RAW       = True
DEBUG_STAGE_TIMINGS    = True
DEBUG_SCENE_DELTA      = True  # [R1] show what triggered (or suppressed) Groq

print("✅ Config set.")
print(f"   Model: {GROQ_MODEL} | multimodal={'YES' if MODEL_IS_MULTIMODAL else 'NO — text-only'}")
print(f"   Groq gates: scene_delta>={SCENE_DELTA_MIN_TRIGGER} "
      f"AND {GROQ_MIN_FRAMES_BETWEEN}+ frames since last call "
      f"AND {GROQ_MIN_INTERVAL}s wall-clock")
print(f"   Session cap: {GROQ_SESSION_CAP if GROQ_SESSION_CAP else 'disabled'} calls")

# ── CELL 4: Initialize Models ────────────────────────────────
yolo_model = YOLO('yolov8x.pt')
print("✅ YOLOv8x loaded.")

groq_client = Groq(api_key=GROQ_API_KEY)
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. "
                     "Set it with: export GROQ_API_KEY='gsk_...'")
print(f"✅ Groq client ready ({GROQ_MODEL}).")

print("Downloading MediaPipe gesture model...")
urllib.request.urlretrieve(
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task",
    "gesture_recognizer.task"
)
print("✅ MediaPipe ready.")

# ── CELL 5: Threat State ─────────────────────────────────────
threat_log          = []
active_threats      = []
pipeline_trace      = []
candidate_log       = []
rejection_log       = []
rover_command       = "PROCEED"
persistence_counter = defaultdict(int)
last_groq_call    = 0.0
groq_call_count   = 0         # [FIX 3] tracks total Groq calls this session
last_groq_frame   = -999      # [FIX 2] frame number of last Groq call
prev_frame_global   = None
frame_count         = 0
current_waypoint_id = None   # [R5] tracks active QML waypoint

def _now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _trace(stage, message, data=None):
    entry = {"frame": frame_count, "stage": stage,
             "message": message, "data": data, "time": _now_iso()}
    pipeline_trace.append(entry)
    return entry

def get_latest_threats():    return active_threats.copy()
def get_rover_command():     return rover_command

def get_waypoints_for_qml():
    checkpoints, seen = [], set()
    for t in threat_log:
        if not t.get("threat_confirmed"): continue
        cid = f"{t['threat_type']}_{len(checkpoints)}"
        if cid not in seen:
            checkpoints.append({
                "id": cid, "x": t["rover_position"]["x"], "y": t["rover_position"]["y"],
                "type": t["threat_type"], "severity": t["severity"],
                "danger_score": t.get("danger_score", 0),
                "specific_description": t.get("specific_description", ""),
            })
            seen.add(cid)
    return checkpoints

def emit_threat(t):
    global active_threats, rover_command
    t["_timestamp_unix"] = time.time()
    threat_log.append(t)
    active_threats.append(t)
    cutoff = time.time() - 10
    active_threats = [x for x in active_threats if x.get("_timestamp_unix", 0) > cutoff]
    sev = t.get("severity", "low")
    if sev == "critical":                             rover_command = "STOP"
    elif sev in ("high", "medium") and rover_command == "PROCEED": rover_command = "REROUTE"
    icons = {"critical":"🔴","high":"🟠","medium":"🟡","low":"🟢","attention":"🔵"}
    label = "✅ THREAT CONFIRMED" if t.get("threat_confirmed") else "ℹ️  ATTENTION (unverified)"
    print(f"\n{icons.get(sev,'⚪')} {label}")
    print(f"   Description  : {t.get('specific_description','N/A')}")
    print(f"   Danger score : {t.get('danger_score','?')}/10  |  Severity: {sev.upper()}")
    print(f"   Source       : {t.get('source','?')}  |  Rover cmd: {rover_command}")
    if t.get("groq_reasoning"):
        print(f"   AI reasoning : {t['groq_reasoning'][:180]}")


# ══════════════════════════════════════════════════════════════
# [R1] SCENE CHANGE DETECTOR
# Compares current YOLO detections against last Groq-validated
# scene. Only unlocks Groq when the delta is meaningful.
# ══════════════════════════════════════════════════════════════

class SceneChangeDetector:
    """
    Gates Groq calls on THREE conditions (all must pass):
      1. Scene delta >= SCENE_DELTA_MIN_TRIGGER (meaningful change)
      2. >= GROQ_MIN_FRAMES_BETWEEN frames since last call [FIX 2]
      3. >= GROQ_MIN_INTERVAL wall-clock seconds since last call
      4. Session cap not exceeded [FIX 3]
    Waypoint transitions bypass all gates except session cap.
    """
    def __init__(self):
        self.last_validated_classes   = set()
        self.last_validated_positions = {}
        self.frames_since_groq      = 0
        self.forced_waypoint_trigger  = False

    def force_trigger(self):
        self.forced_waypoint_trigger = True
        _trace("SCENE_DELTA", "Waypoint transition → Groq force-armed")

    def compute_delta(self, candidates, frame_w, frame_h):
        """
        Returns (should_call_groq: bool, delta_score: float, reasons: list).
        Now enforces frame-count gate and session cap in addition to delta.
        """
        self.frames_since_groq += 1

        # [FIX 3] Session cap — hard stop regardless of everything else
        if GROQ_SESSION_CAP and groq_call_count >= GROQ_SESSION_CAP:
            if DEBUG_SCENE_DELTA:
                print(f"\n   [SCENE DELTA] 🚫 SESSION CAP REACHED "
                      f"({groq_call_count}/{GROQ_SESSION_CAP}) — Groq disabled")
            return False, 0.0, ["session_cap_reached"]

        # Waypoint force-trigger bypasses delta + frame gate (not cap)
        if self.forced_waypoint_trigger:
            self.forced_waypoint_trigger = False
            return True, 99.0, ["waypoint_transition"]

        # [FIX 2] Frame-count gate — checked before computing delta (cheap)
        frames_since_last = frame_count - last_groq_frame
        if frames_since_last < GROQ_MIN_FRAMES_BETWEEN:
            if DEBUG_SCENE_DELTA:
                print(f"\n   [SCENE DELTA] ⏭ FRAME GATE — only {frames_since_last}/"
                      f"{GROQ_MIN_FRAMES_BETWEEN} frames since last call")
            return False, 0.0, [f"frame_gate:{frames_since_last}/{GROQ_MIN_FRAMES_BETWEEN}"]

        current_classes   = {c["yolo_class"] for c in candidates}
        current_positions = {}
        for c in candidates:
            b  = c.get("bbox", {})
            cx = (b.get("x", 0) + b.get("w", 0) / 2) / max(frame_w, 1)
            cy = (b.get("y", 0) + b.get("h", 0) / 2) / max(frame_h, 1)
            current_positions[c["yolo_class"]] = [cx, cy]

        delta_score = 0.0
        reasons     = []

        new_classes = current_classes - self.last_validated_classes
        for cls in new_classes:
            delta_score += SCENE_DELTA_NEW_CLASS_WEIGHT
            reasons.append(f"new_class:{cls}")

        gone_classes = self.last_validated_classes - current_classes
        for cls in gone_classes:
            delta_score += 0.5
            reasons.append(f"class_gone:{cls}")

        for cls in (current_classes & self.last_validated_classes):
            prev = self.last_validated_positions.get(cls)
            curr = current_positions.get(cls)
            if prev and curr:
                move = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
                if move > SCENE_DELTA_BBOX_MOVE_THRESH:
                    delta_score += 0.5
                    reasons.append(f"moved:{cls}_{round(move,2)}")

        for c in candidates:
            if c["candidate_source"] in ("hsv_fire", "hsv_smoke", "cv_structural"):
                delta_score += 0.8
                reasons.append(f"special_candidate:{c['candidate_source']}")
                break

        should_call = delta_score >= SCENE_DELTA_MIN_TRIGGER

        if DEBUG_SCENE_DELTA:
            status = "🟢 GROQ ARMED" if should_call else "⏭ GROQ SKIPPED"
            print(f"\n   [SCENE DELTA] {status} | score={round(delta_score,2)}/{SCENE_DELTA_MIN_TRIGGER} | "
                  f"reasons={reasons[:4]} | calls={groq_call_count}"
                  f"{'/' + str(GROQ_SESSION_CAP) if GROQ_SESSION_CAP else ''}")

        return should_call, delta_score, reasons

    def mark_validated(self, candidates, frame_w, frame_h):
        self.last_validated_classes   = {c["yolo_class"] for c in candidates}
        self.last_validated_positions = {}
        for c in candidates:
            b  = c.get("bbox", {})
            cx = (b.get("x", 0) + b.get("w", 0) / 2) / max(frame_w, 1)
            cy = (b.get("y", 0) + b.get("h", 0) / 2) / max(frame_h, 1)
            self.last_validated_positions[c["yolo_class"]] = [cx, cy]
        self.frames_since_groq = 0
        _trace("SCENE_DELTA", f"Baseline updated: {self.last_validated_classes}")


scene_change_detector = SceneChangeDetector()


# ══════════════════════════════════════════════════════════════
# [R2] OBJECT TRACKER
# Prevents re-sending known objects to Groq when they haven't
# meaningfully changed. Uses IoU dedup + grid-cell memory.
# ══════════════════════════════════════════════════════════════

class ObjectTracker:
    """
    Two-layer deduplication:
      Layer 1 — IoU-based NMS: merges overlapping same-class boxes
                within a single frame's candidate list.
      Layer 2 — Grid-cell memory: remembers (class, grid_cell) pairs
                that have already been validated by Groq this waypoint.
                Suppresses re-sending until the rover moves to a new
                waypoint or the object's grid position changes.
    """
    def __init__(self):
        self.known_objects = {}  # {(class, grid_x, grid_y): last_frame_seen}

    def reset_for_waypoint(self):
        """Clear tracker on waypoint transition — fresh eyes at each checkpoint."""
        self.known_objects = {}
        _trace("TRACKER", "Reset for new waypoint")

    def _iou(self, b1, b2):
        x1 = max(b1["x"], b2["x"])
        y1 = max(b1["y"], b2["y"])
        x2 = min(b1["x"]+b1["w"], b2["x"]+b2["w"])
        y2 = min(b1["y"]+b1["h"], b2["y"]+b2["h"])
        inter = max(0, x2-x1) * max(0, y2-y1)
        if inter == 0: return 0.0
        a1 = b1["w"] * b1["h"]
        a2 = b2["w"] * b2["h"]
        return inter / (a1 + a2 - inter)

    def _grid_cell(self, bbox, frame_w, frame_h):
        cx = bbox.get("x", 0) + bbox.get("w", 0) / 2
        cy = bbox.get("y", 0) + bbox.get("h", 0) / 2
        gx = int(cx / frame_w * GRID_CELLS_X)
        gy = int(cy / frame_h * GRID_CELLS_Y)
        return (min(gx, GRID_CELLS_X-1), min(gy, GRID_CELLS_Y-1))

    def iou_deduplicate(self, candidates):
        """
        Layer 1: within-frame IoU NMS across same-class candidates.
        Returns deduplicated list (higher-confidence box wins).
        """
        if len(candidates) <= 1: return candidates
        keep   = []
        used   = [False] * len(candidates)
        sorted_c = sorted(candidates, key=lambda c: c.get("confidence", 0), reverse=True)
        for i, ca in enumerate(sorted_c):
            if used[i]: continue
            keep.append(ca)
            for j, cb in enumerate(sorted_c[i+1:], start=i+1):
                if used[j]: continue
                if ca["yolo_class"] != cb["yolo_class"]: continue
                if self._iou(ca.get("bbox",{}), cb.get("bbox",{})) > IOU_DEDUP_THRESHOLD:
                    used[j] = True
        removed = len(candidates) - len(keep)
        if removed > 0:
            _trace("TRACKER", f"IoU dedup removed {removed} overlapping boxes")
            if DEBUG_CANDIDATES:
                print(f"   [TRACKER IoU] Removed {removed} duplicate boxes (IoU>{IOU_DEDUP_THRESHOLD})")
        return keep

    def filter_known(self, candidates, frame_w, frame_h):
        """
        Layer 2: suppress candidates whose (class, grid_cell) has
        already been validated by Groq this waypoint.
        Returns (fresh_candidates, suppressed_count).
        """
        # Expire stale entries
        stale_keys = [k for k, f in self.known_objects.items()
                      if frame_count - f > TRACKER_TTL_FRAMES]
        for k in stale_keys: del self.known_objects[k]

        fresh      = []
        suppressed = 0
        for c in candidates:
            key = (c["yolo_class"], *self._grid_cell(c.get("bbox",{}), frame_w, frame_h))
            if key in self.known_objects:
                suppressed += 1
                if DEBUG_CANDIDATES:
                    print(f"   [TRACKER SUPPRESS] {c['yolo_class']} at grid {key[1:]}"
                          f" — already validated {frame_count - self.known_objects[key]} frames ago")
            else:
                fresh.append(c)
        return fresh, suppressed

    def mark_validated(self, candidates, frame_w, frame_h):
        """Record all candidates that were sent to Groq this call."""
        for c in candidates:
            key = (c["yolo_class"], *self._grid_cell(c.get("bbox",{}), frame_w, frame_h))
            self.known_objects[key] = frame_count


object_tracker = ObjectTracker()


# ══════════════════════════════════════════════════════════════
# [R5] WAYPOINT MANAGER
# Notifies both SceneChangeDetector and ObjectTracker when the
# rover transitions to a new QML checkpoint.
# ══════════════════════════════════════════════════════════════

def on_waypoint_transition(new_waypoint_id):
    """
    Call this when the rover WebSocket reports a checkpoint arrival.
    Forces a fresh Groq assessment and clears object memory.
    """
    global current_waypoint_id
    if new_waypoint_id == current_waypoint_id: return
    current_waypoint_id = new_waypoint_id
    scene_change_detector.force_trigger()
    object_tracker.reset_for_waypoint()
    print(f"\n🗺  WAYPOINT TRANSITION → {new_waypoint_id} | Groq armed, tracker cleared")
    _trace("WAYPOINT", f"Transitioned to {new_waypoint_id}")


# ══════════════════════════════════════════════════════════════
# [R3] STRIPPED ENRICHMENT PRIMITIVES
# Only compute what Groq cannot infer from pixels directly.
# Removed: get_dominant_colors, estimate_texture, estimate_material,
#          classify_environment, analyze_object_relationships
# Kept:    frame region, vertical zone, relative distance, posture
#          (aspect ratio — trivially cheap, and Groq needs coords)
# ══════════════════════════════════════════════════════════════

def infer_environment_tags(all_cls):
    """
    [NEW] Cheap environment inference from detected classes only.
    Zero CV computation — pure set lookups on already-computed YOLO output.
    Replaces the expensive classify_environment() that was stripped in R3.
    This gives Groq the context it needs to distinguish
    'TV in office = normal furniture' from 'TV blocking corridor = hazard'.
    """
    cls_set = set(all_cls)
    tags    = []

    # Workspace / office
    if {'laptop', 'chair'} & cls_set:
        tags.append("office_or_workspace")
    if {'laptop', 'tv', 'monitor'} & cls_set and 'person' in cls_set:
        tags.append("occupied_workspace")

    # Residential
    if 'bed' in cls_set:
        tags.append("residential_bedroom")
    if {'toilet', 'sink'} & cls_set:
        tags.append("bathroom")
    if {'couch', 'tv'} & cls_set and 'laptop' not in cls_set:
        tags.append("living_area")

    # Outdoor / transit
    if {'car', 'truck', 'bus', 'motorcycle'} & cls_set:
        tags.append("vehicle_or_outdoor_area")

    # Clutter / chaos — potential distress context
    furniture_count = sum(1 for c in all_cls
                          if c in {'chair','couch','bed','dining table','bench'})
    if furniture_count >= 4:
        tags.append("heavily_cluttered_possible_evacuation_or_damage")

    if not tags:
        tags.append("environment_unknown")
    return tags
    thirds = frame_width / 3
    if x_center < thirds:     return "left"
    if x_center < 2 * thirds: return "center"
    return "right"

def get_vertical_zone(y_center, frame_height):
    if y_center < frame_height * 0.33: return "upper"
    if y_center < frame_height * 0.66: return "mid_height"
    return "floor_level"

def get_relative_distance(bbox_area, frame_area):
    ratio = bbox_area / frame_area
    if ratio > 0.20: return "near_under_1m"
    if ratio > 0.08: return "mid_1_to_3m"
    if ratio > 0.02: return "far_3_to_6m"
    return "distant_over_6m"

def estimate_posture(x1, y1, x2, y2):
    """Aspect ratio only — no CV computation required."""
    h, w = y2-y1, x2-x1
    if h == 0: return "unknown"
    a = h / max(w, 1)
    if a > 2.8: return "standing"
    if a > 2.0: return "standing_or_walking"
    if a > 1.4: return "seated_or_crouching"
    if a > 0.7: return "prone_or_fallen"
    return "collapsed_or_crawling"

def get_anomaly_score(cls_name, v_zone, frame_region, posture="n/a"):
    """
    Lightweight anomaly score — checks posture field (not v_zone) for
    prone/fallen persons. v_zone alone is insufficient because a fallen
    person detected at mid-height still has posture=prone_or_fallen.
    """
    score = 0.0
    if cls_name == "person":
        # [BUG FIX] Check posture string, not v_zone — posture is computed from
        # aspect ratio and is reliable; v_zone just reflects bbox position in frame
        if any(p in posture for p in ("prone", "fallen", "collapsed", "crawling")):
            score += 0.8   # high priority — could be injured survivor
        elif "floor_level" in v_zone:
            score += 0.3   # person at floor level even if posture ambiguous
        elif "crouching" in posture or "kneeling" in posture:
            score += 0.2
    if cls_name in ("backpack", "suitcase", "handbag") and frame_region == "center":
        score += 0.4
    if cls_name in ("knife", "scissors", "baseball bat"):
        score += 0.7
    return round(min(score, 1.0), 2)


# ══════════════════════════════════════════════════════════════
# CANDIDATE GENERATORS
# ══════════════════════════════════════════════════════════════

GROQ_TRIGGER_CLASSES = {
    'person','backpack','handbag','suitcase','knife','scissors',
    'cell phone','laptop','bottle','baseball bat','umbrella',
    'tie','sports ball','fire hydrant'
}
NAV_HAZARD_CLASSES = {
    'chair','couch','bench','dining table','bed','toilet',
    'refrigerator','bicycle','motorcycle','car','truck','bus',
    'potted plant','tv','clock','vase'
}

def generate_yolo_candidates(frame, yolo_results):
    """
    [R3] Enrichment stripped to essentials.
    Only bbox, class, confidence, aspect-ratio posture, frame region,
    vertical zone, relative distance, and anomaly score are computed.
    Everything else (colors, texture, material, environment, relationships)
    is dropped — Groq sees the image directly.
    """
    names            = yolo_results.names
    frame_h, frame_w = frame.shape[:2]
    frame_area       = frame_h * frame_w
    all_cls          = [names[int(b.cls)] for b in yolo_results.boxes]
    n_people         = sum(1 for c in all_cls if c == "person")
    crowd = (["no_people","single","small_group_2_3","group_4_7","crowd_8_plus"]
             [min(n_people, 4)])

    scene_summary = {
        "crowd_density":    crowd,
        "total_detections": len(yolo_results.boxes),
        "detected_classes": list(set(all_cls)),
        # [NEW] Cheap class-based environment inference — replaces stripped classify_environment.
        # Gives Groq context to distinguish 'TV in office' vs 'TV blocking corridor'.
        "environment_tags": infer_environment_tags(all_cls),
    }

    _trace("YOLO", f"Raw detections: {len(yolo_results.boxes)} — {list(set(all_cls))}")

    groq_candidates = []
    nav_hazards       = []

    for box in yolo_results.boxes:
        cls_name   = names[int(box.cls)]
        confidence = float(box.conf)
        if confidence < CONF_OBJECT: continue   # [R4] higher gate than v5

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw, bh          = x2-x1, y2-y1
        bbox_area       = bw * bh
        x_c, y_c        = (x1+x2)/2, (y1+y2)/2

        # [R3] Only cheap spatial computations — no k-means, no Canny, no Hough
        region   = get_frame_region(x_c, frame_w)
        v_zone   = get_vertical_zone(y_c, frame_h)
        distance = get_relative_distance(bbox_area, frame_area)

        # ── Nav hazard → self-emit (time-critical) ───────────
        if cls_name in NAV_HAZARD_CLASSES:
            persistence_counter[cls_name] += 1
            if persistence_counter[cls_name] < PERSISTENCE_THRESHOLD: continue
            if not (frame_w*0.12 < x_c < frame_w*0.88):               continue

            # [BUG FIX] Require the BOTTOM of the bbox to be in the lower 60% of
            # frame. Old check (y2 < 0.35) only skipped objects whose TOP was high —
            # wall-mounted TVs and shelved objects still passed. A real rover hazard
            # must actually reach the floor zone.
            if y2 < frame_h * 0.60: continue

            # [NEW] Context gate: in a clear office/workspace, TVs and monitors on
            # stands are normal furniture — not rover hazards — unless clearly fallen
            # (bbox height > 40% of frame = lying flat/knocked over).
            env_tags = infer_environment_tags(all_cls)
            is_normal_office_furniture = (
                cls_name in ('tv', 'monitor', 'refrigerator')
                and 'office_or_workspace' in env_tags
                and bh < frame_h * 0.40
            )
            if is_normal_office_furniture:
                if DEBUG_CANDIDATES:
                    print(f"\n   [NAV HAZARD SKIPPED] {cls_name} — normal office furniture "
                          f"(env={env_tags[0]}, height={round(bh/frame_h*100)}% frame)")
                continue
            fill_w = round(bw / frame_w * 100)
            low_ly = (y1 + bh) > frame_h * 0.75
            nav_hazards.append({
                "threat_type":         "pathway_blocked",
                "specific_description":(f"{cls_name.replace('_',' ').capitalize()} blocking "
                                        f"{fill_w}% of corridor at {region} — "
                                        f"{'floor-level' if low_ly else 'mid-height'} obstruction"),
                "object":              cls_name,
                "category":            "navigation_hazard",
                "subject_type":        "navigation_obstacle",
                "threat_confirmed":    True,
                "severity":            "high",
                "danger_score":        6,
                "confidence":          round(confidence, 2),
                "source":              "yolo_nav",
                "frame_region":        region,
                "vertical_zone":       v_zone,
                "relative_distance":   distance,
                "persistent":          True,
                "bbox":                {"x":x1,"y":y1,"w":bw,"h":bh},
                "groq_reasoning":      "Nav hazard self-confirmed — time-critical.",
                "key_indicators":      [f"blocking_{fill_w}pct_width", region],
                "timestamp":           _now_iso(),
                "rover_position":      {"x":0.0,"y":0.0},
            })
            if DEBUG_CANDIDATES:
                print(f"\n   [NAV HAZARD → SELF-EMIT] {cls_name} blocking {fill_w}% at {region}")
            continue

        # ── Groq-trigger → build lean candidate ────────────
        if cls_name in GROQ_TRIGGER_CLASSES and confidence >= CONF_PERSON:
            posture       = estimate_posture(x1, y1, x2, y2) if cls_name == "person" else "n/a"
            # [BUG FIX] Pass posture to anomaly score — previously passed v_zone which
            # caused prone persons to score 0.0 when detected at mid-height in frame
            anomaly_score = get_anomaly_score(cls_name, v_zone, region, posture)
            candidate = {
                "candidate_source":  "yolo",
                "yolo_class":        cls_name,
                "confidence":        round(confidence, 2),
                "bbox":              {"x":x1,"y":y1,"w":bw,"h":bh},
                "frame_region":      region,
                "vertical_zone":     v_zone,
                "relative_distance": distance,
                "posture":           posture,         # aspect-ratio only, free
                "anomaly_score":     anomaly_score,   # lightweight 2-signal score
                # [R3] REMOVED: dominant_colors, texture, material_estimate,
                #      scene_environment, object_relationships — Groq sees image
            }
            groq_candidates.append(candidate)
            candidate_log.append({**candidate, "frame": frame_count})
            if DEBUG_CANDIDATES:
                print(f"\n   [YOLO CANDIDATE → QUEUE] {cls_name} conf={round(confidence,2)} "
                      f"| posture={posture} | anomaly={anomaly_score} "
                      f"| {region} {v_zone} {distance}")

    return groq_candidates, nav_hazards, scene_summary


def generate_hsv_candidates(frame):
    """
    [R4] HSV_FIRE_MIN_AREA raised 800→1500.
    Circularity hint preserved — it's cheap and genuinely helps Groq
    reject solid circular objects (bottles, balls) vs. real fire.
    """
    hsv              = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_h, frame_w = frame.shape[:2]
    frame_area       = frame_h * frame_w
    candidates       = []

    fire_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,160,160]),   np.array([15,255,255])),
        cv2.inRange(hsv, np.array([165,160,160]), np.array([180,255,255]))
    )
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    for c in cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if area < HSV_FIRE_MIN_AREA: continue   # [R4] raised threshold
        x, y, w, h   = cv2.boundingRect(c)
        fill_pct      = round(area / frame_area * 100, 1)
        perimeter     = cv2.arcLength(c, True)
        circularity   = (4 * math.pi * area / (perimeter**2)) if perimeter > 0 else 0
        zone          = "upper" if (y+h/2) < frame_h*0.5 else "lower"
        candidate = {
            "candidate_source":  "hsv_fire",
            "yolo_class":        "fire_candidate",
            "bbox":              {"x":x,"y":y,"w":w,"h":h},
            "frame_region":      get_frame_region((x+x+w)/2, frame_w),
            "vertical_zone":     zone,
            "relative_distance": "near" if fill_pct > 8 else "mid",
            "fill_percent":      fill_pct,
            "shape_circularity": round(circularity, 3),
            "shape_note":       ("irregular_consistent_with_fire" if circularity < 0.4
                                 else "regular_shape_likely_solid_object"),
            "anomaly_score":     0.6,
        }
        candidates.append(candidate)
        candidate_log.append({**candidate, "frame": frame_count})
        if DEBUG_CANDIDATES:
            print(f"\n   [HSV FIRE → QUEUE] fill={fill_pct}% | circularity={round(circularity,3)} "
                  f"| {'⚠ irregular' if circularity<0.4 else '✓ regular (likely not fire)'}")

    smoke_mask     = cv2.morphologyEx(
        cv2.inRange(hsv, np.array([0,0,120]), np.array([180,45,220])),
        cv2.MORPH_OPEN, np.ones((15,15), np.uint8)
    )
    smoke_coverage = np.sum(smoke_mask > 0) / frame_area
    if smoke_coverage > HSV_SMOKE_MIN_COVERAGE:
        candidate = {
            "candidate_source": "hsv_smoke",
            "yolo_class":       "smoke_candidate",
            "bbox":             {"x":0,"y":0,"w":frame_w,"h":frame_h},
            "frame_region":     "full_scene",
            "vertical_zone":    "full",
            "relative_distance":"near",
            "coverage_percent": round(smoke_coverage*100, 1),
            "anomaly_score":    0.7 if smoke_coverage > 0.35 else 0.45,
        }
        candidates.append(candidate)
        candidate_log.append({**candidate, "frame": frame_count})
        if DEBUG_CANDIDATES:
            print(f"\n   [HSV SMOKE → QUEUE] coverage={round(smoke_coverage*100,1)}%")

    return candidates


def generate_structural_candidate(frame):
    """Unchanged from v5 — already lean."""
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray, (5,5), 0)
    fh, fw   = frame.shape[:2]
    fa       = fh * fw
    edges    = cv2.Canny(blurred, 40, 120)
    upper_d  = np.sum(edges[:fh//2,:] > 0) / (fa/2)
    lines    = cv2.HoughLinesP(edges,1,np.pi/180,40,
                                minLineLength=int(fw*0.08),maxLineGap=8)
    n_diag   = 0
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            ang = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
            if 20<ang<70 or 110<ang<160: n_diag += 1
    contours,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    small_irreg = sum(1 for c in contours if 50 < cv2.contourArea(c) < 2000 and len(c) > 8)
    hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    burn_ratio  = np.sum(cv2.inRange(hsv,np.array([0,0,0]),np.array([25,80,60]))>0) / fa
    if upper_d < 0.18 and n_diag < 10 and small_irreg < 20 and burn_ratio < 0.08:
        return None
    candidate = {
        "candidate_source":    "cv_structural",
        "yolo_class":          "structural_anomaly_candidate",
        "bbox":                {"x":0,"y":0,"w":fw,"h":fh//2},
        "frame_region":        "upper_zone",
        "vertical_zone":       "upper",
        "relative_distance":   "far",
        "upper_edge_density":  round(upper_d, 4),
        "diagonal_lines":      n_diag,
        "irregular_contours":  small_irreg,
        "burn_ratio":          round(burn_ratio, 4),
        "anomaly_score":       round(min((upper_d*3 + n_diag*0.03 + burn_ratio*2), 1.0), 2),
    }
    if DEBUG_CANDIDATES:
        print(f"\n   [STRUCTURAL → QUEUE] upper_d={round(upper_d,3)} "
              f"| diag={n_diag} | irreg={small_irreg}")
    return candidate


def generate_motion_candidate(frame, prev_frame):
    """Unchanged from v5. Handles mismatched frame sizes (waypoint transitions, resolution changes)."""
    if prev_frame is None: return None
    # Resize prev_frame to current frame size if they differ (different image sources / camera hiccup)
    if prev_frame.shape[:2] != frame.shape[:2]:
        prev_frame = cv2.resize(prev_frame, (frame.shape[1], frame.shape[0]))
    gray_c  = cv2.GaussianBlur(cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY),(21,21),0)
    gray_p  = cv2.GaussianBlur(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),(21,21),0)
    diff    = cv2.absdiff(gray_p, gray_c)
    _,thr   = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    thr     = cv2.dilate(thr, None, iterations=2)
    cntrs,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(cntrs, key=cv2.contourArea, default=None)
    if largest is None or cv2.contourArea(largest) < MOTION_MIN_AREA: return None
    x, y, w, h = cv2.boundingRect(largest)
    fill_pct   = round(cv2.contourArea(largest)/(frame.shape[0]*frame.shape[1])*100, 1)
    region     = get_frame_region(x+w/2, frame.shape[1])
    candidate  = {
        "candidate_source":  "motion",
        "yolo_class":        "motion_candidate",
        "bbox":              {"x":x,"y":y,"w":w,"h":h},
        "frame_region":      region,
        "vertical_zone":     get_vertical_zone(y+h/2, frame.shape[0]),
        "relative_distance": get_relative_distance(w*h, frame.shape[0]*frame.shape[1]),
        "motion_fill_pct":   fill_pct,
        "anomaly_score":     0.4,
    }
    if DEBUG_CANDIDATES:
        print(f"\n   [MOTION → QUEUE] fill={fill_pct}% | region={region}")
    return candidate


# ══════════════════════════════════════════════════════════════
# [R6] COMPRESSED GROQ PROMPT
# Cut ~60% of token count. Removed: embedded examples, verbose
# scoring guide, redundant rules. Kept: core task, output schema,
# critical FP-rejection rules, and the scoring scale.
# ══════════════════════════════════════════════════════════════

SENTINEL_UNIFIED_PROMPT = """You are SENTINEL's threat validation AI for a military/rescue reconnaissance rover.

Evaluate these CANDIDATES from local CV detectors (YOLO, HSV, motion, structural). Confirm or reject each. Most will be false positives.

CANDIDATES:
{candidates_json}

SCENE CONTEXT:
{scene_context}

Return ONLY valid JSON — no markdown, no backticks:
{{
  "scene_description": "1-2 sentence precise description of what is happening",
  "environment_assessment": "safe_area | active_threat_zone | rescue_needed | structural_hazard | unknown",
  "assessments": [
    {{
      "candidate_source": "<from input>",
      "yolo_class": "<from input>",
      "confirmed": true/false,
      "rejection_reason": "If false: specific reason. Empty if confirmed.",
      "specific_description": "If confirmed: precise non-generic description. Empty if rejected.",
      "subject_type": "hostile_threat | armed_individual | survivor_injured | survivor_mobile | civilian_nonthreat | suspicious_unattended_object | fire | smoke | structural_damage | navigation_obstacle | non_threat",
      "danger_score": 0,
      "severity": "critical | high | medium | low | attention",
      "reasoning": "2-3 sentences citing candidate fields (posture, shape_note, anomaly_score, environment_tags, _text_description) and scene context.",
      "key_indicators": ["specific", "visual", "evidence"],
      "recommended_action": "stop | reroute | flag_and_monitor | log_only | ignore",
      "rover_approach_risk": "safe_to_approach | approach_with_caution | do_not_approach"
    }}
  ],
  "overall_danger_score": 0,
  "operator_summary": "One sentence — the most important thing happening right now"
}}

SCORING: 0-2 safe | 3-4 log only | 5-6 flag+reroute | 7-8 stop+alert | 9-10 emergency

CRITICAL RULES — READ environment_tags FROM SCENE CONTEXT FIRST:
- If environment_tags includes 'office_or_workspace' or 'occupied_workspace': laptops, TVs, monitors, chairs, desks, and standard office furniture are COMPLETELY NORMAL. Score 0, type non_threat, reject unless the object is clearly anomalous (fallen, broken, out of place).
- A person sitting, standing, or working in an office/workspace is NOT a threat. Score 0-1, type civilian_nonthreat. Only flag if posture=prone/fallen, behavior suggests distress, or context is clearly wrong.
- If environment_tags includes 'heavily_cluttered_possible_evacuation_or_damage': raise suspicion on unattended bags and prone persons.
- Prone/fallen person (candidate anomaly_score >= 0.8): ALWAYS score 5+ minimum, type survivor_injured. This is a priority rescue signal regardless of environment.
- Fire: confirm ONLY if image shows actual flame (irregular edges, luminosity). Use shape_note + shape_circularity from candidate. Reject solid objects and bright reflections.
- Smoke: confirm ONLY if visible atmospheric haze or diffusion. Reject gray walls, shadows, and lighting variation.
- Never write generic descriptions. Be specific about clothing, position, object type visible in image."""


def assess_all_candidates(frame, all_candidates, scene_summary):
    """
    [R1+R2+R6] Scene-gated, deduplicated, compressed Groq call.
    """
    global last_groq_call, groq_call_count, last_groq_frame
    frame_h, frame_w = frame.shape[:2]

    if not all_candidates:
        _trace("GROQ", "No candidates — skip")
        return [], []

    # [R2] Layer 1: IoU dedup within this frame's candidate list
    all_candidates = object_tracker.iou_deduplicate(all_candidates)

    # [R2] Layer 2: Suppress already-validated (class, grid_cell) pairs
    all_candidates, suppressed_count = object_tracker.filter_known(
        all_candidates, frame_w, frame_h)
    if suppressed_count:
        _trace("TRACKER", f"Suppressed {suppressed_count} known objects")

    if not all_candidates:
        _trace("GROQ", "All candidates suppressed by tracker — skip")
        return [], []

    # [R1] Scene change gate
    should_call, delta_score, delta_reasons = scene_change_detector.compute_delta(
        all_candidates, frame_w, frame_h)

    now = time.time()
    rate_ok = (now - last_groq_call) >= GROQ_MIN_INTERVAL

    if not should_call:
        _trace("GROQ", f"Scene unchanged (delta={delta_score:.2f}) — skip")
        return [], []
    if not rate_ok:
        wait = round(GROQ_MIN_INTERVAL - (now - last_groq_call), 1)
        _trace("GROQ", f"Rate limited — {wait}s remaining")
        print(f"   [GROQ] Rate limit — {wait}s to next call")
        return _fallback_unverified(all_candidates), []

    last_groq_call  = now
    last_groq_frame = frame_count        # [FIX 2] record frame number of this call
    groq_call_count += 1                 # [FIX 3] increment session counter
    t_start = time.time()
    _trace("GROQ", f"Calling Groq: {len(all_candidates)} candidates | "
                     f"delta={delta_score:.2f} reasons={delta_reasons[:3]}")
    print(f"\n{'─'*55}")
    print(f"🤖 GROQ CALL #{groq_call_count}"
          + (f"/{GROQ_SESSION_CAP}" if GROQ_SESSION_CAP else "")
          + f" — {len(all_candidates)} candidates (delta={round(delta_score,2)})")
    for c in all_candidates:
        print(f"   → [{c['candidate_source']}] {c['yolo_class']} "
              f"| anomaly={c.get('anomaly_score',0)}")

    try:
        # Build per-candidate text descriptions to augment the prompt with spatial metadata.
        # This improves reasoning quality when the model cannot resolve fine details from pixels alone.
        def _candidate_text_desc(c):
            parts = [
                f"class={c['yolo_class']}",
                f"source={c['candidate_source']}",
                f"confidence={c.get('confidence', '?')}",
                f"region={c.get('frame_region','?')}",
                f"vertical_zone={c.get('vertical_zone','?')}",
                f"distance={c.get('relative_distance','?')}",
                f"anomaly_score={c.get('anomaly_score','?')}",
            ]
            if c.get("posture") and c["posture"] != "n/a":
                parts.append(f"posture={c['posture']}")
            if c.get("shape_note"):
                parts.append(f"shape_note={c['shape_note']}")
            if c.get("shape_circularity") is not None:
                parts.append(f"circularity={c['shape_circularity']}")
            if c.get("coverage_percent") is not None:
                parts.append(f"coverage={c['coverage_percent']}%")
            if c.get("motion_fill_pct") is not None:
                parts.append(f"motion_fill={c['motion_fill_pct']}%")
            if c.get("upper_edge_density") is not None:
                parts.append(f"upper_edge_density={c['upper_edge_density']}")
            return " | ".join(parts)

        # Augment candidates_json with inline text descriptions
        candidates_with_desc = []
        for c in all_candidates:
            augmented = dict(c)
            augmented["_text_description"] = _candidate_text_desc(c)
            candidates_with_desc.append(augmented)

        prompt = SENTINEL_UNIFIED_PROMPT.format(
            candidates_json = json.dumps(candidates_with_desc, indent=2),
            scene_context   = json.dumps({
                "crowd_density":    scene_summary.get("crowd_density"),
                "detected_classes": scene_summary.get("detected_classes", []),
                "environment_tags": scene_summary.get("environment_tags", []),
            }, indent=2)
        )

        if DEBUG_GROQ_PROMPT:
            print(f"\n[GROQ PROMPT — {len(prompt)} chars]\n{prompt[:800]}...\n")

        # Build Groq chat messages — include image as base64 for pixel-level reasoning.
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                ],
            }
        ]

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            timeout=GROQ_REQUEST_TIMEOUT,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        raw = raw.strip()

        if DEBUG_GROQ_RAW:
            print(f"\n[GROQ RAW]\n{raw[:600]}{'...' if len(raw)>600 else ''}\n")

        elapsed = round(time.time() - t_start, 2)
        parsed  = json.loads(raw)
        if DEBUG_STAGE_TIMINGS:
            print(f"   ⏱ Groq: {elapsed}s | prompt ~{len(prompt)//4} tokens")

        # [R1+R2] Update both trackers on successful call
        scene_change_detector.mark_validated(all_candidates, frame_w, frame_h)
        object_tracker.mark_validated(all_candidates, frame_w, frame_h)

        confirmed_threats = []
        rejected          = []

        print(f"\n🤖 GROQ VERDICTS:")
        for a in parsed.get("assessments", []):
            is_confirmed = a.get("confirmed", False)
            danger_score = a.get("danger_score", 0)
            subject_type = a.get("subject_type", "")
            source       = a.get("candidate_source", "?")
            cls          = a.get("yolo_class", "?")
            tag          = next((c for c in all_candidates
                                 if c.get("yolo_class")==cls
                                 and c.get("candidate_source")==source), {})

            if not is_confirmed:
                entry = {"frame": frame_count, "candidate_source": source,
                         "yolo_class": cls, "rejection_reason": a.get("rejection_reason",""),
                         "reasoning": a.get("reasoning","")}
                rejected.append(entry)
                rejection_log.append(entry)
                print(f"   ❌ REJECTED [{source}] {cls} — {a.get('rejection_reason','')[:80]}")
                _trace("GROQ_REJECT", f"Rejected {cls}")
                continue

            if danger_score < DANGER_THRESHOLD and "survivor" not in subject_type:
                print(f"   ⬇ BELOW THRESHOLD [{source}] {cls} score={danger_score}")
                continue

            print(f"   ✅ CONFIRMED [{source}] {cls} → score={danger_score} | {subject_type}")
            confirmed_threats.append({
                "threat_type":         cls,
                "specific_description":a.get("specific_description", cls),
                "subject_type":        subject_type,
                "threat_confirmed":    True,
                "category":            "groq_validated",
                "danger_score":        danger_score,
                "severity":            a.get("severity", "low"),
                "confidence":          tag.get("confidence", 0.0),
                "groq_reasoning":      a.get("reasoning", ""),
                "key_indicators":      a.get("key_indicators", []),
                "recommended_action":  a.get("recommended_action", "log_only"),
                "rover_approach_risk": a.get("rover_approach_risk", "approach_with_caution"),
                "scene_description":   parsed.get("scene_description", ""),
                "operator_summary":    parsed.get("operator_summary", ""),
                "environment":         parsed.get("environment_assessment", "unknown"),
                "frame_region":        tag.get("frame_region", "unknown"),
                "vertical_zone":       tag.get("vertical_zone", "unknown"),
                "relative_distance":   tag.get("relative_distance", "unknown"),
                "posture":             tag.get("posture", "n/a"),
                "anomaly_score":       tag.get("anomaly_score", 0.0),
                "source":              f"groq+{source}",
                "persistent":          True,
                "bbox":                tag.get("bbox", {"x":0,"y":0,"w":0,"h":0}),
                "timestamp":           _now_iso(),
                "rover_position":      {"x":0.0,"y":0.0},
            })

        print(f"{'─'*55}")
        print(f"🤖 {len(confirmed_threats)} confirmed | {len(rejected)} rejected | "
              f"overall danger={parsed.get('overall_danger_score',0)}/10")
        if parsed.get("operator_summary"):
            print(f"📡 {parsed['operator_summary']}")

        return confirmed_threats, rejected

    except json.JSONDecodeError as e:
        print(f"⚠ JSON parse error: {e}")
        return _fallback_unverified(all_candidates), []
    except Exception as e:
        print(f"⚠ Groq error: {e}")
        return _fallback_unverified(all_candidates), []


def _fallback_unverified(candidates):
    return [{
        "threat_type":         c["yolo_class"],
        "specific_description":(f"UNVERIFIED: {c['yolo_class']} at "
                                 f"{c.get('frame_region','?')}, {c.get('relative_distance','?')} "
                                 f"— Groq unavailable"),
        "subject_type":        "unverified_pending_assessment",
        "threat_confirmed":    False,
        "category":            "fallback_unverified",
        "danger_score":        3,
        "severity":            "medium",
        "confidence":          c.get("confidence", 0.0),
        "groq_reasoning":      "Groq unavailable",
        "source":              "fallback",
        "bbox":                c.get("bbox", {"x":0,"y":0,"w":0,"h":0}),
        "timestamp":           _now_iso(),
        "rover_position":      {"x":0.0,"y":0.0},
        "_rejection_reason":   "Groq unavailable",
    } for c in candidates]


# ── Gesture Recognition (unchanged from v5) ─────────────────
GESTURE_COMMAND_MAP = {
    "Open_Palm":"STOP","Closed_Fist":"CONFIRM_DETECTION",
    "Thumb_Up":"MARK_SAFE","Pointing_Up":"FORWARD",
    "Victory":"REROUTE","ILoveYou":"TURN_RIGHT",
}

def detect_gesture(frame):
    try:
        options = mp_vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path="gesture_recognizer.task"))
        with mp_vision.GestureRecognizer.create_from_options(options) as r:
            result = r.recognize(mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if not result.gestures: return None
        top = result.gestures[0][0].category_name
        return {"gesture":top,"command":GESTURE_COMMAND_MAP.get(top,"NONE"),"timestamp":_now_iso()}
    except Exception as e:
        print(f"⚠ Gesture: {e}"); return None


# ══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════

def run_sentinel_pipeline(frame, verbose=True, waypoint_id=None):
    global prev_frame_global, rover_command, frame_count
    frame_count  += 1
    rover_command = "PROCEED"
    t_pipeline    = time.time()
    frame_h, frame_w = frame.shape[:2]

    # [R5] Notify waypoint system if checkpoint changed
    if waypoint_id is not None:
        on_waypoint_transition(waypoint_id)

    print(f"\n{'='*62}")
    print(f"SENTINEL v6 — FRAME {frame_count}"
          + (f" | WAYPOINT {waypoint_id}" if waypoint_id else ""))
    print(f"{'='*62}")

    # ── Stage 1: Generate all candidates ────────────────────
    t1 = time.time()
    yolo_raw = yolo_model(frame, conf=CONF_OBJECT, imgsz=YOLO_IMGSZ, verbose=False)[0]

    # [R3] generate_yolo_candidates is stripped — no expensive enrichment
    yolo_candidates, nav_hazards, scene_summary = generate_yolo_candidates(frame, yolo_raw)
    hsv_candidates   = generate_hsv_candidates(frame)
    struct_candidate = generate_structural_candidate(frame)
    motion_candidate = generate_motion_candidate(frame, prev_frame_global)
    prev_frame_global = frame.copy()

    all_candidates = yolo_candidates + hsv_candidates
    if struct_candidate: all_candidates.append(struct_candidate)
    if motion_candidate: all_candidates.append(motion_candidate)

    if DEBUG_STAGE_TIMINGS:
        print(f"   ⏱ Stage 1 (candidates): {round(time.time()-t1,2)}s "
              f"[no k-means/Hough enrichment]")
    print(f"\n   Stage 1: YOLO={len(yolo_candidates)} | nav={len(nav_hazards)} | "
          f"HSV={len(hsv_candidates)} | struct={'1' if struct_candidate else '0'} | "
          f"motion={'1' if motion_candidate else '0'} | total={len(all_candidates)}")

    for h in nav_hazards:
        emit_threat(h)

    # ── Stage 2: Gated, deduplicated Groq validation ──────
    # [R1] SceneChangeDetector gates the call
    # [R2] ObjectTracker deduplicates within this call
    # [R6] Compressed prompt reduces latency
    t2 = time.time()
    confirmed_threats, rejected = assess_all_candidates(frame, all_candidates, scene_summary)
    for t in confirmed_threats:
        emit_threat(t)

    if DEBUG_STAGE_TIMINGS:
        print(f"   ⏱ Stage 2 (Groq): {round(time.time()-t2,2)}s")

    gesture = detect_gesture(frame)
    if gesture:
        print(f"\n✋ {gesture['gesture']} → {gesture['command']}")

    all_threats = nav_hazards + confirmed_threats
    total_time  = round(time.time() - t_pipeline, 2)

    output = {
        "frame":             frame_count,
        "nav_hazards":       nav_hazards,
        "confirmed_threats": confirmed_threats,
        "rejected_count":    len(rejected),
        "gesture":           gesture,
        "rover_command":     rover_command,
        "all_threats":       all_threats,
        "qml_waypoints":     get_waypoints_for_qml(),
        "scene_summary":     scene_summary,
        "all_candidates":    all_candidates,
        "pipeline_time_s":   total_time,
    }

    print(f"\n{'='*62}")
    print(f"COMPLETE — {total_time}s | threats={len(all_threats)} | "
          f"rejected={len(rejected)} | cmd={rover_command}")
    print(f"{'='*62}")
    return output


# ── Visualization (unchanged from v5) ────────────────────────
SEVERITY_COLORS_BGR = {"critical":(50,50,255),"high":(0,140,255),
                        "medium":(0,200,255),"low":(0,200,0),"attention":(200,200,0)}
SOURCE_COLORS_BGR   = {"yolo_nav":(0,165,255),"groq+yolo":(255,50,50),
                        "groq+hsv_fire":(0,50,255),"groq+hsv_smoke":(100,100,200),
                        "groq+cv_structural":(0,200,200),"groq+motion":(200,200,0),
                        "fallback":(150,150,150)}

def draw_rich_overlay(frame, output):
    out = frame.copy()
    for c in output["all_candidates"]:
        b = c.get("bbox",{})
        x,y,w,h = b.get("x",0),b.get("y",0),b.get("w",0),b.get("h",0)
        if w>0 and h>0:
            cv2.rectangle(out,(x,y),(x+w,y+h),(80,80,80),1)
            cv2.putText(out,f"[{c['candidate_source']}?]",
                        (x,max(y-4,10)),cv2.FONT_HERSHEY_SIMPLEX,0.35,(80,80,80),1)
    for t in output["all_threats"]:
        b   = t.get("bbox",{})
        col = SOURCE_COLORS_BGR.get(t.get("source","?"),
              SEVERITY_COLORS_BGR.get(t.get("severity","low"),(200,200,200)))
        x,y,w,h = b.get("x",0),b.get("y",0),b.get("w",0),b.get("h",0)
        if w>0 and h>0:
            cv2.rectangle(out,(x,y),(x+w,y+h),col,3)
            desc  = t.get("specific_description","") or t["threat_type"]
            words = desc.split()
            lines = [" ".join(words[i:i+4]) for i in range(0,min(len(words),12),4)]
            lines.insert(0,f"{t.get('danger_score','?')}/10 [{t.get('source','?')[:6]}]")
            for li,line in enumerate(lines[:4]):
                cv2.putText(out,line,(x,max(y-8+li*14,10+li*14)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1)
    cmd  = output["rover_command"]
    cmap = {"STOP":(50,50,255),"REROUTE":(0,140,255),"PROCEED":(0,200,0)}
    cv2.rectangle(out,(0,0),(300,45),(0,0,0),-1)
    cv2.putText(out,f"CMD: {cmd}",(8,32),cv2.FONT_HERSHEY_SIMPLEX,1.0,
                cmap.get(cmd,(200,200,0)),2)
    cv2.putText(out,
                f"F{output['frame']} | cands={len(output['all_candidates'])} "
                f"confirmed={len(output['all_threats'])} rejected={output['rejected_count']}",
                (8,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
    return out

def display_results(frame, output):
    annotated = draw_rich_overlay(frame, output)
    fig       = plt.figure(figsize=(22, 10))
    ax1 = fig.add_subplot(1,3,(1,2))
    ax1.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    ax1.axis('off')
    ax1.set_title(f"SENTINEL v6 F{output['frame']} — {output['rover_command']} "
                  f"({output['pipeline_time_s']}s)", fontsize=13, fontweight='bold')
    legend_patches = [
        mpatches.Patch(color='gray',   label='Candidate (considered)'),
        mpatches.Patch(color='orange', label='Nav hazard (self-emit)'),
        mpatches.Patch(color='red',    label='Groq confirmed YOLO'),
        mpatches.Patch(color='blue',   label='Groq confirmed fire'),
        mpatches.Patch(color='cyan',   label='Groq confirmed structural'),
    ]
    ax1.legend(handles=legend_patches, loc='lower left', fontsize=7, framealpha=0.7)
    ax2 = fig.add_subplot(1,3,3)
    ax2.axis('off')
    threats = output["all_threats"]
    if threats:
        rows = [["Description","Score","Sev","Source","Approach"]]
        for t in threats:
            desc = t.get("specific_description","") or t["threat_type"]
            rows.append([desc[:38]+"…" if len(desc)>38 else desc,
                         f"{t.get('danger_score','?')}/10",
                         t.get("severity","?")[:5].upper(),
                         t.get("source","?")[:10],
                         t.get("rover_approach_risk","?")[:12]])
        tbl = ax2.table(cellText=rows[1:],colLabels=rows[0],cellLoc='left',loc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1.0,1.6)
        ax2.set_title("Confirmed Threats", fontsize=11)
    else:
        ax2.text(0.5,0.6,"✅ All candidates rejected\nScene clear",
                 ha='center',va='center',fontsize=12,color='green')
        ax2.set_title("Confirmed Threats", fontsize=11)
    plt.tight_layout(); plt.show()


# ── Debug utilities (unchanged from v5) ──────────────────────
def print_rejection_log():
    print(f"\n{'='*62}\nREJECTION LOG ({len(rejection_log)} total)\n{'='*62}")
    for r in rejection_log:
        print(f"\n  Frame {r['frame']} | [{r['candidate_source']}] {r['yolo_class']}")
        print(f"  Reason: {r['rejection_reason'][:120]}")

def print_pipeline_trace(last_n=30):
    print(f"\n{'='*62}\nPIPELINE TRACE (last {last_n})\n{'='*62}")
    for e in pipeline_trace[-last_n:]:
        d = json.dumps(e.get("data",""), separators=(',',':'))[:60] if e.get("data") else ""
        print(f"  [{e['stage']:20}] F{e['frame']} — {e['message'][:60]} {d}")

def print_session_stats():
    confirmed = [t for t in threat_log if t.get("threat_confirmed")]
    survivors = [t for t in threat_log if "survivor" in t.get("subject_type","")]
    groq_calls = sum(1 for e in pipeline_trace if e["stage"]=="GROQ"
                       and "Calling" in e.get("message",""))
    skipped_calls = sum(1 for e in pipeline_trace if e["stage"]=="GROQ"
                        and "skip" in e.get("message","").lower())
    print(f"\n{'='*62}\nSESSION STATS\n{'='*62}")
    print(f"  Frames processed    : {frame_count}")
    print(f"  Total candidates    : {len(candidate_log)}")
    print(f"  Threats emitted     : {len(threat_log)}")
    print(f"  Confirmed threats   : {len(confirmed)}")
    print(f"  Survivors found     : {len(survivors)}")
    print(f"  Groq calls fired    : {groq_calls}")
    print(f"  Groq calls skipped  : {skipped_calls}  [R1 savings]")
    if groq_calls + skipped_calls > 0:
        skip_pct = round(skipped_calls/(groq_calls+skipped_calls)*100)
        print(f"  Scene-gate savings  : {skip_pct}% of potential Groq calls avoided")
    print(f"  Rejections by Groq  : {len(rejection_log)}")
    if candidate_log:
        src_counts = defaultdict(int)
        for c in candidate_log: src_counts[c["candidate_source"]] += 1
        print(f"\n  Candidates by source:")
        for src, cnt in sorted(src_counts.items(), key=lambda x: -x[1]):
            rej = sum(1 for r in rejection_log if r["candidate_source"]==src)
            print(f"    {src:25}: {cnt} total, {rej} rejected "
                  f"({round(rej/cnt*100) if cnt else 0}% rejection rate)")




# ============================================================
# CELLS 13–15: Multi-frame loop + DroidCam live stream
# ============================================================
# INPUT MODES (auto-detected):
#   • DroidCam live stream  → CELL 16 (run_droidcam_stream)
#   • .mp4 / .avi / .mov    → video, frame-by-frame
#   • .zip of images        → extracted, sorted, sequential
#   • single image          → single-frame debug run
#
# GROQ GATING (all three gates must pass):
#   Gate 1 — SceneChangeDetector: delta >= SCENE_DELTA_MIN_TRIGGER
#   Gate 2 — Frame count: >= GROQ_MIN_FRAMES_BETWEEN since last call
#   Gate 3 — Wall clock: >= GROQ_MIN_INTERVAL seconds since last call
#   Hard cap: GROQ_SESSION_CAP total calls per session
# ============================================================

# ── CELL 13: Loop configuration ──────────────────────────────
import zipfile, os, glob

FRAME_SKIP       = 3      # video only: process every Nth raw frame (1 = every frame)
MAX_FRAMES       = 0      # 0 = unlimited; set e.g. 100 for quick test
DISPLAY_EVERY_N  = 10     # show annotated image every N processed frames (0 = never)
SUMMARY_EVERY_N  = 20     # print threat summary every N processed frames (0 = never)
SAVE_ANNOTATED   = True   # write annotated frames to /content/sentinel_output/

OUTPUT_DIR = "/content/sentinel_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✅ Loop config set.")
print(f"   FRAME_SKIP={FRAME_SKIP} | MAX_FRAMES={MAX_FRAMES} | "
      f"DISPLAY_EVERY_N={DISPLAY_EVERY_N} | SAVE_ANNOTATED={SAVE_ANNOTATED}")


# ── CELL 14: Input loader ─────────────────────────────────────

def load_input():
    """
    Upload a video, zip of images, or single image.
    Returns an iterator of (frame_index, bgr_frame) tuples.
    """
    print("\nUpload a video (.mp4/.avi/.mov), zip of images, or single image:")
    uploaded = files.upload()
    if not uploaded:
        print("❌ No file uploaded.")
        return None, None

    filename = list(uploaded.keys())[0]
    ext      = os.path.splitext(filename)[1].lower()
    print(f"✅ Received: {filename} ({ext})")

    if ext in ('.mp4', '.avi', '.mov', '.mkv'):
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            print(f"❌ Could not open video: {filename}")
            return None, None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"   Video: {total_frames} frames @ {fps:.1f}fps | {w}×{h}")
        effective = total_frames // FRAME_SKIP
        if MAX_FRAMES: effective = min(effective, MAX_FRAMES)
        print(f"   Will process ~{effective} frames (every {FRAME_SKIP})")

        def video_iter():
            raw_idx = 0
            processed = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                raw_idx += 1
                if raw_idx % FRAME_SKIP != 0: continue
                processed += 1
                yield raw_idx, frame
                if MAX_FRAMES and processed >= MAX_FRAMES: break
            cap.release()

        return video_iter(), "video"

    elif ext == '.zip':
        extract_dir = "/content/sentinel_frames"
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(filename, 'r') as z:
            z.extractall(extract_dir)
        exts  = ('*.jpg','*.jpeg','*.png','*.bmp','*.webp')
        paths = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(extract_dir, '**', e), recursive=True))
        paths.sort()
        if not paths:
            print("❌ No images found in zip.")
            return None, None
        if MAX_FRAMES: paths = paths[:MAX_FRAMES]
        print(f"   Found {len(paths)} images in zip")

        def zip_iter():
            for idx, path in enumerate(paths):
                frame = cv2.imread(path)
                if frame is None:
                    print(f"   ⚠ Skipping unreadable: {os.path.basename(path)}")
                    continue
                yield idx, frame

        return zip_iter(), "zip"

    else:
        frame = cv2.imread(filename)
        if frame is None:
            print(f"❌ Could not read image: {filename}")
            return None, None
        print(f"   Single image: {frame.shape[1]}×{frame.shape[0]}")

        def single_iter():
            yield 0, frame

        return single_iter(), "single"


# ── CELL 15: Main loop ────────────────────────────────────────

def run_loop():
    """
    Multi-frame processing loop for video, zip, or single image.
    DroidCam live stream: use run_droidcam_stream() in CELL 16 instead.
    """
    frame_iter, input_type = load_input()
    if frame_iter is None:
        return

    processed_count       = 0
    groq_fired          = 0
    groq_skipped_delta  = 0
    groq_skipped_rate   = 0
    total_threats         = 0
    total_rejected        = 0
    loop_start            = time.time()

    print(f"\n{'='*62}")
    print(f"SENTINEL LOOP STARTED — input={input_type}")
    print(f"Scene-change gating: delta>={SCENE_DELTA_MIN_TRIGGER} required")
    print(f"Rate limit: >={GROQ_MIN_INTERVAL}s between calls")
    print(f"{'='*62}\n")

    for raw_idx, frame in frame_iter:
        processed_count += 1
        t_frame = time.time()

        output = run_sentinel_pipeline(frame, verbose=False)

        recent_traces = [e for e in pipeline_trace if e["frame"] == frame_count]
        groq_trace    = [e for e in recent_traces if e["stage"] == "GROQ"]
        last_g        = groq_trace[-1]["message"] if groq_trace else ""

        if "Calling Groq" in last_g:
            groq_fired += 1
            groq_status = "🤖 GROQ CALLED"
        elif "Scene unchanged" in last_g or "skip" in last_g.lower():
            groq_skipped_delta += 1
            groq_status = "⏭  scene unchanged"
        elif "Rate limit" in last_g:
            groq_skipped_rate += 1
            groq_status = "⏳ rate limited"
        else:
            groq_status = "⚪ no candidates"

        total_threats  += len(output["all_threats"])
        total_rejected += output["rejected_count"]
        frame_ms        = round((time.time() - t_frame) * 1000)

        cmd_icon   = {"STOP":"🛑","REROUTE":"⚠️ ","PROCEED":"✅"}
        threat_str = ""
        if output["all_threats"]:
            descs = [t.get("specific_description","") or t["threat_type"]
                     for t in output["all_threats"]]
            threat_str = " | " + " + ".join(d[:40] for d in descs[:2])
            if len(descs) > 2: threat_str += f" +{len(descs)-2} more"

        print(f"[F{frame_count:04d}/raw{raw_idx}] "
              f"{cmd_icon.get(output['rover_command'],'  ')} {output['rover_command']:7} | "
              f"{groq_status:22} | "
              f"threats={len(output['all_threats'])} rej={output['rejected_count']} | "
              f"{frame_ms}ms{threat_str}")

        for t in output["all_threats"]:
            icons = {"critical":"  🔴","high":"  🟠","medium":"  🟡",
                     "low":"  🟢","attention":"  🔵"}
            print(f"{icons.get(t['severity'],'    ')} "
                  f"[{t.get('source','?')[:12]:12}] "
                  f"score={t.get('danger_score','?'):2}/10 | "
                  f"{t.get('specific_description','')[:70]}")

        if DISPLAY_EVERY_N and processed_count % DISPLAY_EVERY_N == 0:
            annotated = draw_rich_overlay(frame, output)
            plt.figure(figsize=(14, 7))
            plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"Frame {frame_count} — {output['rover_command']} — {groq_status}",
                      fontsize=11)
            plt.tight_layout()
            plt.show()

        if SAVE_ANNOTATED:
            annotated = draw_rich_overlay(frame, output)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_count:06d}.jpg"), annotated)

        if SUMMARY_EVERY_N and processed_count % SUMMARY_EVERY_N == 0:
            elapsed  = round(time.time() - loop_start, 1)
            fps_eff  = round(processed_count / elapsed, 1) if elapsed > 0 else 0
            total_g  = groq_fired + groq_skipped_delta + groq_skipped_rate
            skip_pct = round((groq_skipped_delta + groq_skipped_rate) / max(total_g,1) * 100)
            print(f"\n{'─'*62}")
            print(f"SUMMARY @ frame {frame_count} ({elapsed}s | {fps_eff} fps effective)")
            print(f"  Groq  : {groq_fired} fired | {groq_skipped_delta} scene-skipped | "
                  f"{groq_skipped_rate} rate-skipped | {skip_pct}% avoided")
            print(f"  Threats: {total_threats} emitted | {total_rejected} rejected")
            print(f"{'─'*62}\n")

    elapsed  = round(time.time() - loop_start, 1)
    fps_eff  = round(processed_count / elapsed, 1) if elapsed > 0 else 0
    total_g  = groq_fired + groq_skipped_delta + groq_skipped_rate
    skip_pct = round((groq_skipped_delta + groq_skipped_rate) / max(total_g,1) * 100)

    print(f"\n{'='*62}\nLOOP COMPLETE\n{'='*62}")
    print(f"  Frames processed : {processed_count} in {elapsed}s ({fps_eff} fps)")
    print(f"  Groq calls       : {groq_fired} fired | {skip_pct}% avoided")
    print(f"  Threats emitted  : {total_threats} | Rejected: {total_rejected}")

    confirmed = [t for t in threat_log if t.get("threat_confirmed")]
    survivors = [t for t in threat_log if "survivor" in t.get("subject_type","")]
    print(f"  Confirmed        : {len(confirmed)} | Survivors: {len(survivors)}")

    if SAVE_ANNOTATED and processed_count > 0:
        print(f"\n  Annotated frames → {OUTPUT_DIR}/")
        print(f"  Download: shutil.make_archive('/content/sentinel_annotated','zip','{OUTPUT_DIR}')")

    print(f"\n  QML waypoints: {len(get_waypoints_for_qml())}")
    for wp in get_waypoints_for_qml():
        print(f"    {wp['id']} | score={wp['danger_score']} | "
              f"{wp.get('specific_description','')[:55]}")

    print_rejection_log()
    print_pipeline_trace(last_n=40)


# ── CELL 16: DroidCam live stream ────────────────────────────
# ─────────────────────────────────────────────────────────────
# SETUP (do this before running):
#   iPhone: Install DroidCam app (free, iOS App Store)
#   Laptop: Install DroidCam client — https://www.dev47apps.com
#
#   USB mode (recommended — lower latency, no WiFi drops):
#     1. Connect iPhone via USB
#     2. Trust computer on iPhone
#     3. Open DroidCam app on iPhone, select "USB"
#     4. Open DroidCam client on laptop — device appears automatically
#     5. DROIDCAM_SOURCE below = 0 (or 1/2 if another webcam exists)
#
#   WiFi mode (fallback):
#     1. iPhone and laptop on same WiFi network
#     2. DroidCam app shows an IP + port (default 4747)
#     3. Set DROIDCAM_SOURCE = "http://<iPhone_IP>:4747/video"
#
# HOW FRAME CAPTURE WORKS:
#   cv2.VideoCapture opens the DroidCam device as a standard webcam.
#   A tight read loop grabs frames continuously.
#   DROIDCAM_PROCESS_EVERY_N_FRAMES controls how many raw frames to skip
#   between pipeline runs — this is your effective "screenshot interval".
#   Skipped frames are read and discarded (prevents buffer buildup).
#   Every Nth frame is passed to run_sentinel_pipeline() which applies
#   all existing gating (SceneChangeDetector, frame gate, session cap).
#
# STOPPING: interrupt the cell (square stop button in Colab).
# ─────────────────────────────────────────────────────────────

# ── DroidCam config ───────────────────────────────────────────
DROIDCAM_SOURCE              = 0     # USB: integer device index (0, 1, 2...)
                                     # WiFi: "http://<iPhone_IP>:4747/video"
DROIDCAM_PROCESS_EVERY_N_FRAMES = 6  # pipeline runs on every 6th raw frame
                                     # DroidCam streams at ~30fps →
                                     # 30/6 = 5 pipeline calls/sec max
                                     # (Groq gates reduce this further)
DROIDCAM_MAX_PIPELINE_FRAMES = 0     # 0 = run until interrupted
                                     # set e.g. 300 to auto-stop after 300 processed
DROIDCAM_DISPLAY_EVERY_N     = 15    # show annotated frame in Colab every N processed
DROIDCAM_SAVE_ANNOTATED      = True  # save annotated frames to OUTPUT_DIR


def run_droidcam_stream(waypoint_id=None):
    """
    Opens DroidCam (iPhone camera via USB or WiFi) as a cv2 capture device.
    Grabs frames continuously, processes every Nth frame through the full
    SENTINEL pipeline. All existing gating (SceneChangeDetector, frame gate,
    session cap) applies normally — DroidCam is just a frame source.

    Args:
        waypoint_id: Optional string. If set, signals a waypoint transition
                     on the first frame (forces fresh Groq call, clears tracker).
                     Pass a new waypoint_id mid-run by calling
                     on_waypoint_transition("checkpoint_N") from another cell.
    """
    cap = cv2.VideoCapture(DROIDCAM_SOURCE)

    if not cap.isOpened():
        print(f"❌ Could not open DroidCam source: {DROIDCAM_SOURCE}")
        print("   USB: check DroidCam client is running and iPhone is trusted")
        print("   WiFi: verify IP and that DroidCam app shows 'Connected'")
        return

    # Try to read one test frame to confirm stream is live
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("❌ DroidCam opened but no frames received. Check DroidCam app is active.")
        cap.release()
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n✅ DroidCam stream open: {w}×{h} @ {actual_fps:.0f}fps")
    print(f"   Source       : {DROIDCAM_SOURCE}")
    print(f"   Processing   : every {DROIDCAM_PROCESS_EVERY_N_FRAMES} raw frames "
          f"(~{actual_fps/DROIDCAM_PROCESS_EVERY_N_FRAMES:.1f} pipeline fps max)")
    print(f"   Groq gates   : delta>={SCENE_DELTA_MIN_TRIGGER} "
          f"AND {GROQ_MIN_FRAMES_BETWEEN}+ frames AND {GROQ_MIN_INTERVAL}s")
    print(f"   Session cap  : {GROQ_SESSION_CAP} Groq calls max")
    print(f"   Stop         : interrupt this cell (■ button)\n")

    if waypoint_id:
        on_waypoint_transition(waypoint_id)

    raw_frame_idx     = 0
    processed_count   = 0
    loop_start        = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠ Frame read failed — stream dropped. Retrying...")
                time.sleep(0.1)
                continue

            raw_frame_idx += 1

            # Discard frames we're skipping — prevents OpenCV buffer buildup
            # which would cause the pipeline to process stale frames
            if raw_frame_idx % DROIDCAM_PROCESS_EVERY_N_FRAMES != 0:
                continue

            processed_count += 1

            # ── Run full pipeline on this frame ───────────────
            output = run_sentinel_pipeline(frame, verbose=False)

            # ── Console status line ───────────────────────────
            elapsed  = round(time.time() - loop_start, 1)
            cmd_icon = {"STOP":"🛑","REROUTE":"⚠️ ","PROCEED":"✅"}
            threat_summary = ""
            if output["all_threats"]:
                descs = [t.get("specific_description","") or t["threat_type"]
                         for t in output["all_threats"]]
                threat_summary = " | " + " + ".join(d[:35] for d in descs[:2])
                if len(descs) > 2: threat_summary += f" +{len(descs)-2}"

            print(f"[{elapsed:6.1f}s | F{frame_count:04d}] "
                  f"{cmd_icon.get(output['rover_command'],'  ')} {output['rover_command']:7} | "
                  f"Groq calls: {groq_call_count}"
                  f"{'/' + str(GROQ_SESSION_CAP) if GROQ_SESSION_CAP else ''} | "
                  f"threats={len(output['all_threats'])}{threat_summary}")

            for t in output["all_threats"]:
                icons = {"critical":"  🔴","high":"  🟠","medium":"  🟡",
                         "low":"  🟢","attention":"  🔵"}
                print(f"{icons.get(t['severity'],'    ')} "
                      f"[{t.get('source','?')[:12]:12}] "
                      f"score={t.get('danger_score','?'):2}/10 | "
                      f"{t.get('specific_description','')[:70]}")

            # ── Periodic Colab display ────────────────────────
            if DROIDCAM_DISPLAY_EVERY_N and processed_count % DROIDCAM_DISPLAY_EVERY_N == 0:
                annotated = draw_rich_overlay(frame, output)
                plt.figure(figsize=(12, 6))
                plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(f"DroidCam F{frame_count} | {output['rover_command']} | "
                          f"Groq calls: {groq_call_count}", fontsize=11)
                plt.tight_layout()
                plt.show()

            # ── Save annotated frame ──────────────────────────
            if DROIDCAM_SAVE_ANNOTATED:
                annotated = draw_rich_overlay(frame, output)
                cv2.imwrite(
                    os.path.join(OUTPUT_DIR, f"droidcam_{frame_count:06d}.jpg"),
                    annotated
                )

            # ── Session cap reached — stop gracefully ─────────
            if GROQ_SESSION_CAP and groq_call_count >= GROQ_SESSION_CAP:
                print(f"\n🚫 Session cap reached ({GROQ_SESSION_CAP} calls) — stopping stream.")
                break

            # ── Max frames reached ────────────────────────────
            if DROIDCAM_MAX_PIPELINE_FRAMES and processed_count >= DROIDCAM_MAX_PIPELINE_FRAMES:
                print(f"\n✅ Reached max pipeline frames ({DROIDCAM_MAX_PIPELINE_FRAMES}) — stopping.")
                break

    except KeyboardInterrupt:
        print("\n⏹ Stream interrupted by user.")

    finally:
        cap.release()
        elapsed = round(time.time() - loop_start, 1)
        print(f"\n{'='*62}")
        print(f"DROIDCAM SESSION COMPLETE")
        print(f"{'='*62}")
        print(f"  Duration         : {elapsed}s")
        print(f"  Raw frames read  : {raw_frame_idx}")
        print(f"  Pipeline runs    : {processed_count}")
        print(f"  Groq calls       : {groq_call_count}")
        confirmed = [t for t in threat_log if t.get("threat_confirmed")]
        survivors = [t for t in threat_log if "survivor" in t.get("subject_type","")]
        print(f"  Confirmed threats: {len(confirmed)}")
        print(f"  Survivors found  : {len(survivors)}")
        print(f"  Rover command    : {get_rover_command()}")
        if DROIDCAM_SAVE_ANNOTATED:
            print(f"  Saved frames     : {OUTPUT_DIR}/droidcam_*.jpg")
        print(f"\n  QML waypoints: {len(get_waypoints_for_qml())}")
        for wp in get_waypoints_for_qml():
            print(f"    {wp['id']} | score={wp['danger_score']} | "
                  f"{wp.get('specific_description','')[:55]}")
        print_session_stats()


# ── To run DroidCam stream: ───────────────────────────────────
# run_droidcam_stream()
#
# To run uploaded video/image instead:
# run_loop()
#
# To signal a waypoint transition mid-stream from another cell:
# on_waypoint_transition("checkpoint_3")
# ─────────────────────────────────────────────────────────────

print("\n── GUI ───────────────────────────────────────────────")
for t in get_latest_threats():
    print(f"  [{t['source']}] {t.get('specific_description','')[:70]}")

print("\n── QML waypoints ─────────────────────────────────────")
for wp in get_waypoints_for_qml():
    print(f"  {wp['id']} | {wp.get('specific_description','')[:60]}")

print(f"\n── Arduino: {get_rover_command()}")
