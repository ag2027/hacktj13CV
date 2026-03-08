import argparse
import asyncio
import base64
import importlib
import json
import os
import threading
import time
import uuid
from collections import deque
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

try:
    import cv2
except Exception:
    cv2 = None


# ── Shared State ─────────────────────────────────────────────


class GatewayState:
    def __init__(self):
        self.lock = threading.Lock()
        self.camera_status = "disconnected"
        self.cv_status = "disconnected"
        self.qml_status = "unknown"
        self.camera_error = None
        self.cv_error = None
        self.droidcam_stream_url = None
        self.camera_source = None
        self.latest_feed = {
            "type": "feed",
            "seq": 0,
            "frame": 0,
            "timestamp": time.time(),
            "image_jpeg_base64": None,
            "detections": [],
            "confirmed_threats": [],
            "nav_hazards": [],
            "gesture": None,
            "rover_command": None,
            "pipeline_time_s": 0.0,
            "camera_connected": False,
            "cv_connected": False,
        }
        self.cv_stats = {
            "type": "cv_stats",
            "seq": 0,
            "timestamp": time.time(),
            "groq_calls": 0,
            "groq_cap": None,
            "pipeline_latency_ms": 0,
            "scene_change_score": 0.0,
            "frames_processed": 0,
        }
        self.terac_history: deque[dict[str, Any]] = deque(maxlen=100)
        self.session_log: deque[dict[str, Any]] = deque(maxlen=200)

    def update_feed(self, payload: dict[str, Any]):
        with self.lock:
            next_seq = self.latest_feed["seq"] + 1
            self.latest_feed = {
                **self.latest_feed,
                **payload,
                "type": "feed",
                "seq": next_seq,
                "timestamp": time.time(),
            }

    def update_cv_stats(self, **updates: Any):
        with self.lock:
            self.cv_stats = {
                **self.cv_stats,
                **updates,
                "seq": self.cv_stats["seq"] + 1,
                "timestamp": time.time(),
            }

    def snapshot_status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "camera": {"status": self.camera_status, "error": self.camera_error},
                "cv": {"status": self.cv_status, "error": self.cv_error},
                "qml": {"status": self.qml_status},
                "droidcam_stream_url": self.droidcam_stream_url,
                "camera_source": self.camera_source,
                "feed_seq": self.latest_feed["seq"],
                "cv_stats": dict(self.cv_stats),
            }

    def snapshot_feed(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.latest_feed)

    def snapshot_cv_stats(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.cv_stats)

    def add_terac_dispatch(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "status": "stored",
            "payload": payload,
        }
        with self.lock:
            self.terac_history.appendleft(record)
            self.session_log.appendleft(
                {
                    "time": time.time(),
                    "type": "terac_dispatch",
                    "message": f"Terac payload stored ({len(payload.get('threats', []))} threats)",
                }
            )
        return record

    def get_terac_history(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.terac_history)[:limit]


# ── CV Pipeline Adapter ─────────────────────────────────────


class CVPipelineAdapter:
    def __init__(
        self,
        state: GatewayState,
        source: Optional[str],
        process_every_n: int,
        jpeg_quality: int,
    ):
        self.state = state
        self.source = source
        self.process_every_n = max(1, process_every_n)
        self.jpeg_quality = max(40, min(jpeg_quality, 95))
        self.stop_event = threading.Event()
        self.thread = None
        self.cv_module = None
        self.import_error = None
        self._load_pipeline_module()

    def _load_pipeline_module(self):
        try:
            self.cv_module = importlib.import_module("cv_server")
            self.state.cv_status = "connected"
            self.state.cv_error = None
        except Exception as exc:
            self.cv_module = None
            self.import_error = str(exc)
            self.state.cv_status = "degraded"
            self.state.cv_error = self.import_error

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def _build_payload(self, frame, output, annotated):
        encoded = None
        if cv2 is not None and annotated is not None:
            ok, buffer = cv2.imencode(
                ".jpg",
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if ok:
                encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")

        return {
            "frame": output.get("frame", 0),
            "image_jpeg_base64": encoded,
            "detections": output.get("threat_summary", []),
            "confirmed_threats": output.get("confirmed_threats", []),
            "nav_hazards": output.get("nav_hazards", []),
            "gesture": output.get("gesture"),
            "rover_command": output.get("rover_command"),
            "qml_waypoints": output.get("qml_waypoints", []),
            "scene_summary": output.get("scene_summary", {}),
            "pipeline_time_s": output.get("pipeline_time_s", 0.0),
            "camera_connected": True,
            "cv_connected": self.cv_module is not None,
        }

    def _build_fallback_output(self):
        return {
            "frame": int(time.time() * 1000),
            "threat_summary": [],
            "confirmed_threats": [],
            "nav_hazards": [],
            "gesture": None,
            "rover_command": None,
            "qml_waypoints": [],
            "scene_summary": {"note": "CV pipeline unavailable; streaming raw frames only."},
            "pipeline_time_s": 0.0,
        }

    def _candidate_capture_sources(self):
        primary = self.source
        candidates = [primary]
        if isinstance(primary, str):
            src = primary.strip()
            low = src.lower()
            if low.startswith("http"):
                if "/video" in low:
                    candidates.append(src.replace("/video", "/mjpegfeed"))
                    candidates.append(src.replace("/video", "/shot.jpg"))
                elif "/mjpegfeed" in low:
                    candidates.append(src.replace("/mjpegfeed", "/video"))
                    candidates.append(src.replace("/mjpegfeed", "/shot.jpg"))

        dedup = []
        for c in candidates:
            if c not in dedup:
                dedup.append(c)
        return dedup

    def _open_capture(self):
        if cv2 is None:
            return None, None

        backends = [None]
        for name in ("CAP_FFMPEG", "CAP_MSMF", "CAP_DSHOW"):
            backend = getattr(cv2, name, None)
            if backend is not None:
                backends.append(backend)

        for source in self._candidate_capture_sources():
            open_arg = int(source) if isinstance(source, str) and source.isdigit() else source
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(open_arg) if backend is None else cv2.VideoCapture(open_arg, backend)
                except Exception:
                    cap = None
                if cap is not None and cap.isOpened():
                    return cap, str(source)
                if cap is not None:
                    cap.release()

        return None, None

    def _run_loop(self):
        if cv2 is None:
            self.state.camera_status = "disconnected"
            self.state.camera_error = "opencv-python is not available"
            return
        if self.source in (None, "", "off"):
            self.state.camera_status = "disconnected"
            self.state.camera_error = "No camera source configured"
            return

        cap = None
        active_source = str(self.source)
        raw_index = 0
        frames_processed = 0

        try:
            while not self.stop_event.is_set():
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap, opened_source = self._open_capture()
                    if cap is None:
                        self.state.camera_status = "disconnected"
                        self.state.camera_error = f"Could not open camera source: {self.source}"
                        time.sleep(1.0)
                        continue
                    active_source = opened_source
                    self.state.camera_source = active_source
                    self.state.camera_status = "connected"
                    self.state.camera_error = None

                ok, frame = cap.read()
                if not ok or frame is None:
                    self.state.camera_status = "degraded"
                    self.state.camera_error = f"Frame read failed ({active_source})"
                    cap.release()
                    cap = None
                    time.sleep(0.25)
                    continue

                raw_index += 1
                if raw_index % self.process_every_n != 0:
                    continue

                started = time.time()
                output = self._build_fallback_output()
                annotated = frame
                if self.cv_module is not None:
                    try:
                        output = self.cv_module.run_sentinel_pipeline(frame, verbose=False)
                        annotated = self.cv_module.draw_rich_overlay(frame, output)
                        self.state.cv_status = "connected"
                        self.state.cv_error = None
                    except Exception as exc:
                        self.state.cv_status = "degraded"
                        self.state.cv_error = str(exc)
                        output = self._build_fallback_output()
                        output["scene_summary"] = {"error": str(exc)}
                        annotated = frame

                elapsed_ms = round((time.time() - started) * 1000)
                frames_processed += 1

                payload = self._build_payload(frame, output, annotated)
                payload["pipeline_time_s"] = round(elapsed_ms / 1000, 3)
                self.state.update_feed(payload)

                # Update CV stats from pipeline trace if available
                groq_calls = 0
                groq_cap = None
                scene_score = 0.0
                if self.cv_module is not None:
                    groq_calls = getattr(self.cv_module, "groq_call_count", 0)
                    groq_cap = getattr(self.cv_module, "GROQ_SESSION_CAP", None)
                    scene_detector = getattr(self.cv_module, "scene_change_detector", None)
                    if scene_detector is not None:
                        scene_score = getattr(scene_detector, "frames_since_groq", 0)

                self.state.update_cv_stats(
                    groq_calls=groq_calls,
                    groq_cap=groq_cap,
                    pipeline_latency_ms=elapsed_ms,
                    scene_change_score=scene_score,
                    frames_processed=frames_processed,
                )
        finally:
            if cap is not None:
                cap.release()
            self.state.camera_status = "disconnected"

# ── FastAPI App ──────────────────────────────────────────────


def resolve_camera_source(camera_source: Optional[str], droidcam_stream_url: str) -> str:
    raw_source = "" if camera_source is None else str(camera_source).strip()
    if raw_source.lower() in ("", "auto", "none"):
        return droidcam_stream_url if droidcam_stream_url else "0"
    if raw_source == "0" and droidcam_stream_url:
        # If DroidCam URL is configured, prefer it over default webcam index 0.
        return droidcam_stream_url
    return raw_source


def build_app(args):
    state = GatewayState()
    droidcam_stream_url = (args.droidcam_stream_url or "").strip()
    camera_source = resolve_camera_source(args.camera_source, droidcam_stream_url)

    cv_adapter = CVPipelineAdapter(
        state=state,
        source=camera_source,
        process_every_n=args.process_every_n,
        jpeg_quality=args.jpeg_quality,
    )

    app = FastAPI(title="SENTINEL Gateway", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.gateway = state
    app.state.cv_adapter = cv_adapter
    app.state.qml_base_url = args.qml_base_url.rstrip("/")
    state.droidcam_stream_url = droidcam_stream_url
    state.camera_source = camera_source

    @app.on_event("startup")
    async def on_startup():
        cv_adapter.start()

    @app.on_event("shutdown")
    async def on_shutdown():
        cv_adapter.stop()

    # ── REST endpoints ───────────────────────────────────────

    @app.get("/api/status")
    async def get_status():
        status = state.snapshot_status()
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                response = await client.get(app.state.qml_base_url)
            status["qml"] = {
                "status": "connected" if response.status_code < 500 else "degraded",
                "base_url": app.state.qml_base_url,
            }
            state.qml_status = status["qml"]["status"]
        except Exception as exc:
            status["qml"] = {
                "status": "disconnected",
                "base_url": app.state.qml_base_url,
                "error": str(exc),
            }
            state.qml_status = "disconnected"
        return status

    @app.post("/api/simulate")
    async def simulate(payload: dict[str, Any]):
        target_url = f"{app.state.qml_base_url}/api/simulate"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(target_url, json=payload)
                response.raise_for_status()
            state.qml_status = "connected"
            return response.json()
        except Exception as exc:
            state.qml_status = "degraded"
            raise HTTPException(status_code=503, detail=f"QML simulator unavailable: {exc}") from exc

    @app.post("/api/terac")
    async def terac_dispatch(payload: dict[str, Any]):
        """Store a threat payload for dispatch to the Terac API.

        Expected payload shape:
        {
            "threats": [
                {
                    "type": "...",
                    "severity": "...",
                    "danger_score": 0-10,
                    "bbox": {"x":0,"y":0,"w":0,"h":0},
                    "coordinates": {"lat": ..., "lng": ...},
                    "description": "..."
                }
            ],
            "image_base64": "...",
            "session_id": "..."
        }
        """
        threats = payload.get("threats")
        if not isinstance(threats, list):
            raise HTTPException(status_code=400, detail="'threats' must be an array")
        record = state.add_terac_dispatch(payload)
        return {
            "ok": True,
            "dispatch_id": record["id"],
            "status": record["status"],
            "threats_count": len(threats),
            "timestamp": record["timestamp"],
        }

    @app.get("/api/terac/history")
    async def terac_history(limit: int = 20):
        limit = max(1, min(limit, 100))
        return {"dispatches": state.get_terac_history(limit)}

    # ── WebSocket endpoints ──────────────────────────────────

    @app.websocket("/ws/feed")
    async def ws_feed(websocket: WebSocket):
        await websocket.accept()
        last_seq = -1
        try:
            while True:
                payload = state.snapshot_feed()
                if payload["seq"] != last_seq:
                    await websocket.send_json(payload)
                    last_seq = payload["seq"]
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return

    @app.websocket("/ws/cv_stats")
    async def ws_cv_stats(websocket: WebSocket):
        await websocket.accept()
        last_seq = -1
        try:
            while True:
                payload = state.snapshot_cv_stats()
                if payload["seq"] != last_seq:
                    await websocket.send_json(payload)
                    last_seq = payload["seq"]
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    return app


# ── CLI ──────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="SENTINEL FastAPI gateway")
    parser.add_argument("--host", default=os.environ.get("GATEWAY_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GATEWAY_PORT", "8090")))
    parser.add_argument("--camera-source", default=os.environ.get("DROIDCAM_SOURCE", "auto"))
    parser.add_argument("--process-every-n", type=int, default=int(os.environ.get("DROIDCAM_PROCESS_EVERY_N", "6")))
    parser.add_argument("--jpeg-quality", type=int, default=int(os.environ.get("GATEWAY_JPEG_QUALITY", "80")))
    parser.add_argument("--qml-base-url", default=os.environ.get("QML_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--droidcam-stream-url", default=os.environ.get("DROIDCAM_STREAM_URL", ""))
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


