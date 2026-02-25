"""FastAPI dashboard server — WebSocket telemetry, MJPEG camera, behavior API."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

if TYPE_CHECKING:
    from pincer.runtime.runtime import PincerRuntime

logger = logging.getLogger("pincer.dashboard")

DIST_DIR = Path(__file__).parent / "frontend" / "dist"


class BehaviorRequest(BaseModel):
    name: str
    params: dict = {}


def create_app(runtime: PincerRuntime) -> FastAPI:
    app = FastAPI(title="Pincer Dashboard")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state = runtime.state

    @app.websocket("/ws")
    async def ws_telemetry(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                data = state.snapshot()
                await ws.send_text(json.dumps(data))
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            pass

    @app.get("/stream")
    async def mjpeg_stream():
        def generate():
            last_seq = -1
            while True:
                frame, seq = state.get_frame()
                if frame is not None and seq != last_seq:
                    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    last_seq = seq
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                    )
                time.sleep(1.0 / 15.0)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/state")
    async def get_state():
        return state.snapshot()

    @app.post("/api/behavior/start")
    async def start_behavior(req: BehaviorRequest):
        try:
            runtime.run_behavior(req.name, **req.params)
            return {"status": "started", "name": req.name}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    @app.post("/api/behavior/stop")
    async def stop_behavior():
        runtime.stop_behavior()
        return {"status": "stopped"}

    @app.post("/api/torque/enable")
    async def enable_torque():
        try:
            runtime.set_torque(True)
            return {"status": "ok", "torque_enabled": True}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    @app.post("/api/torque/disable")
    async def disable_torque():
        try:
            runtime.set_torque(False)
            return {"status": "ok", "torque_enabled": False}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    @app.get("/api/recording")
    async def get_recording():
        return {"status": "ok", **runtime.get_recording_status()}

    @app.post("/api/recording/start")
    async def start_recording():
        try:
            runtime.start_recording()
            return {"status": "ok", **runtime.get_recording_status()}
        except Exception as exc:
            return {"status": "error", "message": str(exc), **runtime.get_recording_status()}

    @app.post("/api/recording/stop")
    async def stop_recording():
        try:
            runtime.stop_recording()
            return {"status": "ok", **runtime.get_recording_status()}
        except Exception as exc:
            return {"status": "error", "message": str(exc), **runtime.get_recording_status()}

    # Serve static frontend build if it exists
    if DIST_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="frontend")

    return app


# ---- Background thread lifecycle ----

_server_thread: threading.Thread | None = None
_server_instance: uvicorn.Server | None = None


def start_dashboard(runtime: PincerRuntime) -> str:
    """Start the dashboard in a background daemon thread.  Returns the URL."""
    global _server_thread, _server_instance

    config = runtime.config
    app = create_app(runtime)

    uvi_config = uvicorn.Config(
        app,
        host=config.dashboard_host,
        port=config.dashboard_port,
        log_level="warning",
    )
    _server_instance = uvicorn.Server(uvi_config)
    _server_instance.install_signal_handlers = lambda: None

    _server_thread = threading.Thread(
        target=_server_instance.run,
        name="pincer-dashboard",
        daemon=True,
    )
    _server_thread.start()

    url = f"http://localhost:{config.dashboard_port}"
    logger.info("Dashboard started at %s", url)
    return url


def stop_dashboard() -> None:
    """Signal the background uvicorn server to shut down."""
    global _server_instance
    if _server_instance is not None:
        _server_instance.should_exit = True
        _server_instance = None
