"""Rerun episode recorder for runtime telemetry."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("pincer.runtime.rerun")

CAMERA_LOG_PERIOD_S = 0.5  # 2 Hz camera snapshots in episode logs


@dataclass
class RecorderStatus:
    enabled: bool
    run_dir: str
    active_episode: str | None
    last_episode: str | None
    episode_count: int
    error: str | None


class RerunEpisodeRecorder:
    """Manages start/stop episode recording and logs selected runtime signals."""

    def __init__(self, run_root: Path) -> None:
        self._lock = threading.Lock()

        self.run_root = run_root.expanduser()
        self.run_root.mkdir(parents=True, exist_ok=True)

        self.run_dir = self.run_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir = self.run_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        self._write_run_links()

        self._episode_index = len(list(self.episodes_dir.glob("episode_*.rrd")))
        self._active_episode_path: Path | None = None
        self._last_episode_path: Path | None = None

        self._tick = 0
        self._next_camera_log_wall_time_s = 0.0
        self._started_wall_time_s = 0.0
        self._last_ik_status: bool | None = None

        self._stream: Any = None
        self._rr: Any | None = None
        self._error: str | None = None
        try:
            import rerun as rr

            self._rr = rr
        except Exception as exc:  # pragma: no cover - import failure depends on environment
            self._error = f"Rerun unavailable: {exc}"
            logger.warning(self._error)

    def status(self) -> RecorderStatus:
        with self._lock:
            return RecorderStatus(
                enabled=self._stream is not None,
                run_dir=str(self.run_dir),
                active_episode=str(self._active_episode_path) if self._active_episode_path else None,
                last_episode=str(self._last_episode_path) if self._last_episode_path else None,
                episode_count=self._episode_index,
                error=self._error,
            )

    def start_recording(self) -> Path:
        """Start a new recording episode and return its target .rrd path."""
        with self._lock:
            if self._rr is None:
                raise RuntimeError(self._error or "Rerun is unavailable.")
            if self._stream is not None:
                if self._active_episode_path is None:
                    raise RuntimeError("Recorder is active but has no episode path.")
                return self._active_episode_path

            self._episode_index += 1
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_name = f"episode_{self._episode_index:04d}_{stamp}.rrd"
            episode_path = self.episodes_dir / episode_name

            recording_id = f"pincer-{stamp}-{self._episode_index:04d}"
            self._stream = self._rr.RecordingStream("pincer-runtime", recording_id=recording_id)

            self._active_episode_path = episode_path
            self._tick = 0
            self._next_camera_log_wall_time_s = 0.0
            self._started_wall_time_s = time.time()
            self._last_ik_status = None
            self._error = None

            self._rr.log(
                "meta/run_dir",
                self._rr.TextDocument(str(self.run_dir)),
                recording=self._stream,
                static=True,
            )
            self._rr.log(
                "meta/episode_path",
                self._rr.TextDocument(str(episode_path)),
                recording=self._stream,
                static=True,
            )

            logger.info("Rerun recording started: %s", episode_path)
            return episode_path

    def stop_recording(self) -> Path | None:
        """Stop active recording and save to disk. Returns .rrd path if saved."""
        with self._lock:
            if self._stream is None:
                return None
            if self._active_episode_path is None:
                raise RuntimeError("Recorder is active but has no episode path.")

            stream = self._stream
            episode_path = self._active_episode_path
            elapsed_s = max(time.time() - self._started_wall_time_s, 0.0)
            ticks = self._tick

            self._stream = None
            self._active_episode_path = None
            self._started_wall_time_s = 0.0
            self._last_ik_status = None

            try:
                self._rr.save(episode_path, recording=stream)
                self._rr.disconnect(recording=stream)
            except Exception as exc:
                self._error = f"Failed to save Rerun episode: {exc}"
                logger.exception(self._error)
                raise RuntimeError(self._error) from exc

            self._last_episode_path = episode_path
            self._write_episode_links(episode_path)
            self._append_episode_manifest(episode_path, elapsed_s, ticks)

            logger.info("Rerun recording saved: %s", episode_path)
            return episode_path

    def log_tick(self, snapshot: dict, frame_bgr: np.ndarray | None, wall_time_s: float) -> None:
        """Log one telemetry tick while recording is active."""
        with self._lock:
            if self._stream is None or self._rr is None:
                return

            rr = self._rr
            stream = self._stream

            try:
                rr.set_time("tick", sequence=self._tick, recording=stream)
                rr.set_time("wall_time", timestamp=wall_time_s, recording=stream)

                joints = snapshot.get("joints", {})
                arm = joints.get("arm", []) or []
                head = joints.get("head", []) or []
                gripper = joints.get("gripper")

                self._log_vector("joint/arm", arm, rr, stream)
                self._log_vector("joint/head", head, rr, stream)
                if gripper is not None:
                    rr.log("joint/gripper", rr.Scalars(float(gripper)), recording=stream)

                command = snapshot.get("command", {})
                cmd_arm = command.get("arm")
                if cmd_arm is not None:
                    self._log_vector("command/arm", cmd_arm, rr, stream)

                ee = snapshot.get("ee")
                if ee is not None:
                    self._log_vector("ee/pose", ee, rr, stream)
                    if len(ee) == 3:
                        rr.log("ee/point", rr.Points3D([list(map(float, ee))]), recording=stream)

                ee_target = command.get("ee_target")
                ik = snapshot.get("ik", {})
                if ee_target is None:
                    ee_target = ik.get("target")
                if ee_target is not None:
                    self._log_vector("command/ee_target", ee_target, rr, stream)
                    if len(ee_target) == 3:
                        rr.log(
                            "command/ee_target_point",
                            rr.Points3D([list(map(float, ee_target))]),
                            recording=stream,
                        )

                ik_error = float(ik.get("error", 0.0))
                ik_iterations = int(ik.get("iterations", 0))
                ik_converged = bool(ik.get("converged", False))
                rr.log("ik/error_m", rr.Scalars(ik_error), recording=stream)
                rr.log("ik/iterations", rr.Scalars(float(ik_iterations)), recording=stream)
                rr.log("ik/converged", rr.Scalars(1.0 if ik_converged else 0.0), recording=stream)

                if self._last_ik_status is None or self._last_ik_status != ik_converged:
                    status_text = "converged" if ik_converged else "solving"
                    rr.log("ik/status", rr.TextLog(status_text), recording=stream)
                    self._last_ik_status = ik_converged

                loop = snapshot.get("loop", {})
                loop_hz = float(loop.get("hz", snapshot.get("loop_hz", 0.0)))
                loop_dt_ms = float(loop.get("dt_ms", 0.0))
                loop_latency_ms = float(loop.get("latency_ms", loop_dt_ms))
                loop_overrun_ms = float(loop.get("overrun_ms", 0.0))
                rr.log("loop/hz", rr.Scalars(loop_hz), recording=stream)
                rr.log("loop/dt_ms", rr.Scalars(loop_dt_ms), recording=stream)
                rr.log("loop/latency_ms", rr.Scalars(loop_latency_ms), recording=stream)
                rr.log("loop/overrun_ms", rr.Scalars(loop_overrun_ms), recording=stream)

                if frame_bgr is not None and wall_time_s >= self._next_camera_log_wall_time_s:
                    frame_rgb = frame_bgr[..., ::-1] if frame_bgr.ndim == 3 and frame_bgr.shape[-1] == 3 else frame_bgr
                    rr.log("camera/color", rr.Image(frame_rgb).compress(jpeg_quality=70), recording=stream)
                    self._next_camera_log_wall_time_s = wall_time_s + CAMERA_LOG_PERIOD_S

                self._tick += 1
            except Exception as exc:
                self._error = f"Rerun logging error: {exc}"
                logger.exception(self._error)

    def _log_vector(self, path: str, values: list[float] | np.ndarray, rr: Any, stream: Any) -> None:
        arr = np.asarray(values, dtype=float).flatten()
        for i, value in enumerate(arr):
            rr.log(f"{path}/{i}", rr.Scalars(float(value)), recording=stream)

    def _append_episode_manifest(self, episode_path: Path, elapsed_s: float, ticks: int) -> None:
        manifest_path = self.run_dir / "episodes.jsonl"
        record = {
            "episode_index": self._episode_index,
            "path": str(episode_path),
            "duration_s": elapsed_s,
            "ticks": ticks,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _write_run_links(self) -> None:
        self._write_link(self.run_root / "latest", self.run_dir)

    def _write_episode_links(self, episode_path: Path) -> None:
        self._write_link(self.run_dir / "latest_episode.rrd", episode_path)
        self._write_link(self.run_root / "latest_episode.rrd", episode_path)

    def _write_link(self, link_path: Path, target: Path) -> None:
        try:
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            relative_target = Path(os.path.relpath(target, link_path.parent))
            link_path.symlink_to(relative_target)
        except Exception:
            logger.debug("Unable to create symlink %s -> %s", link_path, target)
