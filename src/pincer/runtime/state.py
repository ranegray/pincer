"""Thread-safe in-memory store for robot telemetry.

Written by the runtime read loop and behaviors (main / worker threads),
read by the dashboard server (background thread).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class IKSnapshot:
    """Snapshot of the IK solver state."""

    target: np.ndarray | None = None
    error: float = 0.0
    iterations: int = 0
    converged: bool = False


@dataclass
class DetectionSnapshot:
    """Snapshot of the latest detection results."""

    bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    centroids: list[tuple[int, int]] = field(default_factory=list)
    label: str = ""


@dataclass
class CommandSnapshot:
    """Snapshot of the latest commanded state."""

    arm: np.ndarray | None = None
    ee_target: np.ndarray | None = None


@dataclass
class LoopTimingSnapshot:
    """Snapshot of loop timing statistics."""

    hz: float = 0.0
    dt_ms: float = 0.0
    latency_ms: float = 0.0
    overrun_ms: float = 0.0


@dataclass
class RecordingSnapshot:
    """Snapshot of rerun recording status."""

    enabled: bool = False
    run_dir: str = ""
    active_episode: str | None = None
    last_episode: str | None = None
    episode_count: int = 0
    error: str | None = None


class RobotState:
    """Thread-safe shared state between the runtime and dashboard.

    All public methods acquire a single lock, which is fine at 10 Hz.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stamp: float = 0.0

        # Joint state
        self._q_arm: np.ndarray = np.zeros(5)
        self._q_head: np.ndarray = np.zeros(2)
        self._gripper: float = 0.0
        self._loads: np.ndarray | None = None

        # End-effector
        self._ee_pos: np.ndarray = np.zeros(3)

        # IK
        self._ik = IKSnapshot()

        # Last command
        self._command = CommandSnapshot()

        # Camera frame (raw BGR numpy array — NOT jpeg-encoded)
        self._frame: np.ndarray | None = None
        self._frame_seq: int = 0

        # Detection overlay
        self._detection = DetectionSnapshot()

        # Loop rate
        self._loop = LoopTimingSnapshot()

        # Behavior status
        self._behavior_name: str = ""
        self._behavior_status: str = "idle"

        # Torque
        self._torque_enabled: bool = True

        # Recording status
        self._recording = RecordingSnapshot()

    # ---- Writer methods (called by runtime / behaviors) ----

    def update_joints(
        self,
        q_arm: np.ndarray,
        q_head: np.ndarray,
        gripper: float,
        loads: np.ndarray | None = None,
    ) -> None:
        with self._lock:
            self._q_arm = q_arm.copy()
            self._q_head = q_head.copy()
            self._gripper = gripper
            self._loads = loads.copy() if loads is not None else None
            self._stamp = time.monotonic()

    def update_ee(self, pos: np.ndarray) -> None:
        with self._lock:
            self._ee_pos = pos.copy()

    def update_ik(
        self,
        target: np.ndarray | None = None,
        error: float = 0.0,
        iterations: int = 0,
        converged: bool = False,
    ) -> None:
        with self._lock:
            self._ik = IKSnapshot(
                target=target.copy() if target is not None else None,
                error=error,
                iterations=iterations,
                converged=converged,
            )

    def update_command(self, arm: np.ndarray | None = None, ee_target: np.ndarray | None = None) -> None:
        with self._lock:
            self._command = CommandSnapshot(
                arm=arm.copy() if arm is not None else None,
                ee_target=ee_target.copy() if ee_target is not None else None,
            )

    def update_frame(self, bgr: np.ndarray) -> None:
        """Store the latest camera frame.  No copy — caller overwrites each tick."""
        with self._lock:
            self._frame = bgr
            self._frame_seq += 1

    def update_detection(
        self,
        bboxes: list[tuple[int, int, int, int]],
        centroids: list[tuple[int, int]],
        label: str = "",
    ) -> None:
        with self._lock:
            self._detection = DetectionSnapshot(
                bboxes=list(bboxes),
                centroids=list(centroids),
                label=label,
            )

    def update_loop_hz(self, hz: float) -> None:
        with self._lock:
            self._loop.hz = hz
            self._loop.dt_ms = 1000.0 / hz if hz > 0 else 0.0
            self._loop.latency_ms = self._loop.dt_ms
            self._loop.overrun_ms = 0.0

    def update_loop_timing(
        self,
        hz: float,
        dt_ms: float,
        latency_ms: float,
        overrun_ms: float,
    ) -> None:
        with self._lock:
            self._loop = LoopTimingSnapshot(
                hz=hz,
                dt_ms=dt_ms,
                latency_ms=latency_ms,
                overrun_ms=overrun_ms,
            )

    def update_behavior(self, name: str, status: str) -> None:
        with self._lock:
            self._behavior_name = name
            self._behavior_status = status

    def update_torque(self, enabled: bool) -> None:
        with self._lock:
            self._torque_enabled = enabled

    def update_recording(
        self,
        *,
        enabled: bool,
        run_dir: str,
        active_episode: str | None,
        last_episode: str | None,
        episode_count: int,
        error: str | None,
    ) -> None:
        with self._lock:
            self._recording = RecordingSnapshot(
                enabled=enabled,
                run_dir=run_dir,
                active_episode=active_episode,
                last_episode=last_episode,
                episode_count=episode_count,
                error=error,
            )

    # ---- Reader methods (called by dashboard server) ----

    def snapshot(self) -> dict:
        """Return a JSON-serializable dict of the current state."""
        with self._lock:
            return {
                "stamp": self._stamp,
                "joints": {
                    "arm": self._q_arm.tolist(),
                    "head": self._q_head.tolist(),
                    "gripper": self._gripper,
                    "loads": self._loads.tolist() if self._loads is not None else None,
                },
                "ee": self._ee_pos.tolist(),
                "ik": {
                    "target": self._ik.target.tolist() if self._ik.target is not None else None,
                    "error": self._ik.error,
                    "iterations": self._ik.iterations,
                    "converged": self._ik.converged,
                },
                "command": {
                    "arm": self._command.arm.tolist() if self._command.arm is not None else None,
                    "ee_target": (
                        self._command.ee_target.tolist() if self._command.ee_target is not None else None
                    ),
                },
                "detection": {
                    "bboxes": self._detection.bboxes,
                    "centroids": self._detection.centroids,
                    "label": self._detection.label,
                },
                "loop_hz": self._loop.hz,
                "loop": {
                    "hz": self._loop.hz,
                    "dt_ms": self._loop.dt_ms,
                    "latency_ms": self._loop.latency_ms,
                    "overrun_ms": self._loop.overrun_ms,
                },
                "behavior": {
                    "name": self._behavior_name,
                    "status": self._behavior_status,
                },
                "torque_enabled": self._torque_enabled,
                "recording": {
                    "enabled": self._recording.enabled,
                    "run_dir": self._recording.run_dir,
                    "active_episode": self._recording.active_episode,
                    "last_episode": self._recording.last_episode,
                    "episode_count": self._recording.episode_count,
                    "error": self._recording.error,
                },
            }

    def get_frame(self) -> tuple[np.ndarray | None, int]:
        """Return (frame, sequence_number).  Frame is raw BGR or None."""
        with self._lock:
            return self._frame, self._frame_seq
