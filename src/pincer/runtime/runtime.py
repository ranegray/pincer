"""Core Pincer runtime — owns hardware, reads state, runs behaviors."""

from __future__ import annotations

import logging
import math
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pincer.runtime.rerun_recorder import RerunEpisodeRecorder
from pincer.runtime.state import RobotState

logger = logging.getLogger("pincer.runtime")

GRIPPER_MOTOR = "left_arm_gripper"


@dataclass
class RuntimeConfig:
    """Configuration for the Pincer runtime."""

    port: str = "/dev/ttyACM0"
    camera_serial: str = "310222077874"
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8080
    read_hz: float = 10.0
    mock: bool = False
    run_root: str = "runs/pincer-runtime"


class PincerRuntime:
    """Persistent runtime that owns all robot hardware.

    Start with ``run()`` (blocks until SIGINT) or ``start()`` / ``stop()``
    for programmatic control.
    """

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.state = RobotState()
        self.recorder = RerunEpisodeRecorder(Path(self.config.run_root))
        self._sync_recording_state()

        # Hardware — initialised in start()
        self.robot = None
        self.bus = None
        self.camera = None
        self.limits: dict[str, tuple[float, float]] = {}
        self.model = None
        self.data = None
        self.base_fid: int = 0
        self.ee_fid: int = 0

        # Threads
        self._read_stop = threading.Event()
        self._read_thread: threading.Thread | None = None
        self._behavior_thread: threading.Thread | None = None
        self._active_behavior = None
        self._bus_lock = threading.Lock()

        # Signal handling
        self._running = threading.Event()

    # ---- Lifecycle ----

    def start(self) -> None:
        """Initialise hardware, start read loop and dashboard."""
        if self.config.mock:
            self._start_mock()
        else:
            self._start_hardware()

        # Start read loop
        self._read_stop.clear()
        self._read_thread = threading.Thread(
            target=self._read_loop if not self.config.mock else self._mock_read_loop,
            name="pincer-read-loop",
            daemon=True,
        )
        self._read_thread.start()

        # Start dashboard
        self._start_dashboard()
        self._running.set()

    def stop(self) -> None:
        """Shut down everything cleanly."""
        self._running.clear()

        # Stop active behavior
        self.stop_behavior()

        # Stop read loop
        self._read_stop.set()
        if self._read_thread is not None:
            self._read_thread.join(timeout=2.0)

        # Stop dashboard
        self._stop_dashboard()

        # Disconnect hardware
        if not self.config.mock:
            self._stop_hardware()

        # Stop recording and flush active episode.
        try:
            self.stop_recording()
        except Exception:
            logger.exception("Failed to finalize active rerun recording")

        logger.info("Runtime stopped.")

    def run(self) -> None:
        """Start the runtime and block until SIGINT."""
        stop_event = threading.Event()

        def on_signal(sig, frame):
            del sig, frame
            print("\nShutting down...")
            stop_event.set()

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        self.start()
        logger.info(
            "Runtime started.  Dashboard at http://localhost:%d  Press Ctrl+C to stop.",
            self.config.dashboard_port,
        )

        # Use a timeout loop so the main thread can receive signals.
        while not stop_event.is_set():
            stop_event.wait(timeout=0.5)
        self.stop()

    # ---- Hardware init / teardown ----

    def _start_hardware(self) -> None:
        from lerobot.robots.xlerobot.config_xlerobot import XLerobotConfig
        from lerobot.robots.xlerobot.xlerobot import XLerobot

        from pincer.cameras.d435 import D435, D435Config
        from pincer.ik.constants import ARM_MOTORS, HEAD_MOTORS
        from pincer.ik.model import build_arm_model
        from pincer.robots.xlerobot_motor_utils import (
            compute_arm_limits,
            configure_arm_motors,
            read_q,
        )

        # Motor bus
        self.robot = XLerobot(XLerobotConfig(port1=self.config.port, use_degrees=True))
        self.robot.bus1.connect()
        self.bus = self.robot.bus1

        bus1_calib = {k: v for k, v in self.robot.calibration.items() if k in self.bus.motors}
        if not bus1_calib:
            raise RuntimeError(
                f"No bus1 calibration at {self.robot.calibration_fpath}. "
                "Run calibration once before using the runtime."
            )
        self.bus.calibration = bus1_calib
        self.bus.write_calibration(bus1_calib)
        logger.info("Calibration restored for robot '%s'.", self.robot.id)

        configure_arm_motors(self.bus)
        self.limits = compute_arm_limits(self.bus)

        # Camera
        self.camera = D435(D435Config(serial=self.config.camera_serial, width=640, height=480, fps=30))
        self.camera.start()
        logger.info("Camera started (serial=%s).", self.config.camera_serial)

        # IK model
        self.model, self.data, self.base_fid, self.ee_fid = build_arm_model(self.limits)

        # Initial joint read for EE position
        q_arm = read_q(self.bus, ARM_MOTORS)
        try:
            q_head = read_q(self.bus, HEAD_MOTORS)
        except Exception:
            q_head = np.zeros(2, dtype=float)

        gripper_pos = 0.0
        try:
            g = self.bus.sync_read("Present_Position", [GRIPPER_MOTOR])
            gripper_pos = float(g[GRIPPER_MOTOR])
        except Exception:
            pass

        self.state.update_joints(q_arm, q_head, gripper_pos)
        self.state.update_command(arm=q_arm)
        logger.info("Hardware initialised.")

    def _stop_hardware(self) -> None:
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
        if self.robot is not None:
            try:
                if self.robot.bus1.is_connected:
                    self.robot.bus1.disconnect(self.robot.config.disable_torque_on_disconnect)
            except Exception:
                pass

    # ---- Read loop ----

    def _read_loop(self) -> None:
        from pincer.ik.constants import ARM_MOTORS, HEAD_MOTORS
        from pincer.ik.solver import ee_in_base
        from pincer.robots.xlerobot_motor_utils import read_q

        period = 1.0 / self.config.read_hz
        while not self._read_stop.is_set():
            t0 = time.monotonic()
            frame_for_log = None
            try:
                with self._bus_lock:
                    q_arm = read_q(self.bus, ARM_MOTORS)
                    try:
                        q_head = read_q(self.bus, HEAD_MOTORS)
                    except Exception:
                        q_head = np.zeros(2, dtype=float)

                    gripper_pos = 0.0
                    try:
                        g = self.bus.sync_read("Present_Position", [GRIPPER_MOTOR])
                        gripper_pos = float(g[GRIPPER_MOTOR])
                    except Exception:
                        pass

                p_ee = ee_in_base(q_arm, self.model, self.data, self.base_fid, self.ee_fid)
                self.state.update_joints(q_arm, q_head, gripper_pos)
                self.state.update_ee(p_ee)

                # Camera
                if self.camera is not None:
                    color_image, _, _ = self.camera.read()
                    self.state.update_frame(color_image)
                    frame_for_log = color_image
            except Exception:
                logger.exception("Error in read loop")

            elapsed = time.monotonic() - t0
            hz = 1.0 / max(elapsed, 1e-6)
            latency_ms = elapsed * 1000.0
            overrun_ms = max(elapsed - period, 0.0) * 1000.0
            self.state.update_loop_timing(
                hz=hz,
                dt_ms=latency_ms,
                latency_ms=latency_ms,
                overrun_ms=overrun_ms,
            )
            self.recorder.log_tick(
                snapshot=self.state.snapshot(),
                frame_bgr=frame_for_log,
                wall_time_s=time.time(),
            )

            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    # ---- Mock mode ----

    def _start_mock(self) -> None:
        logger.info("Starting in mock mode (no hardware).")

    def _mock_read_loop(self) -> None:
        period = 1.0 / self.config.read_hz
        t_start = time.monotonic()
        while not self._read_stop.is_set():
            t0 = time.monotonic()
            t = t0 - t_start

            # Synthetic joint positions (slow sinusoidal motion)
            q_arm = np.array([
                20.0 * math.sin(t * 0.5),
                45.0 + 10.0 * math.sin(t * 0.3),
                -30.0 + 15.0 * math.sin(t * 0.4),
                15.0 * math.sin(t * 0.6),
                5.0 * math.sin(t * 0.7),
            ])
            q_head = np.array([5.0 * math.sin(t * 0.2), -3.0 * math.cos(t * 0.2)])
            gripper = 50.0 + 40.0 * math.sin(t * 0.1)
            loads = np.abs(np.random.normal(200, 80, size=5))

            self.state.update_joints(q_arm, q_head, gripper, loads)
            self.state.update_command(arm=q_arm)

            # Synthetic EE position
            ee = np.array([
                0.15 + 0.05 * math.sin(t * 0.3),
                -0.05 + 0.03 * math.cos(t * 0.3),
                0.20 + 0.02 * math.sin(t * 0.5),
            ])
            self.state.update_ee(ee)

            elapsed = time.monotonic() - t0
            hz = 1.0 / max(elapsed, 1e-6)
            latency_ms = elapsed * 1000.0
            overrun_ms = max(elapsed - period, 0.0) * 1000.0
            self.state.update_loop_timing(
                hz=hz,
                dt_ms=latency_ms,
                latency_ms=latency_ms,
                overrun_ms=overrun_ms,
            )
            self.recorder.log_tick(
                snapshot=self.state.snapshot(),
                frame_bgr=None,
                wall_time_s=time.time(),
            )

            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    # ---- Dashboard ----

    def _start_dashboard(self) -> None:
        from pincer.dashboard.server import start_dashboard

        start_dashboard(self)

    def _stop_dashboard(self) -> None:
        from pincer.dashboard.server import stop_dashboard

        stop_dashboard()

    # ---- Torque control ----

    def set_torque(self, enabled: bool) -> None:
        """Enable or disable torque on bus1."""
        if self.bus is None:
            raise RuntimeError("Hardware not initialised.")
        with self._bus_lock:
            if enabled:
                self.bus.enable_torque()
            else:
                self.bus.disable_torque()
        self.state.update_torque(enabled)
        logger.info("Torque %s.", "enabled" if enabled else "disabled")

    # ---- Recording control ----

    def start_recording(self) -> None:
        """Start recording telemetry to a new rerun episode (.rrd)."""
        episode_path = self.recorder.start_recording()
        logger.info("Recording started: %s", episode_path)
        self._sync_recording_state()

    def stop_recording(self) -> None:
        """Stop active telemetry recording and save .rrd episode."""
        episode_path = self.recorder.stop_recording()
        if episode_path is not None:
            logger.info("Recording saved: %s", episode_path)
        self._sync_recording_state()

    def get_recording_status(self) -> dict:
        status = self.recorder.status()
        return {
            "enabled": status.enabled,
            "run_dir": status.run_dir,
            "active_episode": status.active_episode,
            "last_episode": status.last_episode,
            "episode_count": status.episode_count,
            "error": status.error,
        }

    def _sync_recording_state(self) -> None:
        status = self.recorder.status()
        self.state.update_recording(
            enabled=status.enabled,
            run_dir=status.run_dir,
            active_episode=status.active_episode,
            last_episode=status.last_episode,
            episode_count=status.episode_count,
            error=status.error,
        )

    # ---- Behavior engine ----

    def run_behavior(self, name: str, **kwargs) -> None:
        """Start a behavior in a worker thread."""
        if self._active_behavior is not None and self._behavior_thread is not None:
            if self._behavior_thread.is_alive():
                raise RuntimeError(
                    f"Behavior '{self._active_behavior.name}' is already running. "
                    "Stop it first."
                )

        from pincer.runtime.behaviors import get_behavior

        cls = get_behavior(name)
        behavior = cls(self)
        self._active_behavior = behavior
        self.state.update_behavior(name, "running")

        def _run_wrapper():
            try:
                behavior.run(**kwargs)
                self.state.update_behavior(name, "completed")
            except Exception:
                logger.exception("Behavior '%s' failed", name)
                self.state.update_behavior(name, "error")
            finally:
                self._active_behavior = None

        self._behavior_thread = threading.Thread(
            target=_run_wrapper,
            name=f"pincer-behavior-{name}",
            daemon=True,
        )
        self._behavior_thread.start()
        logger.info("Started behavior '%s'.", name)

    def stop_behavior(self) -> None:
        """Request the active behavior to stop and wait for it."""
        if self._active_behavior is not None:
            name = self._active_behavior.name
            self._active_behavior.stop()
            if self._behavior_thread is not None:
                self._behavior_thread.join(timeout=5.0)
            self._active_behavior = None
            self.state.update_behavior("", "idle")
            logger.info("Stopped behavior '%s'.", name)
