"""Motor bus helpers for the XLerobot arm."""

import numpy as np

from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from pincer.ik.constants import ARM_MOTORS


def configure_arm_motors(bus: FeetechMotorsBus) -> None:
    """Set position mode and PID gains on all arm motors."""
    bus.disable_torque()
    for m in ARM_MOTORS:
        bus.write("Operating_Mode", m, OperatingMode.POSITION.value)
        bus.write("Torque_Limit", m, 800)
        bus.write("Acceleration", m, 80)
        bus.write("P_Coefficient", m, 16)
        bus.write("I_Coefficient", m, 0)
        bus.write("D_Coefficient", m, 32)
    bus.enable_torque()


def compute_arm_limits(bus: FeetechMotorsBus) -> dict[str, tuple[float, float]]:
    """Derive per-motor position limits (degrees) from calibration data."""
    limits: dict[str, tuple[float, float]] = {}
    for m in ARM_MOTORS:
        cal = bus.calibration.get(m)
        if cal is None:
            raise RuntimeError(f"Missing calibration for motor '{m}'.")
        max_res = bus.model_resolution_table[bus.motors[m].model] - 1
        mid = (cal.range_min + cal.range_max) / 2.0
        lo = (cal.range_min - mid) * 360.0 / max_res + 0.5
        hi = (cal.range_max - mid) * 360.0 / max_res - 0.5
        limits[m] = (float(min(lo, hi)), float(max(lo, hi)))
    return limits


def read_q(bus: FeetechMotorsBus, names: list[str]) -> np.ndarray:
    """Sync-read joint positions for the given motor names (degrees)."""
    q = bus.sync_read("Present_Position", names)
    return np.array([q[n] for n in names], dtype=float)


def clip_arm(q: np.ndarray, limits: dict[str, tuple[float, float]]) -> np.ndarray:
    """Clip arm joint angles to calibrated limits (degrees)."""
    out = q.copy()
    for i, name in enumerate(ARM_MOTORS):
        lo, hi = limits[name]
        out[i] = float(np.clip(out[i], lo, hi))
    return out
