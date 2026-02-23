"""Shared lifecycle and resource-management utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.robots.xlerobot.xlerobot import XLerobot
    from pincer.cameras.d435 import D435


def cleanup(robot: XLerobot | None, camera: D435 | None) -> None:
    """Best-effort teardown of robot and camera resources.

    Safe to call from a ``finally`` block even if neither object was fully
    initialised.
    """
    if camera is not None:
        try:
            camera.stop()
        except Exception:
            pass
    if robot is not None:
        try:
            if robot.bus1.is_connected:
                robot.bus1.disconnect(robot.config.disable_torque_on_disconnect)
        except Exception:
            pass
