"""Manually position the gripper on a target, then read q and model EE pos."""

import numpy as np

from lerobot.robots.xlerobot.xlerobot import XLerobot
from lerobot.robots.xlerobot.config_xlerobot import XLerobotConfig
from pincer.ik.constants import ARM_MOTORS
from pincer.ik.model import build_arm_model
from pincer.ik.solver import ee_in_base
from pincer.robots.xlerobot_motor_utils import compute_arm_limits, read_q

PORT = "/dev/ttyACM0"


def main() -> None:
    robot = XLerobot(XLerobotConfig(port1=PORT, use_degrees=True))
    robot.bus1.connect()
    bus = robot.bus1

    bus1_calib = {k: v for k, v in robot.calibration.items() if k in bus.motors}
    if not bus1_calib:
        raise RuntimeError("No calibration found.")
    bus.calibration = bus1_calib
    bus.write_calibration(bus1_calib)

    limits = compute_arm_limits(bus)
    model, data, base_fid, ee_fid = build_arm_model(limits)

    bus.disable_torque()
    print("Torque disabled. Manually move the gripper tip to the target point.")
    input("Press ENTER when the gripper tip is on the target...")

    q = read_q(bus, ARM_MOTORS)
    p_ee = ee_in_base(q, model, data, base_fid, ee_fid)

    print(f"\nq_motor (deg): {q}")
    print(f"Model EE pos (base, m): {p_ee}")
    print(f"Model EE reach (XY, m): {np.linalg.norm(p_ee[:2]):.4f}")
    print(f"Model EE Z (m): {p_ee[2]:.4f}")

    bus.disable_torque()


if __name__ == "__main__":
    main()
