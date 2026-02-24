"""Detect a colored object, reach to it, grasp, and return to start.

Uses HSV color detection to find the target, IK to reach it, then closes
the gripper and returns the arm to its starting joint configuration.
"""

import signal
import time

import numpy as np
import pyrealsense2 as rs

from lerobot.robots.xlerobot.xlerobot import XLerobot
from lerobot.robots.xlerobot.config_xlerobot import XLerobotConfig
from pincer.cameras.d435 import D435, D435Config
from pincer.detect import ColorDetector
from pincer.ik.constants import ARM_MOTORS, HEAD_MOTORS
from pincer.ik.model import build_arm_model
from pincer.ik.solver import ee_in_base, pan_guess_deg, solve_to_target
from pincer.ik.transforms import build_t_base_camera, camera_to_base
from pincer.robots.xlerobot_motor_utils import clip_arm, compute_arm_limits, configure_arm_motors, read_q
from pincer.utils import cleanup

PORT = "/dev/ttyACM0"
CAMERA_SERIAL = "310222077874"
TABLE_Z_BASE = -0.08
GRIPPER_MOTOR = "left_arm_gripper"
GRIPPER_OPEN = 100.0
GRIPPER_CLOSED = 10.0
GRASP_OFFSET_SIGN = -1.0  # +1 or -1: shifts target in camera X by half object width


def reach_target(bus, p_target, limits, model, data, base_fid, ee_fid, *, running_fn, period=0.1):
    """Re-solve IK each tick and step toward a Cartesian target until convergence."""
    hold = 0
    i = 0

    while running_fn():
        t0 = time.monotonic()

        q_curr = clip_arm(read_q(bus, ARM_MOTORS), limits)
        p_ee = ee_in_base(q_curr, model, data, base_fid, ee_fid)
        err_vec = p_target - p_ee
        err = float(np.linalg.norm(err_vec))

        if err <= 0.01:
            hold += 1
            if hold >= 3:
                return True
        else:
            hold = 0

        q_seed = q_curr.copy()
        if abs(float(err_vec[0])) >= 0.01:
            q_seed[0] = float(np.clip(pan_guess_deg(p_target), *limits["left_arm_shoulder_pan"]))
        q_goal, _, _, _ = solve_to_target(q_seed, p_target, model, data, base_fid, ee_fid, limits)

        step = 10.0 if err > 0.12 else 6.0
        q_cmd = clip_arm(np.clip(q_goal, q_curr - step, q_curr + step), limits)
        bus.sync_write("Goal_Position", {n: float(q_cmd[j]) for j, n in enumerate(ARM_MOTORS)})

        if i % 20 == 0:
            print(f"  err={err:.4f} m, step_deg={step:.1f}")
        i += 1

        sleep_s = period - (time.monotonic() - t0)
        if sleep_s > 0:
            time.sleep(sleep_s)

    return False



def main() -> None:
    robot: XLerobot | None = None
    camera: D435 | None = None
    running = True

    def on_sigint(sig, frame):
        del sig, frame
        nonlocal running
        running = False
        print("\nStopping...")

    signal.signal(signal.SIGINT, on_sigint)

    try:
        # --- Robot setup ---
        robot = XLerobot(XLerobotConfig(port1=PORT, use_degrees=True))
        robot.bus1.connect()
        bus = robot.bus1

        bus1_calib = {k: v for k, v in robot.calibration.items() if k in bus.motors}
        if not bus1_calib:
            raise RuntimeError(
                f"No bus1 calibration at {robot.calibration_fpath}. "
                "Run calibration once before using this script."
            )
        input(f"Calibration found for robot id '{robot.id}'. Press ENTER to restore and continue: ")
        bus.calibration = bus1_calib
        bus.write_calibration(bus1_calib)
        print(f"Bus1 calibration restored. is_calibrated={bus.is_calibrated}\n")

        configure_arm_motors(bus)
        limits = compute_arm_limits(bus)

        # --- Camera setup ---
        camera = D435(D435Config(serial=CAMERA_SERIAL, width=640, height=480, fps=30))
        camera.start()

        # --- Build camera->base transform from current joint config ---
        q_arm = read_q(bus, ARM_MOTORS)
        try:
            q_head = read_q(bus, HEAD_MOTORS)
        except Exception:
            q_head = np.zeros(2, dtype=float)

        T_base_camera = build_t_base_camera(q_arm, q_head)

        # --- Build reduced IK model ---
        model, data, base_fid, ee_fid = build_arm_model(limits)

        # --- Detect color and acquire depth target ---
        detector = ColorDetector.from_color("blue")
        print("Searching for target...")
        while running:
            color_image, _, depth_frame = camera.read()
            detections = detector.detect(color_image)
            if not detections:
                continue
            det = detections[0]
            u, v = det.centroid
            z = depth_frame.get_distance(u, v)
            if z <= 0:
                continue
            bbox_w_px = det.bbox[2]
            fx = camera.intrinsics.fx
            obj_width_m = bbox_w_px * z / fx
            print(
                f"Detected blob at ({u}, {v}), area={det.area:.0f}, "
                f"depth={z:.3f} m, obj_width={obj_width_m:.3f} m"
            )
            x, y, z = rs.rs2_deproject_pixel_to_point(camera.intrinsics, [u, v], z)  # type: ignore[attr-defined]
            # TODO tweak the grasp offset, needs to be slightly smaller
            x += GRASP_OFFSET_SIGN * obj_width_m / 1.5
            p_cam = np.array([x, y, z], dtype=float)
            p_target = camera_to_base(p_cam, T_base_camera)
            p_target[2] = max(p_target[2], TABLE_Z_BASE)
            print(f"Target in base frame (m): {p_target}")
            break
        else:
            print("Interrupted before detection.")
            return

        # --- Open gripper ---
        bus.sync_write("Goal_Position", {GRIPPER_MOTOR: GRIPPER_OPEN})
        print("Gripper opened.")
        time.sleep(0.3)

        # --- Reach to target ---
        print("Reaching to target...")
        reached = reach_target(
            bus, p_target, limits, model, data, base_fid, ee_fid, running_fn=lambda: running
        )
        if not reached:
            print("Did not reach target.")
            return

        print("Reached target. Closing gripper...")
        bus.sync_write("Goal_Position", {GRIPPER_MOTOR: GRIPPER_CLOSED})
        time.sleep(3.0)
        print("Grasp complete.")

        # --- Lift object ---
        p_lift = p_target.copy()
        p_lift[2] += 0.08  # lift 8 cm
        print(f"Lifting to {p_lift}...")
        reach_target(
            bus, p_lift, limits, model, data, base_fid, ee_fid, running_fn=lambda: running
        )
        print("Holding for 5 seconds...")
        time.sleep(5.0)

    finally:
        cleanup(robot, camera)


if __name__ == "__main__":
    main()
