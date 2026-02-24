"""Point-and-reach script for the XLerobot arm.

Reads the depth at the center pixel of the RealSense frame, transforms it
to the robot base frame, and executes IK to move the arm there.
"""

import signal
import time

import numpy as np
import pyrealsense2 as rs

from lerobot.robots.xlerobot.xlerobot import XLerobot
from lerobot.robots.xlerobot.config_xlerobot import XLerobotConfig
from pincer.cameras.d435 import D435, D435Config
from pincer.ik.constants import ARM_MOTORS, BASE_FRAME, CAMERA_FRAME, EE_FRAME, HEAD_MOTORS
from pincer.ik.model import build_arm_model
from pincer.ik.solver import ee_in_base, pan_guess_deg, solve_to_target
from pincer.ik.transforms import build_t_base_camera, camera_to_base
from pincer.robots.xlerobot_motor_utils import clip_arm, compute_arm_limits, configure_arm_motors, read_q
from pincer.utils import cleanup

PORT = "/dev/ttyACM0"
CAMERA_SERIAL = "310222077874"
TABLE_Z_BASE = -0.08  # table surface height in base frame (m); base is ~8-10 cm above table


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

        # --- Acquire depth target at image center ---
        while True:
            _, depth_image, depth_frame = camera.read()
            h, w = depth_image.shape
            u, v = w // 2, h // 2
            z = depth_frame.get_distance(u, v)
            if z <= 0:
                continue
            x, y, z = rs.rs2_deproject_pixel_to_point(camera.intrinsics, [u, v], z)  # type: ignore[attr-defined]
            p_cam = np.array([x, y, z], dtype=float)
            p_target = camera_to_base(p_cam, T_base_camera)
            p_target[2] = max(p_target[2], TABLE_Z_BASE)  # clamp Z to table surface minimum
            print(f"Target in base frame (m): {p_target}")
            break

        # --- Initial IK solve ---
        q_seed = clip_arm(q_arm, limits)
        q_seed[0] = float(np.clip(pan_guess_deg(p_target), *limits["left_arm_shoulder_pan"]))
        q_goal, ok, iters, ik_err = solve_to_target(q_seed, p_target, model, data, base_fid, ee_fid, limits)
        q_goal = clip_arm(q_goal, limits)

        print(f"Frames: {CAMERA_FRAME} -> {BASE_FRAME} -> {EE_FRAME}")
        print(f"IK result: ok={ok}, iters={iters}, planned_err_m={ik_err:.4f}, target_base={p_target}")
        if ik_err > 0.05:
            print(f"[WARN] IK residual is high ({ik_err:.3f} m).")
        print("Running execution loop. Press Ctrl+C to stop.")

        # --- Execution loop ---
        period = 1.0 / 10.0
        hold = 0
        i = 0

        while running:
            t0 = time.monotonic()

            q_curr = clip_arm(read_q(bus, ARM_MOTORS), limits)
            p_ee = ee_in_base(q_curr, model, data, base_fid, ee_fid)
            err_vec = p_target - p_ee
            err = float(np.linalg.norm(err_vec))

            if err <= 0.01:
                hold += 1
                if hold >= 3:
                    print(f"Reached target (err={err:.4f} m).")
                    break
            else:
                hold = 0

            q_seed_step = q_curr.copy()
            if abs(float(err_vec[0])) >= 0.01:
                q_seed_step[0] = float(np.clip(pan_guess_deg(p_target), *limits["left_arm_shoulder_pan"]))
            q_goal, _, _, _ = solve_to_target(q_seed_step, p_target, model, data, base_fid, ee_fid, limits)

            step = 10.0 if err > 0.12 else 6.0
            q_cmd = clip_arm(np.clip(q_goal, q_curr - step, q_curr + step), limits)
            bus.sync_write("Goal_Position", {n: float(q_cmd[j]) for j, n in enumerate(ARM_MOTORS)})

            if i % 20 == 0:
                print(
                    f"err_norm={err:.4f}, step_deg={step:.1f}, "
                    f"q_pan={q_curr[0]:.2f}, q_lift={q_curr[1]:.2f}, q_elbow={q_curr[2]:.2f}"
                )
            i += 1

            sleep_s = period - (time.monotonic() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    finally:
        cleanup(robot, camera)


if __name__ == "__main__":
    main()
