from pathlib import Path
from re import T
import signal
import time
import traceback

import numpy as np
import pyrealsense2 as rs

from lerobot.model.kinematics import RobotKinematics
from lerobot.motors.feetech import OperatingMode
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from pincer.cameras.d435 import D435, D435Config

ARM_MOTOR_NAMES = [
    "left_arm_shoulder_pan",
    "left_arm_shoulder_lift",
    "left_arm_elbow_flex",
    "left_arm_wrist_flex",
    "left_arm_wrist_roll",
]
URDF_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]

TORQUE_LIMIT = 800
ACCELERATION = 40
P_COEFF = 8
I_COEFF = 0
D_COEFF = 32
MAX_STEP_DEG = 12.0
LOOP_HZ = 10.0
MAX_TARGET_RADIUS_M = 2.0
STATUS_EVERY_N = 10
TARGET_FILTER_ALPHA = 0.2
MIN_CMD_DELTA_DEG = 0.75
T_BASE_CAMERA_PATH = Path(__file__).resolve().parent / "calibration" / "T_base_camera.npy"


def cleanup(robot: XLerobot | None, camera: D435 | None):
    print("\nCleaning up...")
    if camera is not None:
        try:
            camera.stop()
            print("Camera stopped.")
        except Exception:
            pass

    if robot is not None:
        try:
            if robot.bus1.is_connected:
                robot.bus1.disconnect(robot.config.disable_torque_on_disconnect)
                print("Bus1 disconnected.")
        except Exception:
            pass

    print("Done.")


def apply_limits(robot: XLerobot):
    bus = robot.bus1
    print("Applying motor safety limits:")
    print(f"    Torque_Limit = {TORQUE_LIMIT} / 1000")
    print(f"    Acceleration = {ACCELERATION} / 254")
    print(f"    P={P_COEFF}, I={I_COEFF}, D={D_COEFF}\n")

    bus.disable_torque()
    for name in ARM_MOTOR_NAMES:
        bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
        bus.write("Torque_Limit", name, TORQUE_LIMIT)
        bus.write("Acceleration", name, ACCELERATION)
        bus.write("P_Coefficient", name, P_COEFF)
        bus.write("I_Coefficient", name, I_COEFF)
        bus.write("D_Coefficient", name, D_COEFF)
    bus.enable_torque()


def main():
    robot: XLerobot | None = None
    camera: D435 | None = None
    run_state = {"stop": False}

    def signal_handler(sig, frame):
        run_state["stop"] = True
        print("\nStop requested, exiting loop...")

    signal.signal(signal.SIGINT, signal_handler)

    try:
        robot = XLerobot(XLerobotConfig(port1="/dev/ttyACM0", use_degrees=True))
        robot.bus1.connect()
        robot.bus1.disable_torque()
        robot.bus1.configure_motors()
        robot.bus1.enable_torque()
        apply_limits(robot)

        camera = D435(D435Config(serial="310222077874", width=640, height=480, fps=30))
        camera.start()

        kin = RobotKinematics(
            urdf_path=str(Path(__file__).resolve().parents[1] / "assets" / "xlerobot" / "xlerobot_front.urdf"),
            target_frame_name="Fixed_Jaw_tip",
            joint_names=URDF_JOINT_NAMES,
        )

        t_base_camera = np.load(T_BASE_CAMERA_PATH)
        assert t_base_camera.shape == (4, 4), "T_base_camera must be a 4x4 matrix."
        print(f"Starting smoke loop at {LOOP_HZ:.1f} Hz. Press Ctrl+C to stop.")

        period_s = 1.0 / LOOP_HZ
        iter_idx = 0
        p_base_filtered: np.ndarray | None = None
        while not run_state["stop"]:
            t0 = time.monotonic()
            color_image, depth_image, depth_frame = camera.read()

            # 1) RealSense camera point (meters, camera frame)
            h, w = depth_image.shape
            u, v = w // 2, h // 2
            z = depth_frame.get_distance(u, v)
            if z <= 0:
                if iter_idx % STATUS_EVERY_N == 0:
                    print("Skipping frame: invalid depth at center pixel.")
                iter_idx += 1
                continue

            x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(camera.intrinsics, [u, v], z)  # type: ignore[attr-defined]

            # 2) Camera -> robot base transform
            p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
            p_base = t_base_camera @ p_cam
            if np.linalg.norm(p_base[:3]) > MAX_TARGET_RADIUS_M:
                if iter_idx % STATUS_EVERY_N == 0:
                    print(f"Skipping frame: target out of range in base frame {p_base[:3]}.")
                iter_idx += 1
                continue
            if p_base_filtered is None:
                p_base_filtered = p_base[:3].copy()
            else:
                p_base_filtered = (1.0 - TARGET_FILTER_ALPHA) * p_base_filtered + TARGET_FILTER_ALPHA * p_base[:3]

            # 3) Read current arm joints from bus1 (in degrees because use_degrees=True)
            q_curr_dict = robot.bus1.sync_read("Present_Position", ARM_MOTOR_NAMES)
            q_curr = np.array([q_curr_dict[m] for m in ARM_MOTOR_NAMES], dtype=float)

            # 4) Build desired EE pose
            t_curr = kin.forward_kinematics(q_curr)
            t_des = np.eye(4)
            t_des[:3, :3] = t_curr[:3, :3]
            t_des[:3, 3] = p_base_filtered

            # 5) IK (input/output joint angles are degrees)
            q_target = kin.inverse_kinematics(q_curr, t_des, orientation_weight=0.0)

            # 6) Send a safe position command to arm motors
            q_cmd = np.clip(q_target, q_curr - MAX_STEP_DEG, q_curr + MAX_STEP_DEG)
            if np.max(np.abs(q_cmd - q_curr)) >= MIN_CMD_DELTA_DEG:
                goal = {name: float(q_cmd[i]) for i, name in enumerate(ARM_MOTOR_NAMES)}
                robot.bus1.sync_write("Goal_Position", goal)

            if iter_idx % STATUS_EVERY_N == 0:
                print(f"p_base_raw={p_base[:3]}, p_base_filt={p_base_filtered}, q_curr={q_curr}, q_cmd={q_cmd}")
            iter_idx += 1

            elapsed = time.monotonic() - t0
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        cleanup(robot, camera)


if __name__ == "__main__":
    main()
