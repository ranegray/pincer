from pathlib import Path
import signal
import time

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
LOOP_HZ = 10.0
MAX_STEP_DEG = 6.0
T_BASE_CAMERA_PATH = Path(__file__).resolve().parent / "calibration" / "T_base_camera.npy"
URDF_PATH = Path(__file__).resolve().parents[1] / "assets" / "xlerobot" / "xlerobot_front.urdf"


def cleanup(robot: XLerobot | None, camera: D435 | None):
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


def apply_limits(robot: XLerobot):
    bus = robot.bus1
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
    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        running = False
        print("\nStopping...")

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        robot = XLerobot(XLerobotConfig(port1="/dev/ttyACM0", use_degrees=True))
        robot.bus1.connect()
        robot.bus1.disable_torque()
        robot.bus1.configure_motors()
        robot.bus1.enable_torque()
        apply_limits(robot)

        camera = D435(D435Config(serial="310222077874", width=640, height=480, fps=30))
        camera.start()

        if not T_BASE_CAMERA_PATH.exists():
            raise FileNotFoundError(f"Missing {T_BASE_CAMERA_PATH}")
        t_base_camera = np.load(T_BASE_CAMERA_PATH)
        if t_base_camera.shape != (4, 4):
            raise ValueError(f"T_base_camera must be 4x4, got {t_base_camera.shape}")

        kin = RobotKinematics(
            urdf_path=str(URDF_PATH),
            target_frame_name="Fixed_Jaw_tip",
            joint_names=URDF_JOINT_NAMES,
        )

        period = 1.0 / LOOP_HZ
        i = 0
        print("Running. Press Ctrl+C to stop.")
        while running:
            t0 = time.monotonic()
            _, depth_image, depth_frame = camera.read()

            h, w = depth_image.shape
            u, v = w // 2, h // 2
            z = depth_frame.get_distance(u, v)
            if z <= 0:
                continue

            x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(camera.intrinsics, [u, v], z)  # type: ignore[attr-defined]
            p_base = t_base_camera @ np.array([x_cam, y_cam, z_cam, 1.0])

            q_curr_dict = robot.bus1.sync_read("Present_Position", ARM_MOTOR_NAMES)
            q_curr = np.array([q_curr_dict[m] for m in ARM_MOTOR_NAMES], dtype=float)

            t_curr = kin.forward_kinematics(q_curr)
            t_des = np.eye(4)
            t_des[:3, :3] = t_curr[:3, :3]
            t_des[:3, 3] = p_base[:3]

            q_target = kin.inverse_kinematics(q_curr, t_des, orientation_weight=0.0)
            q_cmd = np.clip(q_target, q_curr - MAX_STEP_DEG, q_curr + MAX_STEP_DEG)
            robot.bus1.sync_write("Goal_Position", {n: float(q_cmd[j]) for j, n in enumerate(ARM_MOTOR_NAMES)})

            if i % 10 == 0:
                print(f"p_base={p_base[:3]}, q_curr={q_curr}, q_cmd={q_cmd}")
            i += 1

            sleep_s = period - (time.monotonic() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        cleanup(robot, camera)


if __name__ == "__main__":
    main()
