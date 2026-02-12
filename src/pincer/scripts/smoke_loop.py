from pathlib import Path

import numpy as np
import pyrealsense2 as rs

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from pincer.cameras.d435 import D435, D435Config

robot = XLerobot(XLerobotConfig(port1="/dev/ttyACM0", use_degrees=True))
robot.bus1.connect()

camera = D435(D435Config(serial="310222077874", width=640, height=480, fps=30))
camera.start()

ARM_MOTOR_NAMES = [
    "left_arm_shoulder_pan",
    "left_arm_shoulder_lift",
    "left_arm_elbow_flex",
    "left_arm_wrist_flex",
    "left_arm_wrist_roll",
]
URDF_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]

kin = RobotKinematics(
    urdf_path=str(Path(__file__).resolve().parents[1] / "assets" / "xlerobot" / "xlerobot_front.urdf"),
    target_frame_name="Fixed_Jaw_tip",
    joint_names=URDF_JOINT_NAMES,
)

color_image, depth_image, depth_frame = camera.read()

# 1) RealSense camera point (meters, camera frame)
h, w = depth_image.shape
u, v = w // 2, h // 2
z = depth_frame.get_distance(u, v)
x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(camera.intrinsics, [u, v], z)  # type: ignore[attr-defined]

T_base_camera = np.load("calibration/T_base_camera.npy")
assert T_base_camera.shape == (4, 4), "T_base_camera must be a 4x4 matrix."

# 2) Camera -> robot base transform
p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
p_base = T_base_camera @ p_cam

# 3) Read current arm joints from bus1 (in degrees because use_degrees=True)
q_curr_dict = robot.bus1.sync_read("Present_Position", ARM_MOTOR_NAMES)
q_curr = np.array([q_curr_dict[m] for m in ARM_MOTOR_NAMES], dtype=float)

# 4) Build desired EE pose:
t_curr = kin.forward_kinematics(q_curr)
t_des = np.eye(4)
t_des[:3, :3] = t_curr[:3, :3]
t_des[:3, 3] = p_base[:3]

# 5) IK (input/output joint angles are degrees)
q_target = kin.inverse_kinematics(q_curr, t_des)

print(f"p_base (m): {p_base[:3]}")
print(f"q_curr: {q_curr}")
print(f"q_target: {q_target}")

# 6) Send a safe position command to arm motors
if np.linalg.norm(p_base[:3]) > 2.0:
    raise RuntimeError(
        f"Refusing to command arm: target in base frame looks invalid ({p_base[:3]} m). "
        "Check T_base_camera calibration/units."
    )

# Limit one-step motion to avoid jumps from IK spikes.
max_step_deg = 12.0
q_cmd = np.clip(q_target, q_curr - max_step_deg, q_curr + max_step_deg)
goal = {name: float(q_cmd[i]) for i, name in enumerate(ARM_MOTOR_NAMES)}
robot.bus1.sync_write("Goal_Position", goal)
print(f"q_cmd: {q_cmd}")
