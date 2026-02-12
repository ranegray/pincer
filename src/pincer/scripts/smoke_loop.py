import numpy as np
import pyrealsense2 as rs

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from pincer.cameras.d435 import D435, D435Config

robot = XLerobot(XLerobotConfig(port1="/dev/ttyACM0"))
robot.bus1.connect()

camera = D435(D435Config(serial="310222077874", width=640, height=480, fps=30))
camera.start()

kin = RobotKinematics(
    urdf_path="./SO101/so101_new_calib.urdf",  # replace with XLerobot URDF
    target_frame_name="gripper_frame_link",
    joint_names=list(robot.bus1.motors.keys()),
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

# 3) Read current joints from bus1
motor_names = list(robot.bus1.motors.keys())
q_curr_dict = robot.bus1.sync_read("Present_Position", motor_names)
q_curr = np.array([q_curr_dict[m] for m in motor_names], dtype=float)

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
