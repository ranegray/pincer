import signal
import time
from pathlib import Path

import numpy as np
import pink
import pinocchio as pin
import pyrealsense2 as rs
from pink.tasks import FrameTask, PostureTask

from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from pincer.cameras.d435 import D435, D435Config

PORT = "/dev/ttyACM0"
CAMERA_SERIAL = "310222077874"
URDF_PATH = Path(__file__).resolve().parents[1] / "assets" / "xlerobot" / "xlerobot_front.urdf"

BASE_FRAME = "Base_2"
CAMERA_FRAME = "head_camera_depth_optical_frame"
EE_FRAME = "Fixed_Jaw_tip_2"

ARM_MOTORS = [
    "left_arm_shoulder_pan",
    "left_arm_shoulder_lift",
    "left_arm_elbow_flex",
    "left_arm_wrist_flex",
    "left_arm_wrist_roll",
]
ARM_JOINTS = ["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"]
HEAD_MOTORS = ["head_motor_1", "head_motor_2"]
HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]


# ---------------------------------------------------------------------------
# Motor <-> URDF convention mapping
# ---------------------------------------------------------------------------


def arm_motor_to_urdf(q_deg: np.ndarray) -> np.ndarray:
    out = q_deg.copy()
    out[1] = 90.0 - out[1]
    out[2] = out[2] + 90.0
    return out


def arm_urdf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    out = q_deg.copy()
    out[1] = 90.0 - out[1]
    out[2] = out[2] - 90.0
    return out


def head_motor_to_urdf(q_deg: np.ndarray) -> np.ndarray:
    out = q_deg.copy()
    out[0] = -out[0]  # critical for correct camera left/right mapping
    return out


# ---------------------------------------------------------------------------
# Robot setup helpers
# ---------------------------------------------------------------------------


def configure_arm_motors(bus: FeetechMotorsBus) -> None:
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
    q = bus.sync_read("Present_Position", names)
    return np.array([q[n] for n in names], dtype=float)


def clip_arm(q: np.ndarray, limits: dict[str, tuple[float, float]]) -> np.ndarray:
    out = q.copy()
    for i, n in enumerate(ARM_MOTORS):
        lo, hi = limits[n]
        out[i] = float(np.clip(out[i], lo, hi))
    return out


# ---------------------------------------------------------------------------
# IK helpers
# ---------------------------------------------------------------------------


def motor_to_pin_q(q_motor: np.ndarray, model: pin.Model) -> np.ndarray:
    q_pin = pin.neutral(model)
    for name, q_deg in zip(ARM_JOINTS, arm_motor_to_urdf(q_motor)):
        jid = model.getJointId(name)
        q_pin[model.joints[jid].idx_q] = np.deg2rad(float(q_deg))
    return q_pin


def pin_to_motor_q(q_pin: np.ndarray, model: pin.Model) -> np.ndarray:
    q_urdf = np.zeros(len(ARM_JOINTS), dtype=float)
    for i, name in enumerate(ARM_JOINTS):
        jid = model.getJointId(name)
        q_urdf[i] = np.rad2deg(float(q_pin[model.joints[jid].idx_q]))
    return arm_urdf_to_motor(q_urdf)


def ee_in_base(
    q_motor: np.ndarray,
    model: pin.Model,
    data: pin.Data,
    base_fid: int,
    ee_fid: int,
) -> np.ndarray:
    q_pin = motor_to_pin_q(q_motor, model)
    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)
    oMb = data.oMf[base_fid]
    oMe = data.oMf[ee_fid]
    return oMb.rotation.T @ (oMe.translation - oMb.translation)


def pan_guess_deg(target: np.ndarray) -> float:
    return float(np.rad2deg(np.arctan2(float(target[0]), max(1e-6, -float(target[1])))))


def solve_to_target(
    q_seed_local: np.ndarray,
    target: np.ndarray,
    model: pin.Model,
    data: pin.Data,
    base_fid: int,
    ee_fid: int,
    limits: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, bool, int, float]:
    q = q_seed_local.copy()
    cfg = pink.Configuration(model, data, motor_to_pin_q(q, model))
    ee_task = FrameTask(EE_FRAME, position_cost=10.0, orientation_cost=0.0)
    posture_task = PostureTask(cost=1e-2)
    posture_task.set_target(motor_to_pin_q(q_seed_local, model))

    prev_err = float("inf")
    flat_steps = 0
    for step in range(300):
        pin.forwardKinematics(model, data, cfg.q)
        pin.updateFramePlacements(model, data)
        oMb = data.oMf[base_fid]
        oMe = data.oMf[ee_fid]
        p_ee = oMb.rotation.T @ (oMe.translation - oMb.translation)
        err = float(np.linalg.norm(target - p_ee))
        if err <= 0.005:
            return q, True, step, err

        target_world = oMb.rotation @ target + oMb.translation
        ee_task.set_target(pin.SE3(oMe.rotation, target_world))
        dq = pink.solve_ik(cfg, [ee_task, posture_task], 0.01, solver="quadprog")
        cfg.integrate_inplace(dq, 0.01)

        q_next = clip_arm(pin_to_motor_q(cfg.q, model), limits)
        cfg = pink.Configuration(model, data, motor_to_pin_q(q_next, model))

        if float(np.max(np.abs(q_next - q))) <= 0.02 and (prev_err - err) <= 2e-4:
            flat_steps += 1
        else:
            flat_steps = 0
        q = q_next
        prev_err = err
        if flat_steps >= 25:
            return q, False, step + 1, err

    final_err = float(np.linalg.norm(target - ee_in_base(q, model, data, base_fid, ee_fid)))
    return q, False, 300, final_err


# ---------------------------------------------------------------------------
# IK model construction
# ---------------------------------------------------------------------------


def build_ik_model(
    limits: dict[str, tuple[float, float]],
    q_arm: np.ndarray,
    q_head: np.ndarray,
) -> tuple[pin.Model, pin.Data, int, int, np.ndarray]:
    """Build a reduced (arm-only) IK model and compute t_base_camera from the full model.

    Returns (model, data, base_fid, ee_fid, t_base_camera).
    """
    full = pin.buildModelFromUrdf(str(URDF_PATH))
    full_data = full.createData()

    # Set joint values on the full model to compute camera->base transform.
    q_full = pin.neutral(full)
    for name, q_deg in zip(ARM_JOINTS, arm_motor_to_urdf(q_arm)):
        jid = full.getJointId(name)
        q_full[full.joints[jid].idx_q] = np.deg2rad(float(q_deg))
    for name, q_deg in zip(HEAD_JOINTS, head_motor_to_urdf(q_head)):
        jid = full.getJointId(name)
        q_full[full.joints[jid].idx_q] = np.deg2rad(float(q_deg))

    pin.forwardKinematics(full, full_data, q_full)
    pin.updateFramePlacements(full, full_data)

    oMbase = full_data.oMf[full.getFrameId(BASE_FRAME)]
    oMcam = full_data.oMf[full.getFrameId(CAMERA_FRAME)]
    t_base_camera = np.eye(4)
    t_base_camera[:3, :3] = oMbase.rotation.T @ oMcam.rotation
    t_base_camera[:3, 3] = oMbase.rotation.T @ (oMcam.translation - oMbase.translation)

    # Build reduced model (arm joints only).
    keep = set(ARM_JOINTS)
    lock = [jid for jid in range(1, full.njoints) if full.joints[jid].nq > 0 and full.names[jid] not in keep]
    model = pin.buildReducedModel(full, lock, pin.neutral(full))
    data = model.createData()

    ee_fid = model.getFrameId(EE_FRAME)
    base_fid = model.getFrameId(BASE_FRAME)
    if ee_fid >= model.nframes or base_fid >= model.nframes:
        raise RuntimeError("Missing EE or base frame in IK model.")

    # Apply limits to reduced model in URDF convention.
    for m, j in zip(ARM_MOTORS, ARM_JOINTS):
        lo_m, hi_m = limits[m]
        if m == "left_arm_shoulder_lift":
            lo_u, hi_u = 90.0 - hi_m, 90.0 - lo_m
        elif m == "left_arm_elbow_flex":
            lo_u, hi_u = lo_m + 90.0, hi_m + 90.0
        else:
            lo_u, hi_u = lo_m, hi_m
        lo_u, hi_u = float(min(lo_u, hi_u)), float(max(lo_u, hi_u))
        jid = model.getJointId(j)
        idx = model.joints[jid].idx_q
        model.lowerPositionLimit[idx] = np.deg2rad(lo_u)
        model.upperPositionLimit[idx] = np.deg2rad(hi_u)

    return model, data, base_fid, ee_fid, t_base_camera


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup(robot: XLerobot | None, camera: D435 | None) -> None:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
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
        # Connect robot and restore calibration.
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

        # Configure arm motors.
        configure_arm_motors(bus)

        limits = compute_arm_limits(bus)

        # Start camera.
        camera = D435(D435Config(serial=CAMERA_SERIAL, width=640, height=480, fps=30, align_to_color=False))
        camera.start()

        # Read joints and build IK model + camera->base transform.
        q_seed = read_q(bus, ARM_MOTORS)
        try:
            q_head = read_q(bus, HEAD_MOTORS)
        except Exception:
            q_head = np.zeros(2, dtype=float)

        model, data, base_fid, ee_fid, t_base_camera = build_ik_model(limits, q_seed, q_head)

        # Capture center-depth target in base frame.
        while True:
            _, depth_image, depth_frame = camera.read()
            h, w = depth_image.shape
            u, v = w // 2, h // 2
            z = depth_frame.get_distance(u, v)
            if z <= 0:
                continue
            x, y, z = rs.rs2_deproject_pixel_to_point(camera.intrinsics, [u, v], z)  # type: ignore[attr-defined]
            p_target = (t_base_camera @ np.array([x, y, z, 1.0], dtype=float))[:3]
            break

        # Initial IK solve.
        q_seed = clip_arm(q_seed, limits)
        q_seed[0] = float(np.clip(pan_guess_deg(p_target), *limits["left_arm_shoulder_pan"]))
        q_goal, ok, iters, ik_err = solve_to_target(q_seed, p_target, model, data, base_fid, ee_fid, limits)
        q_goal = clip_arm(q_goal, limits)

        print(f"Frames: {CAMERA_FRAME} -> {BASE_FRAME} -> {EE_FRAME}")
        print(f"IK result: ok={ok}, iters={iters}, planned_err_m={ik_err:.4f}, target_base={p_target}")
        if ik_err > 0.05:
            print(f"[WARN] IK residual is high ({ik_err:.3f} m).")
        print("Running one-shot execution. Press Ctrl+C to stop.")

        # Execution loop.
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
                    print(f"Reached target in base frame (ee_err_m={err:.4f}).")
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
