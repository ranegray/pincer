"""Detect a colored object, reach to it, grasp, and lift.

Port of ``pincer.scripts.detect_and_grasp`` as a runtime Behavior.
"""

from __future__ import annotations

import time

import numpy as np
import pyrealsense2 as rs

from pincer.detect import ColorDetector
from pincer.ik.constants import ARM_MOTORS, HEAD_MOTORS
from pincer.ik.solver import ee_in_base, pan_guess_deg, solve_to_target
from pincer.ik.transforms import build_t_base_camera, camera_to_base
from pincer.robots.xlerobot_motor_utils import clip_arm, read_q
from pincer.runtime.behaviors.base import Behavior

GRIPPER_MOTOR = "left_arm_gripper"
GRIPPER_OPEN = 100.0
GRIPPER_CLOSED = 10.0
TABLE_Z_BASE = -0.08
GRASP_OFFSET_SIGN = -1.0


class DetectAndGrasp(Behavior):
    name = "detect_and_grasp"

    def run(self, *, color: str = "blue", **kwargs) -> None:
        rt = self.runtime
        bus = rt.bus
        camera = rt.camera
        limits = rt.limits
        model = rt.model
        data = rt.data
        base_fid = rt.base_fid
        ee_fid = rt.ee_fid
        state = rt.state
        bus_lock = rt._bus_lock

        # Build camera->base transform from current joint config
        with bus_lock:
            q_arm = read_q(bus, ARM_MOTORS)
            try:
                q_head = read_q(bus, HEAD_MOTORS)
            except Exception:
                q_head = np.zeros(2, dtype=float)

        T_base_camera = build_t_base_camera(q_arm, q_head)

        # Detect color target
        detector = ColorDetector.from_color(color)
        print(f"[detect_and_grasp] Searching for '{color}' target...")

        p_target = None
        while not self.should_stop:
            color_image, _, depth_frame = camera.read()
            state.update_frame(color_image)

            detections = detector.detect(color_image)
            if detections:
                det = detections[0]
                state.update_detection(
                    bboxes=[det.bbox],
                    centroids=[det.centroid],
                    label=color,
                )
            else:
                state.update_detection([], [], "")
                continue

            u, v = det.centroid
            z = depth_frame.get_distance(u, v)
            if z <= 0:
                continue

            bbox_w_px = det.bbox[2]
            fx = camera.intrinsics.fx
            obj_width_m = bbox_w_px * z / fx
            print(
                f"[detect_and_grasp] Detected at ({u}, {v}), "
                f"depth={z:.3f} m, width={obj_width_m:.3f} m"
            )

            x, y, z = rs.rs2_deproject_pixel_to_point(camera.intrinsics, [u, v], z)
            x += GRASP_OFFSET_SIGN * obj_width_m / 1.5
            p_cam = np.array([x, y, z], dtype=float)
            p_target = camera_to_base(p_cam, T_base_camera)
            p_target[2] = max(p_target[2], TABLE_Z_BASE)
            print(f"[detect_and_grasp] Target in base frame: {p_target}")
            break

        if p_target is None or self.should_stop:
            print("[detect_and_grasp] Cancelled before detection.")
            return

        # Open gripper
        with bus_lock:
            bus.sync_write("Goal_Position", {GRIPPER_MOTOR: GRIPPER_OPEN})
        print("[detect_and_grasp] Gripper opened.")
        time.sleep(0.3)

        # Reach to target
        print("[detect_and_grasp] Reaching...")
        state.update_ik(target=p_target, error=0.0, iterations=0, converged=False)
        reached = self._reach(p_target, limits, model, data, base_fid, ee_fid, bus, bus_lock, state)

        if not reached:
            print("[detect_and_grasp] Did not reach target (cancelled or failed).")
            return

        # Close gripper
        print("[detect_and_grasp] Closing gripper...")
        with bus_lock:
            bus.sync_write("Goal_Position", {GRIPPER_MOTOR: GRIPPER_CLOSED})
        time.sleep(3.0)

        if self.should_stop:
            return

        # Lift
        p_lift = p_target.copy()
        p_lift[2] += 0.08
        print(f"[detect_and_grasp] Lifting to {p_lift}...")
        state.update_ik(target=p_lift, error=0.0, iterations=0, converged=False)
        self._reach(p_lift, limits, model, data, base_fid, ee_fid, bus, bus_lock, state)

        print("[detect_and_grasp] Holding for 5 seconds...")
        deadline = time.monotonic() + 5.0
        while not self.should_stop and time.monotonic() < deadline:
            time.sleep(0.1)

        print("[detect_and_grasp] Done.")

    def _reach(self, p_target, limits, model, data, base_fid, ee_fid, bus, bus_lock, state):
        """Re-solve IK each tick and step toward a Cartesian target."""
        period = 0.1
        hold = 0
        i = 0

        while not self.should_stop:
            t0 = time.monotonic()

            with bus_lock:
                q_curr = clip_arm(read_q(bus, ARM_MOTORS), limits)
            p_ee = ee_in_base(q_curr, model, data, base_fid, ee_fid)
            state.update_ee(p_ee)

            err_vec = p_target - p_ee
            err = float(np.linalg.norm(err_vec))

            if err <= 0.01:
                hold += 1
                if hold >= 3:
                    state.update_ik(target=p_target, error=err, iterations=i, converged=True)
                    return True
            else:
                hold = 0

            state.update_ik(target=p_target, error=err, iterations=i, converged=False)

            q_seed = q_curr.copy()
            if abs(float(err_vec[0])) >= 0.01:
                q_seed[0] = float(np.clip(pan_guess_deg(p_target), *limits["left_arm_shoulder_pan"]))
            q_goal, _, _, _ = solve_to_target(q_seed, p_target, model, data, base_fid, ee_fid, limits)

            step = 10.0 if err > 0.12 else 6.0
            q_cmd = clip_arm(np.clip(q_goal, q_curr - step, q_curr + step), limits)
            state.update_command(arm=q_cmd, ee_target=p_target)

            with bus_lock:
                bus.sync_write("Goal_Position", {n: float(q_cmd[j]) for j, n in enumerate(ARM_MOTORS)})

            if i % 20 == 0:
                print(f"[detect_and_grasp]   err={err:.4f} m, step_deg={step:.1f}")
            i += 1

            sleep_s = period - (time.monotonic() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

        return False
