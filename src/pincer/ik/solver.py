"""Pink-based IK solver for the XLerobot arm."""

import numpy as np
import pink
import pinocchio as pin
from pink.tasks import FrameTask, PostureTask

from pincer.ik.constants import ARM_MOTORS, BASE_FRAME, EE_FRAME
from pincer.ik.model import motor_to_pin_q, pin_to_motor_q


def ee_in_base(
    q_motor: np.ndarray,
    model: pin.Model,
    data: pin.Data,
    base_fid: int,
    ee_fid: int,
) -> np.ndarray:
    """Return the end-effector position expressed in the base frame (meters)."""
    q_pin = motor_to_pin_q(q_motor, model)
    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)
    oMb = data.oMf[base_fid]
    oMe = data.oMf[ee_fid]
    return oMb.rotation.T @ (oMe.translation - oMb.translation)


def pan_guess_deg(target: np.ndarray) -> float:
    """Rough shoulder-pan seed angle (degrees) pointing toward a base-frame target."""
    return float(np.rad2deg(np.arctan2(float(target[0]), max(1e-6, -float(target[1])))))


def solve_to_target(
    q_seed: np.ndarray,
    target: np.ndarray,
    model: pin.Model,
    data: pin.Data,
    base_fid: int,
    ee_fid: int,
    limits: dict[str, tuple[float, float]],
    *,
    max_iters: int = 300,
    pos_tol: float = 0.005,
    dt: float = 0.01,
    flat_patience: int = 25,
) -> tuple[np.ndarray, bool, int, float]:
    """Solve IK for the arm to reach *target* (base frame, meters).

    Parameters
    ----------
    q_seed:
        Initial arm joint angles in motor convention (degrees).
    target:
        Desired end-effector position in base frame (meters), shape (3,).
    model, data:
        Reduced arm-only Pinocchio model and data objects.
    base_fid, ee_fid:
        Frame IDs for the base and end-effector frames.
    limits:
        Per-motor position limits (degrees) used for clipping.
    max_iters:
        Maximum IK iterations.
    pos_tol:
        Position error threshold for convergence (meters).
    dt:
        Integration step size.
    flat_patience:
        Number of consecutive flat steps before early exit.

    Returns
    -------
    (q_solution, converged, iters, final_error)
        q_solution in motor convention (degrees).
    """
    from pincer.robots.xlerobot_motor_utils import clip_arm

    q = q_seed.copy()
    cfg = pink.Configuration(model, data, motor_to_pin_q(q, model))
    ee_task = FrameTask(EE_FRAME, position_cost=10.0, orientation_cost=0.0)
    posture_task = PostureTask(cost=1e-2)
    posture_task.set_target(motor_to_pin_q(q_seed, model))

    prev_err = float("inf")
    flat_steps = 0

    for step in range(max_iters):
        pin.forwardKinematics(model, data, cfg.q)
        pin.updateFramePlacements(model, data)
        oMb = data.oMf[base_fid]
        oMe = data.oMf[ee_fid]
        p_ee = oMb.rotation.T @ (oMe.translation - oMb.translation)
        err = float(np.linalg.norm(target - p_ee))

        if err <= pos_tol:
            return q, True, step, err

        target_world = oMb.rotation @ target + oMb.translation
        ee_task.set_target(pin.SE3(oMe.rotation, target_world))
        dq = pink.solve_ik(cfg, [ee_task, posture_task], dt, solver="quadprog")
        cfg.integrate_inplace(dq, dt)

        q_next = clip_arm(pin_to_motor_q(cfg.q, model), limits)
        cfg = pink.Configuration(model, data, motor_to_pin_q(q_next, model))

        if float(np.max(np.abs(q_next - q))) <= 0.02 and (prev_err - err) <= 2e-4:
            flat_steps += 1
        else:
            flat_steps = 0
        q = q_next
        prev_err = err

        if flat_steps >= flat_patience:
            return q, False, step + 1, err

    final_err = float(np.linalg.norm(target - ee_in_base(q, model, data, base_fid, ee_fid)))
    return q, False, max_iters, final_err
