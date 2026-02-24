"""Trapezoidal velocity profile trajectory in joint space."""

import numpy as np


class TrapezoidalTrajectory:
    """Time-synchronized trapezoidal velocity trajectory for multiple joints.

    All joints start and finish together. The slowest joint (largest displacement
    relative to velocity/acceleration limits) dictates the total duration, and
    the other joints scale down to match.

    Parameters
    ----------
    q_start:
        Starting joint angles (degrees), shape (n,).
    q_goal:
        Goal joint angles (degrees), shape (n,).
    max_vel:
        Maximum joint velocity (deg/s). Applied per-joint before time sync.
    max_accel:
        Maximum joint acceleration (deg/s²). Applied per-joint before time sync.
    """

    def __init__(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        max_vel: float = 60.0,
        max_accel: float = 120.0,
    ) -> None:
        self.q_start = q_start.astype(float).copy()
        self.q_goal = q_goal.astype(float).copy()
        self._n = len(q_start)

        displacements = np.abs(self.q_goal - self.q_start)

        # Compute per-joint minimum durations using trapezoidal profile
        # For each joint: if displacement is small enough, it's a triangular
        # profile (never reaches max_vel). Otherwise it's trapezoidal.
        t_accels = np.empty(self._n)
        t_totals = np.empty(self._n)

        for i in range(self._n):
            d = displacements[i]
            if d < 1e-6:
                t_accels[i] = 0.0
                t_totals[i] = 0.0
                continue

            # Time to accelerate to max_vel
            t_a = max_vel / max_accel
            # Distance covered during accel + decel (triangular)
            d_tri = max_accel * t_a * t_a  # = max_vel^2 / max_accel

            if d <= d_tri:
                # Triangular profile: never reaches max_vel
                t_a_actual = np.sqrt(d / max_accel)
                t_accels[i] = t_a_actual
                t_totals[i] = 2.0 * t_a_actual
            else:
                # Trapezoidal: accel, cruise, decel
                t_cruise = (d - d_tri) / max_vel
                t_accels[i] = t_a
                t_totals[i] = 2.0 * t_a + t_cruise

        # Synchronize: all joints take the same total time (slowest joint wins)
        self._duration = float(np.max(t_totals)) if np.any(t_totals > 0) else 0.0

        if self._duration < 1e-6:
            # No motion needed
            self._duration = 0.0
            self._t_accel = np.zeros(self._n)
            self._t_cruise_end = np.zeros(self._n)
            self._v_cruise = np.zeros(self._n)
            self._accel = np.zeros(self._n)
            return

        # Recompute per-joint profiles scaled to the synchronized duration.
        # Each joint must cover its displacement in exactly self._duration.
        # We keep max_accel fixed and solve for the per-joint cruise velocity
        # and accel time that fill the total duration.
        self._t_accel = np.empty(self._n)
        self._t_cruise_end = np.empty(self._n)
        self._v_cruise = np.empty(self._n)
        self._accel = np.empty(self._n)

        T = self._duration
        for i in range(self._n):
            d = displacements[i]
            if d < 1e-6:
                self._t_accel[i] = 0.0
                self._t_cruise_end[i] = T
                self._v_cruise[i] = 0.0
                self._accel[i] = 0.0
                continue

            # With total time T and symmetric accel/decel:
            # d = a * t_a^2 + v_c * (T - 2*t_a)
            # v_c = a * t_a
            # d = a * t_a^2 + a * t_a * (T - 2*t_a) = a * t_a * T - a * t_a^2
            # d = a * t_a * (T - t_a)
            #
            # Also constrained: t_a <= T/2 (can't accel longer than half the time)
            # and a <= max_accel
            #
            # Use max_accel, solve for t_a:
            # max_accel * t_a * (T - t_a) = d
            # -max_accel * t_a^2 + max_accel * T * t_a - d = 0
            # t_a = (T - sqrt(T^2 - 4*d/max_accel)) / 2

            discriminant = T * T - 4.0 * d / max_accel
            if discriminant < 0:
                # Not enough accel capacity — use triangular (t_a = T/2)
                t_a = T / 2.0
                a = d / (t_a * t_a)  # reduced accel to fit
            else:
                t_a = (T - np.sqrt(discriminant)) / 2.0
                t_a = min(t_a, T / 2.0)
                a = max_accel

            v_c = a * t_a
            self._t_accel[i] = t_a
            self._t_cruise_end[i] = T - t_a
            self._v_cruise[i] = v_c
            self._accel[i] = a

        self._signs = np.sign(self.q_goal - self.q_start)

    @property
    def duration(self) -> float:
        """Total trajectory duration in seconds."""
        return self._duration

    def sample(self, t: float) -> np.ndarray:
        """Return interpolated joint positions at time *t* (seconds)."""
        if self._duration < 1e-6 or t >= self._duration:
            return self.q_goal.copy()
        if t <= 0.0:
            return self.q_start.copy()

        q = self.q_start.copy()
        for i in range(self._n):
            ta = self._t_accel[i]
            tc_end = self._t_cruise_end[i]
            a = self._accel[i]
            vc = self._v_cruise[i]
            s = self._signs[i]

            if t <= ta:
                # Accelerating
                d = 0.5 * a * t * t
            elif t <= tc_end:
                # Cruising
                d = 0.5 * a * ta * ta + vc * (t - ta)
            else:
                # Decelerating
                t_decel = t - tc_end
                d = 0.5 * a * ta * ta + vc * (tc_end - ta) + vc * t_decel - 0.5 * a * t_decel * t_decel
            q[i] += s * d

        return q
