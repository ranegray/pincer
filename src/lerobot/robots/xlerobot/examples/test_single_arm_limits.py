#!/usr/bin/env python
"""
Simple single-arm test to verify torque, acceleration, and PID limits.

Uses SO100Follower which handles calibration automatically:
  - First run: interactive calibration (half-turn homing + range recording), saved to file
  - Subsequent runs: loads calibration from file (or type 'c' to recalibrate)

Usage:
    python -m lerobot.robots.xlerobot.examples.test_single_arm_limits

What it does:
    1. Connects and calibrates via SO100Follower (same as the other examples)
    2. Overrides motor limits with your chosen torque, acceleration, and PID values
    3. Enables torque so the arm holds position — push joints to feel resistance
    4. Nudges each joint ±3.5° so you can see how smooth/jerky the motion is
    5. Reads load feedback from each motor
    6. Disconnects cleanly

Safety:
    - Starts with LOW torque (200/1000) and LOW acceleration (10)
    - Nudge is only ±40 raw steps (~3.5 degrees)
    - Press Ctrl+C at any time to disable torque and exit
"""

import signal
import sys
import time
import traceback

from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so_follower import SO100Follower, SOFollowerRobotConfig


# ── CONFIG: Change these to test different limits ──────────────────────────
PORT = "/dev/ttyACM0"

# Torque: 0-1000 (1000 = max). Start low!
TORQUE_LIMIT = 200

# Acceleration: 0-254. Lower = slower ramp-up. 0 = use Maximum_Acceleration default.
ACCELERATION = 10

# PID: Lower P = softer, slower response. Default P=32, D=32, I=0
P_COEFF = 8
I_COEFF = 0
D_COEFF = 32

# How far to nudge each joint (raw steps, 4096 = full revolution, 40 ≈ 3.5°)
NUDGE_STEPS = 40

# How long to hold each nudge position (seconds)
HOLD_TIME = 1.5
# ───────────────────────────────────────────────────────────────────────────


def cleanup(robot: SO100Follower):
    """Disable torque and disconnect."""
    print("\nDisabling torque and disconnecting...")
    try:
        robot.disconnect()
    except Exception:
        pass
    print("Done. Arm is free.")


def apply_limits(robot: SO100Follower):
    """Override motor parameters with our test limits (torque off during writes)."""
    bus = robot.bus
    motors = list(bus.motors.keys())

    print("Applying motor limits:")
    print(f"  Torque_Limit  = {TORQUE_LIMIT}/1000")
    print(f"  Acceleration  = {ACCELERATION}")
    print(f"  P={P_COEFF}  I={I_COEFF}  D={D_COEFF}")
    print()

    bus.disable_torque()
    for name in motors:
        bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
        bus.write("Torque_Limit", name, TORQUE_LIMIT)
        bus.write("Acceleration", name, ACCELERATION)
        bus.write("P_Coefficient", name, P_COEFF)
        bus.write("I_Coefficient", name, I_COEFF)
        bus.write("D_Coefficient", name, D_COEFF)
    bus.enable_torque()


def nudge_test(robot: SO100Follower):
    """Move each joint ± NUDGE_STEPS to observe motion smoothness."""
    bus = robot.bus
    motors = list(bus.motors.keys())

    # Read current positions (raw, no normalization)
    positions = bus.sync_read("Present_Position", motors)
    print("Current positions (raw):")
    for name, pos in positions.items():
        print(f"  {name:16s} = {pos}")
    print()

    print(f"Nudge test: moving each joint ±{NUDGE_STEPS} steps (~{NUDGE_STEPS * 360 / 4096:.1f}deg)")
    print("Watch how smoothly each joint moves.\n")

    for name in motors:
        home = positions[name]

        # Forward
        target_fwd = home + NUDGE_STEPS
        print(f"  {name}: {home} -> {target_fwd} (forward)")
        bus.write("Goal_Position", name, target_fwd)
        time.sleep(HOLD_TIME)

        # Backward
        target_bwd = home - NUDGE_STEPS
        print(f"  {name}: {target_fwd} -> {target_bwd} (backward)")
        bus.write("Goal_Position", name, target_bwd)
        time.sleep(HOLD_TIME)

        # Return home
        print(f"  {name}: {target_bwd} -> {home} (home)")
        bus.write("Goal_Position", name, home)
        time.sleep(HOLD_TIME)
        print()

    # Read load feedback
    print("Present_Load after nudge test:")
    loads = bus.sync_read("Present_Load", motors)
    for name, load in loads.items():
        print(f"  {name:16s} load = {load}")
    print()


def main():
    # Build robot using SO100Follower (handles calibration automatically)
    config = SOFollowerRobotConfig(port=PORT)
    robot = SO100Follower(config)

    # Ctrl+C handler
    def signal_handler(sig, frame):
        cleanup(robot)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # ── Step 1: Connect + calibrate ──
        # First run: will run interactive calibration (half-turn homing + range-of-motion)
        # Subsequent runs: loads from file, or type 'c' to recalibrate
        print(f"Connecting on {PORT}...")
        robot.connect()
        print("Connected and calibrated!\n")

        # ── Step 2: Override with test limits ──
        apply_limits(robot)
        print("Torque ENABLED with test limits.\n")

        # ── Step 3: Manual push test ──
        input(
            ">> Push each joint gently to feel the resistance.\n"
            "   With low torque you should be able to overpower the motors easily.\n"
            "   Press ENTER when ready to start the nudge test..."
        )

        # ── Step 4: Nudge test ──
        nudge_test(robot)

        # ── Step 5: Hold for more manual testing ──
        input(
            ">> Arm is holding position. Push joints again to feel resistance.\n"
            "   Press ENTER to disable torque and exit..."
        )

        cleanup(robot)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        cleanup(robot)


if __name__ == "__main__":
    main()
