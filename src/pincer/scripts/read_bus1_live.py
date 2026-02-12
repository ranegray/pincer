import time

from lerobot.robots.xlerobot import XLerobot, XLerobotConfig


def main():
    robot = XLerobot(XLerobotConfig(port1="/dev/ttyACM0"))
    robot.bus1.connect()
    motors = list(robot.bus1.motors.keys())

    try:
        robot.bus1.disable_torque()
        print("Torque disabled. Move joints by hand. Ctrl+C to stop.\n")

        while True:
            pos = robot.bus1.sync_read("Present_Position", motors)
            line = " | ".join(f"{m}: {pos[m]}" for m in motors)
            print(line, end="\r", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            robot.bus1.enable_torque()
        except Exception:
            pass
        robot.bus1.disconnect(robot.config.disable_torque_on_disconnect)


if __name__ == "__main__":
    main()
