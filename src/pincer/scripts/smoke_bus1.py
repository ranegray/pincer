from lerobot.robots.xlerobot import XLerobot, XLerobotConfig


def main():
    # connect to XLerobot only bus1 (so101 + head camera)
    robot = XLerobot(XLerobotConfig(port1="/dev/ttyACM0"))
    robot.bus1.connect()

    # read positions and print them
    print("Reading motor positions on bus1...")
    for name in robot.bus1.motors:
        pos = robot.bus1.read("Present_Position", name)
        print(f"Motor {name}: Position = {pos}")

    # cleanup
    print("\nDisconnecting from bus1...")
    robot.bus1.disconnect(robot.config.disable_torque_on_disconnect)
    print("Done.")


if __name__ == "__main__":
    main()
