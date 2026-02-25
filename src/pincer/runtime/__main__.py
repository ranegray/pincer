"""Entry point: python -m pincer.runtime"""

import argparse
import logging

from pincer.runtime.runtime import PincerRuntime, RuntimeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Pincer robot runtime")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for motor bus")
    parser.add_argument("--camera-serial", default="310222077874", help="RealSense camera serial")
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard bind address")
    parser.add_argument("--dashboard-port", type=int, default=8080, help="Dashboard port")
    parser.add_argument("--mock", action="store_true", help="Run with synthetic data (no hardware)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = RuntimeConfig(
        port=args.port,
        camera_serial=args.camera_serial,
        dashboard_host=args.host,
        dashboard_port=args.dashboard_port,
        mock=args.mock,
    )

    runtime = PincerRuntime(config)
    runtime.run()


if __name__ == "__main__":
    main()
