import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(description="Create a 4x4 transform matrix from translation and RPY.")
    parser.add_argument("--tx", type=float, required=True, help="Translation x (meters)")
    parser.add_argument("--ty", type=float, required=True, help="Translation y (meters)")
    parser.add_argument("--tz", type=float, required=True, help="Translation z (meters)")
    parser.add_argument("--roll", type=float, default=0.0, help="Roll angle")
    parser.add_argument("--pitch", type=float, default=0.0, help="Pitch angle")
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw angle")
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="Interpret roll/pitch/yaw as degrees (default is radians).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration/T_base_marker.npy"),
        help="Output .npy path for 4x4 transform.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rot = Rotation.from_euler("xyz", [args.roll, args.pitch, args.yaw], degrees=args.degrees)
    t = np.eye(4, dtype=float)
    t[:3, :3] = rot.as_matrix()
    t[:3, 3] = np.array([args.tx, args.ty, args.tz], dtype=float)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, t)

    print("Saved transform:")
    print(np.array2string(t, precision=6, suppress_small=True))
    print(f"npy: {args.output}")


if __name__ == "__main__":
    main()
