import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pincer.cameras.d435 import D435, D435Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate fixed T_base_camera using one ArUco marker and known T_base_marker."
    )
    parser.add_argument("--serial", required=True, help="RealSense serial number.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--marker-id", type=int, default=0, help="Target ArUco marker ID.")
    parser.add_argument("--marker-size-m", type=float, default=0.08, help="Marker side length in meters.")
    parser.add_argument(
        "--dictionary",
        type=str,
        default="DICT_6X6_250",
        help="ArUco dictionary name (e.g., DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, DICT_7X7_1000).",
    )
    parser.add_argument(
        "--t-base-marker",
        type=Path,
        required=True,
        help="Path to 4x4 numpy file (.npy) containing T_base_marker.",
    )
    parser.add_argument("--num-samples", type=int, default=20, help="Accepted observations to collect.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration/T_base_camera.npy"),
        help="Output path for T_base_camera (.npy).",
    )
    return parser.parse_args()


def aruco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module not found. Install opencv-contrib-python.")
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary '{name}'.")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def invert_transform(t_ab: np.ndarray) -> np.ndarray:
    r_ab = t_ab[:3, :3]
    p_ab = t_ab[:3, 3]
    t_ba = np.eye(4, dtype=float)
    t_ba[:3, :3] = r_ab.T
    t_ba[:3, 3] = -r_ab.T @ p_ab
    return t_ba


def compose_transform(r_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=float)
    t[:3, :3] = r_mat
    t[:3, 3] = t_vec.reshape(3)
    return t


def average_rotations(rot_mats: list[np.ndarray]) -> np.ndarray:
    quats = np.array([Rotation.from_matrix(r).as_quat() for r in rot_mats], dtype=float)
    ref = quats[0]
    for i in range(1, len(quats)):
        if np.dot(ref, quats[i]) < 0:
            quats[i] = -quats[i]
    quat_mean = np.mean(quats, axis=0)
    quat_mean /= np.linalg.norm(quat_mean)
    return Rotation.from_quat(quat_mean).as_matrix()


def main():
    args = parse_args()
    t_base_marker = np.load(args.t_base_marker)
    if t_base_marker.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix in {args.t_base_marker}, got {t_base_marker.shape}.")

    dictionary = aruco_dictionary(args.dictionary)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    camera = D435(D435Config(serial=args.serial, width=args.width, height=args.height, fps=args.fps))
    camera.start()

    intr = camera.intrinsics
    camera_matrix = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

    t_base_camera_samples: list[np.ndarray] = []
    reprojection_errors: list[float] = []

    print("ArUco extrinsics calibration started.")
    print(f"Target marker id: {args.marker_id}, size: {args.marker_size_m} m")
    print("Controls: press 'c' to capture, 'q' to quit.\n")

    marker_object_points = np.array(
        [
            [-args.marker_size_m / 2, args.marker_size_m / 2, 0.0],
            [args.marker_size_m / 2, args.marker_size_m / 2, 0.0],
            [args.marker_size_m / 2, -args.marker_size_m / 2, 0.0],
            [-args.marker_size_m / 2, -args.marker_size_m / 2, 0.0],
        ],
        dtype=np.float32,
    )

    try:
        while len(t_base_camera_samples) < args.num_samples:
            color_image, _, _ = camera.read()
            corners, ids, _ = detector.detectMarkers(color_image)

            vis = color_image.copy()
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

            cv2.putText(
                vis,
                f"Samples: {len(t_base_camera_samples)}/{args.num_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(vis, "c=capture, q=quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("aruco_extrinsics_calibration", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key != ord("c"):
                continue

            if ids is None:
                print("No markers detected.")
                continue

            ids_flat = ids.flatten().tolist()
            if args.marker_id not in ids_flat:
                print(f"Marker id {args.marker_id} not detected.")
                continue

            idx = ids_flat.index(args.marker_id)
            marker_corners = corners[idx].reshape(-1, 2).astype(np.float32)

            ok, rvec, tvec = cv2.solvePnP(
                marker_object_points,
                marker_corners,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                print("solvePnP failed.")
                continue

            projected, _ = cv2.projectPoints(marker_object_points, rvec, tvec, camera_matrix, dist_coeffs)
            reproj_error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - marker_corners, axis=1)))

            r_camera_marker, _ = cv2.Rodrigues(rvec)
            t_camera_marker = compose_transform(r_camera_marker, tvec)
            t_base_camera = t_base_marker @ invert_transform(t_camera_marker)

            t_base_camera_samples.append(t_base_camera)
            reprojection_errors.append(reproj_error)
            print(
                f"Accepted sample {len(t_base_camera_samples)}/{args.num_samples} "
                f"(reprojection error: {reproj_error:.3f} px)"
            )

    finally:
        camera.stop()
        cv2.destroyAllWindows()

    if len(t_base_camera_samples) < 3:
        raise RuntimeError("Not enough samples collected. Need at least 3.")

    translations = np.array([t[:3, 3] for t in t_base_camera_samples], dtype=float)
    rotations = [t[:3, :3] for t in t_base_camera_samples]
    t_mean = np.mean(translations, axis=0)
    r_mean = average_rotations(rotations)

    t_base_camera_est = np.eye(4, dtype=float)
    t_base_camera_est[:3, :3] = r_mean
    t_base_camera_est[:3, 3] = t_mean

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, t_base_camera_est)

    trans_std = np.std(translations, axis=0)
    print("\nCalibration complete.")
    print(f"Samples used: {len(t_base_camera_samples)}")
    print(
        "Reprojection error (px): "
        f"mean={np.mean(reprojection_errors):.3f}, std={np.std(reprojection_errors):.3f}, "
        f"max={np.max(reprojection_errors):.3f}"
    )
    print(f"Translation std (m): x={trans_std[0]:.4f}, y={trans_std[1]:.4f}, z={trans_std[2]:.4f}")
    print(f"Saved matrix: {args.output}")
    print("T_base_camera:")
    print(np.array2string(t_base_camera_est, precision=6, suppress_small=True))


if __name__ == "__main__":
    main()
