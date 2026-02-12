import cv2
import pyrealsense2 as rs

from pincer.cameras.d435 import D435, D435Config


def main():
    camera = D435(D435Config(serial="310222077874", width=640, height=480, fps=30))
    camera.start()

    try:
        while True:
            color_image, depth_image, depth_frame = camera.read()

            h, w = depth_image.shape
            u, v = w // 2, h // 2
            z = depth_frame.get_distance(u, v)  # depth in meters
            xyz = rs.rs2_deproject_pixel_to_point(camera.intrinsics, [u, v], z)

            vis_depth = cv2.convertScaleAbs(depth_image, alpha=0.03)
            vis_depth = cv2.applyColorMap(vis_depth, cv2.COLORMAP_JET)
            cv2.circle(color_image, (u, v), 5, (0, 255, 0), -1)
            cv2.putText(
                color_image,
                f"Depth: {z:.2f}m XYZ: [{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}]",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Color", color_image)
            cv2.imshow("Depth", vis_depth)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

