import cv2
import numpy as np
import pyrealsense2 as rs

SERIAL = "310222077874"

# start and configure the realsense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(SERIAL)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# align depth to color
align = rs.align(rs.stream.color)

# get intrinsics for depth to point cloud conversion
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
intrinsics = depth_stream.get_intrinsics()

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        h, w = depth_image.shape
        u, v = w // 2, h // 2
        z = depth_frame.get_distance(u, v)  # depth in meters
        xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], z)

        vis_depth = cv2.convertScaleAbs(depth_image, alpha=0.03)
        vis_depth = cv2.applyColorMap(vis_depth, cv2.COLORMAP_JET)
        cv2.circle(color_image, (u, v), 5, (0, 255, 0), -1)
        cv2.putText(color_image, f"Depth: {z:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Color", color_image)
        cv2.imshow("Depth", vis_depth)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
