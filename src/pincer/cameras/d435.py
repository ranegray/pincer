from dataclasses import dataclass

import numpy as np
import pyrealsense2 as rs


@dataclass
class D435Config:
    serial: str
    width: int = 640
    height: int = 480
    fps: int = 30
    align_to_color: bool = True


class D435:
    def __init__(self, config: D435Config):
        self.config = config
        self.pipeline = rs.pipeline()
        self._profile = None
        self._align = rs.align(rs.stream.color) if config.align_to_color else None
        self._intrinsics = None

    @property
    def intrinsics(self):
        if self._intrinsics is None:
            raise RuntimeError("Camera is not started. Call start() first.")
        return self._intrinsics

    def start(self) -> None:
        cfg = rs.config()
        cfg.enable_device(self.config.serial)
        cfg.enable_stream(
            rs.stream.color, self.config.width, self.config.height, rs.format.bgr8, self.config.fps
        )
        cfg.enable_stream(
            rs.stream.depth, self.config.width, self.config.height, rs.format.z16, self.config.fps
        )
        self._profile = self.pipeline.start(cfg)
        if config.align_to_color:
            color_stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
            self._intrinsics = color_stream.get_intrinsics()
        else:
            depth_stream = self._profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self._intrinsics = depth_stream.get_intrinsics()

    def read(self) -> tuple[np.ndarray, np.ndarray, rs.depth_frame]:
        frames = self.pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture color/depth frame from D435.")

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image, depth_frame

    def stop(self) -> None:
        self.pipeline.stop()
