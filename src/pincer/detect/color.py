"""HSV-based color detection for BGR camera frames."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    """A single color blob detected in the image."""

    centroid: tuple[int, int]
    area: float
    bbox: tuple[int, int, int, int]
    contour: np.ndarray


@dataclass
class ColorDetectorConfig:
    """Configuration for HSV color thresholding."""

    hsv_low: tuple[int, int, int]
    hsv_high: tuple[int, int, int]
    min_area: int = 100
    blur_ksize: int = 5


COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "green": [((35, 80, 50), (85, 255, 255))],
    "blue": [((100, 120, 50), (130, 255, 255))],
}


def color_config(name: str, **kwargs) -> ColorDetectorConfig:
    """Build a ColorDetectorConfig from a color name like "red", "green", or "blue"."""
    key = name.lower()
    if key not in COLOR_RANGES:
        raise ValueError(f"Unknown color {name!r}. Choose from: {', '.join(COLOR_RANGES)}")
    ranges = COLOR_RANGES[key]
    return ColorDetectorConfig(hsv_low=ranges[0][0], hsv_high=ranges[0][1], **kwargs)


class ColorDetector:
    def __init__(self, config: ColorDetectorConfig):
        self.config = config
        self._extra_ranges: list[tuple[np.ndarray, np.ndarray]] = []

    @classmethod
    def from_color(cls, name: str, **kwargs) -> "ColorDetector":
        """Create a ColorDetector from a color name like "red", "green", or "blue"."""
        key = name.lower()
        if key not in COLOR_RANGES:
            raise ValueError(f"Unknown color {name!r}. Choose from: {', '.join(COLOR_RANGES)}")
        ranges = COLOR_RANGES[key]
        detector = cls(ColorDetectorConfig(hsv_low=ranges[0][0], hsv_high=ranges[0][1], **kwargs))
        for low, high in ranges[1:]:
            detector._extra_ranges.append((np.array(low), np.array(high)))
        return detector

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        blurred = cv2.GaussianBlur(bgr, (self.config.blur_ksize, self.config.blur_ksize), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.config.hsv_low), np.array(self.config.hsv_high))
        for low, high in self._extra_ranges:
            mask |= cv2.inRange(hsv, low, high)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            detections.append(Detection(centroid=(cx, cy), area=area, bbox=(x, y, w, h), contour=cnt))

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections
