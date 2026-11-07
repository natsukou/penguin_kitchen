"""A fixed-color marker baseline, not semantic recognition of real utensils."""

from dataclasses import dataclass

import cv2
import numpy as np

from .types import Detection, SafetyError

MARKER_BGR = {
    "spoon": (0, 255, 255),
    "food_container": (255, 0, 0),
    "bowl": (0, 0, 255),
    "mat": (0, 255, 0),
}


@dataclass(frozen=True)
class Calibration:
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_base: tuple[tuple[float, ...], ...]

    def matrix(self) -> np.ndarray:
        matrix = np.asarray(self.camera_to_base, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise SafetyError("Invalid camera extrinsic matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise SafetyError("Extrinsic must be homogeneous")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise SafetyError("Extrinsic rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
            raise SafetyError("Extrinsic rotation must have determinant +1")
        if not all(np.isfinite(v) for v in (self.fx, self.fy, self.cx, self.cy)):
            raise SafetyError("Non-finite camera intrinsics")
        if self.fx <= 0 or self.fy <= 0:
            raise SafetyError("Camera focal length must be positive")
        return matrix

    def unproject(self, u: float, v: float, depth_m: float) -> tuple[float, float, float]:
        if not np.isfinite([u, v, depth_m]).all() or depth_m <= 0:
            raise SafetyError("Invalid pixel/depth")
        point = np.array(
            [
                (u - self.cx) * depth_m / self.fx,
                (v - self.cy) * depth_m / self.fy,
                depth_m,
                1.0,
            ]
        )
        return tuple(float(x) for x in (self.matrix() @ point)[:3])


def detect_markers(
    bgr: np.ndarray, depth_m: np.ndarray, calibration: Calibration
) -> dict[str, Detection]:
    """Depth must already be aligned to RGB and expressed in meters."""
    if bgr.ndim != 3 or bgr.shape[2] != 3 or depth_m.shape != bgr.shape[:2]:
        raise SafetyError("RGB/depth dimensions differ")
    if bgr.dtype != np.uint8:
        raise SafetyError("RGB input must be uint8 BGR")
    calibration.matrix()
    detections = {}
    for name, color in MARKER_BGR.items():
        mask = cv2.inRange(bgr, np.array(color, np.uint8), np.array(color, np.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 25]
        if not candidates:
            continue
        if len(candidates) != 1:
            raise SafetyError(f"Ambiguous marker: {name}")
        label = candidates[0]
        values = depth_m[labels == label]
        valid = values[np.isfinite(values) & (values > 0)]
        if len(valid) < 0.8 * len(values):
            raise SafetyError(f"Insufficient valid depth: {name}")
        u, v = centroids[label]
        position = calibration.unproject(float(u), float(v), float(np.median(valid)))
        detections[name] = Detection(name, position, 1.0)
    return detections
