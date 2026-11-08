"""Reconstruct a G1 scene from a synchronized capture in robot base coordinates."""

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from .recognition import Landmark
from .vision import Calibration

HEAD_RGB = ("head-left-rgb", "head-right-rgb")
WRIST_SIDES = ("left", "right")
MAX_CAMERA_SKEW_NS = 50_000_000
MAX_LANDMARK_DISAGREEMENT_M = 0.03


def pose_matrix(pose):
    """Convert a G1 xyz + xyzw transform to a rigid homogeneous matrix."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Missing or invalid transform")
    if not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=0.0025):
        raise ValueError("Transform quaternion is not unit length")
    x, y, z, w = pose[3:] / np.linalg.norm(pose[3:])
    matrix = np.eye(4)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = pose[:3]
    return matrix


def _calibration(frame):
    intrinsics = frame["intrinsics"]
    if (intrinsics["width"], intrinsics["height"]) != (frame["width"], frame["height"]):
        raise ValueError("Intrinsics and frame dimensions differ")
    k = np.asarray(intrinsics["K"], dtype=float).reshape(3, 3)
    distortion = np.asarray(intrinsics["D"], dtype=float)
    if not np.isfinite(distortion).all():
        raise ValueError("Invalid lens distortion coefficients")
    matrix = pose_matrix(frame["camera_to_base_pose"])
    Calibration(k[0, 0], k[1, 1], k[0, 2], k[1, 2], tuple(map(tuple, matrix))).matrix()
    return k, distortion, matrix


def _pixel_rays(frame, pixels):
    k, distortion, matrix = _calibration(frame)
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if not np.isfinite(pixels).all():
        raise ValueError("Non-finite image coordinate")
    lens = frame["intrinsics"]["distortion_model"].lower()
    if lens == "kannala_brandt" and distortion.shape == (4,):
        xy = cv2.fisheye.undistortPoints(pixels[:, None, :], k, distortion)[:, 0, :]
    elif lens == "inverse brown conrady" and distortion.shape == (5,):
        # Follow the inverse Brown-Conrady deprojection in librealsense.
        k1, k2, p1, p2, k3 = distortion
        reference = (pixels - k[:2, 2]) / np.diag(k)[:2]
        x, y = reference.T.copy()
        for _ in range(10):
            r2 = x * x + y * y
            radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            sx, sy = x * radial, y * radial
            dx = 2 * p1 * sx * sy + p2 * (r2 + 2 * sx * sx)
            dy = 2 * p2 * sx * sy + p1 * (r2 + 2 * sy * sy)
            x = (reference[:, 0] - dx) / radial
            y = (reference[:, 1] - dy) / radial
        xy = np.column_stack((x, y))
    else:
        raise ValueError(f"Unsupported G1 camera distortion model: {lens}")
    rays = np.column_stack((xy, np.ones(len(xy)))) @ matrix[:3, :3].T
    if not np.isfinite(rays).all():
        raise ValueError("Camera rays are not finite")
    return matrix[:3, 3], rays


def triangulate_head_point(frames, left_pixel, right_pixel):
    """Triangulate a manually matched feature from the two head images."""
    left = frames["head-left-rgb"]
    right = frames["head-right-rgb"]
    for frame, (u, v) in ((left, left_pixel), (right, right_pixel)):
        if not 0 <= u < frame["width"] or not 0 <= v < frame["height"]:
            raise ValueError("Stereo pixel lies outside its image")
    a, (u,) = _pixel_rays(left, [left_pixel])
    b, (v,) = _pixel_rays(right, [right_pixel])
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    if np.linalg.norm(np.cross(u, v)) < 0.01:
        raise ValueError("Stereo rays are too close to parallel")
    s, t = np.linalg.lstsq(np.column_stack((u, -v)), b - a, rcond=None)[0]
    p, q = a + s * u, b + t * v
    separation = float(np.linalg.norm(p - q))
    if min(s, t) <= 0 or separation > 0.03:
        raise ValueError(
            "Stereo points do not resolve to a common feature in front of both cameras"
        )
    return {"point_m": ((p + q) / 2).tolist(), "ray_separation_m": separation}


def _wrist_depth(directory, frames, side):
    rgb = frames[f"{side}-wrist-rgb"]
    depth = frames[f"{side}-wrist-depth"]
    if any(rgb[key] != depth[key] for key in ("width", "height", "optical_frame", "intrinsics")):
        raise ValueError(f"{side} wrist RGB and depth calibration disagree")
    width, height = depth["width"], depth["height"]
    if not (
        isinstance(width, int)
        and isinstance(height, int)
        and width > 0
        and height > 0
        and width * height <= 4_000_000
    ):
        raise ValueError(f"{side} wrist has invalid depth dimensions")
    scale = depth["depth_scale"]
    if depth["format"].lower() != "16uc1" or not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{side} wrist requires scaled raw 16UC1 depth")
    with gzip.open(directory / f"{side}-wrist-depth.bin.gz", "rb") as stream:
        payload = stream.read(width * height * 2 + 1)
    if len(payload) != width * height * 2:
        raise ValueError(f"{side} wrist depth size does not match its metadata")
    return depth, np.frombuffer(payload, dtype="<u2").reshape(height, width), scale


def _wrist_cloud(directory, frames, side):
    depth, image, scale = _wrist_depth(directory, frames, side)
    rows, columns = np.nonzero(image[::3, ::3])
    rows, columns = rows * 3, columns * 3
    meters = image[rows, columns] / scale
    valid = meters <= 4.0
    if not valid.any():
        raise ValueError(f"{side} wrist contains no usable depth points")
    pixels = np.column_stack((columns[valid], rows[valid]))
    origin, rays = _pixel_rays(depth, pixels)
    return (origin + rays * meters[valid, None]).astype("<f4")


def _synchronized_frames(directory, required):
    report = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    frames = report["frames"]
    if any(label not in frames for label in required):
        raise ValueError("The G1 capture is missing required frames")
    stamps = [frames[label]["timestamp_ns"] for label in required]
    if (
        any(type(stamp) is not int or stamp <= 0 for stamp in stamps)
        or max(stamps) - min(stamps) > MAX_CAMERA_SKEW_NS
    ):
        raise ValueError("The G1 frame set is outside the 50 ms capture window")
    if any(
        frame.get("transform_timestamp_ns") != frame["timestamp_ns"]
        for frame in (frames[label] for label in required)
    ):
        raise ValueError("A camera transform does not match its frame timestamp")
    return frames, (max(stamps) - min(stamps)) / 1e6


def _wrist_landmark(directory, frames, side, pixel):
    frame, image, scale = _wrist_depth(directory, frames, side)
    height, width = image.shape
    if not (0 <= pixel[0] <= width - 1 and 0 <= pixel[1] <= height - 1):
        raise ValueError(f"{side} wrist landmark lies outside its image")
    u, v = (round(value) for value in pixel)
    meters = float(image[v, u]) / scale
    if not 0 < meters <= 4.0:
        raise ValueError(f"{side} wrist landmark has no usable depth")
    origin, (ray,) = _pixel_rays(frame, [(u, v)])
    return origin + ray * meters


def locate_landmark(directory: Path, landmark: Landmark) -> dict:
    """Check one matched target feature against head stereo and wrist depth."""
    views = landmark.views
    wrists = [side for side in WRIST_SIDES if f"{side}-wrist-rgb" in views]
    if any(name not in views for name in HEAD_RGB) or not wrists:
        raise ValueError("The same feature is required in both head images and a wrist image")
    required = HEAD_RGB + tuple(
        f"{side}-wrist-{kind}" for side in wrists for kind in ("rgb", "depth")
    )
    frames, skew_ms = _synchronized_frames(directory, required)
    for camera in (*HEAD_RGB, *(f"{side}-wrist-rgb" for side in wrists)):
        if views[camera].timestamp_ns != frames[camera]["timestamp_ns"]:
            raise ValueError(f"Landmark timestamp does not match captured frame: {camera}")

    stereo = triangulate_head_point(frames, *(views[name].pixel_px for name in HEAD_RGB))
    wrist_checks = {}
    for side in wrists:
        point = _wrist_landmark(directory, frames, side, views[f"{side}-wrist-rgb"].pixel_px)
        difference = float(np.linalg.norm(point - stereo["point_m"]))
        if not np.isfinite(difference) or difference > MAX_LANDMARK_DISAGREEMENT_M:
            raise ValueError(f"{side} wrist depth disagrees with head stereo")
        wrist_checks[side] = {"point_m": point.tolist(), "difference_m": difference}
    return {
        "landmark": landmark.name,
        "frame_id": "base_link",
        "head_stereo": stereo,
        "wrist_depth": wrist_checks,
        "max_camera_skew_ms": skew_ms,
    }


def fuse_capture(directory, stereo_points):
    required = HEAD_RGB + tuple(
        f"{side}-wrist-{kind}" for side in WRIST_SIDES for kind in ("rgb", "depth")
    )
    frames, skew_ms = _synchronized_frames(directory, required)

    left = _wrist_cloud(directory, frames, "left")
    right = _wrist_cloud(directory, frames, "right")
    landmarks = {
        name: triangulate_head_point(frames, left_pixel, right_pixel)
        for name, left_pixel, right_pixel in stereo_points
    }
    cloud = np.concatenate((left, right))
    path = directory / "wrist-clouds-base-link.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\ncomment base_link units meters\n"
        f"element vertex {len(cloud)}\nproperty float x\nproperty float y\n"
        "property float z\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(cloud.tobytes())
    result = {
        "coordinate_frame": "base_link",
        "max_camera_skew_ms": skew_ms,
        "points_by_wrist": {"left": len(left), "right": len(right)},
        "head_landmarks": landmarks,
        "cloud": path.name,
        "evidence": "diagnostic_capture_only",
    }
    (directory / "fusion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument(
        "--head-point",
        nargs=5,
        action="append",
        default=[],
        metavar=("NAME", "LEFT_U", "LEFT_V", "RIGHT_U", "RIGHT_V"),
    )
    args = parser.parse_args()
    matches = [
        (name, (float(lu), float(lv)), (float(ru), float(rv)))
        for name, lu, lv, ru, rv in args.head_point
    ]
    print(json.dumps(fuse_capture(args.capture, matches), indent=2))


if __name__ == "__main__":
    main()
