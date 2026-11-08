"""Read-only G1 tabletop range and gripper diagnostics in native SDK units."""

import json
import math
from pathlib import Path

import numpy as np

from .g1_multiview import (
    MAX_CAMERA_SKEW_NS,
    _pixel_rays,
    _synchronized_frames,
    _wrist_depth,
    pose_matrix,
)
from .types import TaskError

# SDK 1.10.0 input envelope. Measured tool limits must be supplied separately.
SDK_MAX_WIDTH_M = 0.12
SDK_MAX_SPEED_MPS = 0.2
SDK_MAX_EFFORT_NM = 100.0


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_tool_profile(path: Path, side: str, flange_frame: str) -> dict:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if (
        profile["side"] != side
        or profile["flange_frame"] != flange_frame
        or not isinstance(profile["calibration_id"], str)
        or not profile["calibration_id"].strip()
        or not isinstance(profile["tool_model"], str)
        or not profile["tool_model"].strip()
    ):
        raise TaskError("Tool profile does not identify the selected calibrated G1 arm")
    offset = profile["tcp_offset_in_flange_m"]
    if not isinstance(offset, list) or len(offset) != 3 or not all(_finite(v) for v in offset):
        raise TaskError("Tool profile requires a finite TCP offset in flange coordinates")
    limits = profile["limits"]
    low, high = limits["width_range_m"]
    speed, effort = limits["max_speed_mps"], limits["max_effort_nm"]
    if not all(_finite(v) for v in (low, high, speed, effort)) or not (
        0 <= low < high <= SDK_MAX_WIDTH_M
        and 0 < speed <= SDK_MAX_SPEED_MPS
        and 0 < effort <= SDK_MAX_EFFORT_NM
    ):
        raise TaskError("Calibrated tool limits must lie within the G1 SDK envelope")
    return profile


def _grip_check(command, profile, feedback, stamp):
    if command is None:
        return {"status": "no_command_requested"}
    width, speed, torque = command
    if not all(_finite(v) for v in command) or not (
        0 <= width <= SDK_MAX_WIDTH_M
        and 0 < speed <= SDK_MAX_SPEED_MPS
        and 0 < torque <= SDK_MAX_EFFORT_NM
    ):
        return {"status": "outside_sdk_envelope"}
    if profile is None:
        return {"status": "tool_calibration_missing"}
    limits = profile["limits"]
    low, high = limits["width_range_m"]
    if not (
        low <= width <= high
        and speed <= limits["max_speed_mps"]
        and torque <= limits["max_effort_nm"]
    ):
        return {"status": "outside_calibrated_limits"}
    if (
        not feedback
        or type(feedback.get("timestamp_ns")) is not int
        or abs(feedback["timestamp_ns"] - stamp) > MAX_CAMERA_SKEW_NS
        or feedback.get("moving") is not False
        or not _finite(feedback.get("width_m"))
        or not low <= feedback["width_m"] <= high
        or not _finite(feedback.get("effort_nm"))
        or abs(feedback["effort_nm"]) > limits["max_effort_nm"]
    ):
        return {"status": "gripper_feedback_unverified"}
    return {
        "status": "within_provided_profile_for_offline_review",
        "width_m": width,
        "speed_mps": speed,
        "effort_nm": torque,
    }


def _surface(frame, image, scale, pixel):
    u, v = pixel
    h, w = image.shape
    if not (2 <= u < w - 2 and 2 <= v < h - 2):
        raise TaskError("Select a wrist pixel at least two pixels inside the image")
    patch = image[v - 2 : v + 3, u - 2 : u + 3] / scale
    center = patch[2, 2]
    values = patch[(patch > 0) & (patch <= 4)]
    if not 0 < center <= 4 or len(values) < 9:
        raise TaskError("Insufficient depth at this surface pixel")
    z = float(np.median(values))
    spread = float(np.percentile(values, 90) - np.percentile(values, 10))
    if abs(center - z) > 0.015 or spread > 0.03:
        raise TaskError("Depth patch straddles surfaces; choose a point inside the object")
    origin, (ray,) = _pixel_rays(frame, [pixel])
    return origin, origin + z * ray, z, spread, len(values)


def _table_coordinates(point, landmarks):
    origin, forward, lateral = landmarks
    x, y = forward - origin, lateral - origin
    if min(np.linalg.norm(x), np.linalg.norm(y)) < 0.05:
        raise TaskError("Separate the marked table points by at least 5 cm")
    x = x / np.linalg.norm(x)
    z = np.cross(x, y)
    if np.linalg.norm(z) < 0.04:
        raise TaskError("Table points must not be collinear")
    z = z / np.linalg.norm(z)
    if z[2] < 0.97:
        raise TaskError("Mark a near-level tabletop with axes ordered to put Z upwards")
    y = np.cross(z, x)
    axes = np.column_stack((x, y, z))
    return {
        "origin_base_m": origin.tolist(),
        "axes_in_base_link": {"x": x.tolist(), "y": y.tolist(), "z": z.tolist()},
        "surface_in_table_m": (axes.T @ (point - origin)).tolist(),
    }


def measure_tabletop(
    capture: Path,
    side: str,
    pixel: tuple[int, int],
    *,
    table_pixels=None,
    profile_path: Path | None = None,
    command: tuple[float, float, float] | None = None,
) -> dict:
    if side not in ("left", "right"):
        raise TaskError("Select left or right wrist")
    rgb, depth = (f"{side}-wrist-{kind}" for kind in ("rgb", "depth"))
    frames, skew_ms = _synchronized_frames(capture, (rgb, depth))
    frame, image, scale = _wrist_depth(capture, frames, side)
    origin, point, z, spread, samples = _surface(frame, image, scale, pixel)
    table = None
    if table_pixels is not None:
        table = _table_coordinates(
            point, [_surface(frame, image, scale, anchor)[1] for anchor in table_pixels]
        )
        table["source"] = "three_manually_marked_surface_points"
        table["wrist_pixels_px"] = [list(anchor) for anchor in table_pixels]

    meta = json.loads((capture / "capture.json").read_text(encoding="utf-8"))
    stamp = frame["timestamp_ns"]
    flange_frame = f"{side}_arm_end_effector_mount_link"
    flange = meta.get("flanges", {}).get(side)
    feedback = meta.get("grippers", {}).get(side)
    profile = load_tool_profile(profile_path, side, flange_frame) if profile_path else None
    flange_range, tcp_range = None, None
    if (
        flange
        and flange.get("timestamp_ns") == stamp
        and flange.get("frame_id") == flange_frame
        and flange.get("pose_in_base_link") is not None
    ):
        flange_to_base = pose_matrix(flange["pose_in_base_link"])
        flange_range = float(np.linalg.norm(point - flange_to_base[:3, 3]))
        if profile is not None:
            offset = np.asarray([*profile["tcp_offset_in_flange_m"], 1.0])
            tcp = (flange_to_base @ offset)[:3]
            tcp_range = float(np.linalg.norm(point - tcp))
    return {
        "frame_id": "base_link",
        "side": side,
        "wrist_pixel_px": list(pixel),
        "timestamp_ns": stamp,
        "rgb_depth_skew_ms": skew_ms,
        "surface_base_m": point.tolist(),
        "optical_depth_m": z,
        "depth_spread_p10_p90_m": spread,
        "depth_samples": samples,
        "camera_to_surface_m": float(np.linalg.norm(point - origin)),
        "base_horizontal_range_m": float(np.linalg.norm(point[:2])),
        "table_frame": table,
        "flange_to_surface_m": flange_range,
        "tcp_to_surface_m": tcp_range,
        "tcp_calibration_id": profile["calibration_id"] if profile else None,
        "gripper_feedback": feedback,
        "gripper_feedback_skew_ms": (
            abs(feedback["timestamp_ns"] - stamp) / 1e6
            if feedback and type(feedback.get("timestamp_ns")) is int
            else None
        ),
        "gripper_command_check": _grip_check(command, profile, feedback, stamp),
        "reachability": "requires_ik_and_collision_scene",
        "contact_force_n": None,
        "dispatch_allowed": False,
    }
