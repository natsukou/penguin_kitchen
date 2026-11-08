import gzip
import json

import numpy as np
import pytest

from pet_meal.g1_multiview import fuse_capture, triangulate_head_point
from pet_meal.g1_tabletop import measure_tabletop
from pet_meal.policy import recommend_scene_grasp
from pet_meal.recognition import load_scene


def frame(camera_x, lens="inverse brown conrady", distortion=None):
    return {
        "width": 3,
        "height": 3,
        "timestamp_ns": 1_000_000_000,
        "transform_timestamp_ns": 1_000_000_000,
        "optical_frame": "camera",
        "camera_to_base_pose": [camera_x, 0, 0, 0, 0, 0, 1],
        "intrinsics": {
            "width": 3,
            "height": 3,
            "K": [10, 0, 1, 0, 10, 1, 0, 0, 1],
            "D": [0] * 5 if distortion is None else distortion,
            "distortion_model": lens,
        },
        "format": "16UC1",
        "depth_scale": 1000,
    }


def test_head_stereo_reconstructs_known_base_point():
    frames = {
        "head-left-rgb": frame(0, "kannala_brandt", [0] * 4),
        "head-right-rgb": frame(0.1, "kannala_brandt", [0] * 4),
    }
    point = triangulate_head_point(frames, (1, 1), (1 - 10 * np.arctan(0.1), 1))
    assert point["point_m"] == pytest.approx([0, 0, 1])
    assert point["ray_separation_m"] < 1e-10


def test_fuse_wrist_clouds_requires_time_matched_transforms(tmp_path):
    frames = {
        f"{side}-wrist-{kind}": frame(x)
        for side, x in (("left", 0), ("right", 0.1))
        for kind in ("rgb", "depth")
    }
    frames.update({f"head-{side}-rgb": frame(0) for side in ("left", "right")})
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    for side in ("left", "right"):
        data = np.full((3, 3), 1000, dtype="<u2").tobytes()
        (tmp_path / f"{side}-wrist-depth.bin.gz").write_bytes(gzip.compress(data))

    result = fuse_capture(tmp_path, [])
    assert result["points_by_wrist"] == {"left": 1, "right": 1}
    data = (tmp_path / result["cloud"]).read_bytes().split(b"end_header\n", 1)[1]
    points = np.frombuffer(data, dtype="<f4").reshape(-1, 3)
    np.testing.assert_allclose(points, [[-0.1, -0.1, 1], [0, -0.1, 1]])

    frames["right-wrist-depth"]["transform_timestamp_ns"] += 1
    capture.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    with pytest.raises(ValueError, match="transform does not match"):
        fuse_capture(tmp_path, [])


def test_grasp_guidance_requires_frame_bound_geometry_and_gripper_feedback(tmp_path):
    frames = {
        "head-left-rgb": frame(0, "kannala_brandt", [0] * 4),
        "head-right-rgb": frame(0.1, "kannala_brandt", [0] * 4),
        "left-wrist-rgb": frame(0),
        "left-wrist-depth": frame(0),
    }
    (tmp_path / "capture.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
    depth = tmp_path / "left-wrist-depth.bin.gz"
    depth.write_bytes(gzip.compress(np.full((3, 3), 1000, dtype="<u2").tobytes()))
    landmark = {
        "name": "cap_center",
        "views": {
            camera: {"timestamp_ns": frames[camera]["timestamp_ns"], "pixel_px": [u, 1]}
            for camera, u in (
                ("head-left-rgb", 1),
                ("head-right-rgb", 1 - 10 * np.arctan(0.1)),
                ("left-wrist-rgb", 1),
            )
        },
    }
    feedback = {"held_object": "unknown", "spoon_loaded": None, "bowl_loaded": None}
    payload = {
        "timestamp_ns": None,
        "feedback": feedback,
        "detections": {
            "bottle_01": {
                "object_id": "bottle_01",
                "name": "bottle",
                "category": "bottle",
                "description": "standing bottle",
                "position_m": [9, 9, 9],
                "frame_id": "base_link",
                "confidence": 0.95,
                "landmark": landmark,
            }
        },
    }
    observation = tmp_path / "scene.json"

    def recommend():
        observation.write_text(json.dumps(payload), encoding="utf-8")
        return recommend_scene_grasp(
            load_scene(observation), object_id="bottle_01", capture=tmp_path
        )

    result = recommend()
    assert result["next_action"] == "verify_gripper"
    assert result["geometry"]["head_stereo"]["point_m"] == pytest.approx([0, 0, 1])
    assert result["geometry"]["wrist_depth"]["left"]["difference_m"] < 1e-10

    feedback["held_object"] = None
    result = recommend()
    assert result["next_action"] == "plan_grasp"
    assert result["dispatch_allowed"] is False

    landmark["views"]["head-left-rgb"]["timestamp_ns"] += 1
    result = recommend()
    assert result["next_action"] == "localize_target"
    assert "timestamp does not match" in result["geometry_error"]

    landmark["views"]["head-left-rgb"]["timestamp_ns"] -= 1
    depth.write_bytes(gzip.compress(np.full((3, 3), 1200, dtype="<u2").tobytes()))
    result = recommend()
    assert result["geometry"] is None
    assert result["next_action"] == "localize_target"
    assert "disagrees" in result["geometry_error"]


def test_surface_range_distinguishes_camera_flange_tcp_and_feedback_time(tmp_path):
    wrist = frame(0)
    wrist["width"] = wrist["height"] = 7
    wrist["intrinsics"].update(width=7, height=7, K=[10, 0, 3, 0, 10, 3, 0, 0, 1])
    stamp = wrist["timestamp_ns"]
    report = {
        "frames": {f"right-wrist-{kind}": wrist for kind in ("rgb", "depth")},
        "flanges": {
            "right": {
                "frame_id": "right_arm_end_effector_mount_link",
                "timestamp_ns": stamp,
                "pose_in_base_link": [0, 0, 0.25, 0, 0, 0, 1],
            }
        },
        "grippers": {
            "right": {"timestamp_ns": stamp, "width_m": 0.08, "effort_nm": 2, "moving": False}
        },
    }
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps(report), encoding="utf-8")
    depth = np.full((7, 7), 1000, dtype="<u2")
    (tmp_path / "right-wrist-depth.bin.gz").write_bytes(gzip.compress(depth.tobytes()))
    path = tmp_path / "tool.json"
    path.write_text(
        json.dumps(
            {
                "side": "right",
                "flange_frame": "right_arm_end_effector_mount_link",
                "calibration_id": "fixture",
                "tool_model": "fixture",
                "tcp_offset_in_flange_m": [0, 0, 0.10],
                "limits": {
                    "width_range_m": [0.01, 0.10],
                    "max_speed_mps": 0.02,
                    "max_effort_nm": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    actual = measure_tabletop(tmp_path, "right", (3, 3))
    assert actual["camera_to_surface_m"] == pytest.approx(1.0)
    assert actual["flange_to_surface_m"] == pytest.approx(0.75)
    assert actual["tcp_to_surface_m"] is None
    assert actual["contact_force_n"] is None

    def review(command):
        return measure_tabletop(tmp_path, "right", (3, 3), profile_path=path, command=command)

    actual = review((0.05, 0.01, 2.5))
    assert actual["tcp_to_surface_m"] == pytest.approx(0.65)
    assert actual["gripper_command_check"]["status"] == "within_provided_profile_for_offline_review"
    assert actual["dispatch_allowed"] is False
    assert review((0.05, 0.01, 4))["gripper_command_check"]["status"] == "outside_calibrated_limits"

    report["grippers"]["right"]["timestamp_ns"] += 50_000_001
    capture.write_text(json.dumps(report), encoding="utf-8")
    assert review((0.05, 0.01, 2.5))["gripper_command_check"]["status"] == (
        "gripper_feedback_unverified"
    )
