"""Capture time-aligned G1 RGB-D frames and their individual base transforms."""

import argparse
import gzip
import json
import math
import time
from pathlib import Path

CAMERAS = (
    ("left-wrist-rgb", "LEFT_ARM_CAMERA", "left_arm_camera_color_optical_frame", False),
    ("left-wrist-depth", "LEFT_ARM_DEPTH_CAMERA", "left_arm_camera_color_optical_frame", True),
    ("right-wrist-rgb", "RIGHT_ARM_CAMERA", "right_arm_camera_color_optical_frame", False),
    ("right-wrist-depth", "RIGHT_ARM_DEPTH_CAMERA", "right_arm_camera_color_optical_frame", True),
    ("head-left-rgb", "HEAD_LEFT_CAMERA", "head_left_camera_color_optical_frame", False),
    ("head-right-rgb", "HEAD_RIGHT_CAMERA", "head_right_camera_color_optical_frame", False),
)
MAX_SKEW_NS = 50_000_000


def matched_pose(robot, frame, timestamp):
    pose, received = robot.get_transform("base_link", frame, timestamp, 350)
    pose = list(pose)
    if (
        len(pose) == 7
        and received == timestamp
        and all(math.isfinite(value) for value in pose)
        and abs(sum(value**2 for value in pose[3:]) - 1) < 0.005
    ):
        return pose
    return None


def gripper_feedback(robot, groups):
    report = {}
    for side, group in groups:
        state = robot.get_gripper_state(group)
        if state is not None:
            report[side] = {
                "width_m": state.width,
                "moving": state.is_moving,
                "effort_nm": state.effort,
                "timestamp_ns": state.timestamp_ns,
            }
    return report


def capture(output):
    from galbot_sdk.g1 import G1JointGroup, GalbotRobot, SensorType

    sensors = [getattr(SensorType, name) for _, name, _, _ in CAMERAS]
    head_anchor = SensorType.HEAD_LEFT_CAMERA
    head_first = [head_anchor] + [sensor for sensor in sensors if sensor != head_anchor]
    output.mkdir(parents=True, exist_ok=False)
    robot = GalbotRobot()
    report = {"frames": {}, "synchronized": False}
    try:
        if not robot.init(set(sensors), True):
            raise RuntimeError("G1 multi-camera initialization failed")
        time.sleep(1.2)
        best = {}
        best_joint_state = None
        best_grippers = {}
        groups = (("left", G1JointGroup.left_gripper), ("right", G1JointGroup.right_gripper))
        best_score = (0, float("-inf"))
        for attempt in range(5):
            # The SDK uses the latest timestamp of the first camera as its anchor.
            sync_order = head_first if attempt % 2 == 0 else sensors
            observation = robot.get_synced_observation(sync_order, True)
            if observation:
                frames = {
                    sensor: frame
                    for frame_map in (observation.rgb_data_map, observation.depth_data_map)
                    for sensor, frame in frame_map.items()
                    if sensor in sensors and frame and frame.header.timestamp_ns > 0
                }
                stamps = [frame.header.timestamp_ns for frame in frames.values()]
                if stamps:
                    skew = max(stamps) - min(stamps)
                    score = (len(frames), -skew)
                    if score > best_score:
                        best, best_score = frames, score
                        best_joint_state = observation.joint_state
                        best_grippers = gripper_feedback(robot, groups)
                    if len(frames) == len(sensors) and skew <= MAX_SKEW_NS:
                        break
            time.sleep(0.12)
        if not best:
            raise RuntimeError("No frames arrived from the synchronized camera request")
        report["max_camera_skew_ms"] = -best_score[1] / 1e6
        report["synchronized"] = len(best) == len(sensors) and -best_score[1] <= MAX_SKEW_NS
        if best_joint_state is not None:
            report["joint_state"] = {
                "timestamp_ns": best_joint_state.timestamp_ns,
                "positions_rad": {
                    state.joint_name: state.position for state in best_joint_state.joint_state_vec
                },
            }
        time.sleep(0.35)
        for (label, enum_name, optical, is_depth), sensor in zip(CAMERAS, sensors):
            frame = best.get(sensor)
            if frame is None:
                continue
            data = bytes(frame.data)
            if not data or len(data) > 8_000_000:
                raise RuntimeError("Unexpected camera payload size for " + enum_name)
            timestamp = frame.header.timestamp_ns
            pose = matched_pose(robot, optical, timestamp)
            intrinsics = robot.get_camera_intrinsic(sensor) or {}
            filename = label + ".bin.gz"
            with gzip.open(output / filename, "xb", compresslevel=1) as stream:
                stream.write(data)
            report["frames"][label] = {
                "sensor": enum_name,
                "timestamp_ns": timestamp,
                "file": filename,
                "width": frame.width,
                "height": frame.height,
                "format": str(frame.format),
                "depth_scale": frame.depth_scale if is_depth else None,
                "optical_frame": optical,
                "reported_frame": frame.header.frame_id,
                "camera_to_base_pose": pose,
                "transform_timestamp_ns": timestamp if pose is not None else None,
                "intrinsics": {
                    key: intrinsics[key]
                    for key in ("K", "D", "width", "height", "distortion_model")
                    if key in intrinsics
                },
            }
        report["flanges"] = {}
        for side in ("left", "right"):
            depth_frame = report["frames"].get(f"{side}-wrist-depth")
            if depth_frame is not None:
                timestamp = depth_frame["timestamp_ns"]
                frame = f"{side}_arm_end_effector_mount_link"
                report["flanges"][side] = {
                    "frame_id": frame,
                    "timestamp_ns": timestamp,
                    "pose_in_base_link": matched_pose(robot, frame, timestamp),
                }
        report["grippers"] = best_grippers
    finally:
        try:
            (output / "capture.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        finally:
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = capture(args.output)
    print("Captured {} cameras".format(len(report["frames"])))


if __name__ == "__main__":
    main()
