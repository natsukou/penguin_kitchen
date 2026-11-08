"""G1 tabletop task contracts and deterministic, plan-only orchestration."""

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from .g1_multiview import MAX_CAMERA_SKEW_NS
from .policy import recommend_scene_grasp
from .recognition import _number, _stamp, _text, _vector, load_scene
from .scene_choice import parse_grasp_choice
from .types import TaskError, Vec3

SKILLS = ("grasp", "pick_place", "shake_then_place")
TCP = "left_gripper_tcp_link"
FINGERS = ("left_gripper_l_finger_link", "left_gripper_r_finger_link")
TOOL_FRAMES = {
    "galbot_gripper": (TCP, FINGERS),
    "galbot_ge102in": (
        "left_ge102_contact_link",
        ("left_ge102_left_active_link2", "left_ge102_right_active_link2"),
    ),
}
Pose = tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class TaskRequest:
    skill: str
    object_id: str | None = None
    category: str | None = None
    repeat_count: int | None = None

    def __post_init__(self):
        if self.skill not in SKILLS:
            raise TaskError("Unsupported G1 tabletop skill")
        if (self.object_id is None) == (self.category is None):
            raise TaskError("Specify exactly one object ID or category")
        for value in (self.object_id, self.category):
            if value is not None and (
                len(_text(value, "Target selector")) > 128 or value != value.strip()
            ):
                raise TaskError("Invalid target selector")
        if self.repeat_count is not None and (
            self.skill != "shake_then_place"
            or type(self.repeat_count) is not int
            or not 1 <= self.repeat_count <= 3
        ):
            raise TaskError("Shake repeat count must be an integer between 1 and 3")


@dataclass(frozen=True)
class BottleGeometry:
    center_m: Vec3
    radius_m: float
    height_m: float
    grip_height_m: float
    table_center_m: Vec3
    table_dimensions_m: Vec3
    point_cloud: Path
    resolution_m: float


@dataclass(frozen=True)
class SceneSnapshot:
    revision: str
    object_id: str
    capture_timestamp_ns: int
    joint_timestamp_ns: int
    joint_positions_rad: dict[str, float]
    gripper_width_m: float
    geometry: BottleGeometry
    geometry_bound_to_capture: bool
    tool_model: str = "galbot_gripper"


@dataclass(frozen=True)
class StagePlan:
    stage_id: str
    motion: str
    tcp_waypoints_base: tuple[Pose, ...]
    payload_attached: bool
    completion_evidence: str
    tool_model: str = "galbot_gripper"

    @property
    def allowed_target_contacts(self) -> tuple[str, ...]:
        return (
            TOOL_FRAMES[self.tool_model][1]
            if self.payload_attached or self.stage_id == "close"
            else ()
        )


@dataclass(frozen=True)
class StageResult:
    stage_id: str
    status: str
    reason: str
    trajectory_points: int = 0
    sdk_status: str | None = None
    planned_tcp_pose_base: Pose | None = None
    goal_position_error_m: float | None = None
    goal_orientation_error_rad: float | None = None


class StagePlanner(Protocol):
    evidence_level: str

    def review(
        self, snapshot: SceneSnapshot, stages: tuple[StagePlan, ...], checkpoint: Callable[[], None]
    ) -> list[StageResult]: ...


def _load_snapshot(capture: Path, target: dict, localized: dict) -> SceneSnapshot:
    meta_bytes = (capture / "capture.json").read_bytes()
    model_bytes = (capture / "geometry.json").read_bytes()
    meta, model = json.loads(meta_bytes), json.loads(model_bytes)
    tool_model = _text(model.get("tool_model", "galbot_gripper"), "Tool model")
    if tool_model not in TOOL_FRAMES:
        raise TaskError("Unsupported G1 left gripper model")
    capture_hash = model.get("capture_sha256")
    if capture_hash is not None and capture_hash != hashlib.sha256(meta_bytes).hexdigest():
        raise TaskError("Geometry capture_sha256 does not match capture.json")
    if model["cap"]["frame_id"] != "base_link":
        raise TaskError("Bottle geometry must be in base_link")
    cap = _vector(localized["head_stereo"]["point_m"], 3, "Localized cap")
    model_cap = _vector(model["cap"]["head_stereo"]["point_m"], 3, "Modeled cap")
    if math.dist(cap, model_cap) > 0.001:
        raise TaskError("Bottle geometry does not match the localized target")

    cylinder = model["cylinder"]
    center = _vector(cylinder["center_m"], 3, "Cylinder center")
    radius = _number(cylinder["radius_m"], "Cylinder radius")
    height = _number(cylinder["height_m"], "Cylinder height")
    grip_height = _number(cylinder["grip_height_m"], "Grip height")
    if not (
        radius > 0
        and height > 0
        and math.dist(cap[:2], center[:2]) <= radius
        and abs(cap[2] - center[2] - height / 2) <= 0.015
        and center[2] - height / 2 < grip_height < center[2] + height / 2
    ):
        raise TaskError("Cylinder does not describe an upright bottle at the selected landmark")
    obstacles = model["obstacles"]
    table = obstacles["table_box"]
    table_center = _vector(table["center_m"], 3, "Table center")
    table_dimensions = _vector(table["dimensions_m"], 3, "Table dimensions")
    resolution = _number(obstacles["voxel_m"], "Cloud resolution")
    cloud = (capture / _text(obstacles["point_cloud"], "Point cloud")).resolve()
    if (
        min(table_dimensions) <= 0
        or resolution <= 0
        or cloud.suffix.lower() != ".pcd"
        or not cloud.is_relative_to(capture.resolve())
        or not cloud.is_file()
    ):
        raise TaskError("A measured table and local PCD collision scene are required")

    frames = {camera: row["timestamp_ns"] for camera, row in target["landmark"]["views"].items()}
    stamp = _stamp(frames["head-left-rgb"], "Capture timestamp")
    joints, gripper = meta["joint_state"], meta["grippers"]["left"]
    joint_stamp = _stamp(joints["timestamp_ns"], "Joint timestamp")
    gripper_stamp = _stamp(gripper["timestamp_ns"], "Gripper timestamp")
    wrist_stamp = _stamp(meta["frames"]["left-wrist-depth"]["timestamp_ns"], "Left wrist timestamp")
    if (
        max(abs(joint_stamp - stamp), abs(wrist_stamp - stamp), abs(gripper_stamp - wrist_stamp))
        > MAX_CAMERA_SKEW_NS
    ):
        raise TaskError("Joint and gripper measurements must match the camera capture")
    positions = {
        name: _number(value, f"Joint {name}") for name, value in joints["positions_rad"].items()
    }
    width = _number(gripper["width_m"], "Gripper width")
    if not positions or gripper["moving"] is not False or not 2 * radius < width <= 0.12:
        raise TaskError("Capture must show a stationary gripper open wider than the bottle")

    digest = hashlib.sha256()
    for data in (meta_bytes, model_bytes, json.dumps(target, sort_keys=True).encode()):
        digest.update(data)
    with cloud.open("rb") as stream:
        digest.update(hashlib.file_digest(stream, "sha256").digest())
    return SceneSnapshot(
        digest.hexdigest(),
        target["object_id"],
        stamp,
        joint_stamp,
        positions,
        width,
        BottleGeometry(
            center, radius, height, grip_height, table_center, table_dimensions, cloud, resolution
        ),
        capture_hash is not None,
        tool_model,
    )


def _pose(position: Vec3, yaw: float, pitch: float, roll: float = 0.0) -> Pose:
    cz, sz = math.cos(yaw / 2), math.sin(yaw / 2)
    cy, sy = math.cos(pitch / 2), math.sin(pitch / 2)
    cx, sx = math.cos(roll / 2), math.sin(roll / 2)
    return (
        *position,
        cz * cy * sx - sz * sy * cx,
        cz * sy * cx + sz * cy * sx,
        sz * cy * cx - cz * sy * sx,
        cz * cy * cx + sz * sy * sx,
    )


def bottle_stages(
    request: TaskRequest,
    geometry: BottleGeometry,
    yaw: float,
    pitch: float = 0.0,
    grip_height_m: float | None = None,
    tool_model: str = "galbot_gripper",
) -> tuple[StagePlan, ...]:
    """Candidate TCP waypoints for an upright bottle, before IK or collision checks."""
    x, y, _ = geometry.center_m
    z = geometry.grip_height_m if grip_height_m is None else grip_height_m
    standoff = 0.08  # Travel for the approach stage, not a clearance radius around the bottle.
    horizontal = standoff * math.cos(pitch)
    grasp = _pose((x, y, z), yaw, pitch)
    pregrasp = _pose(
        (
            x - horizontal * math.cos(yaw),
            y - horizontal * math.sin(yaw),
            z + standoff * math.sin(pitch),
        ),
        yaw,
        pitch,
    )
    lifted = _pose((x, y, z + 0.025), yaw, pitch)
    stages = [
        StagePlan("pregrasp", "joint_path", (pregrasp,), False, "arm_reached"),
        StagePlan("insert", "cartesian_line", (grasp,), False, "arm_reached"),
        StagePlan("close", "gripper", (), False, "target_held"),
        StagePlan("test_lift", "cartesian_line", (lifted,), True, "target_followed_off_table"),
    ]
    if request.skill == "shake_then_place":
        height = z + 0.07
        neutral = _pose((x, y, height), yaw, pitch)
        swing = (
            _pose((x, y, height), yaw, pitch, 0.10),
            _pose((x, y, height), yaw, pitch, -0.10),
            neutral,
        )
        stages.extend(
            (
                StagePlan("raise_for_shake", "cartesian_line", (neutral,), True, "target_held"),
                StagePlan(
                    "shake", "joint_path", swing * (request.repeat_count or 1), True, "target_held"
                ),
            )
        )
    if request.skill != "grasp":
        stages.extend(
            (
                StagePlan("lower", "cartesian_line", (grasp,), True, "target_supported"),
                StagePlan("release", "gripper", (), True, "target_released_at_origin"),
                StagePlan("retreat", "cartesian_line", (pregrasp,), False, "arm_reached"),
            )
        )
    return tuple(replace(stage, tool_model=tool_model) for stage in stages)


def plan_task(
    request: TaskRequest,
    capture: Path,
    observation: Path,
    vision_choice: Path | None,
    planner: StagePlanner | None,
    emit: Callable[[str, dict], None],
    checkpoint: Callable[[], None],
) -> dict:
    scene = load_scene(observation)
    selector = {"object_id": request.object_id, "category": request.category}
    if vision_choice is not None:
        matches = recommend_scene_grasp(scene, **selector)["candidates"]
        with vision_choice.open(encoding="utf-8-sig") as stream:
            choice = parse_grasp_choice(stream.read(2049), {obj["object_id"] for obj in matches})
        emit("task_visual_choice", choice)
        if choice["action"] != "grasp":
            return {"state": "held", "reason": choice["action"]}
        selector = {"object_id": choice["object_id"]}
    checkpoint()
    decision = recommend_scene_grasp(scene, capture=capture, **selector)
    emit("task_target_resolved", decision)
    if decision["next_action"] != "plan_grasp":
        return {"state": "held", "reason": decision["reason"]}
    target = decision["target"]
    if target["category"] != "bottle":
        return {"state": "held", "reason": "grasp_geometry_model_unavailable"}
    checkpoint()
    try:
        snapshot = _load_snapshot(capture, target, decision["geometry"])
    except (OSError, ValueError, KeyError, TypeError, TaskError) as exc:
        return {"state": "held", "reason": "scene_geometry_unavailable", "detail": str(exc)}
    emit("task_snapshot", asdict(snapshot))
    geometry = snapshot.geometry
    heading = math.atan2(geometry.center_m[1], geometry.center_m[0])
    grip = geometry.grip_height_m
    raised_grip = grip + min(0.008, (geometry.center_m[2] + geometry.height_m / 2 - grip) / 2)
    poses = (
        (heading, 0.0, grip),
        (heading - 0.20, 0.0, grip),
        (heading + 0.20, 0.0, grip),
        (heading - 0.15, 0.18, grip),
        (heading - 0.15, 0.35, raised_grip),
        (heading - 0.35, 0.35, raised_grip),
    )
    candidates = []
    for yaw, pitch, height in poses:
        checkpoint()
        stages = bottle_stages(request, geometry, yaw, pitch, height, snapshot.tool_model)
        candidate_id = f"side_{len(candidates) + 1}"
        plan = {
            "candidate_id": candidate_id,
            "scene_revision": snapshot.revision,
            "reference_frame": "base_link",
            "tcp_frame": TOOL_FRAMES[snapshot.tool_model][0],
            "grasp_sample": {"yaw_rad": yaw, "pitch_rad": pitch, "grip_height_m": height},
            "stages": [
                {**asdict(stage), "allowed_target_contacts": stage.allowed_target_contacts}
                for stage in stages
            ],
        }
        emit("task_candidate", plan)
        results = [] if planner is None else planner.review(snapshot, stages, checkpoint)
        checkpoint()
        for result in results:
            emit(
                "task_stage_reviewed",
                {
                    "candidate_id": candidate_id,
                    "scene_revision": snapshot.revision,
                    **asdict(result),
                },
            )
        candidates.append({**plan, "planning_results": [asdict(row) for row in results]})
    reason = "sdk_planning_not_run" if planner is None else "full_stage_plan_unverified"
    return {
        "state": "preview",
        "reason": reason,
        "scene_revision": snapshot.revision,
        "object_id": snapshot.object_id,
        "target_geometry": {
            "center_m": geometry.center_m,
            "grip_height_m": geometry.grip_height_m,
        },
        "candidates": candidates,
    }
