"""Galbot SDK 1.10 planning adapter for the HPU. This module never acquires control."""

import importlib
import math
import platform
from collections.abc import Callable
from uuid import uuid4

from .g1_task import TOOL_FRAMES, SceneSnapshot, StagePlan, StageResult
from .types import TaskError

ARM = "left_arm"
CHAINS = ("leg", "head", ARM, "right_arm")
JOINT_COUNTS = (5, 2, 7, 7)
IDENTITY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


class GalbotStagePlanner:
    evidence_level = "galbot_sdk_plan_only"

    def __init__(self):
        if platform.system() != "Linux":
            raise TaskError("Galbot SDK planning requires the configured Linux HPU")
        self.sdk = importlib.import_module("galbot_sdk.g1")
        self.motion = self.sdk.GalbotMotion()
        if not self.motion.init():
            raise TaskError("Galbot motion planner initialization failed")

    def _joints(self, snapshot: SceneSnapshot) -> list[float]:
        result = []
        for chain, size in zip(CHAINS, JOINT_COUNTS, strict=True):
            names = list(self.motion.get_chain_joint_names(chain))
            if len(names) != size or not set(names) <= snapshot.joint_positions_rad.keys():
                raise TaskError(f"Capture does not match the G1 {chain} joint names")
            result.extend(snapshot.joint_positions_rad[name] for name in names)
        return result

    def _state(self, joints):
        state = self.sdk.RobotStates()
        state.whole_body_joint = joints
        state.base_state = IDENTITY
        return state

    def _plan_segment(self, stage, joints):
        tcp, _ = TOOL_FRAMES[stage.tool_model]
        config = self.sdk.PlannerConfig()
        config.is_direct_execute = False
        config.is_blocking = True
        config.is_check_collision = True
        config.enable_env_collision_check = True
        config.is_relative_pose = False
        config.actuate_type = self.sdk.ActuateType.ACTUATE_WITH_CHAIN_ONLY
        config.reference_frame = "base_link"
        config.timeout_second = 5.0
        waypoints = []
        for pose in stage.tcp_waypoints_base:
            target = self.sdk.MotionPlanChainTarget()
            target.chain_name = ARM
            target.mode = self.sdk.MotionPlanTargetMode.kCartesian
            target.cart.chain_name = ARM
            target.cart.frame_id = tcp
            target.cart.reference_frame = "base_link"
            target.cart.pose = self.sdk.Pose(list(pose))
            waypoints.append([target])
        method = (
            self.motion.move_line if stage.motion == "cartesian_line" else self.motion.motion_plan
        )
        status, trajectories = method(waypoints, config, self._state(joints))
        if status != self.sdk.MotionStatus.SUCCESS:
            return StageResult(
                stage.stage_id, "rejected", "sdk_plan_failed", sdk_status=str(status)
            )
        points = trajectories.get(ARM, [])
        if (
            set(trajectories) != {ARM}
            or not points
            or any(
                len(row) != JOINT_COUNTS[2] or not all(math.isfinite(v) for v in row)
                for row in points
            )
        ):
            return StageResult(
                stage.stage_id, "rejected", "unexpected_trajectory", sdk_status=str(status)
            )
        offset = sum(JOINT_COUNTS[:2])
        end = joints[:offset] + list(points[-1]) + joints[offset + JOINT_COUNTS[2] :]
        fk_status, actual = self.motion.forward_kinematics_by_state(
            tcp, self._state(end), "base_link"
        )
        goal = stage.tcp_waypoints_base[-1]
        if (
            fk_status != self.sdk.MotionStatus.SUCCESS
            or len(actual) != 7
            or not all(math.isfinite(v) for v in actual)
            or abs(math.hypot(*actual[3:]) - 1.0) > 0.005
        ):
            return StageResult(
                stage.stage_id, "rejected", "endpoint_fk_unverified", sdk_status=str(fk_status)
            )
        position_error = math.dist(actual[:3], goal[:3])
        dot = sum(a * b for a, b in zip(actual[3:], goal[3:], strict=True))
        orientation_error = 2 * math.acos(min(1.0, abs(dot) / math.hypot(*actual[3:])))
        metrics = {
            "planned_tcp_pose_base": tuple(actual),
            "goal_position_error_m": position_error,
            "goal_orientation_error_rad": orientation_error,
        }
        if position_error > 0.005 or orientation_error > 2 * math.acos(0.999):
            return StageResult(
                stage.stage_id,
                "rejected",
                "endpoint_outside_tolerance",
                len(points),
                str(status),
                **metrics,
            )
        joints[:] = end
        return StageResult(
            stage.stage_id, "planned", "sdk_collision_checked", len(points), str(status), **metrics
        )

    def review(
        self,
        snapshot: SceneSnapshot,
        stages: tuple[StagePlan, ...],
        checkpoint: Callable[[], None],
    ) -> list[StageResult]:
        if not snapshot.geometry_bound_to_capture:
            raise TaskError("SDK planning requires geometry tied to the capture.json SHA-256")
        tcp, fingers = TOOL_FRAMES[snapshot.tool_model]
        if not {tcp, *fingers} <= set(self.motion.get_link_names()):
            raise TaskError(f"Load the matching {snapshot.tool_model} model in the G1 planner")
        if any(stage.tool_model != snapshot.tool_model for stage in stages):
            raise TaskError("Stage tool model does not match the capture")
        joints = self._joints(snapshot)
        shape = snapshot.geometry
        prefix = "pet_meal_" + uuid4().hex[:16]
        obstacles = (
            ("table", "box", (*shape.table_center_m, *IDENTITY[3:]), shape.table_dimensions_m, ""),
            ("clutter", "point_cloud", IDENTITY, (1.0, 1.0, 1.0), str(shape.point_cloud)),
            (
                "bottle",
                "cylinder",
                (*shape.center_m, *IDENTITY[3:]),
                (shape.radius_m, shape.height_m, 0.0),
                "",
            ),
        )
        registered = []
        results = []
        blocked = None
        try:
            for name, kind, pose, scale, key in obstacles:
                checkpoint()
                obstacle_id = f"{prefix}_{name}"
                status = self.motion.add_obstacle(
                    obstacle_id,
                    kind,
                    pose,
                    scale,
                    key=key,
                    target_frame="base_link",
                    reference_joint_positions=joints,
                    reference_base_pose=IDENTITY,
                    resolution=shape.resolution_m,
                )
                if status != self.sdk.MotionStatus.SUCCESS:
                    raise TaskError(f"Failed to register G1 obstacle {name}: {status}")
                registered.append(obstacle_id)
            for stage in stages:
                checkpoint()
                if blocked:
                    result = StageResult(stage.stage_id, "blocked", blocked)
                elif stage.motion == "gripper":
                    result = StageResult(
                        stage.stage_id, "blocked", "contact_and_payload_model_unverified"
                    )
                    blocked = "previous_stage_unverified"
                else:
                    result = self._plan_segment(stage, joints)
                    if result.status != "planned":
                        blocked = "previous_stage_unverified"
                results.append(result)
        finally:
            cleanup_errors = []
            for obstacle_id in reversed(registered):
                try:
                    status = self.motion.remove_obstacle(obstacle_id)
                    if status != self.sdk.MotionStatus.SUCCESS:
                        cleanup_errors.append(f"{obstacle_id}: {status}")
                except Exception as exc:
                    cleanup_errors.append(f"{obstacle_id}: {exc}")
            if cleanup_errors:
                raise TaskError("G1 planning scene cleanup failed: " + ", ".join(cleanup_errors))
        return results
