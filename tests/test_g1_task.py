import gzip
import json
import threading
from types import SimpleNamespace

import numpy as np

from pet_meal.g1_multiview import pose_matrix
from pet_meal.g1_planner import ARM, CHAINS, JOINT_COUNTS, GalbotStagePlanner
from pet_meal.g1_task import TOOL_FRAMES, BottleGeometry, SceneSnapshot, TaskRequest, bottle_stages
from pet_meal.hub import ControlHub


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _scene(root):
    root.mkdir(parents=True)
    stamp = 1_000_000_000
    frames = {}
    for label, x in [
        ("head-left-rgb", 0),
        ("head-right-rgb", 0.1),
        ("left-wrist-rgb", 0),
        ("left-wrist-depth", 0),
    ]:
        head = label.startswith("head")
        frames[label] = {
            "width": 3,
            "height": 3,
            "timestamp_ns": stamp,
            "transform_timestamp_ns": stamp,
            "optical_frame": "camera",
            "camera_to_base_pose": [x, 0, 0, 0, 0, 0, 1],
            "intrinsics": {
                "width": 3,
                "height": 3,
                "K": [10, 0, 1, 0, 10, 1, 0, 0, 1],
                "D": [0] * (4 if head else 5),
                "distortion_model": "kannala_brandt" if head else "inverse brown conrady",
            },
            "format": "16UC1",
            "depth_scale": 1000,
        }
    _write(
        root / "capture.json",
        {
            "frames": frames,
            "joint_state": {"timestamp_ns": stamp, "positions_rad": {"left": 0}},
            "grippers": {"left": {"timestamp_ns": stamp, "width_m": 0.10, "moving": False}},
        },
    )
    depth = np.full((3, 3), 1000, dtype="<u2")
    (root / "left-wrist-depth.bin.gz").write_bytes(gzip.compress(depth.tobytes()))
    (root / "obstacles.pcd").write_text("PCD fixture", encoding="ascii")
    _write(
        root / "geometry.json",
        {
            "tool_model": "galbot_ge102in",
            "cap": {"frame_id": "base_link", "head_stereo": {"point_m": [0, 0, 1]}},
            "cylinder": {
                "center_m": [0, 0, 0.90],
                "radius_m": 0.025,
                "height_m": 0.20,
                "grip_height_m": 0.90,
            },
            "obstacles": {
                "point_cloud": "obstacles.pcd",
                "voxel_m": 0.008,
                "table_box": {"center_m": [0, 0, 0.4], "dimensions_m": [1, 1, 0.8]},
            },
        },
    )
    first = {
        "object_id": "b1",
        "name": "Blue bottle",
        "description": "blue cap",
        "category": "bottle",
        "confidence": 0.96,
        "position_m": None,
        "frame_id": None,
        "landmark": {
            "name": "cap",
            "views": {
                label: {"timestamp_ns": stamp, "pixel_px": [u, 1]}
                for label, u in [
                    ("head-left-rgb", 1),
                    ("head-right-rgb", 1 - 10 * np.arctan(0.1)),
                    ("left-wrist-rgb", 1),
                ]
            },
        },
    }
    observation = {
        "timestamp_ns": stamp,
        "detections": {
            "b1": first,
            "b2": {**first, "object_id": "b2", "landmark": None},
        },
        "feedback": {"held_object": None, "spoon_loaded": None, "bowl_loaded": None},
    }
    return _write(root / "scene.json", observation)


def test_tabletop_imported_choice_through_hub_records_stages(tmp_path, config):
    capture = tmp_path / "capture"
    scene = _scene(capture)
    choice = _write(tmp_path / "choice.json", {"action": "grasp", "object_id": "b1"})
    hub = ControlHub(tmp_path, config)
    try:
        request = TaskRequest("shake_then_place", category="bottle", repeat_count=2)
        hub.start_tabletop(request, capture, scene)
        hub.wait()
        assert hub.snapshot()["summary"]["reason"] == "multiple_instances"
        hub.start_tabletop(request, capture, scene, vision_choice=choice)
        hub.wait()
        state = hub.snapshot()
        assert state["mode"] == "g1_plan_only"
        result = state["summary"]
        assert result["state"] == "preview" and result["object_id"] == "b1"
        assert result["dispatch_allowed"] is False and result["motion_commands_sent"] == 0
        assert state["confirmed_actions"] == 0
        assert len(result["candidates"]) == 6
        assert result["candidates"][0]["tcp_frame"] == "left_ge102_contact_link"
        stages = result["candidates"][0]["stages"]
        assert [stage["stage_id"] for stage in stages] == [
            "pregrasp",
            "insert",
            "close",
            "test_lift",
            "raise_for_shake",
            "shake",
            "lower",
            "release",
            "retreat",
        ]
        assert len(stages[5]["tcp_waypoints_base"]) == 6
        assert stages[3]["completion_evidence"] == "target_followed_off_table"
        assert stages[1]["allowed_target_contacts"] == ()
        assert stages[2]["allowed_target_contacts"] == TOOL_FRAMES["galbot_ge102in"][1]
        tilted = result["candidates"][-1]
        assert tilted["grasp_sample"]["pitch_rad"] > 0
        start, end = [row["tcp_waypoints_base"][0] for row in tilted["stages"][:2]]
        np.testing.assert_allclose(np.asarray(end[:3]) - start[:3], 0.08 * pose_matrix(end)[:3, 0])
        assert hub.run_history()[0]["summary"]["run_id"] == state["run_id"]
    finally:
        hub.close()


def test_tabletop_stop_does_not_wait_for_inflight_planning_call(tmp_path, config):
    capture = tmp_path / "capture"
    scene = _scene(capture)
    entered, release = threading.Event(), threading.Event()

    class Planner:
        evidence_level = "test"

        def review(self, snapshot, stages, checkpoint):
            entered.set()
            assert release.wait(3)
            checkpoint()

    hub = ControlHub(tmp_path, config)
    try:
        hub.start_tabletop(TaskRequest("grasp", object_id="b1"), capture, scene, planner=Planner())
        assert entered.wait(3)
        assert hub.command("stop")["status"] == "stopping"
        release.set()
        hub.wait()
        assert hub.snapshot()["summary"]["state"] == "cancelled"
    finally:
        release.set()
        hub.close()


def test_sdk_review_chains_start_states_and_keeps_collision_checks(tmp_path):
    planner = GalbotStagePlanner.__new__(GalbotStagePlanner)
    planner.sdk = SimpleNamespace(
        PlannerConfig=SimpleNamespace,
        RobotStates=SimpleNamespace,
        ActuateType=SimpleNamespace(ACTUATE_WITH_CHAIN_ONLY=0),
        Pose=list,
        MotionPlanChainTarget=lambda: SimpleNamespace(cart=SimpleNamespace()),
        MotionPlanTargetMode=SimpleNamespace(kCartesian=0),
        MotionStatus=SimpleNamespace(SUCCESS=0),
    )
    shape = BottleGeometry(
        (0.5, 0, 0.9), 0.025, 0.2, 0.9, (0.5, 0, 0.4), (1, 1, 0.8), tmp_path / "cloud.pcd", 0.008
    )
    model = "galbot_ge102in"
    tcp, fingers = TOOL_FRAMES[model]
    stages = bottle_stages(TaskRequest("grasp", object_id="b1"), shape, 0.0, tool_model=model)
    joint_names = {
        chain: [f"{chain}_{i}" for i in range(count)]
        for chain, count in zip(CHAINS, JOINT_COUNTS, strict=True)
    }
    snapshot = SceneSnapshot(
        "revision",
        "b1",
        1,
        1,
        dict.fromkeys(sum(joint_names.values(), []), 0.0),
        0.1,
        shape,
        True,
        model,
    )
    added, removed = [], []

    def pregrasp(waypoints, config, start):
        assert config.is_check_collision and config.enable_env_collision_check
        assert config.is_direct_execute is False
        assert all(q == 0.0 for q in start.whole_body_joint)
        assert waypoints[0][0].cart.pose == list(stages[0].tcp_waypoints_base[0])
        assert waypoints[0][0].cart.frame_id == tcp
        return 0, {ARM: [[0.1] * 7]}

    def insert(waypoints, config, start):
        assert config.is_check_collision and config.enable_env_collision_check
        assert config.is_direct_execute is False
        assert start.whole_body_joint[7:14] == [0.1] * 7
        return 1, {}

    def add_obstacle(obstacle_id, kind, *args, **kwargs):
        added.append((obstacle_id, kind))
        return 0

    def remove_obstacle(obstacle_id):
        removed.append(obstacle_id)
        return 0

    planner.motion = SimpleNamespace(
        get_chain_joint_names=joint_names.__getitem__,
        get_link_names=lambda: [tcp, *fingers],
        add_obstacle=add_obstacle,
        remove_obstacle=remove_obstacle,
        motion_plan=pregrasp,
        move_line=insert,
        forward_kinematics_by_state=lambda *args: (0, stages[0].tcp_waypoints_base[0]),
    )
    results = planner.review(snapshot, stages, lambda: None)
    assert [row.status for row in results] == ["planned", "rejected", "blocked", "blocked"]
    assert results[0].planned_tcp_pose_base == stages[0].tcp_waypoints_base[0]
    assert results[0].goal_position_error_m == 0
    assert results[0].goal_orientation_error_rad == 0
    assert [kind for _, kind in added] == ["box", "point_cloud", "cylinder"]
    assert removed == [obstacle_id for obstacle_id, _ in reversed(added)]


def test_tabletop_simulation_verifies_feedback_and_stops_before_unconfirmed_lift(tmp_path, config):
    capture = tmp_path / "capture"
    scene = _scene(capture)
    task = TaskRequest("shake_then_place", object_id="b1", repeat_count=2)
    hub = ControlHub(tmp_path, config)
    try:
        hub.start_tabletop(task, capture, scene, simulation_fault="none")
        hub.wait()
        snapshot = hub.snapshot()
        result = snapshot["summary"]
        assert snapshot["mode"] == "g1_simulation"
        assert result["success"] and result["simulated_verified_stages"] == 9
        assert result["shake_cycles"] == 2 and result["motion_commands_sent"] == 0
        rows = result["simulated_stage_results"]
        assert rows[3]["bottle_lift_m"] > 0.02
        assert abs(rows[-1]["bottle_lift_m"]) < 1e-9 and rows[-1]["held_object"] is None

        hub.start_tabletop(task, capture, scene, simulation_fault="grasp_unconfirmed")
        hub.wait()
        failed = hub.snapshot()["summary"]
        assert not failed["success"] and failed["state"] == "held"
        assert failed["shake_cycles"] == 0 and failed["motion_commands_sent"] == 0
        assert [row["stage_id"] for row in failed["simulated_stage_results"]] == [
            "pregrasp",
            "insert",
            "close",
        ]
        assert failed["simulated_stage_results"][-1]["status"] == "rejected"
    finally:
        hub.close()
