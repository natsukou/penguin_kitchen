import time
from types import SimpleNamespace

import pytest

from scripts import g1_hpu_worker as worker


def test_unreviewed_and_fast_trajectories_are_rejected():
    points = [[0.02] + [0] * 7, [0.04] + [0] * 7]
    gripper = {
        "stage_id": "close", "width_m": 0.06, "speed_mps": 0.01, "effort_nm": 2,
        "accept_width_m": [0.06, 0.09],
    }
    plan = {
        "schema_version": 1, "robot_sn": "test", "tool_model": worker.TOOL,
        "joint_names": worker.ARM_NAMES, "scene_revision": "0" * 64,
        "candidate_id": "side_1", "object_id": "bottle_01", "review_id": "review",
        "skill": "grasp", "expires_ns": time.time_ns() + 60_000_000_000,
        "stages": [
            {"stage_id": "pregrasp", "points": points},
            {"stage_id": "insert", "points": points},
            gripper,
            {"stage_id": "test_lift", "points": points},
        ],
    }
    request = {"robot_sn": "test"}
    profile = {
        **request, "joint_limits_rad": [[-2, 2]] * 7, "max_joint_speed_rad_s": 0.1,
        "max_gripper_speed_mps": 0.02, "max_gripper_effort_nm": 4,
    }
    worker.validate_plan(plan, request, profile)
    with pytest.raises(worker.Rejected, match="complete ordered stages"):
        worker.validate_plan({**plan, "stages": plan["stages"][:2]}, request, profile)
    points[1][1] = 0.1
    with pytest.raises(worker.Rejected, match="segment speed"):
        worker.validate_plan(plan, request, profile)


def test_trajectory_assigns_sdk_lists_and_explicit_joint_order():
    sdk = SimpleNamespace(
        Trajectory=SimpleNamespace, TrajectoryPoint=SimpleNamespace, JointCommand=SimpleNamespace,
    )
    trajectory = worker.make_trajectory(sdk, [[0.02, *range(7)], [0.04, *range(1, 8)]])
    assert trajectory.joint_names == worker.ARM_NAMES
    assert trajectory.points[1].time_from_start_second == 0.04
    assert [command.position for command in trajectory.points[1].joint_command_vec] == list(
        range(1, 8)
    )


def test_scene_verifier_requires_fresh_evidence_for_the_dispatched_command(monkeypatch, tmp_path):
    context = {"command_id": "command", "object_id": "bottle", "scene_revision": "0" * 64}
    row = {
        **context, "check": "target_held", "source": "live_rgbd", "passed": True,
        "evidence_id": "image", "timestamp_ns": time.time_ns(),
    }
    monkeypatch.setattr(
        worker.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=worker.json.dumps(row)),
    )
    profile = {"verifier_argv": [str(tmp_path / "verify")]}
    assert worker.scene_evidence(profile, context, "target_held")["passed"] is True
    with pytest.raises(worker.Rejected, match="stale"):
        worker.scene_evidence(profile, context, "target_held", row["timestamp_ns"])
    row["command_id"] = "another-command"
    with pytest.raises(worker.Rejected, match="unbound"):
        worker.scene_evidence(profile, context, "target_held")
