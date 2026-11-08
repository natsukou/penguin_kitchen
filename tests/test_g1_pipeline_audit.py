from types import SimpleNamespace

import pytest

from scripts import test_g1_pipeline as pipeline


def test_real_audit_requires_correlated_fresh_object_feedback(monkeypatch):
    preview_id, real_id = "1" * 32, "2" * 32
    candidate, revision, robot, digest = "side_2", "a" * 64, "robot", "b" * 64
    preview = {
        "run_id": preview_id,
        "status": "preview",
        "summary": {"candidates": [{"candidate_id": candidate}]},
    }
    real = {
        "run_id": real_id,
        "mode": "g1_hardware",
        "status": "complete",
        "summary": {
            "evidence_level": "galbot_hpu_xcu",
            "motion_commands_attempted": 9,
            "motion_commands_accepted": 9,
            "confirmed_stages": 9,
            "robot_sn": robot,
            "plan_sha256": digest,
        },
    }
    snapshot = {
        "revision": revision,
        "capture_timestamp_ns": 10,
        "geometry_bound_to_capture": True,
        "tool_model": "galbot_ge102in",
    }
    target = {"object_id": "bottle_01"}
    monkeypatch.setattr(pipeline, "inspect_decision", lambda *_: (target, {}, snapshot))
    events = []
    for n, stage in enumerate(pipeline.STAGES):
        context = {
            "run_id": real_id,
            "robot_sn": robot,
            "plan_sha256": digest,
            "scene_revision": revision,
            "candidate_id": candidate,
            "object_id": "bottle_01",
            "stage_id": stage,
            "command_id": f"command-{n}",
            "xcu_mode": "WORKING_MODE",
            "tool_model": snapshot["tool_model"],
        }
        stamp = 100 + 10 * n
        before = {
            "check": "scene_clear",
            "source": "live_rgbd",
            "passed": True,
            "timestamp_ns": stamp - 1,
            "evidence_id": "clearance",
        }
        actual = {"timestamp_ns": stamp + 1}
        if stage in ("close", "release"):
            actual.update(width_m=0.06, moving=False)
        else:
            actual.update(
                trajectory_status="TrajectoryControlStatus.COMPLETED",
                joint_names=pipeline.JOINTS,
                positions_rad=[0.1] * 7,
            )
        after = {
            "check": pipeline.OBJECT_CHECKS.get(stage),
            "source": "live_rgbd",
            "passed": True,
            "timestamp_ns": stamp + 2,
            "evidence_id": "post-action-frame",
        }
        for name, fields in zip(
            pipeline.HPU_EVENTS,
            (
                {"scene_evidence": before},
                {"sent_ns": stamp},
                {"sent_ns": stamp, "sdk_status": "ControlStatus.SUCCESS"},
                {"actual": actual},
                {"object_evidence": after},
            ),
        ):
            events.append(
                {
                    "evidence_level": "galbot_hpu_xcu",
                    "event": name,
                    "payload": {**context, **fields},
                }
            )
    monkeypatch.setattr(pipeline, "run_events", lambda run_id: events if run_id == real_id else [])
    api = SimpleNamespace(request=lambda path: real if path.endswith(real_id) else preview)
    args = SimpleNamespace(decision_run_id=preview_id, hardware_run_id=real_id)
    report = {}
    pipeline.audit(api, args, report)
    assert report["hardware_audit"]["verified_stages"] == 9

    release = events[7 * len(pipeline.HPU_EVENTS) + 4]["payload"]
    release["object_evidence"]["timestamp_ns"] -= 1
    with pytest.raises(pipeline.CheckError, match="release: missing linked post-action"):
        pipeline.audit(api, args, {})

    real["mode"] = "g1_simulation"
    with pytest.raises(pipeline.CheckError, match="Physical run has not completed"):
        pipeline.audit(api, args, {})
