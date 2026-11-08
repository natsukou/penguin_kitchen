"""Rehearse the G1 bottle pipeline and audit real receipts without dispatching motion."""

import argparse
import gzip
import http.client
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

WORKSPACE = Path(__file__).resolve().parents[1]
RUNS = WORKSPACE / "artifacts/hub-runs"
CAPTURE = "g1-grasp-geometry-20261107T180009.730799Z"
REPLAY_OBSERVATION = "g1-task-offline-test/observation-simulated-empty-gripper.json"
CHOICE = f"{CAPTURE}/bailian-scene-selection/choice-import-fixture.json"
STAGES = (
    "pregrasp",
    "insert",
    "close",
    "test_lift",
    "raise_for_shake",
    "shake",
    "lower",
    "release",
    "retreat",
)
OBJECT_CHECKS = {
    "close": "target_held",
    "test_lift": "target_followed_off_table",
    "raise_for_shake": "target_held",
    "shake": "target_held",
    "lower": "target_supported",
    "release": "target_released_at_origin",
}
HPU_EVENTS = (
    "hpu_stage_checked",
    "hpu_dispatch_intent",
    "hpu_sdk_receipt",
    "hpu_motion_verified",
    "hpu_stage_verified",
)
JOINTS = [f"left_arm_joint{number}" for number in range(1, 8)]
TEST_ERRORS = (OSError, KeyError, ValueError, TypeError, RuntimeError)


class CheckError(RuntimeError):
    pass


def check(condition, message):
    if not condition:
        raise CheckError(message)


class LocalApi:
    def __init__(self, port):
        self.port = port
        self.token = None

    def request(self, path, body=None, expected=200):
        headers = {}
        if body is not None:
            headers = {"Content-Type": "application/json", "X-Control-Token": self.token}
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=27)
        try:
            connection.request(
                "GET" if body is None else "POST",
                path,
                body=None if body is None else json.dumps(body),
                headers=headers,
            )
            response = connection.getresponse()
            data = json.loads(response.read())
        finally:
            connection.close()
        check(
            response.status == expected,
            f"{path} returned HTTP {response.status}: {data.get('error', '')}",
        )
        return data

    def finish(self, receipt):
        run_id = receipt["run_id"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            run = self.request(f"/api/runs/{run_id}")
            if not run["active"]:
                check(run["run_id"] == run_id and run["summary"], "Run summary is missing")
                return run
            time.sleep(0.1)
        raise CheckError(f"Run {run_id} is still active")


def run_events(run_id):
    check(re.fullmatch("[0-9a-f]{32}", run_id), "Invalid run ID")
    path = RUNS / run_id / "events.jsonl"
    with path.open(encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream]
    check(events and all(row["run_id"] == run_id for row in events), "Run log ID mismatch")
    return events


def payload(events, name):
    matches = [row["payload"] for row in events if row["event"] == name]
    check(len(matches) == 1, f"Expected exactly one {name} event")
    return matches[0]


def inspect_decision(summary, events):
    resolved = payload(events, "task_target_resolved")
    target, geometry = resolved["target"], resolved["geometry"]
    check(
        target and target["category"] == "bottle" and target["object_id"] == summary["object_id"],
        "Recognition did not resolve the planned bottle instance",
    )
    check(
        geometry
        and geometry["frame_id"] == "base_link"
        and 0 <= geometry["max_camera_skew_ms"] <= 50
        and geometry["wrist_depth"]
        and all(0 <= row["difference_m"] <= 0.03 for row in geometry["wrist_depth"].values())
        and len(geometry["head_stereo"]["point_m"]) == 3
        and all(math.isfinite(v) for v in geometry["head_stereo"]["point_m"]),
        "Multiview localization did not pass the synchronization or geometry contract",
    )
    snapshot = payload(events, "task_snapshot")
    candidates = summary["candidates"]
    check(
        len(candidates) == 6
        and snapshot["revision"] == summary["scene_revision"]
        and snapshot["object_id"] == target["object_id"]
        and all(
            row["scene_revision"] == snapshot["revision"]
            and row["reference_frame"] == "base_link"
            and tuple(stage["stage_id"] for stage in row["stages"]) == STAGES
            for row in candidates
        ),
        "Decision did not generate the nine bottle stages from the resolved scene",
    )
    return target, geometry, snapshot


def detect_bottle(weights, capture, target):
    import cv2
    import numpy as np

    from pet_meal.yolo import predict_image

    directory = (WORKSPACE / "artifacts" / capture).resolve()
    metadata = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    frame = metadata["frames"]["head-left-rgb"]
    source = (directory / frame["file"]).resolve()
    check(source.is_relative_to(directory), "Head image is outside its capture directory")
    height, width = frame["height"], frame["width"]
    size = width * height * 3 // 2
    check(frame["format"].lower() == "nv12" and 0 < size <= 8_000_000, "Invalid head RGB format")
    with gzip.open(source, "rb") as stream:
        pixels = stream.read(size + 1)
    check(len(pixels) == size, "Head image payload does not match the NV12 dimensions")
    nv12 = np.frombuffer(pixels, dtype=np.uint8).reshape(height * 3 // 2, width)
    x, y = target["landmark"]["views"]["head-left-rgb"]["pixel_px"]
    with TemporaryDirectory() as scratch:
        image = Path(scratch) / "head-left.png"
        check(
            cv2.imwrite(str(image), cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)),
            "Could not decode captured head image",
        )
        prediction = predict_image(weights, image, Path(scratch) / "detected")
    check(
        prediction["input_shape"] == [frame["height"], frame["width"]],
        "Detector image dimensions differ from the referenced capture",
    )
    boxes = [
        item
        for item in prediction["detections"]
        if item["label"] == "bottle"
        and item["box_xyxy_px"][0] <= x <= item["box_xyxy_px"][2]
        and item["box_xyxy_px"][1] <= y <= item["box_xyxy_px"][3]
    ]
    return {
        "status": "pass" if boxes else "fail",
        "source": "recorded_image_local_yolo",
        "image": str(source),
        "landmark_px": [x, y],
        "detections": prediction["detections"],
        "matched_bottle": max(boxes, key=lambda row: row["confidence"], default=None),
    }


def rehearse(api, args, report):
    task = api.request("/api/tabletop/interpret", {"text": "抓起水瓶，摇一下再放回"})["task"]
    check(
        task
        == {
            "skill": "shake_then_place",
            "object_id": None,
            "category": "bottle",
            "repeat_count": 1,
        },
        "Text did not resolve to the intended bottle task",
    )
    report["task"] = task
    request = {"task": task, "capture": args.capture, "observation": args.observation}
    if args.vision_choice:
        request["vision_choice"] = args.vision_choice
    path = "/api/tabletop/plan"
    preview = api.finish(api.request(path, {**request, "request_id": uuid.uuid4().hex}, 202))
    summary = preview["summary"]
    report["decision_run_id"] = preview["run_id"]
    check(
        preview["mode"] == "g1_plan_only"
        and preview["status"] == "preview"
        and summary["dispatch_allowed"] is False
        and summary["motion_commands_sent"] == 0,
        f"Planning preview was blocked: {summary.get('reason')} {summary.get('detail', '')}",
    )
    events = run_events(preview["run_id"])
    target, geometry, snapshot = inspect_decision(summary, events)
    report["recognition"] = {
        "status": "pass",
        "source": "recorded_scene_observation",
        "object_id": target["object_id"],
        "choice_source": "imported" if args.vision_choice else "scene_category",
    }
    if args.weights:
        try:
            report["image_recognition"] = detect_bottle(args.weights, args.capture, target)
        except (*TEST_ERRORS, ImportError) as exc:
            report["image_recognition"] = {"status": "fail", "reason": str(exc)}
    report["processing"] = {
        "status": "pass",
        "scene_revision": snapshot["revision"],
        "geometry": geometry,
    }
    report["decision"] = {
        "status": "pass",
        "candidates": len(summary["candidates"]),
        "stages": STAGES,
    }

    path = "/api/tabletop/simulate"
    command = {**request, "fault": "none", "request_id": uuid.uuid4().hex}
    run = api.finish(api.request(path, command, 202))
    replay = api.request(path, command, 202)
    result = run["summary"]
    check(
        run["mode"] == "g1_simulation"
        and run["status"] == "complete"
        and result["scene_revision"] == snapshot["revision"]
        and result["motion_commands_sent"] == 0
        and replay["replayed"] is True
        and replay["run_id"] == run["run_id"],
        "Simulation receipt, scene binding, or idempotent retry failed",
    )
    report["control"] = {
        "status": "pass",
        "source": "simulation_api",
        "run_id": run["run_id"],
        "replay_verified": True,
    }
    rows = result["simulated_stage_results"]
    check(
        tuple(row["stage_id"] for row in rows) == STAGES
        and all(row["status"] == "verified" for row in rows)
        and rows[2]["held_object"] == target["object_id"]
        and rows[3]["bottle_lift_m"] >= 0.015
        and rows[5]["shake_cycles"] == 1
        and rows[7]["held_object"] is None
        and math.dist(rows[7]["bottle_position_m"], rows[0]["bottle_position_m"]) <= 1e-5,
        "Simulated grasp, lift, shake or return evidence is missing",
    )
    held = api.finish(
        api.request(
            path,
            {
                **command,
                "fault": "grasp_unconfirmed",
                "request_id": uuid.uuid4().hex,
            },
            202,
        )
    )
    failed = held["summary"]
    check(
        held["status"] == "held"
        and failed["reason"] == "simulated_stage_unconfirmed"
        and [row["stage_id"] for row in failed["simulated_stage_results"]] == list(STAGES[:3])
        and failed["simulated_stage_results"][-1]["status"] == "rejected"
        and failed["shake_cycles"] == 0
        and failed["motion_commands_sent"] == 0,
        "Unconfirmed grasp advanced beyond close",
    )
    report["action"] = {
        "status": "pass",
        "source": "simulation_feedback",
        "verified_stages": len(rows),
        "lift_m": rows[3]["bottle_lift_m"],
        "shake_cycles": rows[5]["shake_cycles"],
        "failed_grasp_run_id": held["run_id"],
        "failed_grasp_stopped_at": "close",
    }


def inspect_hardware(api):
    probe = api.request("/api/hardware/probe", {})
    check(
        probe["state"] == "ready", f"HPU probe did not return live feedback: {probe.get('reason')}"
    )
    xcu, device, arm, gripper = (probe[key] for key in ("xcu", "device", "arm", "gripper"))
    check(
        xcu["robot_sn"]
        and xcu["robot_sn"] == device["serial_number"]
        and arm["joint_names"] == JOINTS
        and len(arm["positions_rad"]) == len(JOINTS)
        and all(math.isfinite(position) for position in arm["positions_rad"])
        and 0 <= gripper["width_m"] <= 0.12,
        "HPU/XCU identity or measured joint/gripper response is invalid",
    )
    arm_status = probe["arm_trajectory_status"]
    blockers = []
    if xcu["mode"] != "WORKING_MODE" or xcu["switch_status"] != "SUCCESS":
        blockers.append("xcu_working_mode_unconfirmed")
    if arm_status not in (
        ["TrajectoryControlStatus.COMPLETED"],
        ["TrajectoryControlStatus.STOPPED_UNREACHED"],
    ):
        blockers.append("arm_idle_channel_unconfirmed")
    if probe["motion_enabled"] is not True:
        blockers.append("motion_disabled")
    return {
        "readback": "pass",
        "robot_sn": xcu["robot_sn"],
        "sdk_version": probe["sdk_version"],
        "xcu_mode": xcu["mode"],
        "arm_trajectory_status": arm_status,
        "gripper_width_m": gripper["width_m"],
        "blockers": blockers,
        "reviewed_plan_and_live_verifier": "not_checked_by_probe",
        "physical_actions": "not_run",
    }


def audit(api, args, report):
    preview = api.request(f"/api/runs/{args.decision_run_id}")
    real = api.request(f"/api/runs/{args.hardware_run_id}")
    check(preview["status"] == "preview", "The decision run did not complete a preview")
    summary = real["summary"]
    check(
        real["mode"] == "g1_hardware"
        and real["status"] == "complete"
        and summary["evidence_level"] == "galbot_hpu_xcu"
        and summary["motion_commands_attempted"] == len(STAGES)
        and summary["motion_commands_accepted"] == summary["confirmed_stages"] == len(STAGES),
        "Physical run has not completed all nine stages",
    )
    target, _, snapshot = inspect_decision(preview["summary"], run_events(preview["run_id"]))
    check(
        snapshot["geometry_bound_to_capture"] is True
        and snapshot["tool_model"] == "galbot_ge102in",
        "Physical audit requires geometry tied to the capture and the E102 tool",
    )
    events = run_events(real["run_id"])
    receipts = [row for row in events if row["event"] in HPU_EVENTS]
    check(
        all(row["evidence_level"] == "galbot_hpu_xcu" for row in receipts)
        and not any(row["event"] == "hpu_stop_receipt" for row in events)
        and [row["event"] for row in receipts] == list(HPU_EVENTS) * len(STAGES),
        "Physical receipts are incomplete, out of order or contain a stop",
    )
    candidates = {row["candidate_id"] for row in preview["summary"]["candidates"]}
    command_ids = set()
    candidate_id = receipts[0]["payload"]["candidate_id"]
    check(candidate_id in candidates, "Executed candidate is absent from the decision run")
    for i, stage in enumerate(STAGES):
        start = i * len(HPU_EVENTS)
        checked, dispatched, sdk, measured, verified = (
            row["payload"] for row in receipts[start : start + len(HPU_EVENTS)]
        )
        context = {
            "run_id": real["run_id"],
            "robot_sn": summary["robot_sn"],
            "plan_sha256": summary["plan_sha256"],
            "scene_revision": snapshot["revision"],
            "candidate_id": candidate_id,
            "object_id": target["object_id"],
            "stage_id": stage,
            "command_id": dispatched["command_id"],
            "xcu_mode": "WORKING_MODE",
            "tool_model": snapshot["tool_model"],
        }
        check(
            all(
                all(row[key] == value for key, value in context.items())
                for row in (checked, dispatched, sdk, measured, verified)
            )
            and context["command_id"]
            and context["command_id"] not in command_ids
            and sdk["sdk_status"] == "ControlStatus.SUCCESS"
            and sdk["sent_ns"] == dispatched["sent_ns"] > snapshot["capture_timestamp_ns"],
            f"{stage}: scene or command association failed",
        )
        command_ids.add(context["command_id"])
        before, actual, after = (
            checked["scene_evidence"],
            measured["actual"],
            verified["object_evidence"],
        )
        check(
            before["check"] == "scene_clear"
            and before["passed"] is True
            and before["evidence_id"]
            and before["source"] in ("live_rgbd", "onsite_observed")
            and before["timestamp_ns"] <= dispatched["sent_ns"] < actual["timestamp_ns"],
            f"{stage}: missing clearance evidence or fresh measured feedback",
        )
        if stage in ("close", "release"):
            check(
                0 <= actual["width_m"] <= 0.12 and actual["moving"] is False,
                f"{stage}: no stationary gripper readback",
            )
        else:
            check(
                actual["trajectory_status"] == "TrajectoryControlStatus.COMPLETED"
                and actual["joint_names"] == JOINTS
                and len(actual["positions_rad"]) == len(JOINTS)
                and all(math.isfinite(value) for value in actual["positions_rad"]),
                f"{stage}: no completed XCU trajectory with measured joint positions",
            )
        if stage in OBJECT_CHECKS:
            check(
                after
                and after["check"] == OBJECT_CHECKS[stage]
                and after["passed"] is True
                and after["timestamp_ns"] > actual["timestamp_ns"]
                and after["evidence_id"]
                and after["source"] in ("live_rgbd", "onsite_observed"),
                f"{stage}: missing linked post-action object evidence",
            )
    report["hardware_audit"] = {
        "status": "pass",
        "decision_run_id": preview["run_id"],
        "hardware_run_id": real["run_id"],
        "verified_stages": len(command_ids),
        "scene_revision": snapshot["revision"],
        "recognition_source": "decision_run_observation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--capture", default=CAPTURE, help="Path relative to artifacts")
    parser.add_argument(
        "--observation", default=REPLAY_OBSERVATION, help="Path relative to artifacts"
    )
    parser.add_argument(
        "--vision-choice", default=CHOICE, help="Path relative to artifacts, or empty"
    )
    parser.add_argument(
        "--weights", type=Path, help="Optional local YOLO weights for the head image"
    )
    parser.add_argument("--output", type=Path, help="Write report to a new JSON file")
    parser.add_argument("--decision-run-id", help="Decision preview run ID for post-run audit")
    parser.add_argument("--hardware-run-id", help="Physical run ID for post-run audit")
    parser.add_argument(
        "--require-live-ready", action="store_true", help="Exit 2 on probe blockers"
    )
    args = parser.parse_args()
    if bool(args.decision_run_id) != bool(args.hardware_run_id):
        parser.error("Post-run audit requires both run IDs")
    if args.output and args.output.exists():
        parser.error("Report output already exists")
    api = LocalApi(args.port)
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "port": args.port,
        "test_dispatches_physical_motion": False,
    }
    exit_code = 0
    try:
        status = api.request("/api/status")
        api.token = status["control_token"]
        if args.hardware_run_id:
            audit(api, args, report)
        else:
            check(
                not status["active"] and not status["stop_latched"], "Hub is active or stop-latched"
            )
            if status["hardware_transport_configured"]:
                try:
                    report["live_hardware"] = inspect_hardware(api)
                except TEST_ERRORS as exc:
                    report["live_hardware"] = {
                        "readback": "fail",
                        "blockers": ["hpu_probe_failed"],
                        "reason": str(exc),
                    }
            else:
                report["live_hardware"] = {
                    "readback": "not_run",
                    "blockers": ["hpu_not_configured"],
                }
            rehearse(api, args, report)
            if report.get("image_recognition", {}).get("status") == "fail":
                exit_code = 1
            elif args.require_live_ready and report["live_hardware"]["blockers"]:
                exit_code = 2
    except TEST_ERRORS as exc:
        report["error"] = str(exc)
        exit_code = 1
    report["exit_code"] = exit_code
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(serialized, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
