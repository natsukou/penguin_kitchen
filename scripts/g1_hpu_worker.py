"""G1 SDK 1.10 HPU process. Runs on the Orin system Python (3.8)."""

import argparse
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

WIRE = "PETMEAL-G1/1 "
ARM = "left_arm"
GRIPPER = "left_gripper"
TOOL = "galbot_ge102in"
TOOL_LINKS = {
    "left_ge102_contact_link",
    "left_ge102_left_active_link2",
    "left_ge102_right_active_link2",
}
ARM_NAMES = ["left_arm_joint%d" % index for index in range(1, 8)]
PREFIX = ["pregrasp", "insert", "close", "test_lift"]
SUFFIX = ["lower", "release", "retreat"]
STAGES = {
    "grasp": PREFIX,
    "pick_place": PREFIX + SUFFIX,
    "shake_then_place": PREFIX + ["raise_for_shake", "shake"] + SUFFIX,
}
AFTER = {
    "close": "target_held",
    "test_lift": "target_followed_off_table",
    "raise_for_shake": "target_held",
    "shake": "target_held",
    "lower": "target_supported",
    "release": "target_released_at_origin",
}
GRIP_STAGES = {"close", "release"}
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SHA = re.compile(r"[0-9a-f]{64}")
RUN_ID = re.compile(r"[0-9a-f]{32}")


class Rejected(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise Rejected(reason)


def matches(pattern, value):
    return isinstance(value, str) and bool(pattern.fullmatch(value))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def fresh(stamp, age_ns=500_000_000, after_ns=0):
    return type(stamp) is int and stamp > after_ns and 0 <= time.time_ns() - stamp <= age_ns


def group_status(robot, sdk):
    rows = robot.check_trajectory_execution_status([ARM])
    require(len(rows) == 1, "XCU did not return exactly one arm trajectory status")
    require(
        rows[0]
        in (
            sdk.TrajectoryControlStatus.RUNNING,
            sdk.TrajectoryControlStatus.COMPLETED,
        ),
        "XCU arm trajectory status: " + str(rows[0]),
    )
    return rows[0]


def read_arm(robot, after_ns=0):
    rows = robot.get_joint_states([], ARM_NAMES)
    require([row.joint_name for row in rows] == ARM_NAMES, "Unexpected arm joint readback order")
    require(
        all(
            fresh(row.timestamp_ns, after_ns=after_ns)
            and finite(row.position)
            and finite(row.velocity)
            for row in rows
        ),
        "Missing or stale arm state",
    )
    return {
        "joint_names": ARM_NAMES,
        "positions_rad": [row.position for row in rows],
        "velocities_rad_s": [row.velocity for row in rows],
        "timestamp_ns": min(row.timestamp_ns for row in rows),
    }


def read_gripper(robot, after_ns=0):
    state = robot.get_gripper_state(GRIPPER)
    require(
        state is not None
        and fresh(state.timestamp_ns, after_ns=after_ns)
        and finite(state.width)
        and 0 <= state.width <= 0.12
        and finite(state.effort),
        "Missing or stale left-gripper state",
    )
    return {
        "width_m": state.width,
        "effort_nm": state.effort,
        "moving": bool(state.is_moving),
        "timestamp_ns": state.timestamp_ns,
    }


def query_xcu(profile, password):
    """Use the official read-only q/v commands through a host-key-verified PTY."""
    import paramiko

    host, user = profile["xcu_host"], profile["xcu_user"]
    with paramiko.SSHClient() as client:
        client.load_system_host_keys()
        client.connect(
            host,
            username=user,
            password=password,
            timeout=4,
            auth_timeout=4,
            look_for_keys=password is None,
            allow_agent=password is None,
        )
        channel = client.get_transport().open_session(timeout=4)
        try:
            channel.get_pty()
            channel.exec_command("cd /data/bin && LD_LIBRARY_PATH=/data/lib ./sys_tool")
            end = time.monotonic() + 7
            next_query = time.monotonic() + 0.8
            transcript = ""
            serial = profile["robot_sn"]
            while time.monotonic() < end:
                if channel.recv_ready():
                    transcript = (transcript + channel.recv(8192).decode(errors="replace"))[-65536:]
                modes = re.findall(
                    r"mode_type\s*:\s*([A-Z_]+)\s+mode_switch_state\s*\{\s*status\s*:\s*([A-Z_]+)",
                    transcript,
                )
                if modes and re.search(
                    r"(?<![\w-])" + re.escape(serial) + r"(?![\w-])", transcript
                ):
                    return {
                        "robot_sn": serial,
                        "mode": modes[-1][0],
                        "switch_status": modes[-1][1],
                        "xcu_host": host,
                        "source": "xcu_sys_tool",
                        "timestamp_ns": time.time_ns(),
                    }
                if time.monotonic() >= next_query:
                    channel.sendall(b"q\n")
                    channel.sendall(b"v\n")
                    next_query += 1.3
                time.sleep(0.08)
        finally:
            channel.close()
    raise Rejected("XCU sys_tool did not report the expected serial and mode")


def validate_plan(plan, request, profile):
    require(plan.get("schema_version") == 1, "Unsupported HPU plan schema")
    require(
        plan.get("robot_sn") == request["robot_sn"] == profile["robot_sn"]
        and plan.get("tool_model") == TOOL
        and plan.get("joint_names") == ARM_NAMES,
        "Plan robot, E102 tool or joint order does not match HPU",
    )
    require(
        matches(SHA, plan.get("scene_revision"))
        and matches(ID, plan.get("candidate_id"))
        and matches(ID, plan.get("object_id"))
        and matches(ID, plan.get("review_id")),
        "Plan lacks scene, candidate, target or review identity",
    )
    expiry = plan.get("expires_ns")
    require(
        type(expiry) is int and 0 < expiry - time.time_ns() <= 600_000_000_000,
        "Reviewed plan must be current and expire within ten minutes",
    )
    stages = plan.get("stages")
    require(
        isinstance(stages, list)
        and all(isinstance(stage, dict) for stage in stages)
        and [stage.get("stage_id") for stage in stages] == STAGES.get(plan.get("skill")),
        "Plan does not contain the complete ordered stages for its skill",
    )
    limits = profile["joint_limits_rad"]
    require(
        len(limits) == 7
        and all(
            len(pair) == 2 and all(finite(v) for v in pair) and pair[0] < pair[1] for pair in limits
        ),
        "HPU profile requires seven calibrated joint limits",
    )
    speed = profile["max_joint_speed_rad_s"]
    require(finite(speed) and 0 < speed <= 0.2, "HPU commissioning speed outside supported bound")
    for stage in stages:
        if stage["stage_id"] in GRIP_STAGES:
            require(
                set(stage) == {"stage_id", "width_m", "speed_mps", "effort_nm", "accept_width_m"},
                "Invalid reviewed gripper segment fields",
            )
            width, velocity, effort = (stage[k] for k in ("width_m", "speed_mps", "effort_nm"))
            accept = stage["accept_width_m"]
            require(
                all(finite(value) for value in (width, velocity, effort))
                and 0 <= width <= 0.12
                and 0 < velocity <= profile["max_gripper_speed_mps"] <= 0.2
                and 0 < effort <= profile["max_gripper_effort_nm"] <= 100
                and isinstance(accept, list)
                and len(accept) == 2
                and all(finite(value) for value in accept)
                and 0 <= accept[0] <= accept[1] <= 0.12,
                "Gripper segment exceeds the reviewed HPU tool limits",
            )
            continue
        require(set(stage) == {"stage_id", "points"}, "Invalid reviewed trajectory fields")
        points = stage["points"]
        require(isinstance(points, list) and 2 <= len(points) <= 1000, "Invalid trajectory size")
        previous = None
        for point in points:
            require(
                isinstance(point, list)
                and len(point) == 8
                and all(finite(v) for v in point)
                and all(lo <= value <= hi for value, (lo, hi) in zip(point[1:], limits))
                and 0.02 <= point[0] <= 30,
                "Trajectory point violates time, shape or joint limits",
            )
            if previous is not None:
                interval = point[0] - previous[0]
                require(
                    interval >= 0.02 - 1e-9
                    and max(abs(a - b) for a, b in zip(point[1:], previous[1:]))
                    <= speed * interval,
                    "Trajectory waypoint order or segment speed exceeds reviewed limits",
                )
            previous = point


def make_trajectory(sdk, rows):
    trajectory = sdk.Trajectory()
    trajectory.joint_names = ARM_NAMES
    points = []
    for stamp, *positions in rows:
        point = sdk.TrajectoryPoint()
        point.time_from_start_second = stamp
        commands = []
        for position in positions:
            command = sdk.JointCommand()
            command.position = position
            commands.append(command)
        point.joint_command_vec = commands
        points.append(point)
    trajectory.points = points
    return trajectory


def scene_evidence(profile, context, expected, after_ns=0):
    """Only an HPU-configured verifier supplies live scene/object evidence."""
    command = profile.get("verifier_argv")
    require(
        isinstance(command, list) and command and Path(command[0]).is_absolute(),
        "No HPU live scene verifier is configured",
    )
    result = subprocess.run(
        command,
        input=json.dumps({**context, "check": expected, "after_ns": after_ns}),
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    row = json.loads(result.stdout)
    require(
        all(row.get(key) == context[key] for key in ("command_id", "scene_revision", "object_id"))
        and row.get("check") == expected
        and row.get("source") in ("live_rgbd", "onsite_observed")
        and row.get("passed") is True
        and matches(ID, row.get("evidence_id"))
        and fresh(row.get("timestamp_ns"), 3_000_000_000, after_ns),
        "HPU scene verifier returned stale, unbound or negative evidence",
    )
    return {
        key: row[key]
        for key in (
            "check",
            "source",
            "passed",
            "evidence_id",
            "timestamp_ns",
        )
    }


class Session:
    def __init__(self, request):
        self.request = request
        self.last_heartbeat = time.monotonic()
        self.cancelled = False
        self.sent = 0
        self.events = None

    def read_control(self):
        try:
            for line in sys.stdin:
                row = json.loads(line)
                if row == {"operation": "cancel"}:
                    break
                if row != {"operation": "heartbeat"}:
                    break
                self.last_heartbeat = time.monotonic()
        finally:
            self.cancelled = True

    def checkpoint(self):
        require(
            not self.cancelled and time.monotonic() - self.last_heartbeat <= 2.0,
            "HPU session cancelled or heartbeat lost",
        )

    def emit(self, event, payload):
        row = {
            "type": "event",
            "run_id": self.request["run_id"],
            "event": "hpu_" + event,
            "payload": {**payload, "hpu_timestamp_ns": time.time_ns()},
        }
        if self.events is not None:
            self.events.write(json.dumps(row, allow_nan=False) + "\n")
            self.events.flush()
            os.fsync(self.events.fileno())
        write_frame(row)


def write_frame(row):
    # Native SDK messages also use stdout. The client only consumes these frames.
    print(WIRE + json.dumps(row, allow_nan=False, separators=(",", ":")), flush=True)


def execute_stage(session, robot, sdk, profile, stage, context):
    stage_id = stage["stage_id"]
    session.emit(
        "stage_checked",
        {
            **context,
            "scene_evidence": scene_evidence(profile, context, "scene_clear"),
        },
    )
    session.checkpoint()
    is_grip = stage_id in GRIP_STAGES
    arm = read_arm(robot)
    require(max(map(abs, arm["velocities_rad_s"])) <= 0.02, "Arm is not stationary")
    if not is_grip:
        points = stage["points"]
        start_delta = max(abs(a - b) for a, b in zip(arm["positions_rad"], points[0][1:]))
        require(
            start_delta <= min(0.01, profile["max_joint_speed_rad_s"] * points[0][0]),
            "Live arm state differs from the reviewed trajectory start",
        )
        prior = robot.check_trajectory_execution_status([ARM])
        require(
            prior
            in (
                [sdk.TrajectoryControlStatus.COMPLETED],
                [sdk.TrajectoryControlStatus.STOPPED_UNREACHED],
            ),
            "XCU did not confirm the arm trajectory channel is idle",
        )
    else:
        require(not read_gripper(robot)["moving"], "Left gripper is already moving")
    session.checkpoint()
    sent_ns = time.time_ns()
    session.emit("dispatch_intent", {**context, "sent_ns": sent_ns})
    session.sent += 1
    try:
        if is_grip:
            status = robot.set_gripper_command(
                GRIPPER,
                stage["width_m"],
                stage["speed_mps"],
                stage["effort_nm"],
                False,
            )
        else:
            status = robot.execute_joint_trajectory(make_trajectory(sdk, points), False)
        session.emit("sdk_receipt", {**context, "sdk_status": str(status), "sent_ns": sent_ns})
        require(status == sdk.ControlStatus.SUCCESS, "SDK rejected the stage command")
        duration = 8 if is_grip else points[-1][0] + 2
        began = time.monotonic()
        while time.monotonic() - began < duration:
            session.checkpoint()
            if is_grip:
                actual = read_gripper(robot)
                low, high = stage["accept_width_m"]
                if (
                    time.monotonic() - began >= 0.5
                    and actual["timestamp_ns"] > sent_ns
                    and not actual["moving"]
                    and low <= actual["width_m"] <= high
                ):
                    break
            else:
                controller = group_status(robot, sdk)
                actual = read_arm(robot)
                if (
                    time.monotonic() - began >= points[-1][0]
                    and controller == sdk.TrajectoryControlStatus.COMPLETED
                    and actual["timestamp_ns"] > sent_ns
                    and max(map(abs, actual["velocities_rad_s"])) <= 0.02
                    and max(
                        abs(a - b)
                        for a, b in zip(
                            actual["positions_rad"],
                            points[-1][1:],
                        )
                    )
                    <= 0.01
                ):
                    actual["trajectory_status"] = str(controller)
                    break
            time.sleep(0.08)
        else:
            raise Rejected("XCU stage did not reach its measured endpoint before deadline")
        session.emit("motion_verified", {**context, "actual": actual})
        proof = None
        if stage_id in AFTER:
            proof = scene_evidence(profile, context, AFTER[stage_id], actual["timestamp_ns"])
        session.checkpoint()
        session.emit("stage_verified", {**context, "object_evidence": proof})
    except BaseException:
        # Galbot stop_trajectory_execution covers joint trajectories only.
        stop = "gripper_stop_unconfirmed" if is_grip else str(robot.stop_trajectory_execution())
        session.emit("stop_receipt", {**context, "sdk_status": stop})
        raise


def run_plan(profile, session, robot, sdk, password):
    request = session.request
    for key, pattern in (
        ("run_id", RUN_ID),
        ("plan_id", ID),
        ("plan_sha256", SHA),
        ("robot_sn", ID),
    ):
        require(matches(pattern, request.get(key)), "Invalid execution identity: " + key)
    require(profile.get("motion_enabled") is True, "HPU motion is disabled")
    path = Path(profile["plan_dir"]).resolve()
    plan_path = (path / (request["plan_id"] + ".json")).resolve()
    require(plan_path.parent == path, "Reviewed plan must be inside the HPU plan directory")
    raw = plan_path.read_bytes()
    require(
        len(raw) <= 2_000_000 and hashlib.sha256(raw).hexdigest() == request["plan_sha256"],
        "HPU plan does not match the requested SHA-256",
    )
    plan = json.loads(raw)
    validate_plan(plan, request, profile)
    motion = sdk.GalbotMotion()
    require(
        motion.init() and TOOL_LINKS <= set(motion.get_link_names()),
        "E102 planner links are unavailable on HPU",
    )
    require(motion.get_chain_joint_names(ARM) == ARM_NAMES, "SDK left-arm joint names differ")
    for stage in plan["stages"]:
        session.checkpoint()
        xcu = query_xcu(profile, password)
        require(
            xcu["mode"] == "WORKING_MODE" and xcu["switch_status"] == "SUCCESS",
            "XCU is not in confirmed WORKING_MODE",
        )
        require(time.time_ns() < plan["expires_ns"], "Reviewed HPU plan expired")
        context = {
            "run_id": request["run_id"],
            "robot_sn": xcu["robot_sn"],
            "xcu_host": xcu["xcu_host"],
            "xcu_mode": xcu["mode"],
            "xcu_timestamp_ns": xcu["timestamp_ns"],
            "scene_revision": plan["scene_revision"],
            "candidate_id": plan["candidate_id"],
            "object_id": plan["object_id"],
            "stage_id": stage["stage_id"],
            "tool_model": TOOL,
            "command_id": uuid.uuid4().hex,
            "plan_sha256": request["plan_sha256"],
        }
        execute_stage(session, robot, sdk, profile, stage, context)


def execute(profile, request, robot, sdk, password):
    import fcntl

    session = Session(request)

    def cancel(_signal, _frame):
        session.cancelled = True

    signal.signal(signal.SIGHUP, cancel)
    signal.signal(signal.SIGTERM, cancel)
    threading.Thread(target=session.read_control, daemon=True).start()
    state = Path(profile["state_dir"])
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (state / "dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = state / request["run_id"]
        require(not run.exists(), "HPU already received this run; reconcile its on-device journal")
        require(
            not any(
                directory.is_dir()
                and (
                    not (directory / "result.json").is_file()
                    or json.loads((directory / "result.json").read_text())["state"] == "unconfirmed"
                )
                for directory in state.iterdir()
                if matches(RUN_ID, directory.name)
            ),
            "HPU has an unconfirmed run; reconcile its on-device journal",
        )
        run.mkdir(mode=0o700)
        with (run / "events.jsonl").open("x", encoding="utf-8") as session.events:
            try:
                run_plan(profile, session, robot, sdk, password)
                result = {"state": "complete", "reason": "all_hpu_stages_verified"}
            except Exception as exc:
                result = {
                    "state": "unconfirmed" if session.sent else "held",
                    "reason": str(exc),
                }
            result["motion_commands_attempted"] = session.sent
            pending = run / "result.pending"
            with pending.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            pending.replace(run / "result.json")
            return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    args = parser.parse_args()
    request = {}
    robot = None
    xcu = None
    execution_entered = False
    try:
        request = json.loads(sys.stdin.readline(4096))
        require(isinstance(request, dict), "Expected a JSON request")
        password = request.pop("xcu_password", None)
        operation = request.get("operation")
        require(operation in ("probe", "execute"), "Unsupported HPU operation")
        if operation == "execute":
            require(matches(RUN_ID, request.get("run_id")), "Invalid run ID")
        profile = json.loads(args.profile.read_text(encoding="utf-8"))
        xcu = query_xcu(profile, password)
        if operation == "execute":
            require(
                profile.get("motion_enabled") is True
                and xcu["mode"] == "WORKING_MODE"
                and xcu["switch_status"] == "SUCCESS",
                "HPU or XCU mode gate rejected motion",
            )
        import galbot_sdk
        import galbot_sdk.g1 as sdk

        require(galbot_sdk.__version__ == "1.10.0", "Expected Galbot SDK 1.10.0")
        robot = sdk.GalbotRobot()
        require(robot.init(), "Galbot SDK robot initialization failed")
        device = robot.get_device_information()
        require(
            device.get("serial_number") == xcu["robot_sn"], "SDK device serial differs from XCU"
        )
        if operation == "probe":
            result = {
                "state": "ready",
                "hpu_host": socket.gethostname(),
                "xcu": xcu,
                "sdk_version": galbot_sdk.__version__,
                "device": device,
                "arm": read_arm(robot),
                "gripper": read_gripper(robot),
                "arm_trajectory_status": [
                    str(status) for status in robot.check_trajectory_execution_status([ARM])
                ],
                "motion_enabled": profile.get("motion_enabled") is True
                and xcu["mode"] == "WORKING_MODE"
                and xcu["switch_status"] == "SUCCESS",
            }
        else:
            execution_entered = True
            result = execute(profile, request, robot, sdk, password)
    except Exception as exc:
        result = {
            "state": "unconfirmed" if execution_entered else "held",
            "reason": str(exc),
            "xcu": xcu,
            "motion_enabled": False,
            "motion_commands_attempted": None if execution_entered else 0,
        }
    finally:
        # Send the persisted outcome before SDK teardown, which can block on middleware.
        write_frame({"type": "result", "run_id": request.get("run_id"), "result": result})
        if robot is not None:
            robot.request_shutdown()
            robot.wait_for_shutdown()
            robot.destroy()


if __name__ == "__main__":
    main()
