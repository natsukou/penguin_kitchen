import http.client
import json
import threading

import pytest

from pet_meal.g1_task import TaskRequest
from pet_meal.hub import ControlHub
from pet_meal.server import HubServer


@pytest.fixture
def server(tmp_path, config):
    hub = ControlHub(tmp_path, config)
    server = HubServer(hub, 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    yield server
    server.shutdown()
    hub.close()
    server.server_close()
    thread.join(2)


def request(server, method, path, payload=None, token=None, origin=None, host=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=4)
    headers = {}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-Control-Token"] = token
    if origin:
        headers["Origin"] = origin
    if host:
        headers["Host"] = host
    connection.request(method, path, json.dumps(payload) if payload is not None else None, headers)
    response = connection.getresponse()
    data = response.read()
    result = response.status, data
    connection.close()
    return result


def test_dashboard_static_and_status(server):
    assert request(server, "GET", "/")[0] == 200
    assert request(server, "GET", "/app.js")[0] == 200
    assert request(server, "GET", "/style.css")[0] == 200
    status, body = request(server, "GET", "/api/status")
    assert status == 200
    assert json.loads(body)["hardware_motion_enabled"] is False


def test_relevance_reports_are_read_only_and_distinguish_missing_data(server):
    status, body = request(server, "GET", "/api/bailian-vision")
    assert status == 200 and json.loads(body) == {"available": False}
    directory = server.hub.workspace / "artifacts/relevance"
    directory.mkdir(parents=True)
    (directory / "fixtures.json").write_text('{"source":"offline_fixtures","passed":6}')
    status, body = request(server, "GET", "/api/relevance-test")
    assert status == 200 and json.loads(body)["source"] == "offline_fixtures"
    assert not server.hub.snapshot()["active"]


def test_control_requires_local_token_and_origin(server):
    payload = {"command": "start", "step_interval_s": 0}
    assert request(server, "POST", "/api/commands", payload)[0] == 403
    assert (
        request(server, "POST", "/api/commands", payload, server.token, "https://example.com")[0]
        == 403
    )
    assert not server.hub.snapshot()["active"]


def test_hpu_execution_is_separate_and_preserves_correlated_receipts(server):
    class Hpu:
        calls = 0

        def probe(self):
            return {"state": "ready", "xcu": {"mode": "WORKING_MODE"}, "motion_enabled": True}

        def execute(self, task, emit, stopped):
            self.calls += 1
            assert not stopped()
            row = {**task, "command_id": "command1", "stage_id": "pregrasp"}
            emit("hpu_sdk_receipt", {**row, "sdk_status": "ControlStatus.SUCCESS"})
            emit("hpu_stage_verified", row)
            return {"state": "complete", "motion_commands_attempted": 1}

    payload = {
        "plan_id": "reviewed-plan", "plan_sha256": "a" * 64, "robot_sn": "g1-test",
        "request_id": "unique-hardware-request",
    }
    path = "/api/tabletop/execute"
    assert request(server, "POST", path, payload, server.token)[0] == 409
    hpu = server.hub._hpu = Hpu()
    assert request(server, "POST", path, payload)[0] == 403
    assert request(server, "POST", "/api/hardware/probe", {}, server.token)[0] == 200
    assert server.hub.snapshot()["hardware_connected"] is True
    status, body = request(server, "POST", path, payload, server.token)
    assert status == 202
    run_id = json.loads(body)["run_id"]
    server.hub.wait()
    status, body = request(server, "GET", "/api/runs/" + run_id)
    assert status == 200
    result = json.loads(body)
    assert result["mode"] == "g1_hardware"
    assert result["summary"]["motion_commands_accepted"] == 1
    assert result["summary"]["confirmed_stages"] == 1
    status, body = request(server, "POST", path, payload, server.token)
    assert status == 202 and json.loads(body)["replayed"] and hpu.calls == 1
    status, _ = request(server, "POST", path, {**payload, "plan_id": "other"}, server.token)
    assert status == 409 and hpu.calls == 1


def test_hpu_lost_receipt_latches_without_simulation_fallback(server):
    class DisconnectedHpu:
        def execute(self, _task, _emit, _stopped):
            raise TimeoutError("SSH connection lost")

    server.hub._hpu = DisconnectedHpu()
    receipt = server.hub.start_hardware("p", "0" * 64, "g1-test", "lost-link")
    server.hub.wait()
    assert server.hub.snapshot()["stop_latched"] is True
    result = server.hub.run_status(receipt["run_id"])
    assert result["status"] == "unconfirmed"
    assert result["summary"]["motion_commands_accepted"] == 0
    assert not server.hub.snapshot()["frame_available"]


def test_api_full_flow_and_result_export(server):
    assert (
        request(
            server,
            "POST",
            "/api/commands",
            {"command": "start", "step_interval_s": 0},
            server.token,
        )[0]
        == 200
    )
    server.hub.wait()
    status, body = request(server, "GET", "/api/summary")
    assert status == 200 and json.loads(body)["success"] is True
    assert request(server, "GET", "/api/frame")[1].startswith(b"\x89PNG")
    assert len(json.loads(request(server, "GET", "/api/runs")[1])) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"command": "move_joints"},
        {"command": "start", "joint_positions": [0, 0]},
        {"command": "start", "step_interval_s": "fast"},
        {"command": "start", "fault": []},
        [],
        {"command": None},
    ],
)
def test_invalid_control_requests_cannot_dispatch(server, payload):
    assert request(server, "POST", "/api/commands", payload, server.token)[0] in {400, 409}
    assert not server.hub.snapshot()["active"]


def test_no_path_traversal_or_dns_rebinding(server):
    assert request(server, "GET", "/../configs/simulation.json")[0] == 404
    assert request(server, "GET", "/api/status", host="attacker.example")[0] == 403
    assert request(server, "GET", "/api/events?after=invalid")[0] == 400


def test_tabletop_plan_accepts_only_local_artifacts_and_preview_mode(server):
    directory = server.hub.workspace / "artifacts" / "capture"
    directory.mkdir()
    (directory / "scene.json").write_text(
        '{"timestamp_ns":null,"detections":{},"feedback":'
        '{"held_object":"unknown","spoon_loaded":null,"bowl_loaded":null}}',
        encoding="utf-8",
    )
    payload = {
        "capture": "capture",
        "observation": "capture/scene.json",
        "task": {"skill": "grasp", "category": "bottle"},
    }
    post = "/api/tabletop/plan"
    assert request(server, "POST", post, payload)[0] == 403
    assert request(server, "POST", post, {**payload, "capture": ".."}, server.token)[0] == 400
    assert request(server, "POST", post, {**payload, "planner": "sdk"}, server.token)[0] == 400
    assert request(server, "POST", post, payload, server.token)[0] == 202
    server.hub.wait()
    result = json.loads(request(server, "GET", "/api/summary")[1])
    assert result["reason"] == "target_not_detected"
    assert result["dispatch_allowed"] is False

    simulate = "/api/tabletop/simulate"
    assert request(server, "POST", simulate, payload)[0] == 403
    assert request(server, "POST", simulate, {**payload, "execute": True}, server.token)[0] == 400
    invalid_fault = {**payload, "fault": "invalid"}
    assert request(server, "POST", simulate, invalid_fault, server.token)[0] == 409
    assert request(server, "POST", simulate, payload, server.token)[0] == 202
    server.hub.wait()
    snapshot = server.hub.snapshot()
    assert snapshot["mode"] == "g1_simulation"
    assert snapshot["summary"]["reason"] == "target_not_detected"
    assert snapshot["summary"]["motion_commands_sent"] == 0


def test_operator_voice_requires_review_before_planning(server):
    for path in ("/operator", "/operator.js", "/operator.css"):
        assert request(server, "GET", path)[0] == 200
    path = "/api/tabletop/interpret"
    payload = {"text": "请取水瓶后放回。"}
    assert request(server, "POST", path, payload)[0] == 403
    assert request(server, "POST", path, {**payload, "execute": True}, server.token)[0] == 400
    for text, skill, count in (
        ("抓水瓶", "grasp", None),
        (payload["text"], "pick_place", None),
        ("抓起水瓶，摇两下再放回。", "shake_then_place", 2),
    ):
        status, body = request(server, "POST", path, {"text": text}, server.token)
        assert status == 200
        assert json.loads(body) == {
            "task": {
                "skill": skill, "object_id": None, "category": "bottle", "repeat_count": count,
            },
            "dispatch_allowed": False,
        }
    for text in ("不要抓水瓶", "抓起水瓶摇五下再放回", "停止"):
        assert request(server, "POST", path, {"text": text}, server.token)[0] == 409
    snapshot = server.hub.snapshot()
    assert snapshot["status"] == "idle" and snapshot["run_id"] is None


def test_tabletop_receipt_is_queryable_before_worker_and_after_restart(server, monkeypatch):
    capture = server.hub.workspace / "artifacts" / "capture"
    capture.mkdir()
    observation = capture / "scene.json"
    observation.write_text(
        '{"timestamp_ns":null,"detections":{},"feedback":'
        '{"held_object":"unknown","spoon_loaded":null,"bowl_loaded":null}}',
        encoding="utf-8",
    )
    worker = server.hub._run_tabletop
    release = threading.Event()

    def delayed_worker(*args):
        if release.wait(4):
            worker(*args)

    monkeypatch.setattr(server.hub, "_run_tabletop", delayed_worker)
    payload = {
        "request_id": "go-tabletop-001",
        "capture": "capture",
        "observation": "capture/scene.json",
        "task": {"skill": "grasp", "category": "bottle"},
    }
    post = "/api/tabletop/plan"
    try:
        code, body = request(server, "POST", post, payload, server.token)
        accepted = json.loads(body)
        assert code == 202 and len(accepted["run_id"]) == 32 and not accepted["replayed"]
        route = "/api/runs/" + accepted["run_id"]
        code, body = request(server, "GET", route)
        current = json.loads(body)
        assert code == 200 and current["active"] and current["summary"] is None
        assert current["request_id"] == payload["request_id"]
        code, body = request(server, "POST", post, payload, server.token)
        assert code == 202 and json.loads(body)["replayed"]
        assert json.loads(body)["run_id"] == accepted["run_id"]
        changed = {**payload, "task": {"skill": "pick_place", "category": "bottle"}}
        assert request(server, "POST", post, changed, server.token)[0] == 409
    finally:
        release.set()
    server.hub.wait()
    code, body = request(server, "GET", route)
    completed = json.loads(body)
    assert code == 200 and completed["summary"]["run_id"] == accepted["run_id"]
    assert completed["status"] == "held"
    assert request(server, "GET", "/api/runs/" + "0" * 32)[0] == 404
    restarted = ControlHub(server.hub.workspace, server.hub.config)
    try:
        replay = restarted.start_tabletop(
            TaskRequest("grasp", category="bottle"), capture, observation,
            request_id=payload["request_id"],
        )
        assert replay["replayed"] and replay["summary"] == completed["summary"]
        assert len(restarted.run_history()) == 1
    finally:
        restarted.close()
