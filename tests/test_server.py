import http.client
import json
import threading

import pytest

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


def test_dashboard_serves_nutrition_and_recipe_images(server):
    status, body = request(server, "GET", "/")
    assert status == 200 and b'href="/nutrition_ui/index.html"' in body
    status, body = request(server, "GET", "/nutrition_ui/index.html")
    assert status == 200 and b'src="/nutrition_ui/app.js"' in body
    assert request(server, "GET", "/nutrition_ui/app.js")[0] == 200
    assert request(server, "GET", "/nutrition_ui/style.css")[0] == 200
    status, body = request(server, "GET", "/images/raw-tofu.ad4e6520250f.webp")
    assert status == 200 and body[:4] == b"RIFF" and body[8:12] == b"WEBP"
    assert request(server, "GET", "/images/missing.webp")[0] == 404
    assert request(server, "GET", "/images/../server.py")[0] == 404


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
