import json
import threading
import time

import pytest

from pet_meal.control import RunControl
from pet_meal.events import EventBus
from pet_meal.hub import ControlHub
from pet_meal.types import RunCancelled, TaskError


@pytest.fixture
def hub(tmp_path, config):
    instance = ControlHub(tmp_path, config)
    yield instance
    instance.close()


def wait_for(predicate, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Expected state did not arrive")


def test_hub_completes_and_persists_evidence(hub):
    hub.command("start", step_interval_s=0)
    hub.wait()
    status = hub.snapshot()
    assert status["status"] == "complete"
    assert status["summary"]["success"]
    assert status["confirmed_actions"] == 6
    assert status["hardware_connected"] is False
    assert status["hardware_motion_enabled"] is False
    assert status["frame_available"]
    assert hub.latest_frame().startswith(b"\x89PNG")
    assert len(hub.run_history()) == 1
    assert hub.run_history()[0]["summary"]["run_id"] == status["run_id"]
    modules = {m["id"]: m for m in status["modules"]}
    assert modules["hardware"]["status"] == "disabled"
    assert modules["act"]["status"] == "disabled"
    assert modules["policy"]["status"] == "healthy"


def test_double_start_is_rejected(hub):
    hub.command("start", step_interval_s=0.1)
    with pytest.raises(TaskError, match="already active"):
        hub.command("start")


def test_pause_prevents_dispatch_resume_uses_fresh_observation(hub):
    hub.command("start", step_interval_s=0.05)
    wait_for(lambda: hub.snapshot()["confirmed_actions"] >= 1)
    hub.command("pause")
    paused = hub.snapshot()
    commands = len(hub._backend.executed)
    # Longer than the configured observation freshness threshold. Resume must
    # reobserve, not dispatch a target calculated before the pause.
    time.sleep(1.1)
    assert len(hub._backend.executed) == commands
    assert hub.snapshot()["state"] == paused["state"]
    hub.command("resume")
    hub.wait()
    assert hub.snapshot()["summary"]["success"]


def test_stop_while_paused_cancels_and_requires_reset(hub):
    hub.command("start", step_interval_s=0.05)
    hub.command("pause")
    commands = len(hub._backend.executed)
    hub.command("stop")
    hub.wait()
    status = hub.snapshot()
    assert status["status"] == "cancelled"
    assert status["stop_latched"]
    assert hub._backend.stopped
    assert len(hub._backend.executed) == commands
    with pytest.raises(TaskError, match="Reset"):
        hub.command("start")
    hub.command("reset")
    hub.command("start", step_interval_s=0)
    hub.wait()
    assert hub.snapshot()["summary"]["success"]


def test_fault_latches_and_identifies_module(hub):
    hub.command("start", fault="grasp_slip", step_interval_s=0)
    hub.wait()
    status = hub.snapshot()
    assert status["status"] == "failed" and status["stop_latched"]
    assert status["summary"]["failed_module"] == "feedback"
    assert next(m for m in status["modules"] if m["id"] == "feedback")["status"] == "fault"


def test_reset_cannot_replace_active_gate(hub):
    hub.command("start", step_interval_s=0.1)
    with pytest.raises(TaskError, match="Wait"):
        hub.command("reset")


def test_history_survives_new_hub_instance(hub, tmp_path, config):
    hub.command("start", step_interval_s=0)
    hub.wait()
    new_hub = ControlHub(tmp_path, config)
    try:
        assert len(new_hub.run_history()) == 1
        assert new_hub.snapshot()["status"] == "idle"
    finally:
        new_hub.close()


def test_event_bus_detaches_payload_and_reports_cursor_gaps():
    bus = EventBus(capacity=2)
    payload = {"event": "example", "payload": {"value": 1}}
    bus.publish(payload)
    payload["payload"]["value"] = 2
    assert bus.after()["events"][0]["payload"]["value"] == 1
    for i in range(3):
        bus.publish({"event": "next", "payload": {"value": i}})
    result = bus.after(1)
    assert result["truncated"]
    assert [r["cursor"] for r in result["events"]] == [3, 4]
    assert bus.after(100)["truncated"]
    assert len(bus.after(100)["events"]) == 2


def test_stopped_gate_never_dispatches_operation():
    gate = RunControl()
    gate.stop()
    calls = []
    with pytest.raises(RunCancelled):
        gate.execute(lambda: calls.append("unsafe"))
    assert calls == []


def test_concurrent_start_has_one_winner(hub):
    outcomes = []

    def start():
        try:
            hub.command("start", step_interval_s=0.1)
            outcomes.append("started")
        except TaskError:
            outcomes.append("rejected")

    threads = [threading.Thread(target=start) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert sorted(outcomes) == ["rejected", "started"]


@pytest.mark.parametrize("interval", [-1, 6, float("nan"), float("inf")])
def test_invalid_pacing_rejected(interval):
    with pytest.raises(TaskError):
        RunControl(interval)


def test_recorded_events_have_source_hash_and_action_checks(hub):
    hub.command("start", step_interval_s=0)
    hub.wait()
    rows = [json.loads(s) for s in (hub._directory / "events.jsonl").read_text().splitlines()]
    assert len(rows[0]["payload"]["source_sha256"]) == 64
    checked = [r for r in rows if r["event"] == "action_checked"]
    assert len(checked) == 6
