"""Local control plane: one active run, event fanout and module monitoring."""

import hashlib
import json
import re
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .control import RunControl
from .decision import SEQUENCE, TaskRunner
from .events import EventBus
from .g1_link import validate_execution
from .g1_simulation import FAULTS as TABLETOP_FAULTS
from .g1_simulation import simulate_stages
from .g1_task import StagePlanner, TaskRequest, plan_task
from .modules import ModuleRegistry
from .recording import Recorder
from .simulator import FAULTS, Simulator
from .types import RunCancelled, TaskError


def _read_json_object(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


class ControlHub:
    def __init__(self, workspace: Path, config: dict, *, hpu=None):
        self.workspace = workspace
        self.config = deepcopy(config)
        self.root = workspace / "artifacts/hub-runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.bus = EventBus()
        self.registry = ModuleRegistry()
        self._lock = threading.RLock()
        self._command_lock = threading.RLock()
        self._gate = RunControl()
        self._active = False
        self._latched = False
        self._status = "idle"
        self._state = None
        self._run_id = None
        self._directory = None
        self._fault = "none"
        self._latest_observation = None
        self._latest_action = None
        self._summary = None
        self._confirmed = 0
        self._started = None
        self._frame_path = None
        self._thread = None
        self._backend = None
        self._mode = "simulation_only"
        self._tabletop = None
        self._hpu = hpu
        self._hardware_probe = None
        self._probe_monotonic = None
        self._publish({"event": "hub_ready", "payload": {"mode": "simulation_only"}})

    def _publish(self, row: dict):
        with self._lock:
            event = self.bus.publish(row)
            self.registry.consume(event)
            name, payload = event["event"], event.get("payload", {})
            if name == "state_entered":
                self._state = payload["state"]
            elif name == "observation":
                self._latest_observation = payload
            elif name == "action_requested":
                self._latest_action = payload
            elif name == "action_confirmed":
                self._confirmed += 1
            elif name in (
                "tabletop_requested",
                "task_target_resolved",
                "task_stage_reviewed",
                "tabletop_sim_stage_result",
                "hpu_stage_checked",
                "hpu_dispatch_intent",
                "hpu_sdk_receipt",
                "hpu_motion_verified",
                "hpu_stage_verified",
                "hpu_stop_receipt",
            ):
                self._tabletop[name] = payload
                if name == "hpu_stage_verified":
                    self._confirmed += 1
            if self._backend and self._backend.frame_index and self._directory:
                self._frame_path = (
                    self._directory
                    / "frames"
                    / (f"{self._backend.frame_index - 1:03d}-synthetic-rgb.png")
                )

    def _require_idle(self):
        with self._lock:
            if self._active:
                raise TaskError("A run is already active")
            if self._latched:
                raise TaskError("Reset the software stop latch before starting")

    def _launch(self, gate, backend, directory, fault, mode, worker, args=()):
        with self._lock:
            self.registry.reset()
            self._gate, self._backend, self._directory = gate, backend, directory
            self._mode, self._active, self._status = mode, True, "running"
            self._tabletop = {} if mode != "simulation_only" else None
            self._fault, self._state, self._summary = fault, None, None
            self._latest_observation, self._latest_action = None, None
            self._frame_path, self._run_id = None, directory.name
            self._confirmed = 0
            self._started = time.monotonic()
            self._thread = threading.Thread(target=worker, args=args, daemon=True)
            self._thread.start()

    def start_tabletop(
        self,
        request: TaskRequest,
        capture: Path,
        observation: Path,
        *,
        vision_choice: Path | None = None,
        planner: StagePlanner | None = None,
        simulation_fault: str | None = None,
        request_id: str | None = None,
    ) -> dict:
        if simulation_fault is not None and (
            planner is not None or simulation_fault not in TABLETOP_FAULTS
        ):
            raise TaskError("Invalid tabletop simulation configuration")
        mode = "g1_plan_only" if simulation_fault is None else "g1_simulation"
        payload = {
            "task": asdict(request),
            "capture": str(capture.resolve()),
            "observation": str(observation.resolve()),
            "vision_choice": str(vision_choice.resolve()) if vision_choice is not None else None,
            "mode": mode,
            "fault": simulation_fault,
            "planner": planner.evidence_level if planner is not None else None,
        }
        return self._accept_tabletop(
            payload, mode, simulation_fault or "none", request_id, self._run_tabletop,
            (request, capture, observation, vision_choice, planner, simulation_fault, request_id),
        )

    def start_hardware(self, plan_id, plan_sha256, robot_sn, request_id):
        if self._hpu is None:
            raise TaskError("HPU transport is not configured")
        if request_id is None:
            raise TaskError("Physical execution requires a stable request_id")
        request = {"plan_id": plan_id, "plan_sha256": plan_sha256, "robot_sn": robot_sn}
        validate_execution({**request, "run_id": "0" * 32})
        return self._accept_tabletop(
            {**request, "mode": "g1_hardware"}, "g1_hardware", "none", request_id,
            self._run_hardware, (request,),
        )

    def probe_hardware(self):
        if self._hpu is None:
            raise TaskError("HPU transport is not configured")
        with self._command_lock:
            with self._lock:
                if self._active and self._mode == "g1_hardware":
                    raise TaskError("Wait for the current HPU execution to finish")
            try:
                result = self._hpu.probe()
            except Exception as exc:
                result = {"state": "unavailable", "reason": str(exc), "motion_enabled": False}
            with self._lock:
                self._hardware_probe = result
                self._probe_monotonic = time.monotonic()
            return result

    def _accept_tabletop(self, payload, mode, fault, request_id, worker, args):
        if request_id is not None and (
            not isinstance(request_id, str)
            or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", request_id)
        ):
            raise TaskError("Invalid request_id")
        run_id = (
            uuid.uuid4().hex
            if request_id is None
            else uuid.uuid5(uuid.NAMESPACE_URL, f"pet-meal/tabletop/{request_id}").hex
        )
        directory = self.root / run_id
        manifest = directory / "run.json"
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with self._command_lock:
            if request_id is not None and directory.exists():
                previous = _read_json_object(manifest)
                if previous is None or previous.get("run_id") != run_id:
                    raise TaskError("Existing request receipt is unavailable")
                if previous.get("request_sha256") != fingerprint:
                    raise TaskError("request_id already belongs to a different request")
                return {**self.run_status(run_id), "replayed": True}
            self._require_idle()
            try:
                directory.mkdir()
                manifest.write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "mode": mode,
                            "request_id": request_id,
                            "request_sha256": fingerprint,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except FileExistsError as exc:
                raise TaskError("Run receipt already exists. Retry the same request_id") from exc
            except OSError as exc:
                raise TaskError("Unable to persist run receipt") from exc
            self._launch(
                RunControl(), None, directory, fault, mode, worker, args,
            )
            return {**self.snapshot(), "request_id": request_id, "replayed": False}

    def command(self, name: str, *, fault: str = "none", step_interval_s: float = 0.8) -> dict:
        # Do not hold the telemetry lock when waiting for the gate: a worker can
        # publish while holding the gate, so reversing that order would deadlock.
        with self._command_lock:
            with self._lock:
                active, latched, gate = self._active, self._latched, self._gate
            if name == "start":
                self._require_idle()
                if fault not in FAULTS:
                    raise TaskError("Unknown scenario")
                gate = RunControl(step_interval_s)
                directory = self.root / uuid.uuid4().hex
                backend = Simulator(self.config, fault, directory / "frames")
                self._launch(gate, backend, directory, fault, "simulation_only", self._run)
            elif name == "pause":
                if not active:
                    raise TaskError("No active run to pause")
                if self._mode == "g1_hardware":
                    raise TaskError("Use stop for an active HPU execution")
                gate.pause()
                with self._lock:
                    if self._active and self._summary is None:
                        self._status = "paused"
            elif name == "resume":
                if not active or latched:
                    raise TaskError("No paused run available for resume")
                gate.resume()
                with self._lock:
                    if self._active and self._summary is None:
                        self._status = "running"
            elif name == "stop":
                gate.stop()
                with self._lock:
                    self._latched = True
                    if self._active:
                        self._status = "stopping"
                    else:
                        self._status = "stopped"
            elif name == "reset":
                if active:
                    raise TaskError("Wait for the active run to stop before resetting")
                with self._lock:
                    self._latched, self._status = False, "idle"
                    self._gate = RunControl()
            else:
                raise TaskError("Unknown command")
            self._publish(
                {"event": "operator_command", "run_id": self._run_id, "payload": {"command": name}}
            )
            return self.snapshot()

    def _run_tabletop(
        self, request, capture, observation, vision_choice, planner, simulation_fault, request_id
    ):
        level = planner.evidence_level if planner is not None else "recorded_scene_preview"
        if simulation_fault is not None:
            level = "deterministic_tabletop_simulator"
        recorder = None
        result = {"state": "failed", "reason": "recorder_unavailable"}
        try:
            recorder = Recorder(
                self._directory,
                {"mode": self._mode, "evidence": level, "fault": simulation_fault},
                level,
                on_event=self._publish,
                run_id=self._run_id,
            )
            recorder.event(
                "tabletop_requested",
                {
                    "request": asdict(request),
                    "request_id": request_id,
                    "capture": capture,
                    "observation": observation,
                    "vision_choice": vision_choice,
                },
            )
            try:
                result = plan_task(
                    request,
                    capture,
                    observation,
                    vision_choice,
                    planner,
                    recorder.event,
                    lambda: self._gate.checkpoint(paced=False),
                )
                if simulation_fault is not None:
                    result = simulate_stages(
                        result,
                        simulation_fault,
                        recorder.event,
                        lambda: self._gate.checkpoint(paced=False),
                    )
                self._gate.checkpoint(paced=False)
            except RunCancelled:
                result = {"state": "cancelled", "reason": "operator_stop"}
            except Exception as exc:
                result = {"state": "failed", "reason": "task_error", "error": str(exc)}
            summary = self._tabletop_summary(result, level)
            recorder.finish(summary)
        except Exception as exc:
            summary = self._tabletop_summary(
                {"state": "failed", "reason": "recorder_error", "error": str(exc)},
                level,
            )
        finally:
            try:
                if recorder is not None:
                    recorder.close()
            finally:
                with self._lock:
                    self._summary = summary
                    self._status = self._state = summary["state"]
                    self._active = False

    def _tabletop_summary(self, result: dict, level: str) -> dict:
        return {
            **result,
            "run_id": self._run_id,
            "evidence_level": level,
            "success": result["state"] in ("preview", "complete"),
            "dispatch_allowed": False,
            "motion_commands_sent": 0,
            "elapsed_wall_s": time.monotonic() - self._started,
        }

    def _run_hardware(self, request):
        recorder = None
        request = {**request, "run_id": self._run_id}
        accepted = set()
        verified = set()

        def emit(name, payload):
            if (
                payload.get("run_id") != self._run_id
                or payload.get("robot_sn") != request["robot_sn"]
                or payload.get("plan_sha256") != request["plan_sha256"]
            ):
                raise TaskError("HPU stage receipt belongs to a different run or robot")
            if name == "hpu_sdk_receipt" and payload.get("sdk_status") == "ControlStatus.SUCCESS":
                accepted.add(payload["command_id"])
            if name == "hpu_stage_verified":
                command_id = payload["command_id"]
                if command_id not in accepted or command_id in verified:
                    raise TaskError("HPU verification lacks a unique SDK acceptance receipt")
                verified.add(command_id)
            recorder.event(name, payload)

        try:
            recorder = Recorder(
                self._directory, {"mode": "g1_hardware"}, "galbot_hpu_xcu",
                on_event=self._publish, run_id=self._run_id,
            )
            recorder.event("hardware_requested", request)
            result = self._hpu.execute(request, emit, lambda: self._gate.snapshot()["stopped"])
            if result.get("state") not in ("complete", "held", "unconfirmed"):
                raise TaskError("Invalid HPU terminal state")
            if result["state"] == "complete" and (
                not accepted or accepted != verified
                or result.get("motion_commands_attempted") != len(accepted)
            ):
                raise TaskError("HPU completion lacks matching dispatch and stage receipts")
        except Exception as exc:
            result = {"state": "unconfirmed", "reason": str(exc)}
        summary = {
            **result, **request, "success": result["state"] == "complete",
            "evidence_level": "galbot_hpu_xcu", "motion_commands_accepted": len(accepted),
            "confirmed_stages": self._confirmed,
            "elapsed_wall_s": time.monotonic() - self._started,
        }
        try:
            if recorder is not None:
                recorder.finish(summary)
        except Exception as exc:
            summary.update(state="unconfirmed", success=False, reason=str(exc))
        finally:
            try:
                if recorder is not None:
                    recorder.close()
            finally:
                with self._lock:
                    self._summary = summary
                    self._state = self._status = summary["state"]
                    self._active = False
                    self._latched |= summary["state"] == "unconfirmed"

    def _run(self):
        recorder = None
        try:
            recorder = Recorder(
                self._directory,
                self.config,
                self._backend.evidence_level,
                on_event=self._publish,
                run_id=self._run_id,
            )
            recorder.event("scenario", {"fault": self._fault})
            summary = TaskRunner(self._backend, self.config, recorder, self._gate).run()
            with self._lock:
                self._summary = summary
                self._status = summary["state"]
                if not summary["success"]:
                    self._latched = True
        except Exception as exc:
            self._gate.stop()
            self._backend.stop()
            with self._lock:
                self._status, self._latched = "failed", True
                self._summary = {
                    "run_id": self._run_id,
                    "success": False,
                    "state": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            self._publish(
                {
                    "event": "failure",
                    "payload": {"module": "recorder", "error": self._summary["error"]},
                }
            )
        finally:
            if recorder:
                recorder.close()
            with self._lock:
                self._active = False

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = time.monotonic() - self._started if self._started else None
            if not self._active and self._summary:
                elapsed = self._summary.get("elapsed_wall_s")
            probe = self._hardware_probe or {}
            connected = bool(
                self._probe_monotonic is not None
                and time.monotonic() - self._probe_monotonic < 30
                and probe.get("state") == "ready"
            )
            return deepcopy(
                {
                    "server_utc": datetime.now(timezone.utc).isoformat(),
                    "mode": self._mode,
                    "hardware_connected": connected,
                    "hardware_motion_enabled": connected and probe.get("motion_enabled") is True,
                    "hardware_transport_configured": self._hpu is not None,
                    "hardware_probe": self._hardware_probe,
                    "status": self._status,
                    "active": self._active,
                    "stop_latched": self._latched,
                    "run_id": self._run_id,
                    "state": self._state,
                    "fault": self._fault,
                    "confirmed_actions": self._confirmed,
                    "elapsed_wall_s": elapsed,
                    "step_interval_s": self._gate.interval,
                    "sequence": [s.value for s in SEQUENCE],
                    "faults": list(FAULTS),
                    "modules": self.registry.snapshot(),
                    "observation": self._latest_observation,
                    "last_action": self._latest_action,
                    "summary": self._summary,
                    "tabletop": self._tabletop,
                    "frame_available": bool(self._frame_path and self._frame_path.is_file()),
                    "artifact_directory": str(self._directory) if self._directory else None,
                }
            )

    def latest_frame(self) -> bytes | None:
        with self._lock:
            path = self._frame_path
        return path.read_bytes() if path and path.is_file() else None

    def run_history(self) -> list[dict]:
        items = []
        for path in sorted(
            self.root.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )[:25]:
            summary = _read_json_object(path)
            if summary is not None:
                items.append({"summary": summary, "artifact_directory": str(path.parent)})
        return items

    def run_status(self, run_id: str) -> dict | None:
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            return None
        directory = self.root / run_id
        manifest = _read_json_object(directory / "run.json") or {}
        with self._lock:
            if self._run_id == run_id:
                summary = deepcopy(self._summary)
                return {
                    "run_id": run_id,
                    "request_id": manifest.get("request_id"),
                    "mode": self._mode,
                    "status": summary["state"] if summary is not None else self._status,
                    "active": self._active,
                    "summary": summary,
                    "artifact_directory": str(self._directory),
                }
        summary = _read_json_object(directory / "summary.json")
        if summary is None or summary.get("run_id") != run_id:
            summary = None
            if not manifest:
                # Older runs stored a separate directory UUID and recorder UUID.
                for path in self.root.glob("*/summary.json"):
                    previous = _read_json_object(path)
                    if previous is not None and previous.get("run_id") == run_id:
                        directory, summary = path.parent, previous
                        break
        if summary is None and manifest.get("run_id") != run_id:
            return None
        return {
            "run_id": run_id,
            "request_id": manifest.get("request_id"),
            "mode": manifest.get("mode"),
            "status": summary.get("state", "unconfirmed") if summary else "unconfirmed",
            "active": False,
            "summary": summary,
            "artifact_directory": str(directory),
        }

    def wait(self, timeout: float = 10) -> None:
        thread = self._thread
        if thread:
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("Hub worker did not finish")

    def close(self) -> None:
        self.command("stop")
        self.wait()
