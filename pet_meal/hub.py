"""Local control plane: one active run, event fanout and module monitoring."""

import json
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .control import RunControl
from .decision import SEQUENCE, TaskRunner
from .events import EventBus
from .modules import ModuleRegistry
from .recording import Recorder
from .simulator import FAULTS, Simulator
from .types import TaskError


class ControlHub:
    def __init__(self, workspace: Path, config: dict):
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
            if self._backend and self._backend.frame_index and self._directory:
                self._frame_path = (
                    self._directory
                    / "frames"
                    / (f"{self._backend.frame_index - 1:03d}-synthetic-rgb.png")
                )

    def command(self, name: str, *, fault: str = "none", step_interval_s: float = 0.8) -> dict:
        # Do not hold the telemetry lock when waiting for the gate: a worker can
        # publish while holding the gate, so reversing that order would deadlock.
        with self._command_lock:
            with self._lock:
                active, latched, gate = self._active, self._latched, self._gate
            if name == "start":
                if active:
                    raise TaskError("A run is already active")
                if latched:
                    raise TaskError("Reset the software stop latch before starting")
                if fault not in FAULTS:
                    raise TaskError("Unknown scenario")
                gate = RunControl(step_interval_s)
                directory = self.root / uuid.uuid4().hex
                backend = Simulator(self.config, fault, directory / "frames")
                with self._lock:
                    self.registry.reset()
                    self._gate, self._backend, self._directory = gate, backend, directory
                    self._active, self._status = True, "running"
                    self._fault, self._state, self._summary = fault, None, None
                    self._latest_observation, self._latest_action = None, None
                    self._frame_path, self._run_id = None, None
                    self._confirmed = 0
                    self._started = time.monotonic()
                    self._thread = threading.Thread(target=self._run, daemon=True)
                    self._thread.start()
            elif name == "pause":
                if not active:
                    raise TaskError("No active run to pause")
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
            self._publish({"event": "operator_command", "payload": {"command": name}})
        return self.snapshot()

    def _run(self):
        recorder = None
        try:
            recorder = Recorder(
                self._directory,
                self.config,
                self._backend.evidence_level,
                on_event=self._publish,
            )
            with self._lock:
                self._run_id = recorder.run_id
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
            return deepcopy(
                {
                    "server_utc": datetime.now(timezone.utc).isoformat(),
                    "mode": "simulation_only",
                    "hardware_connected": False,
                    "hardware_motion_enabled": False,
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
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                items.append({"summary": summary, "artifact_directory": str(path.parent)})
            except (OSError, ValueError):
                continue
        return items

    def wait(self, timeout: float = 10) -> None:
        thread = self._thread
        if thread:
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError("Hub worker did not finish")

    def close(self) -> None:
        self.command("stop")
        self.wait()
