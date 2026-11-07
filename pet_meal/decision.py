"""Finite-state task orchestration; observations gate every transition."""

import math
import time

from .control import RunControl
from .policy import TARGETS, FixedScenePolicy
from .recording import Recorder
from .safety import SafetyGuard
from .types import Action, Backend, Observation, RunCancelled, State, TaskError

SEQUENCE = (
    State.OBSERVE,
    State.LOCATE_SPOON,
    State.GRASP_SPOON,
    State.ACQUIRE_FOOD,
    State.POUR_FOOD,
    State.RETURN_SPOON,
    State.GRASP_BOWL,
    State.PLACE_BOWL,
    State.VERIFY,
)


class TaskRunner:
    def __init__(
        self, backend: Backend, config: dict, recorder: Recorder, control: RunControl | None = None
    ):
        if backend.evidence_level != "simulation_only":
            raise TaskError("Task runner supports simulation evidence only")
        self.backend, self.config, self.recorder = backend, config, recorder
        self.guard = SafetyGuard(config)
        self.state = State.OBSERVE
        self.spoon_home = None
        self.command_count = 0
        self.confirmed_count = 0
        self.policy = FixedScenePolicy()
        self.control = control or RunControl()
        self.active_module = "decision"

    def observe(self) -> Observation:
        self.active_module = "vision"
        obs = self.backend.observe()
        if obs.source != "synthetic_rgbd_markers" or obs.feedback.source != "simulated_state":
            raise TaskError("Unexpected evidence source for simulation runner")
        self.recorder.event("observation", obs)
        self.active_module = "safety"
        self.guard.validate_observation(obs)
        missing = {"spoon", "food_container", "bowl", "mat"} - obs.detections.keys()
        if missing:
            self.active_module = "vision"
            raise TaskError(f"Missing objects: {sorted(missing)}")
        return obs

    def verify_action(self, action: Action, obs: Observation) -> None:
        self.active_module = "feedback"
        feedback = obs.feedback
        if action.name.startswith("grasp_") and feedback.held_object != action.object_id:
            raise TaskError("Grasp was not confirmed by feedback")
        if action.name == "acquire_food" and not feedback.spoon_loaded:
            raise TaskError("Food acquisition was not confirmed")
        if action.name == "pour_food" and (feedback.spoon_loaded or not feedback.bowl_loaded):
            raise TaskError("Food transfer was not confirmed")
        if action.name in {"return_spoon", "place_bowl"}:
            if feedback.held_object is not None:
                raise TaskError("Release was not confirmed")
            name = "spoon" if action.name == "return_spoon" else "bowl"
            actual = obs.detections[name].position_m
            if (
                math.dist(actual[:2], action.target_m[:2])
                > self.config["safety"]["placement_tolerance_m"]
            ):
                raise TaskError(f"Placement error: {name}")

    def step(self, state: State) -> None:
        self.state = state
        self.active_module = "decision"
        self.recorder.event("state_entered", {"state": state})
        obs = self.observe()
        if state == State.OBSERVE:
            self.spoon_home = obs.detections["spoon"].position_m
        elif state in TARGETS:
            self.active_module = "policy"
            action = self.policy.select(state, obs, self.spoon_home)
            self.recorder.event("action_selected", action)
            self.active_module = "safety"
            self.guard.validate_action(action, obs)
            self.recorder.event("action_checked", action)
            self.recorder.event("action_requested", action)
            self.active_module = "control"
            self.backend.execute(action)
            self.command_count += 1
            self.recorder.event("action_dispatched", action)
            self.verify_action(action, self.observe())
            self.confirmed_count += 1
            self.recorder.event("action_confirmed", action)
        elif state == State.VERIFY:
            self.active_module = "feedback"
            if obs.feedback.held_object is not None or not obs.feedback.bowl_loaded:
                raise TaskError("Final meal state was not confirmed")
            bowl = obs.detections["bowl"].position_m
            mat = obs.detections["mat"].position_m
            if math.dist(bowl[:2], mat[:2]) > self.config["safety"]["placement_tolerance_m"]:
                raise TaskError("Final visual placement verification failed")

    def run(self) -> dict:
        started = time.perf_counter()
        error = None
        stop_error = None
        failed_module = None
        failed_state = None
        try:
            for state in SEQUENCE:
                self.control.checkpoint()
                # Observe, validate, execute and confirm atomically relative to the
                # operator gate. Resuming always starts with a fresh observation.
                self.control.execute(lambda: self.step(state))
            self.state = State.COMPLETE
        except RunCancelled as exc:
            error = str(exc)
            self.guard.latch(error)
            self.recorder.event("cancelled", {"state": self.state, "reason": error})
            self.state = State.CANCELLED
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.guard.latch(error)
            failed_module, failed_state = self.active_module, self.state
            self.recorder.event(
                "failure", {"state": self.state, "error": error, "module": self.active_module}
            )
            self.state = State.FAILED
        finally:
            try:
                self.backend.stop()
            except Exception as exc:
                stop_error = f"{type(exc).__name__}: {exc}"
                self.guard.latch(stop_error)
                self.state = State.FAILED
        summary = {
            "run_id": self.recorder.run_id,
            "state": self.state,
            "success": self.state == State.COMPLETE,
            "evidence_level": self.backend.evidence_level,
            "hardware_acceptance": "not_tested",
            "vision": "synthetic_color_markers_not_semantic_detection",
            "food_quantity": "not_measured",
            "act_policy": "not_trained_or_loaded",
            "elapsed_wall_s": time.perf_counter() - started,
            "physical_demo_duration_s": None,
            "successful_commands": self.command_count,
            "confirmed_actions": self.confirmed_count,
            "error": error,
            "stop_error": stop_error,
            "failed_module": failed_module,
            "failed_state": failed_state,
        }
        self.recorder.finish(summary)
        return summary
