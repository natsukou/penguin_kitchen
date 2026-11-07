import json
from dataclasses import replace

import pytest

from pet_meal.decision import SEQUENCE, TaskRunner
from pet_meal.recording import Recorder
from pet_meal.simulator import FAULTS, Simulator
from pet_meal.types import State, TaskError


def run_task(tmp_path, config, fault="none", backend=None):
    backend = backend or Simulator(config, fault, tmp_path / "frames")
    recorder = Recorder(tmp_path, config, backend.evidence_level)
    try:
        result = TaskRunner(backend, config, recorder).run()
    finally:
        recorder.close()
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    return result, backend, rows


def test_full_task_reobserves_and_records_synthetic_evidence(tmp_path, config):
    result, backend, rows = run_task(tmp_path, config)
    assert result["success"] is True
    assert result["hardware_acceptance"] == "not_tested"
    assert result["physical_demo_duration_s"] is None
    assert result["successful_commands"] == 6
    assert backend.stopped
    assert all(row["evidence_level"] == "simulation_only" for row in rows)
    assert [r["payload"]["state"] for r in rows if r["event"] == "state_entered"] == list(SEQUENCE)
    assert sum(r["event"] == "action_confirmed" for r in rows) == 6
    assert sum(r["event"] == "observation" for r in rows) == len(SEQUENCE) + 6
    assert len(list((tmp_path / "frames").glob("*.png"))) == len(SEQUENCE) + 6


@pytest.mark.parametrize("fault", FAULTS[1:])
def test_injected_fault_stops_and_never_reports_completion(tmp_path, config, fault):
    result, backend, rows = run_task(tmp_path, config, fault)
    assert not result["success"]
    assert result["state"] == State.FAILED
    assert result["error"]
    assert backend.stopped
    assert any(row["event"] == "failure" for row in rows)
    assert not any(
        r["event"] == "action_confirmed" and r["payload"]["name"] == "place_bowl" for r in rows
    )


def test_unconfirmed_grasp_prevents_food_action(tmp_path, config):
    result, backend, _ = run_task(tmp_path, config, "grasp_slip")
    assert "Grasp was not confirmed" in result["error"]
    assert [a.name for a in backend.executed] == ["grasp_spoon"]


def test_failure_to_stop_overrides_success(tmp_path, config):
    class StopFailure(Simulator):
        def stop(self):
            raise TaskError("injected stop failure")

    result, _, _ = run_task(tmp_path, config, backend=StopFailure(config))
    assert not result["success"]
    assert "injected stop failure" in result["stop_error"]


def test_visual_movement_updates_next_action_target(tmp_path, config):
    class MovingContainer(Simulator):
        def execute(self, action):
            super().execute(action)
            if action.name == "grasp_spoon":
                self.positions["food_container"] = (0.7, -0.25, 0.1)

    result, backend, _ = run_task(tmp_path, config, backend=MovingContainer(config))
    assert result["success"]
    acquisition = next(a for a in backend.executed if a.name == "acquire_food")
    assert acquisition.target_m[0] == pytest.approx(0.7, abs=0.003)
    assert acquisition.target_m[1] == pytest.approx(-0.25, abs=0.003)


def test_final_verification_detects_bowl_moved_after_placement(tmp_path, config):
    class DisturbedBowl(Simulator):
        def __init__(self, config):
            super().__init__(config)
            self.post_place_observations = 0

        def observe(self):
            if self.executed and self.executed[-1].name == "place_bowl":
                self.post_place_observations += 1
                if self.post_place_observations == 2:
                    self.positions["bowl"] = (0.48, 0.18, 0.1)
            return super().observe()

    result, _, rows = run_task(tmp_path, config, backend=DisturbedBowl(config))
    assert not result["success"]
    assert "Final visual placement" in result["error"]
    assert any(r["event"] == "state_entered" and r["payload"]["state"] == "verify" for r in rows)


def test_unconfirmed_food_prevents_pouring(tmp_path, config):
    class BadFoodFeedback(Simulator):
        def observe(self):
            obs = super().observe()
            return replace(obs, feedback=replace(obs.feedback, spoon_loaded=False))

    result, backend, _ = run_task(tmp_path, config, backend=BadFoodFeedback(config))
    assert not result["success"]
    assert all(a.name != "pour_food" for a in backend.executed)
