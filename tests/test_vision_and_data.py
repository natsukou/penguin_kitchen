from dataclasses import replace

import cv2
import numpy as np
import pytest

from pet_meal.data import split_episodes
from pet_meal.simulator import Simulator
from pet_meal.types import SafetyError, TaskError
from pet_meal.vision import detect_markers


def test_rgbd_to_base_localization(config):
    sim = Simulator(config)
    obs = sim.observe()
    for name, expected in config["scene"].items():
        assert obs.detections[name].position_m == pytest.approx(expected, abs=0.003)


def test_wrong_rgb_depth_dimensions(config):
    sim = Simulator(config)
    rgb, depth = sim.render()
    with pytest.raises(SafetyError):
        detect_markers(rgb, depth[:100], sim.calibration)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_depth_rejected(config, value):
    sim = Simulator(config)
    rgb, depth = sim.render()
    depth[:] = value
    with pytest.raises(SafetyError):
        detect_markers(rgb, depth, sim.calibration)


def test_duplicate_marker_is_ambiguous(config):
    sim = Simulator(config)
    rgb, depth = sim.render()
    cv2.rectangle(rgb, (10, 10), (30, 30), (0, 255, 255), -1)
    depth[10:31, 10:31] = 1.1
    with pytest.raises(SafetyError, match="Ambiguous"):
        detect_markers(rgb, depth, sim.calibration)


def test_reflection_is_not_valid_camera_rotation(config):
    sim = Simulator(config)
    calibration = replace(sim.calibration, camera_to_base=tuple(map(tuple, np.diag([1, 1, -1, 1]))))
    with pytest.raises(SafetyError, match="determinant"):
        calibration.matrix()


def test_episode_splits_keep_sessions_together_and_are_deterministic():
    episodes = [
        {
            "episode_id": f"ep{i}",
            "session_id": f"session{i // 3}",
            "evidence_level": "hardware_recording",
        }
        for i in range(120)
    ]
    splits = split_episodes(episodes)
    assert splits == split_episodes(list(reversed(episodes)))
    all_ids = sum(splits.values(), [])
    assert len(all_ids) == len(set(all_ids)) == len(episodes)
    owner = {key: split for split, ids in splits.items() for key in ids}
    for start in range(0, 120, 3):
        assert len({owner[f"ep{i}"] for i in range(start, start + 3)}) == 1


def test_synthetic_and_duplicate_episodes_cannot_enter_training():
    with pytest.raises(TaskError, match="Synthetic"):
        split_episodes([{"episode_id": "fake", "evidence_level": "simulation_only"}])
    row = {"episode_id": "same", "evidence_level": "hardware_recording"}
    with pytest.raises(TaskError, match="unique"):
        split_episodes([row, row])
