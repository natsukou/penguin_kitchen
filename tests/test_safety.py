import time
from dataclasses import replace

import pytest

from pet_meal.safety import SafetyGuard
from pet_meal.sdk import GalbotBackend, capture_readonly
from pet_meal.simulator import Simulator
from pet_meal.types import Action, HardwareDisabled, SafetyError, TaskError


@pytest.mark.parametrize(
    "position",
    [
        (float("nan"), 0, 0.1),
        (float("inf"), 0, 0.1),
        (0.81, 0, 0.1),
        (0.4, -0.41, 0.1),
        (0.4, 0, -0.1),
        (0.4, 0),
    ],
)
def test_rejects_invalid_or_outside_targets(config, position):
    with pytest.raises(SafetyError):
        SafetyGuard(config).validate_position(position)


def test_latched_stop_cannot_be_bypassed_by_new_observation(config):
    guard = SafetyGuard(config)
    guard.latch("test stop")
    with pytest.raises(SafetyError, match="latched"):
        guard.validate_observation(Simulator(config).observe())


def test_invalid_frame_and_action_target(config):
    obs = Simulator(config).observe()
    guard = SafetyGuard(config)
    for action in (
        Action("grasp_spoon", (0.7, 0.1, 0.1), "spoon"),
        Action("grasp_spoon", obs.detections["spoon"].position_m, "spoon", "camera"),
        Action("arbitrary_low_level_command", (0.4, 0.1, 0.1), "spoon"),
    ):
        with pytest.raises(SafetyError):
            guard.validate_action(action, obs)


@pytest.mark.parametrize("offset", [-10_000_000_000, 10_000_000_000])
def test_stale_and_future_observation(config, offset):
    obs = replace(Simulator(config).observe(), timestamp_ns=time.monotonic_ns() + offset)
    with pytest.raises(SafetyError, match="observation"):
        SafetyGuard(config).validate_observation(obs)


def test_missing_spoon_ownership_blocks_pour(config):
    obs = Simulator(config).observe()
    action = Action("pour_food", obs.detections["bowl"].position_m, "bowl")
    with pytest.raises(SafetyError, match="not held"):
        SafetyGuard(config).validate_action(action, obs)


def test_real_motion_is_unconditionally_disabled():
    with pytest.raises(HardwareDisabled):
        GalbotBackend().execute(Action("grasp_spoon", (0.4, 0.1, 0.1), "spoon"))


def test_windows_readonly_capture_fails_before_sdk_import(tmp_path, monkeypatch):
    monkeypatch.setattr("pet_meal.sdk.platform.system", lambda: "Windows")
    with pytest.raises(TaskError, match="Linux"):
        capture_readonly(tmp_path / "capture", "LEFT_ARM_CAMERA", "LEFT_ARM_DEPTH_CAMERA")


def test_cannot_change_simulator_config_to_enable_hardware(config):
    config["hardware_motion_enabled"] = True
    with pytest.raises(TaskError):
        Simulator(config)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), 0, -1])
def test_invalid_placement_tolerance_cannot_bypass_acceptance(config, tolerance):
    config["safety"]["placement_tolerance_m"] = tolerance
    with pytest.raises(SafetyError, match="placement_tolerance"):
        SafetyGuard(config)
