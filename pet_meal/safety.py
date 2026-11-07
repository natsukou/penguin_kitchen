import math
import time

from .types import Action, Observation, SafetyError

ACTION_OBJECTS = {
    "grasp_spoon": "spoon",
    "acquire_food": "food_container",
    "pour_food": "bowl",
    "return_spoon": "spoon",
    "grasp_bowl": "bowl",
    "place_bowl": "mat",
}


class SafetyGuard:
    def __init__(self, config: dict):
        self.config = config
        self.latched = False
        self.reason = ""
        safety = config["safety"]
        bounds = safety["workspace_m"]
        if len(bounds) != 3 or any(
            len(b) != 2 or not all(math.isfinite(x) for x in b) or b[0] >= b[1] for b in bounds
        ):
            raise SafetyError("Invalid workspace bounds")
        for field in (
            "max_observation_age_s",
            "max_sensor_skew_s",
            "target_tolerance_m",
            "placement_tolerance_m",
        ):
            if not math.isfinite(safety[field]) or safety[field] <= 0:
                raise SafetyError(f"Invalid safety setting: {field}")
        if not 0 < safety["min_confidence"] <= 1:
            raise SafetyError("Invalid confidence threshold")

    def latch(self, reason: str) -> None:
        self.latched, self.reason = True, reason

    def validate_observation(self, obs: Observation) -> None:
        if self.latched:
            raise SafetyError(f"Stop latched: {self.reason}")
        age = (time.monotonic_ns() - obs.timestamp_ns) / 1e9
        if age < 0 or age > self.config["safety"]["max_observation_age_s"]:
            raise SafetyError("Stale or future observation")
        skew = obs.sensor_skew_ns / 1e9
        if skew < 0 or skew > self.config["safety"]["max_sensor_skew_s"]:
            raise SafetyError("RGB/depth timestamp skew exceeds limit")
        for name, detection in obs.detections.items():
            if name != detection.object_id or detection.frame_id != "base_link":
                raise SafetyError("Detection identity/frame mismatch")
            if not math.isfinite(detection.confidence):
                raise SafetyError("Non-finite confidence")
            if not self.config["safety"]["min_confidence"] <= detection.confidence <= 1:
                raise SafetyError(f"Low or invalid confidence: {name}")
            self.validate_position(detection.position_m)

    def validate_position(self, position) -> None:
        if len(position) != 3 or not all(math.isfinite(x) for x in position):
            raise SafetyError("Target position must contain three finite values")
        if any(
            not lower <= value <= upper
            for value, (lower, upper) in zip(
                position, self.config["safety"]["workspace_m"], strict=True
            )
        ):
            raise SafetyError("Position outside configured workspace")

    def validate_action(self, action: Action, obs: Observation) -> None:
        self.validate_observation(obs)
        self.validate_position(action.target_m)
        if action.frame_id != "base_link" or ACTION_OBJECTS.get(action.name) != action.object_id:
            raise SafetyError("Unknown action, object or coordinate frame")
        detection = obs.detections.get(action.object_id)
        if detection is None:
            raise SafetyError(f"Target not observed: {action.object_id}")
        if (
            action.name != "return_spoon"
            and math.dist(action.target_m, detection.position_m)
            > self.config["safety"]["target_tolerance_m"]
        ):
            raise SafetyError("Action target differs from current visual localization")
        held = obs.feedback.held_object
        if action.name.startswith("grasp_") and held is not None:
            raise SafetyError("Cannot grasp while already holding an object")
        if action.name in {"acquire_food", "pour_food", "return_spoon"} and held != "spoon":
            raise SafetyError("Spoon is not held")
        if action.name == "acquire_food" and obs.feedback.spoon_loaded:
            raise SafetyError("Spoon already loaded")
        if action.name == "pour_food" and not obs.feedback.spoon_loaded:
            raise SafetyError("No food in spoon")
        if action.name == "return_spoon" and obs.feedback.spoon_loaded:
            raise SafetyError("Cannot return loaded spoon")
        if action.name == "place_bowl" and held != "bowl":
            raise SafetyError("Bowl is not held")
        if action.name == "grasp_bowl" and not obs.feedback.bowl_loaded:
            raise SafetyError("Bowl is empty")
