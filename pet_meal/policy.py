"""Deterministic action selection. Learned policies are a separate future adapter."""

from dataclasses import asdict
from pathlib import Path

from .recognition import SceneObservation
from .types import Action, Observation, State, TaskError, Vec3

TARGETS = {
    State.GRASP_SPOON: "spoon",
    State.ACQUIRE_FOOD: "food_container",
    State.POUR_FOOD: "bowl",
    State.RETURN_SPOON: "spoon",
    State.GRASP_BOWL: "bowl",
    State.PLACE_BOWL: "mat",
}


class FixedScenePolicy:
    def select(self, state: State, obs: Observation, spoon_home: Vec3 | None) -> Action:
        target = TARGETS[state]
        if state == State.RETURN_SPOON:
            if spoon_home is None:
                raise TaskError("No visually observed spoon return location")
            position = spoon_home
        else:
            position = obs.detections[target].position_m
        return Action(state.value, position, target)


def recommend_scene_grasp(
    scene: SceneObservation,
    *,
    object_id: str | None = None,
    category: str | None = None,
    capture: Path | None = None,
) -> dict:
    """Select the next step in a grasp task using scene evidence."""
    if (object_id is None) == (category is None):
        raise TaskError("Specify exactly one object_id or category")

    matches = [
        obj
        for obj in scene.detections.values()
        if (obj.object_id == object_id if object_id is not None else obj.category == category)
    ]
    report = {
        "intent": "grasp",
        "requested_target": {"object_id": object_id, "category": category},
        "timestamp_ns": scene.timestamp_ns,
        "object_count": len(scene.detections),
        "candidates": [
            {"object_id": obj.object_id, "name": obj.name, "confidence": obj.confidence}
            for obj in matches
        ],
        "target": None,
        "geometry": None,
        "feedback": asdict(scene.feedback),
        "dispatch_allowed": False,
    }
    if not matches:
        return {**report, "next_action": "reobserve_target", "reason": "target_not_detected"}
    if len(matches) > 1:
        return {**report, "next_action": "disambiguate_target", "reason": "multiple_instances"}

    target = matches[0]
    report["target"] = asdict(target)
    if target.landmark is None or capture is None:
        return {
            **report,
            "next_action": "localize_target",
            "reason": "matched_pixels_missing" if target.landmark is None else "capture_missing",
        }

    from .g1_multiview import locate_landmark

    try:
        report["geometry"] = locate_landmark(capture, target.landmark)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError) as exc:
        return {
            **report,
            "next_action": "localize_target",
            "reason": "geometry_rejected",
            "geometry_error": str(exc),
        }

    held = scene.feedback.held_object
    if held is not None:
        return {
            **report,
            "next_action": "verify_gripper",
            "reason": "gripper_unknown" if held == "unknown" else "gripper_reported_occupied",
        }
    return {**report, "next_action": "plan_grasp", "reason": "geometry_cross_checked"}
