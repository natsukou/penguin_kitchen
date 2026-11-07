"""Deterministic action selection. Learned policies are a separate future adapter."""

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
