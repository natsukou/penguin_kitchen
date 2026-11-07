from dataclasses import dataclass
from enum import Enum
from typing import Protocol

Vec3 = tuple[float, float, float]


class TaskError(RuntimeError):
    pass


class SafetyError(TaskError):
    pass


class HardwareDisabled(TaskError):
    pass


class RunCancelled(TaskError):
    pass


class State(str, Enum):
    OBSERVE = "observe"
    LOCATE_SPOON = "locate_spoon"
    GRASP_SPOON = "grasp_spoon"
    ACQUIRE_FOOD = "acquire_food"
    POUR_FOOD = "pour_food"
    RETURN_SPOON = "return_spoon"
    GRASP_BOWL = "grasp_bowl"
    PLACE_BOWL = "place_bowl"
    VERIFY = "verify"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Detection:
    object_id: str
    position_m: Vec3
    confidence: float
    frame_id: str = "base_link"


@dataclass(frozen=True)
class Feedback:
    held_object: str | None
    spoon_loaded: bool
    bowl_loaded: bool
    source: str


@dataclass(frozen=True)
class Observation:
    timestamp_ns: int  # time.monotonic_ns, not a hardware camera timestamp
    detections: dict[str, Detection]
    feedback: Feedback
    source: str
    sensor_skew_ns: int = 0


@dataclass(frozen=True)
class Action:
    name: str
    target_m: Vec3
    object_id: str
    frame_id: str = "base_link"


class Backend(Protocol):
    evidence_level: str

    def observe(self) -> Observation: ...

    def execute(self, action: Action) -> None: ...

    def stop(self) -> None: ...
