"""Scene-level recognition evidence imported from observation_all.json."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .types import TaskError, Vec3


@dataclass(frozen=True)
class PixelObservation:
    timestamp_ns: int
    pixel_px: tuple[float, float]


@dataclass(frozen=True)
class Landmark:
    name: str
    views: dict[str, PixelObservation]


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    name: str
    category: str
    description: str
    confidence: float
    position_m: Vec3 | None
    frame_id: str | None
    landmark: Landmark | None


@dataclass(frozen=True)
class SceneFeedback:
    held_object: str | None
    spoon_loaded: bool | None
    bowl_loaded: bool | None


@dataclass(frozen=True)
class SceneObservation:
    timestamp_ns: int | None
    detections: dict[str, SceneObject]
    feedback: SceneFeedback


def _fields(value, required, label, *, optional=()):
    required = set(required)
    if not isinstance(value, dict) or not required <= value.keys() <= required.union(optional):
        raise TaskError(f"Invalid fields in {label}")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise TaskError(f"{label} must be nonempty text")
    return value.strip()


def _stamp(value, label):
    if type(value) is not int or value <= 0:
        raise TaskError(f"{label} must be a positive integer timestamp")
    return value


def _number(value, label):
    if type(value) not in (int, float):
        raise TaskError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError:
        raise TaskError(f"{label} must be a finite number") from None
    if not math.isfinite(number):
        raise TaskError(f"{label} must be a finite number")
    return number


def _vector(value, length, label):
    if not isinstance(value, list) or len(value) != length:
        raise TaskError(f"{label} must contain {length} coordinates")
    return tuple(_number(component, label) for component in value)


def _landmark(value):
    value = _fields(value, ("name", "views"), "landmark")
    if not isinstance(value["views"], dict) or not value["views"]:
        raise TaskError("Landmark must contain camera observations")
    views = {}
    for camera, row in value["views"].items():
        _text(camera, "Camera ID")
        row = _fields(row, ("timestamp_ns", "pixel_px"), "landmark view")
        timestamp = _stamp(row["timestamp_ns"], "Camera timestamp_ns")
        u, v = _vector(row["pixel_px"], 2, "pixel_px")
        if min(u, v) < 0:
            raise TaskError("pixel_px must be nonnegative")
        views[camera] = PixelObservation(timestamp, (u, v))
    return Landmark(_text(value["name"], "Landmark name"), views)


def _detection(key, row):
    row = _fields(
        row,
        ("object_id", "name", "category", "description", "confidence", "position_m", "frame_id"),
        "detection",
        optional=("landmark",),
    )
    if _text(key, "Detection key") != key or row["object_id"] != key:
        raise TaskError("Detection key must match its object_id")
    confidence = _number(row["confidence"], "confidence")
    if not 0 <= confidence <= 1:
        raise TaskError("confidence must be within [0, 1]")
    position, frame = row["position_m"], row["frame_id"]
    if (position is None) != (frame is None):
        raise TaskError("position_m and frame_id must be provided together")
    return SceneObject(
        object_id=key,
        name=_text(row["name"], "name"),
        category=_text(row["category"], "category"),
        description=_text(row["description"], "description"),
        confidence=confidence,
        position_m=None if position is None else _vector(position, 3, "position_m"),
        frame_id=None if frame is None else _text(frame, "frame_id"),
        landmark=None if row.get("landmark") is None else _landmark(row["landmark"]),
    )


def _feedback(value):
    value = _fields(value, ("held_object", "spoon_loaded", "bowl_loaded"), "feedback")
    held, spoon, bowl = (value[key] for key in ("held_object", "spoon_loaded", "bowl_loaded"))
    if any(state is not None and type(state) is not bool for state in (spoon, bowl)):
        raise TaskError("Food feedback must be true, false or null")
    return SceneFeedback(None if held is None else _text(held, "held_object"), spoon, bowl)


def load_scene(path: Path) -> SceneObservation:
    """Read recognition evidence while preserving unknown geometry and feedback."""

    def unique_pairs(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise TaskError("Scene JSON contains duplicate keys")
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_pairs)
    except (ValueError, RecursionError) as exc:
        raise TaskError("Scene file is not valid JSON") from exc
    value = _fields(value, ("timestamp_ns", "detections", "feedback"), "scene")
    timestamp = value["timestamp_ns"]
    if not isinstance(value["detections"], dict):
        raise TaskError("detections must be an object indexed by object_id")
    return SceneObservation(
        timestamp_ns=None if timestamp is None else _stamp(timestamp, "Scene timestamp_ns"),
        detections={key: _detection(key, row) for key, row in value["detections"].items()},
        feedback=_feedback(value["feedback"]),
    )
