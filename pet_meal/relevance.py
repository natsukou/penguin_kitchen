"""Typed vision evidence and stage-specific relevance, independent of execution."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .policy import TARGETS
from .types import State, TaskError

ROLES = tuple(dict.fromkeys(TARGETS.values()))
PHASES = {state.value: role for state, role in TARGETS.items()}
VISIBILITY = ("visible", "uncertain", "not_visible")


@dataclass(frozen=True)
class ObjectEvidence:
    role: str
    visibility: str
    evidence: str


def parse_evidence(content: str) -> tuple[ObjectEvidence, ...]:
    def unique_pairs(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise TaskError("Vision JSON contains duplicate keys")
        return result

    if not isinstance(content, str) or len(content) > 8192:
        raise TaskError("Vision response must be a bounded JSON string")
    try:
        value = json.loads(content, object_pairs_hook=unique_pairs)
    except (ValueError, RecursionError) as exc:
        raise TaskError("Vision response is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"objects"}:
        raise TaskError("Vision response must contain only objects")
    rows = value["objects"]
    if not isinstance(rows, list) or len(rows) != len(ROLES):
        raise TaskError("Vision response must report exactly one observation per task role")
    by_role = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"role", "visibility", "evidence"}:
            raise TaskError("Unexpected fields in vision observation")
        role, visibility, evidence = row["role"], row["visibility"], row["evidence"]
        if role not in ROLES or role in by_role:
            raise TaskError("Unknown or repeated task role in vision response")
        if visibility not in VISIBILITY:
            raise TaskError("Invalid visibility value")
        if not isinstance(evidence, str) or not 1 <= len(evidence.strip()) <= 240:
            raise TaskError("Missing or oversized visual evidence")
        by_role[role] = ObjectEvidence(role, visibility, evidence.strip())
    return tuple(by_role[role] for role in ROLES)


def decide_relevance(phase: str, observations: tuple[ObjectEvidence, ...]) -> dict:
    if phase not in PHASES:
        raise TaskError("Unknown relevance phase")
    target = PHASES[phase]
    objects = []
    for observation in observations:
        if observation.visibility != "visible":
            relevance = "insufficient"
        else:
            relevance = "relevant" if observation.role == target else "irrelevant"
        objects.append({**asdict(observation), "relevance": relevance})
    target_result = next((row for row in objects if row["role"] == target), None)
    return {
        "phase": phase,
        "target_role": target,
        "outcome": "candidate"
        if target_result and target_result["relevance"] == "relevant"
        else "hold",
        "objects": objects,
        "dispatch_allowed": False,
    }


def fixture_test() -> dict:
    """Exercise the same evidence parser and decision path without provider I/O."""
    sample = {"spoon": ("visible", "可见有柄的勺"), "bowl": ("visible", "可见开口碗")}
    cases = (
        ("spoon_with_bowl", State.GRASP_SPOON, sample, "candidate", {"bowl": "irrelevant"}),
        ("same_scene_new_phase", State.GRASP_BOWL, sample, "candidate", {"spoon": "irrelevant"}),
        (
            "ambiguous_utensil",
            State.GRASP_SPOON,
            {**sample, "spoon": ("uncertain", "器具边缘被遮挡，形状不完整")},
            "hold",
            {},
        ),
        ("no_spoon", State.GRASP_SPOON, {"bowl": sample["bowl"]}, "hold", {}),
        (
            "bowl_is_not_food_source",
            State.ACQUIRE_FOOD,
            {"bowl": sample["bowl"]},
            "hold",
            {"bowl": "irrelevant"},
        ),
        (
            "visible_mat",
            State.PLACE_BOWL,
            {**sample, "mat": ("visible", "可见有清晰边界的餐垫")},
            "candidate",
            {},
        ),
    )
    results = []
    for name, phase, statuses, expected, other_expectations in cases:
        objects = []
        for role in ROLES:
            visibility, evidence = statuses.get(role, ("not_visible", "画面中未找到可见证据"))
            objects.append({"role": role, "visibility": visibility, "evidence": evidence})
        observations = parse_evidence(json.dumps({"objects": objects}, ensure_ascii=False))
        decision = decide_relevance(phase.value, observations)
        relevant = {row["role"]: row["relevance"] for row in decision["objects"]}
        expected_roles = {
            decision["target_role"]: "relevant" if expected == "candidate" else "insufficient",
            **other_expectations,
        }
        passed = (
            decision["outcome"] == expected
            and not decision["dispatch_allowed"]
            and all(
                relevant[role] == expected_label for role, expected_label in expected_roles.items()
            )
        )
        results.append({"name": name, "passed": passed, "expected": expected, **decision})
    return {
        "source": "offline_fixtures",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": sum(row["passed"] for row in results),
        "total": len(results),
        "cases": results,
    }


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
