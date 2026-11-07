import hashlib
import math

from .types import TaskError


def split_episodes(episodes: list[dict], seed: int = 42, ratios=(0.8, 0.1, 0.1)) -> dict:
    """Partition complete episodes; sessions remain together. Real training only."""
    if len(ratios) != 3 or not all(math.isfinite(r) and r >= 0 for r in ratios):
        raise TaskError("Invalid split ratios")
    if not math.isclose(sum(ratios), 1):
        raise TaskError("Split ratios must sum to one")
    seen = set()
    groups: dict[str, list[str]] = {}
    for episode in episodes:
        key = episode["episode_id"]
        if not isinstance(key, str) or not key or key in seen:
            raise TaskError("Episode IDs must be unique nonempty strings")
        if episode.get("evidence_level") != "hardware_recording":
            raise TaskError("Synthetic/unverified episodes cannot enter the real training split")
        seen.add(key)
        group = episode.get("session_id", key)
        if not isinstance(group, str) or not group:
            raise TaskError("Session ID must be a nonempty string")
        groups.setdefault(group, []).append(key)
    split = {"train": [], "validation": [], "test": []}
    for group, keys in sorted(groups.items()):
        digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        name = (
            "train"
            if fraction < ratios[0]
            else "validation"
            if fraction < ratios[0] + ratios[1]
            else "test"
        )
        split[name].extend(sorted(keys))
    return split
