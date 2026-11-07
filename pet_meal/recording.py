import hashlib
import json
import platform
import uuid
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import __version__


def encode(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)}")


class Recorder:
    def __init__(
        self,
        directory: Path,
        config: dict,
        evidence_level: str,
        on_event: Callable[[dict], None] | None = None,
    ):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.run_id = uuid.uuid4().hex
        self.evidence_level = evidence_level
        self.on_event = on_event
        self.path = directory / "events.jsonl"
        self.stream = self.path.open("x", encoding="utf-8")
        source_digest = hashlib.sha256()
        for source in sorted(Path(__file__).parent.rglob("*")):
            if source.suffix not in {".py", ".html", ".css", ".js"}:
                continue
            source_digest.update(source.relative_to(Path(__file__).parent).as_posix().encode())
            source_digest.update(source.read_bytes())
        self.event(
            "run_started",
            {
                "app_version": __version__,
                "source_sha256": source_digest.hexdigest(),
                "python": platform.python_version(),
                "config_sha256": hashlib.sha256(
                    json.dumps(config, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
                "config": config,
            },
        )

    def event(self, name: str, payload) -> None:
        row = {
            "schema_version": 1,
            "run_id": self.run_id,
            "utc": datetime.now(timezone.utc).isoformat(),
            "evidence_level": self.evidence_level,
            "event": name,
            "payload": payload,
        }
        self.stream.write(json.dumps(row, default=encode, allow_nan=False) + "\n")
        self.stream.flush()
        if self.on_event is not None:
            self.on_event(row)

    def finish(self, summary: dict) -> None:
        self.event("run_finished", summary)
        (self.directory / "summary.json").write_text(
            json.dumps(summary, indent=2, default=encode, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        self.stream.close()
