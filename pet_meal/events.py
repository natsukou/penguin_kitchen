"""Bounded in-process telemetry stream with monotonic cursors."""

import json
import threading
from collections import deque
from datetime import datetime, timezone

from .recording import encode


class EventBus:
    def __init__(self, capacity: int = 1500):
        self._events = deque(maxlen=capacity)
        self._lock = threading.RLock()
        self._cursor = 0

    def publish(self, row: dict) -> dict:
        # Detach references so worker state cannot mutate historical events.
        value = json.loads(json.dumps(row, default=encode, allow_nan=False))
        with self._lock:
            self._cursor += 1
            value["cursor"] = self._cursor
            value.setdefault("utc", datetime.now(timezone.utc).isoformat())
            self._events.append(value)
        return value

    def after(self, cursor: int = 0) -> dict:
        with self._lock:
            oldest = self._events[0]["cursor"] if self._events else self._cursor + 1
            return {
                "cursor": self._cursor,
                "truncated": cursor != 0 and (cursor < oldest - 1 or cursor > self._cursor),
                "events": [
                    row for row in self._events if row["cursor"] > cursor or cursor > self._cursor
                ],
            }
