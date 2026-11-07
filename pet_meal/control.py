"""Single command gate for the software simulator, not a physical emergency stop."""

import math
import threading
import time
from collections.abc import Callable

from .types import RunCancelled, TaskError


class RunControl:
    def __init__(self, step_interval_s: float = 0):
        if not math.isfinite(step_interval_s) or not 0 <= step_interval_s <= 5:
            raise TaskError("Step interval must be between 0 and 5 seconds")
        self.interval = step_interval_s
        self._condition = threading.Condition(threading.RLock())
        self._paused = False
        self._stopped = False

    def pause(self) -> None:
        with self._condition:
            if self._stopped:
                raise TaskError("Stop is latched")
            self._paused = True
            self._condition.notify_all()

    def resume(self) -> None:
        with self._condition:
            if self._stopped:
                raise TaskError("Stop is latched")
            self._paused = False
            self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()

    def _wait_unpaused(self) -> None:
        while self._paused and not self._stopped:
            self._condition.wait()
        if self._stopped:
            raise RunCancelled("Operator requested software stop")

    def checkpoint(self, paced: bool = True) -> None:
        with self._condition:
            self._wait_unpaused()
            remaining = self.interval if paced else 0
            while remaining > 0:
                before = time.monotonic()
                self._condition.wait(remaining)
                remaining -= time.monotonic() - before
                self._wait_unpaused()

    def execute(self, operation: Callable[[], None]) -> None:
        # Serializes pause/stop with command dispatch. Only instantaneous simulator
        # operations belong here; physical control needs a validated interrupt path.
        with self._condition:
            self._wait_unpaused()
            operation()

    def snapshot(self) -> dict:
        with self._condition:
            return {
                "paused": self._paused,
                "stopped": self._stopped,
                "step_interval_s": self.interval,
            }
