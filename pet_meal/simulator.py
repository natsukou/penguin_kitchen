"""Synthetic scene and image rendering; no physics or robot dynamics claims."""

import time
from pathlib import Path

import cv2
import numpy as np

from .types import Action, Feedback, Observation, TaskError
from .vision import MARKER_BGR, Calibration, detect_markers

FAULTS = (
    "none",
    "missing_spoon",
    "stale_frame",
    "grasp_slip",
    "placement_error",
    "depth_missing",
    "camera_skew",
    "command_failure",
)


class Simulator:
    evidence_level = "simulation_only"

    def __init__(self, config: dict, fault: str = "none", frame_dir: Path | None = None):
        if (
            config.get("backend") != "simulation"
            or config.get("hardware_motion_enabled") is not False
        ):
            raise TaskError("Simulator requires an explicit simulation-only configuration")
        if fault not in FAULTS:
            raise TaskError(f"Unknown fault: {fault}")
        self.positions = {key: tuple(value) for key, value in config["scene"].items()}
        self.held = None
        self.spoon_loaded = False
        self.bowl_loaded = False
        self.stopped = False
        self.executed: list[Action] = []
        self.fault = fault
        self.frame_dir = frame_dir
        self.frame_index = 0
        self.calibration = Calibration(
            500,
            500,
            320,
            240,
            ((0, -1, 0, 0.45), (-1, 0, 0, 0), (0, 0, -1, 1.2), (0, 0, 0, 1)),
        )

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((480, 640), dtype=np.float32)
        base_to_camera = np.linalg.inv(self.calibration.matrix())
        for name in ("mat", "food_container", "spoon", "bowl"):
            if self.fault == "missing_spoon" and name == "spoon":
                continue
            camera = base_to_camera @ np.array([*self.positions[name], 1.0])
            u = round(self.calibration.fx * camera[0] / camera[2] + self.calibration.cx)
            v = round(self.calibration.fy * camera[1] / camera[2] + self.calibration.cy)
            radius = 24 if name == "mat" else 8
            cv2.rectangle(
                bgr, (u - radius, v - radius), (u + radius, v + radius), MARKER_BGR[name], -1
            )
            value = 0 if self.fault == "depth_missing" and name == "spoon" else camera[2]
            cv2.rectangle(
                depth, (u - radius, v - radius), (u + radius, v + radius), float(value), -1
            )
        return bgr, depth

    def observe(self) -> Observation:
        timestamp = time.monotonic_ns()
        bgr, depth = self.render()
        if self.frame_dir is not None:
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            path = self.frame_dir / f"{self.frame_index:03d}-synthetic-rgb.png"
            if not cv2.imwrite(str(path), bgr):
                raise TaskError("Failed to record RGB frame")
            np.save(self.frame_dir / f"{self.frame_index:03d}-synthetic-depth-m.npy", depth)
        self.frame_index += 1
        return Observation(
            timestamp - (10_000_000_000 if self.fault == "stale_frame" else 0),
            detect_markers(bgr, depth, self.calibration),
            Feedback(self.held, self.spoon_loaded, self.bowl_loaded, "simulated_state"),
            "synthetic_rgbd_markers",
            100_000_000 if self.fault == "camera_skew" else 0,
        )

    def execute(self, action: Action) -> None:
        if self.stopped:
            raise TaskError("Simulator is stopped")
        if self.fault == "command_failure" and action.name == "pour_food":
            raise TaskError("Injected backend execution failure")
        self.executed.append(action)
        if action.name in {"grasp_spoon", "grasp_bowl"}:
            self.held = action.object_id if self.fault != "grasp_slip" else None
        elif action.name == "acquire_food":
            self.spoon_loaded = True
        elif action.name == "pour_food":
            self.spoon_loaded, self.bowl_loaded = False, True
        elif action.name == "return_spoon":
            self.positions["spoon"] = action.target_m
            self.held = None
        elif action.name == "place_bowl":
            x, y, _ = action.target_m
            self.positions["bowl"] = (x + (0.08 if self.fault == "placement_error" else 0), y, 0.1)
            self.held = None
        else:
            raise TaskError(f"Unsupported simulator action: {action.name}")

    def stop(self) -> None:
        self.stopped = True
