"""Optional local-only semantic detection; never dispatches robot actions."""

import hashlib
import json
import os
from pathlib import Path

import cv2

from .types import TaskError


def predict_image(weights: Path, image: Path, output: Path, device: str = "cpu") -> dict:
    if not weights.is_file() or not image.is_file():
        raise TaskError("Local weights and image are required; automatic downloads are disabled")
    # Keep third-party settings inside the project and disable analytics/autoinstall.
    os.environ["YOLO_CONFIG_DIR"] = str(
        Path(__file__).resolve().parents[1] / "artifacts/yolo-config"
    )
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    model = YOLO(str(weights.resolve()))
    result = model.predict(str(image.resolve()), device=device, verbose=False)[0]
    detections = [
        {
            "class_id": int(box.cls.item()),
            "label": result.names[int(box.cls.item())],
            "confidence": float(box.conf.item()),
            "box_xyxy_px": box.xyxy[0].cpu().tolist(),
        }
        for box in result.boxes
    ]
    output.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(output / "annotated.jpg"), result.plot()):
        raise TaskError("Failed to save annotated image")
    report = {
        "evidence_level": "offline_model_inference_only",
        "weights": str(weights.resolve()),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "image": str(image.resolve()),
        "device": device,
        "input_shape": list(result.orig_shape),
        "detections": detections,
        "speed_ms": result.speed,
        "pet_meal_scene_accuracy": "not_evaluated",
        "robot_control": "not_connected",
    }
    (output / "predictions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
