import hashlib
import json

import cv2
import numpy as np

from pet_meal import tabletop_model


def test_reference_similarity_alone_does_not_label_onsite_food(tmp_path, monkeypatch):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture")
    image = tmp_path / "sample.jpg"
    assert cv2.imwrite(str(image), np.zeros((12, 12, 3), dtype=np.uint8))
    metadata = {
        "schema_version": 2,
        "backbone_sha256": hashlib.sha256(b"fixture").hexdigest(),
        "classes": [
            {"kind": "ingredient", "class_id": "tofu", "onsite_capture_groups": 0},
            {"kind": "ingredient", "class_id": "vegetable", "onsite_capture_groups": 1},
        ],
    }
    bank = tmp_path / "bank.npz"
    np.savez(bank, metadata=json.dumps(metadata), prototypes=np.eye(2))
    monkeypatch.setattr(tabletop_model, "load_local_yolo", lambda _: object())
    monkeypatch.setattr(tabletop_model, "_embed", lambda *_: np.array([[1.0, 0.0]]))

    report = tabletop_model.recognize_references(weights, bank, image, "ingredient")
    hit = report["detections"][0]
    assert hit["nearest_reference"] == "tofu"
    assert hit["class_id"] == "unknown"
    assert hit["status"] == "onsite_references_insufficient"
    assert report["dispatch_allowed"] is False
