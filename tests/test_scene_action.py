import json

import cv2
import numpy as np

from pet_meal.cli import main
from pet_meal.local_vlm import choose_grasp_local
from pet_meal.recognition import SceneObject


def test_scene_cli_selects_by_instance_and_preserves_unknown_feedback(tmp_path, capsys):
    path = tmp_path / "observation_all.json"
    detections = {
        object_id: {
            "object_id": object_id,
            "name": name,
            "category": "bottle",
            "description": name,
            "confidence": confidence,
            "position_m": None,
            "frame_id": None,
        }
        for object_id, name, confidence in (
            ("bottle_01", "蓝盖水瓶", 0.95),
            ("bottle_02", "粉橙水壶", 0.98),
        )
    }
    scene = {
        "timestamp_ns": None,
        "detections": detections,
        "feedback": {"held_object": "unknown", "spoon_loaded": None, "bowl_loaded": None},
    }
    path.write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")
    command = ["scene-action", "--observation", str(path)]

    assert main([*command, "--category", "bottle"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["next_action"] == "disambiguate_target"
    assert result["target"] is None
    assert {row["object_id"] for row in result["candidates"]} == set(detections)

    assert main([*command, "--object-id", "bottle_01"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["target"]["object_id"] == "bottle_01"
    assert result["next_action"] == "localize_target"
    assert result["timestamp_ns"] is None
    assert result["feedback"] == scene["feedback"]
    assert result["dispatch_allowed"] is False

    choice = tmp_path / "visual-choice.json"
    choice.write_text('{"action":"grasp","object_id":"bottle_01"}', encoding="utf-8")
    imported = [*command, "--category", "bottle", "--vision-choice", str(choice)]
    assert main(imported) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["next_action"] == "localize_target"
    assert result["imported_vision"]["selection"]["object_id"] == "bottle_01"
    assert result["dispatch_allowed"] is False

    choice.write_text('{"action":"grasp","object_id":"unlisted"}', encoding="utf-8")
    assert main(imported) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["next_action"] == "hold"
    assert result["dispatch_allowed"] is False

    assert main([*command, "--object-id", "absent"]) == 0
    assert json.loads(capsys.readouterr().out)["next_action"] == "reobserve_target"

    detections["bottle_01"]["object_id"] = "bottle_02"
    path.write_text(json.dumps(scene), encoding="utf-8")
    assert main([*command, "--object-id", "bottle_01"]) == 2
    assert "key must match" in json.loads(capsys.readouterr().out)["error"]


def test_local_vlm_uses_same_action_contract_without_cloud(monkeypatch, tmp_path):
    image = tmp_path / "scene.jpg"
    assert cv2.imwrite(str(image), np.zeros((80, 120, 3), dtype=np.uint8))
    (tmp_path / "model.safetensors").touch()
    candidates = [SceneObject("bottle_01", "蓝盖水瓶", "bottle", "透明", 0.95, None, None, None)]

    def infer(model_dir, jpeg, prompt):
        assert model_dir == tmp_path
        assert jpeg.startswith(b"\xff\xd8")
        assert "bottle_01" in prompt
        return '{"action":"grasp","object_id":"bottle_01"}', "cpu"

    monkeypatch.setattr("pet_meal.local_vlm._infer", infer)
    result = choose_grasp_local(image, "抓蓝盖瓶", candidates, tmp_path)
    assert result["selection"] == {"action": "grasp", "object_id": "bottle_01"}
    assert result["source"] == "local_vlm"
    assert result["dispatch_allowed"] is False
