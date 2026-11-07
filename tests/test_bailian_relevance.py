import io
import json
from urllib.error import HTTPError, URLError

import cv2
import numpy as np
import pytest

from pet_meal import bailian
from pet_meal.relevance import ROLES, decide_relevance, fixture_test, parse_evidence
from pet_meal.types import TaskError


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "scene.jpg"
    assert cv2.imwrite(str(path), np.zeros((80, 120, 3), dtype=np.uint8))
    return path


def provider_content():
    return {
        "objects": [
            {
                "role": role,
                "visibility": "visible" if role in {"spoon", "bowl"} else "not_visible",
                "evidence": "fixture observation",
            }
            for role in ROLES
        ],
    }


def use_fake_provider(monkeypatch, handler):
    class FakeOpener:
        def open(self, request, timeout):
            assert timeout == 25
            return handler(request)

    def opener(redirect_handler):
        assert isinstance(redirect_handler, bailian.NoRedirect)
        return FakeOpener()

    monkeypatch.setattr(bailian, "build_opener", opener)


def test_one_cloud_response_is_reused_across_local_phases(monkeypatch, image):
    calls = []
    key = "test-key-only-0000"

    def fake_response(request):
        calls.append(request.full_url)
        assert request.get_header("Authorization") == f"Bearer {key}"
        data = json.loads(request.data)
        assert data["model"] == bailian.DEFAULT_MODEL
        assert data["response_format"] == {"type": "json_object"}
        assert data["enable_thinking"] is False
        assert data["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;")
        assert "grasp_spoon" not in json.dumps(data)
        return io.BytesIO(
            json.dumps(
                {
                    "id": "request_fixture",
                    "usage": {"total_tokens": 321, "debug": key},
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(provider_content())},
                        }
                    ],
                }
            ).encode()
        )

    use_fake_provider(monkeypatch, fake_response)
    result = bailian.analyze_image(image, key, bailian.DEFAULT_BASE_URL, bailian.DEFAULT_MODEL)
    assert calls == [bailian.DEFAULT_BASE_URL + "/chat/completions"]
    assert result["status"] == "completed"
    assert result["request_id"] == "request_fixture"
    assert result["usage"] == {"total_tokens": 321}
    assert key not in json.dumps(result)
    decisions = result["decisions"]
    assert [d["outcome"] for d in decisions] == ["candidate", "candidate", "hold"]
    assert [d["target_role"] for d in decisions] == ["spoon", "bowl", "food_container"]
    assert all(not d["dispatch_allowed"] for d in decisions)
    assert result["dispatch_allowed"] is False


def test_offline_fixture_matrix_and_phase_switch():
    result = fixture_test()
    assert result["passed"] == result["total"] == 6
    observations = parse_evidence(json.dumps(provider_content()))
    spoon = decide_relevance("grasp_spoon", observations)
    bowl = decide_relevance("grasp_bowl", observations)
    assert {r["role"]: r["relevance"] for r in spoon["objects"]}["spoon"] == "relevant"
    assert {r["role"]: r["relevance"] for r in bowl["objects"]}["spoon"] == "irrelevant"


@pytest.mark.parametrize("change", ["duplicate_role", "unknown_role", "extra_motion", "bad_status"])
def test_rejects_untrusted_provider_fields(change):
    payload = provider_content()
    if change == "duplicate_role":
        payload["objects"][0]["role"] = payload["objects"][1]["role"]
    elif change == "unknown_role":
        payload["objects"][0]["role"] = "knife"
    elif change == "extra_motion":
        payload["objects"][0]["joint_targets"] = [0.0]
    else:
        payload["objects"][0]["visibility"] = "high"
    with pytest.raises(TaskError):
        parse_evidence(json.dumps(payload))


@pytest.mark.parametrize(
    "url",
    [
        "http://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://dashscope.aliyuncs.com.evil.example/compatible-mode/v1",
        "https://x@dashscope.aliyuncs.com/compatible-mode/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1?key=example",
        "https://ws.example.maas.aliyuncs.com/compatible-mode/v1",
    ],
)
def test_rejects_nonofficial_or_modified_endpoint(url):
    with pytest.raises(TaskError):
        bailian.endpoint_for(url)


def test_workspace_endpoint_is_supported():
    url = "https://ws-test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    assert bailian.endpoint_for(url) == url + "/chat/completions"


def test_missing_key_never_attempts_request(monkeypatch, image):
    def fail_if_called(*args):
        pytest.fail("Cloud request attempted with no credential")

    monkeypatch.setattr(bailian, "_request_evidence", fail_if_called)
    result = bailian.analyze_image(image, "", bailian.DEFAULT_BASE_URL, bailian.DEFAULT_MODEL)
    assert result["status"] == "unavailable"
    assert result["request_attempted"] is False
    assert all(d["outcome"] == "hold" for d in result["decisions"])


@pytest.mark.parametrize("failure", ["network", "unauthorized", "truncated", "malformed"])
def test_provider_failure_holds_every_phase_and_redacts_body(monkeypatch, image, failure):
    key = "test-key-only-0000"

    def fake_response(request):
        if failure == "network":
            raise URLError(key)
        if failure == "unauthorized":
            raise HTTPError(request.full_url, 401, key, {}, io.BytesIO(key.encode()))
        if failure == "truncated":
            return io.BytesIO(b'{"choices":[{"finish_reason":"length"}]}')
        return io.BytesIO(
            b'{"choices":[{"finish_reason":"stop","message":{"content":"bad JSON"}}]}'
        )

    use_fake_provider(monkeypatch, fake_response)
    result = bailian.analyze_image(image, key, bailian.DEFAULT_BASE_URL, bailian.DEFAULT_MODEL)
    assert result["status"] == "unavailable"
    assert result["request_attempted"] is True
    assert key not in json.dumps(result)
    assert all(d["outcome"] == "hold" and not d["dispatch_allowed"] for d in result["decisions"])
