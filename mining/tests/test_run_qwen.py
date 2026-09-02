import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis import run_qwen  # noqa: E402


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(
            {
                "model": "qwen3.8:27b",
                "done": True,
                "message": {"role": "assistant", "content": '{"answer": true}'},
            }
        ).encode()


def test_ollama_chat_requests_deterministic_structured_output(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(run_qwen.urllib.request, "urlopen", fake_urlopen)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "boolean"}},
        "required": ["answer"],
    }

    response = run_qwen.ollama_chat(
        "http://localhost:11434/api/chat",
        "qwen3.8:27b",
        "Classify.",
        "Input",
        schema,
        True,
        65_536,
        30,
    )

    assert response["done"] is True
    assert captured["timeout"] == 30
    assert captured["payload"]["format"] == schema
    assert captured["payload"]["think"] is True
    assert captured["payload"]["options"] == {
        "temperature": 0,
        "seed": 42,
        "num_ctx": 65_536,
        "num_predict": 4_096,
    }


def test_rq1_label_validation_rejects_taxonomy_mismatch():
    content = json.dumps(
        {
            "explanation": "Explanation",
            "optimization_comparison": "Comparison",
            "high_level_pattern": "Known",
            "sub_pattern": "Unknown",
        }
    )

    try:
        run_qwen._validated_label(
            run_qwen.Study("rq1", run_qwen.run_rq1, Path(), Path(), Path()),
            content,
            {"Known": ["Expected"]},
        )
    except ValueError as error:
        assert "Invalid sub-pattern" in str(error)
    else:
        raise AssertionError("Expected invalid taxonomy label to fail")


def test_atomic_checkpoint_path_uses_pr_identity(tmp_path):
    study = run_qwen.Study("rq2", run_qwen.run_rq2, Path(), Path(), tmp_path)

    path = run_qwen._checkpoint_path(study, "123:45")
    run_qwen.atomic_write_json(path, {"classification_status": "classified"})

    assert path == tmp_path / "responses" / "123-45.json"
    assert json.loads(path.read_text())["classification_status"] == "classified"
