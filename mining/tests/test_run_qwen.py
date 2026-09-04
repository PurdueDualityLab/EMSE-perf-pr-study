import json
from pathlib import Path
import sys

import pandas as pd

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
        4_096,
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


def test_rq1_label_adjudication_uses_unique_catalog_parent():
    content = json.dumps(
        {
            "explanation": "Explanation",
            "optimization_comparison": "Comparison",
            "high_level_pattern": "Wrong parent",
            "sub_pattern": "Streaming Implementation Or Optimization",
        }
    )

    label, adjudication = run_qwen.pattern_label_with_adjudication(
        content,
        {"I/O": ["Streaming Implementation Or Optimization)"]},
    )

    assert label.high_level_pattern == "I/O"
    assert label.sub_pattern == "Streaming Implementation Or Optimization)"
    assert adjudication["original_high_level_pattern"] == "Wrong parent"
    assert adjudication["corrected_high_level_pattern"] == "I/O"


def test_rq2_label_normalization_deduplicates_set_like_fields():
    content = json.dumps(
        {
            "validation_present": True,
            "validation_types": ["benchmark", "benchmark"],
            "primary_validation_type": "benchmark",
            "evidence_sources": ["description", "description"],
            "metrics": ["10 ms"],
            "evidence_quotes": ["10 ms"],
            "validation_description": "A benchmark is reported.",
        }
    )

    label, normalization = run_qwen.validation_label_with_normalization(content)

    assert label.validation_types == ["benchmark"]
    assert label.evidence_sources == ["description"]
    assert set(normalization["changes"]) == {"validation_types", "evidence_sources"}


def test_bounded_rq2_retry_schema_caps_set_like_fields_without_mutating_source():
    schema = run_qwen.run_rq2.semantic_schema()

    bounded = run_qwen.bounded_rq2_retry_schema(schema)

    assert bounded["properties"]["validation_types"]["maxItems"] == 4
    assert bounded["properties"]["evidence_sources"]["maxItems"] == 5
    assert "maxItems" not in schema["properties"]["validation_types"]
    assert "maxItems" not in schema["properties"]["evidence_sources"]


def test_atomic_checkpoint_path_uses_pr_identity(tmp_path):
    study = run_qwen.Study("rq2", run_qwen.run_rq2, Path(), Path(), tmp_path)

    path = run_qwen._checkpoint_path(study, "123:45")
    run_qwen.atomic_write_json(path, {"classification_status": "classified"})

    assert path == tmp_path / "responses" / "123-45.json"
    assert json.loads(path.read_text())["classification_status"] == "classified"


def test_smoke_manifest_selection_is_deterministic_and_balanced():
    manifest = pd.DataFrame(
        [
            {"repo_id": 2, "number": 2, "sample_arm": "human_candidate"},
            {"repo_id": 1, "number": 3, "sample_arm": "agentic"},
            {"repo_id": 1, "number": 1, "sample_arm": "agentic"},
            {"repo_id": 2, "number": 1, "sample_arm": "human_candidate"},
        ]
    )

    selected = run_qwen.select_manifest(manifest, 1)

    assert selected[["repo_id", "number"]].to_records(index=False).tolist() == [(1, 1), (2, 1)]
    assert selected["sample_arm"].value_counts().to_dict() == {
        "agentic": 1,
        "human_candidate": 1,
    }
