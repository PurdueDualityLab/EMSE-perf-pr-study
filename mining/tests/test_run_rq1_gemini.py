import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "rq1_optimization_patterns"))

from run_rq1_gemini import _parse_outputs, request_for, split_jsonl_lines  # noqa: E402


def test_internal_jobs_are_deterministic_disjoint_and_complete():
    lines = [f'{{"key": "{number}"}}\n' for number in range(5)]

    jobs = split_jsonl_lines(lines, max_requests=2, max_bytes=1_000)

    assert [len(job) for job in jobs] == [2, 2, 1]
    assert [line for job in jobs for line in job] == lines


def test_request_uses_gemini_json_schema():
    request = request_for("Classify this PR")
    config = request["generation_config"]

    assert request["contents"][0]["parts"][0]["text"] == "Classify this PR"
    assert "expert software engineer" in request["system_instruction"]["parts"][0]["text"]
    assert config["response_mime_type"] == "application/json"
    assert config["temperature"] == 0
    assert config["thinking_config"] == {"thinking_level": "MEDIUM"}
    assert config["max_output_tokens"] == 4096
    assert config["response_json_schema"]["additionalProperties"] is False
    assert set(config["response_json_schema"]["required"]) == {
        "explanation",
        "optimization_comparison",
        "high_level_pattern",
        "sub_pattern",
    }
    json.dumps(request)


def test_collected_rows_preserve_retry_key(tmp_path, monkeypatch):
    manifest = pd.DataFrame(
        [{"repo_id": 1, "number": 2, "key": "1:2", "study_contract_sha256": "s"}]
    )
    manifest.to_parquet(tmp_path / "batch_manifest.parquet", index=False)
    (tmp_path / "prepare_metadata.json").write_text(
        json.dumps({"model": "gemini", "catalog_file": "catalog.csv"})
    )
    monkeypatch.setattr(
        "run_rq1_gemini.load_taxonomy", lambda _: pd.DataFrame()
    )
    monkeypatch.setattr(
        "run_rq1_gemini.taxonomy_labels", lambda _: {"A": ["a"]}
    )
    output = json.dumps({"key": "1:2", "error": {"message": "failed"}})

    result = _parse_outputs(tmp_path, [output], {"1:2"})

    assert result["key"].tolist() == ["1:2"]


def test_missing_job_output_becomes_explicit_error(tmp_path, monkeypatch):
    manifest = pd.DataFrame(
        [{"repo_id": 1, "number": 2, "key": "1:2", "study_contract_sha256": "s"}]
    )
    manifest.to_parquet(tmp_path / "batch_manifest.parquet", index=False)
    (tmp_path / "prepare_metadata.json").write_text(
        json.dumps({"model": "gemini", "catalog_file": "catalog.csv"})
    )
    monkeypatch.setattr("run_rq1_gemini.load_taxonomy", lambda _: pd.DataFrame())
    monkeypatch.setattr("run_rq1_gemini.taxonomy_labels", lambda _: {"A": ["a"]})

    result = _parse_outputs(tmp_path, [], {"1:2"})

    assert result["classification_status"].tolist() == ["error"]
    assert result["key"].tolist() == ["1:2"]
