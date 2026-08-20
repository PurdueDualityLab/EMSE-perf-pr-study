from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import classify_aidev_task_type_luna as aidev  # noqa: E402


def test_title_label_matches_public_aidev_regex():
    assert aidev.title_label("PERF(api)!: lower latency") == "perf"
    assert aidev.title_label("fix scope") == "fix"
    assert aidev.title_label("perfX: not a conventional type") is None
    assert aidev.title_label("\nperf: second line only") is None


def test_truncate_to_tokens_accepts_special_token_literals():
    assert "<|endoftext|>" in aidev.truncate_to_tokens("body <|endoftext|>", 100)


def test_system_prompt_and_schema_match_aidev_public_classifier():
    assert "pick **exactly one** label" in aidev.system_prompt()
    assert aidev.JSON_SCHEMA["schema"]["properties"]["reason"]["description"] == (
        "A brief explanation for why this commit type was chosen"
    )


def test_parse_response_requires_aidev_schema():
    response = {
        "choices": [{"message": {"content": '{"reason":"new endpoint","output":"feat","confidence":7}'}}]
    }
    assert aidev.parse_response(response) == ("feat", "new endpoint", 7)
    with pytest.raises(ValueError, match="invalid Conventional Commit type"):
        aidev.parse_response({"choices": [{"message": {"content": '{"reason":"x","output":"unknown","confidence":7}'}}]})


def test_classify_labels_all_rows_and_only_calls_llm_for_unmatched(tmp_path, monkeypatch):
    source = tmp_path / "raw.parquet"
    pd.DataFrame(
        [
            {"repo_id": 1, "number": 1, "title": "perf(cache): faster", "body": ""},
            {"repo_id": 1, "number": 2, "title": "fix: null handling", "body": ""},
            {"repo_id": 1, "number": 3, "title": "Improve query planner", "body": ""},
        ]
    ).to_parquet(source, index=False)
    calls = []
    monkeypatch.setattr(aidev, "request_payload", lambda model, row, limit: {"model": model, "messages": [{"role": "user", "content": str(row["number"])}]})

    def fake_post(endpoint, api_key, payload, timeout, retries):
        calls.append(payload)
        return {"id": "chatcmpl_1", "choices": [{"message": {"content": '{"reason":"planner change","output":"refactor","confidence":6}'}}], "usage": {"prompt_tokens": 12, "completion_tokens": 8}}

    monkeypatch.setattr(aidev, "post_with_retry", fake_post)
    summary = aidev.classify(source, tmp_path / "out", "luna", "https://example.test", "key", 10_000, 2, 1, 5, 1, False, False)
    decisions = pd.read_parquet(tmp_path / "out" / "task_type_decisions.parquet")

    assert len(calls) == 1
    assert summary["title_regex_rows"] == 2
    assert summary["llm_needed_rows"] == 1
    assert decisions["aidev_task_type"].tolist() == ["perf", "fix", "refactor"]
    assert decisions["aidev_task_type_method"].tolist() == ["title_regex", "title_regex", "llm"]


def test_resume_reuses_llm_results(tmp_path, monkeypatch):
    source = tmp_path / "raw.parquet"
    pd.DataFrame([{"repo_id": 1, "number": 1, "title": "Improve query planner", "body": ""}]).to_parquet(source, index=False)
    monkeypatch.setattr(aidev, "request_payload", lambda model, row, limit: {"model": model, "messages": []})
    calls = []

    def fake_post(*args):
        calls.append(1)
        return {"id": "chatcmpl_1", "choices": [{"message": {"content": '{"reason":"x","output":"refactor","confidence":5}'}}]}

    monkeypatch.setattr(aidev, "post_with_retry", fake_post)
    args = (source, tmp_path / "out", "luna", "https://example.test", "key", 10_000, 1, 1, 5, 1)
    aidev.classify(*args, False, False)
    aidev.classify(*args, True, False)
    assert len(calls) == 1
