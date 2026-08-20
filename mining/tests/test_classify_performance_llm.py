from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import classify_performance_llm as llm  # noqa: E402


def response(label="performance"):
    return {
        "id": "resp_123",
        "output_text": f'{{"label":"{label}","rationale":"Evidence in metadata."}}',
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def test_parse_response_rejects_invalid_label():
    with pytest.raises(ValueError, match="invalid label"):
        llm.parse_response({"output_text": '{"label":"maybe","rationale":"x"}'})


def test_classify_writes_results_and_resumes(tmp_path, monkeypatch):
    source = tmp_path / "candidates.parquet"
    pd.DataFrame(
        [
            {"repo_id": 1, "number": 2, "title": "perf: faster", "body": "Reduce latency"},
            {"repo_id": 1, "number": 3, "title": "docs", "body": ""},
        ]
    ).to_parquet(source, index=False)
    calls = []

    def fake_post(endpoint, api_key, payload, timeout):
        calls.append(payload)
        return response("performance" if len(calls) == 1 else "uncertain")

    monkeypatch.setattr(llm, "post_response", fake_post)
    summary = llm.classify(source, tmp_path / "out", "model-x", "https://example.test", "key", 1, 1, False, False)
    output = pd.read_parquet(tmp_path / "out" / "decisions.parquet")

    assert summary["status_counts"] == {"classified": 1, "uncertain": 1}
    assert output["llm_label"].tolist() == ["performance", "uncertain"]
    assert len(calls) == 2

    llm.classify(source, tmp_path / "out", "model-x", "https://example.test", "key", 1, 1, True, False)
    assert len(calls) == 2


def test_retry_errors_retries_only_errors(tmp_path, monkeypatch):
    source = tmp_path / "candidates.parquet"
    pd.DataFrame([{"repo_id": 1, "number": 2, "title": "perf", "body": ""}]).to_parquet(source, index=False)

    monkeypatch.setattr(llm, "post_response", lambda *args: (_ for _ in ()).throw(ValueError("bad reply")))
    llm.classify(source, tmp_path / "out", "model-x", "https://example.test", "key", 1, 1, False, False)
    monkeypatch.setattr(llm, "post_response", lambda *args: response())
    summary = llm.classify(source, tmp_path / "out", "model-x", "https://example.test", "key", 1, 1, True, True)

    assert summary["status_counts"] == {"classified": 1}
