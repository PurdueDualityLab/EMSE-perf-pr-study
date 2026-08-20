from pathlib import Path
import json
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import classify_performance_prs as batch  # noqa: E402


def source(tmp_path, rows):
    path = tmp_path / "input.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_partition_enforces_limits_and_preserves_batch_payload(monkeypatch):
    monkeypatch.setattr(batch.aidev, "request_payload", lambda model, row, limit: {"model": model, "messages": [{"role": "user", "content": row["body"]}]})
    rows = [{"repo_id": 1, "number": number, "title": "plain", "body": "x"} for number in range(1, 4)]
    parts = batch.partition_requests(rows, "luna", 10, 2, 10_000)
    assert [len(part) for part in parts] == [2, 1]
    line = json.loads(parts[0][0][2])
    assert line["custom_id"] == "1#1"
    assert line["url"] == "/v1/chat/completions"
    assert line["body"] == {"model": "luna", "messages": [{"role": "user", "content": "x"}]}
    with pytest.raises(ValueError, match="50000"):
        batch.partition_requests(rows, "luna", 10, 50_001, 10_000)
    with pytest.raises(ValueError, match="exceeds"):
        batch.partition_requests(rows, "luna", 10, 1, 10)


def test_submit_skips_automatic_rows_splits_and_is_idempotent(tmp_path, monkeypatch):
    path = source(tmp_path, [{"repo_id": 1, "number": 1, "title": "fix: x", "body": ""}, {"repo_id": 1, "number": 2, "title": "plain", "body": ""}, {"repo_id": 1, "number": 3, "title": "plain", "body": ""}])
    monkeypatch.setattr(batch.aidev, "request_payload", lambda model, row, limit: {"model": model, "messages": []})
    uploads, creates = [], []
    monkeypatch.setattr(batch, "upload_batch_file", lambda endpoint, key, file: uploads.append(file) or f"file-{len(uploads)}")
    monkeypatch.setattr(batch, "create_batch", lambda endpoint, key, file_id: creates.append(file_id) or {"id": f"batch-{len(creates)}", "status": "validating"})
    result = batch.submit(path, tmp_path / "out", "luna", "https://example.test/v1/chat/completions", "secret", 10, max_requests=1)
    assert result["submitted_batches"] == 2
    assert len(uploads) == len(creates) == 2
    state = json.loads((tmp_path / "out" / "state.json").read_text())
    assert [item["custom_ids"] for item in state["batches"]] == [["1#2"], ["1#3"]]
    batch.submit(path, tmp_path / "out", "luna", "https://example.test/v1/chat/completions", "secret", 10, max_requests=1, resume=True)
    assert len(uploads) == len(creates) == 2
    assert "secret" not in (tmp_path / "out" / "state.json").read_text()


def test_collect_is_order_independent_and_finalize_counts_all_rows(tmp_path, monkeypatch):
    path = source(tmp_path, [{"repo_id": 1, "number": 1, "title": "feat: automatic", "body": ""}, {"repo_id": 1, "number": 2, "title": "plain", "body": ""}, {"repo_id": 1, "number": 3, "title": "plain", "body": ""}])
    out = tmp_path / "out"
    out.mkdir()
    signature = batch.state_signature(path, "luna", "https://example.test/v1/chat/completions", 10)
    state = {"signature": signature, "batches": [{"batch_id": "b1", "custom_ids": ["1#2", "1#3"], "request_count": 2, "status": "completed"}], "completed": {}}
    (out / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(batch, "retrieve_batch", lambda *args: {"status": "completed", "output_file_id": "out1"})
    response = lambda label: {"id": label, "choices": [{"message": {"content": json.dumps({"reason": label, "output": "refactor", "confidence": 6})}}]}
    monkeypatch.setattr(batch, "download_batch_file", lambda *args: "\n".join(json.dumps({"custom_id": key, "response": {"status_code": 200, "body": response(key)}}) for key in ("1#3", "1#2")))
    assert batch.collect(path, out, "luna", "https://example.test/v1/chat/completions", "secret", 10, True)["completed_records"] == 2
    summary = batch.finalize(path, out, "luna", "https://example.test/v1/chat/completions", 10, True)
    decisions = pd.read_parquet(out / "task_type_decisions.parquet")
    assert decisions["number"].tolist() == [1, 2, 3]
    assert decisions["aidev_task_type_method"].tolist() == ["title_regex", "llm", "llm"]
    assert summary["rows"] == 3
    assert summary["title_regex_rows"] == 1
    assert summary["llm_needed_rows"] == 2
