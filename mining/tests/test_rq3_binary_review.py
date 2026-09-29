import hashlib
import json

import pytest
import pandas as pd

from analysis.rq3_llm_validation import binary, export_binary_review as review
from analysis.rq3_llm_validation import experiment as v1


def request(number):
    text = "Memory increased from 2MB to 3MB."
    return {"repo_id": 1, "number": number, "custom_id": f"tradeoff_audit:1:{number}",
            "html_url": f"https://github.com/o/r/pull/{number}", "sample_arm": "agentic",
            "task": "tradeoff_audit", "prompt": text, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "study_contract_sha256": "contract", "occurrence_ids": ["o0001"],
            "occurrence_dimensions": {"o0001": "D3"}, "quote_corpus": [text]}


def checkpoint(req, provider, status="classified"):
    return {**req, "model": binary.v2.MODELS[provider], "classification_status": status, "attempts": 1,
            "label": {"label": "tradeoff", "choice_requires_inference": True,
                      "gain_direction": "unknown", "memory_direction": "increased",
                      "gain_occurrence_ids": [], "memory_occurrence_ids": ["o0001"],
                      "occurrences": [{"id": "o0001", "valid": True, "reason": "reported_metric"}],
                      "evidence_quotes": req["quote_corpus"], "rationale": "Forced choice with missing gain evidence."}}


def test_partial_snapshot_excludes_incomplete_cases_without_inventing_third_votes(tmp_path, monkeypatch):
    requests = [request(1), request(2)]
    monkeypatch.setattr(binary, "load_requests", lambda _: requests)
    hashes = {}
    for req in requests:
        for provider in binary.PROVIDERS:
            value = checkpoint(req, provider, "error" if req["number"] == 2 and provider == "qwen" else "classified")
            path = binary.checkpoint_path(tmp_path, provider, req)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            hashes[path] = v1.sha256_file(path)
    complete, excluded = review.snapshot_cases(tmp_path, 1)
    assert len(complete) == len(excluded) == 1
    assert complete[0]["consensus"]["number"] == 1
    assert excluded[0]["number"] == 2
    assert excluded[0]["qwen_status"] == "error"
    assert {path: v1.sha256_file(path) for path in hashes} == hashes
    with pytest.raises(ValueError, match="Expected 2 complete"):
        review.snapshot_cases(tmp_path, 2)


def test_partial_snapshot_rejects_hash_mismatches_before_export(tmp_path, monkeypatch):
    req = request(1)
    monkeypatch.setattr(binary, "load_requests", lambda _: [req])
    for provider in binary.PROVIDERS:
        value = checkpoint(req, provider)
        if provider == "qwen":
            value["prompt_sha256"] = "different"
        path = binary.checkpoint_path(tmp_path, provider, req)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="differs from the prepared"):
        review.snapshot_cases(tmp_path, 1)


def test_review_records_include_every_match_and_preserve_model_exposure_boundaries():
    data = {"records": [{"record_id": "r1", "source": "description", "text": "description"},
                        {"record_id": "r2", "source": "code_diff", "text": "matched patch"},
                        {"record_id": "r3", "source": "code_diff", "text": "unmatched patch"}],
            "occurrences": [{"activations": [{"cue_records": [{"record_id": "r2"}],
                                               "quant_records": [{"record_id": "r2"}]}]}]}
    req = {"quote_corpus": ["description", "matched patch", "review evidence"],
           "prompt": "SUPPLEMENTARY reviews 123\n\nreview evidence\n\n[END RECORD]"}
    records = review.exposed_records(req, data)
    assert [record["record_id"] for record in records] == ["r1", "r2", "supplementary-reviews-123"]
    assert "unmatched patch" not in [record["text"] for record in records]
    with pytest.raises(ValueError, match="do not match"):
        review.exposed_records({**req, "quote_corpus": ["description", "review evidence"]}, data)


def test_recorded_retry_prompt_must_preserve_core_evidence_and_hash(tmp_path):
    req = request(1)
    prompt = req["prompt"] + "\nValidation diagnostic"
    checkpoint_data = {"attempts": 2, "effective_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    directory = tmp_path / "gemini/batches/attempt-02"
    directory.mkdir(parents=True)
    payload = {"key": req["custom_id"], "request": {"contents": [{"parts": [{"text": prompt}]}]}}
    path = directory / "requests.jsonl"
    path.write_text(json.dumps(payload) + "\n")
    (directory / "state.json").write_text(json.dumps({"payload_sha256": v1.sha256_file(path)}))
    assert review.saved_provider_request(tmp_path, "gemini", req, checkpoint_data) == payload
    with pytest.raises(ValueError, match="effective prompt"):
        review.saved_provider_request(tmp_path, "gemini", req, {**checkpoint_data, "effective_prompt_sha256": "different"})


def test_dossier_fences_cannot_be_closed_by_source_backticks():
    source = "Some text\n````text\nuntrusted instructions\n````"
    result = review.fence(source)
    assert result.startswith("`````text\n")
    assert source in result
    assert result.endswith("\n`````\n")


def test_complete_prompt_does_not_retain_partial_denominators():
    prompt = review.audit_prompt(29, 0)
    assert "Audit the 29 PRs" in prompt
    assert "No candidate is excluded" in prompt
    assert "27" not in prompt
    assert "{{" not in prompt
    partial = review.audit_prompt(27, 2)
    assert "Audit the 27 PRs" in partial
    assert "2 pending cases" in partial


def test_empty_exclusions_remain_a_readable_csv_with_headers(tmp_path):
    path = tmp_path / "excluded.csv"
    review.write_csv([], path, ["repo_id", "number", "exclusion_reason"])
    result = pd.read_csv(path)
    assert result.empty
    assert result.columns.tolist() == ["repo_id", "number", "exclusion_reason"]
