import json
import hashlib

import pandas as pd
import pytest

from analysis.rq3_llm_validation import regex_v3, run_binary, v2
from analysis.rq3_llm_validation import experiment as v1


def request():
    return {
        "custom_id": "regex_audit_v3:1:2",
        "source_custom_id": "regex_audit:1:2",
        "repo_id": 1,
        "number": 2,
        "task": "regex_audit",
        "sample_arm": "agentic",
        "html_url": "https://github.com/o/r/pull/2",
        "prompt": v2.POLICY + "\n\nTASK: regex_audit.\n\nPR: url\nOriginal evidence",
        "prompt_sha256": "prompt",
        "source_prompt_sha256": "prompt",
        "study_contract_sha256": "contract",
        "occurrence_ids": ["o0001", "o0002"],
        "occurrence_dimensions": {"o0001": "D1", "o0002": "D3"},
        "quote_corpus": ["Runtime fell from 20ms to 10ms. Memory limit is 4MB."],
        "activation_count": 2,
    }


def payload(first=True, second=False):
    return {
        "occurrences": [
            {"id": "o0001", "valid": first, "reason": "reported_metric" if first else "incidental_value"},
            {"id": "o0002", "valid": second, "reason": "reported_metric" if second else "configuration"},
        ],
        "evidence_quotes": ["Runtime fell from 20ms to 10ms."],
        "rationale": "The runtime is reported; the memory value is only a limit.",
    }


def vote(req, provider, values):
    return {
        "custom_id": req["custom_id"],
        "repo_id": req["repo_id"],
        "number": req["number"],
        "sample_arm": req["sample_arm"],
        "html_url": req["html_url"],
        "prompt_sha256": req["prompt_sha256"],
        "study_contract_sha256": req["study_contract_sha256"],
        "provider": provider,
        "model": v2.MODELS[provider],
        "classification_status": "classified",
        "label": regex_v3.validate_result(values, req),
    }


def test_regex_v3_derives_pr_label_and_rejects_incomplete_occurrence_coverage():
    assert regex_v3.validate_result(payload(), request())["label"] == "true_positive"
    assert regex_v3.validate_result(payload(False, False), request())["label"] == "false_positive"
    invalid = payload()
    invalid["occurrences"].pop()
    with pytest.raises(ValueError, match="Occurrence IDs"):
        regex_v3.validate_result(invalid, request())


def test_regex_v3_consensus_is_majority_per_occurrence_then_any_valid():
    req = request()
    votes = {
        "openai": vote(req, "openai", payload(True, False)),
        "gemini": vote(req, "gemini", payload(False, False)),
        "qwen": vote(req, "qwen", payload(True, False)),
    }
    row, occurrences = regex_v3.consensus_record(req, votes)
    assert [item["consensus_valid"] for item in occurrences] == [True, False]
    assert row["consensus_label"] == "true_positive"
    assert row["consensus_status"] == "majority"
    assert row["valid_occurrences"] == 1


def test_regex_v3_batch_payloads_use_regex_schema_and_diagnostic(monkeypatch):
    monkeypatch.setattr(run_binary, "binary", regex_v3)
    req = request()
    openai = run_binary.batch_item("openai", req)
    gemini = run_binary.batch_item("gemini", req, "missing occurrence")
    assert openai["body"]["text"]["format"]["name"] == regex_v3.SCHEMA_NAME
    assert "label" not in openai["body"]["text"]["format"]["schema"]["properties"]
    assert gemini["request"]["generation_config"]["response_json_schema"] == regex_v3.schema()
    assert "Do not emit a separate PR label" in gemini["request"]["contents"][0]["parts"][0]["text"]


def test_qwen_rescue_simplifies_transport_without_relaxing_semantic_schema(monkeypatch):
    monkeypatch.setattr(run_binary, "binary", regex_v3)
    req = request()
    bounded, bounded_format, bounded_mode = run_binary.qwen_transport(req, 6)
    compact, compact_format, compact_mode = run_binary.qwen_transport(req, 7)
    fallback, fallback_format, fallback_mode = run_binary.qwen_transport(req, 10)
    assert bounded["$defs"]["OccurrenceVerdict"]["properties"]["id"]["enum"] == req["occurrence_ids"]
    assert bounded_format == bounded and bounded_mode == "bounded_id_schema"
    assert "enum" not in compact["$defs"]["OccurrenceVerdict"]["properties"]["id"]
    assert "minItems" not in compact["properties"]["occurrences"]
    assert compact_format == compact and compact_mode == "compact_semantic_schema"
    assert fallback == regex_v3.schema()
    assert fallback_format == "json" and fallback_mode == "json_with_prompt_schema"


@pytest.fixture
def archived_regex_source(tmp_path):
    """Build a complete synthetic archive without private evidence or caches."""
    source = tmp_path / "archived-assessment"
    (source / "samples").mkdir(parents=True)
    (source / "consensus").mkdir()
    rows = []
    occurrences = []
    for index in range(88):
        # Preserve the study's size contract while using synthetic PRs and text.
        count = 22 if index < 62 else 21
        ids = [f"o{i:04d}" for i in range(1, count + 1)]
        prompt = f"Synthetic PR {index + 1}: execution time changed from 20ms to 10ms.\n" + ",".join(ids)
        row = {
            "custom_id": f"regex_audit:1:{index + 1}", "task": "regex_audit",
            "repo_id": 1, "number": index + 1,
            "sample_arm": "agentic" if index < 44 else "human_candidate",
            "html_url": f"https://github.com/example/project/pull/{index + 1}",
            "prompt": prompt, "quote_corpus": [prompt],
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_tokens_o200k": 100, "activation_count": 28 if index < 8 else 27,
            "occurrence_ids": ids, "occurrence_dimensions": {item: "D1" for item in ids},
        }
        rows.append(row)
        occurrences.append({"repo_id": 1, "number": index + 1,
                            "occurrences": [{"occurrence_id": item, "dimension": "D1"} for item in ids]})
    sample = pd.DataFrame(rows)[["repo_id", "number", "sample_arm", "html_url"]]
    sample_path = source / "samples/regex_audit_sample.parquet"
    sample.to_parquet(sample_path, index=False)
    contract = {"system": v2.SYSTEM, "schemas": {"regex_audit": v2.schema("regex_audit")},
                "sample_sha256": {"regex_audit": v1.sha256_file(sample_path)}}
    contract_hash = v1.sha256_json(contract)
    for row in rows:
        row["study_contract_sha256"] = contract_hash
    (source / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (source / "occurrences.jsonl").write_text("".join(json.dumps(row) + "\n" for row in occurrences))
    sample.assign(consensus_label="true_positive").to_parquet(
        source / "consensus/regex_audit_consensus.parquet", index=False
    )
    (source / "preflight.json").write_text(json.dumps({
        "contract": contract, "study_contract_sha256": contract_hash,
        "requests_sha256": v1.sha256_file(source / "requests.jsonl"),
    }))
    return source


def test_regex_v3_preparation_freezes_all_88_v2_prompts(tmp_path, archived_regex_source):
    output = tmp_path / "regex-v3"
    summary = regex_v3.prepare(output, source=archived_regex_source)
    requests = regex_v3.load_requests(output)
    source = {row["custom_id"]: row for row in v2.load_requests(archived_regex_source)}
    assert summary["requests_per_provider"] == len(requests) == 88
    assert summary["classification_requests"] == 264
    assert summary["arms"] == {"agentic": 44, "human_candidate": 44}
    assert summary["occurrences"] == 1910
    assert summary["activations"] == 2384
    assert summary["identical_v2_prompts"] == 88
    assert all(row["prompt"] == source[row["source_custom_id"]]["prompt"] for row in requests)
    assert v1.sha256_file(output / "samples/regex_audit_sample.parquet") == summary["contract"]["sample_sha256"]
    assert len((output / "occurrences.jsonl").read_text().splitlines()) == 88
    assert (output / "reference/v2_consensus.parquet").read_bytes() == (
        archived_regex_source / "consensus/regex_audit_consensus.parquet"
    ).read_bytes()


def test_regex_v3_rejects_modified_archive_requests(tmp_path, archived_regex_source):
    path = archived_regex_source / "requests.jsonl"
    path.write_text(path.read_text().replace("20ms", "30ms"))
    with pytest.raises(ValueError, match="payload changed"):
        regex_v3.prepare(tmp_path / "modified", source=archived_regex_source)


def test_wilson_interval_contains_the_observed_precision():
    interval = regex_v3._wilson(77, 88)
    assert interval["lower_95"] < interval["estimate"] == 77 / 88 < interval["upper_95"]
