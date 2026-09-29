import json

import pytest

from analysis.rq3_llm_validation import v2
from analysis.rq3_llm_validation import run_v2
from analysis.rq3_llm_validation import v2_consensus
from analysis.rq3_llm_validation.run_v2 import parse_response, request_schema
from analysis.rq3_llm_validation.v2_consensus import consensus_for
from analysis.rq3_llm_validation.v2_evidence import extract_dimensions, group_activations, normalize_with_offsets


def request(task="regex_audit"):
    return {"custom_id": f"{task}:1:2", "task": task, "repo_id": 1, "number": 2,
            "sample_arm": "agentic", "html_url": "https://github.com/o/r/pull/2",
            "prompt_sha256": "prompt", "study_contract_sha256": "contract", "activation_count": 3,
            "occurrence_ids": ["o0001", "o0002"], "occurrence_dimensions": {"o0001": "D1", "o0002": "D3"},
            "quote_corpus": ["Runtime decreased from 20ms to 10ms. Memory increased from 2MB to 4MB."]}


def payload(valid=(True, True)):
    return {"occurrences": [{"id": f"o{i:04d}", "valid": value,
                              "reason": "reported_metric" if value else "configuration"}
                             for i, value in enumerate(valid, 1)],
            "evidence_quotes": ["Runtime decreased from 20ms to 10ms."], "rationale": "Measured results."}


def checkpoint(req, values):
    return {**req, "classification_status": "classified", "label": values}


@pytest.mark.parametrize("raw", [
    "Latency decreased from 20 milliseconds to 10 milliseconds. https://example.org/x \nMemory: 20 MiB",
    "大对象阈值（1MB）\n内存 latency 20秒 -> 10秒", "<!-- remove --> <b>Time</b>: 0.4101774 seconds. 0 Warning(s)\n0 Error(s)",
    "İΟΔΟΣ v1.2.3 abcdef1234 \nRuntime 2µs -> 1μs", "x" * 200 + "\n10x faster",
])
def test_normalization_keeps_original_locators_and_legacy_matches(raw):
    normalized, offsets = normalize_with_offsets(raw)
    old = extract_dimensions(raw)
    new = extract_dimensions(raw, include_positions=True)
    assert {dim: [{key: match[key] for key in ("cue", "quant_kind", "quant", "rule", "snippet")}
                  for match in matches] for dim, matches in new.items()} == old
    for matches in new.values():
        for match in matches:
            for name in ("cue", "quant"):
                start, end = match[f"{name}_start"], match[f"{name}_end"]
                assert normalized[start:end]
                assert 0 <= offsets[start][0] < offsets[end - 1][1] <= len(raw)


def test_deduplication_preserves_repeated_positions_and_cross_dimension_activations():
    values = [{"activation_id": str(i), "source": source, "dimension": dimension,
               "quant_raw_start": start, "quant_raw_end": end}
              for i, (source, dimension, start, end) in enumerate([
                  ("description", "D1", 10, 15), ("description", "D1", 9, 16),
                  ("description", "D1", 30, 35), ("description", "D3", 10, 15),
                  ("issue_comments", "D1", 10, 15)])]
    groups = group_activations(values)
    assert len(groups) == 4
    assert sum(len(g["activations"]) for g in groups) == 5
    assert groups == group_activations(list(reversed(values)))


def test_later_valid_occurrence_determines_pr_even_when_first_is_noise():
    result = v2.validate_result(payload((False, True)), request())
    assert result["label"] == "true_positive"


@pytest.mark.parametrize("mutation", [
    lambda data: data["occurrences"].pop(),
    lambda data: data["occurrences"].append(data["occurrences"][0]),
    lambda data: data["occurrences"][0].update(id="unknown"),
    lambda data: data["occurrences"].reverse(),
])
def test_response_must_cover_every_occurrence_exactly_once_in_order(mutation):
    data = payload()
    mutation(data)
    with pytest.raises(ValueError, match="Occurrence IDs"):
        v2.validate_result(data, request())


def test_quotes_must_come_from_original_supplied_records():
    with pytest.raises(ValueError, match="not uniquely present"):
        v2.validate_result({**payload(), "evidence_quotes": ["Invented measured result."]}, request())


def test_markdown_quote_alignment_restores_original_without_changing_values():
    raw = "+- **First compilation**: PCH built in ~1.15s"
    quote = "First compilation: PCH built in ~1.15s"
    restored = v2.align_quote(quote, [raw])
    assert restored in raw
    assert "~1.15s" in restored
    with pytest.raises(ValueError, match="not uniquely present"):
        v2.align_quote(quote.replace("1.15", "1.14"), [raw])


def test_quote_only_repair_cannot_change_occurrence_decisions(tmp_path, monkeypatch):
    req = request()
    data = {**payload((False, True)), "evidence_quotes": ["Malformed combined citation"]}
    response = {"done": True, "done_reason": "stop", "message": {"content": json.dumps({
        "evidence_quotes": ["Runtime decreased from 20ms to 10ms."]})}}
    monkeypatch.setattr(run_v2, "call_provider", lambda *args: response)
    result = run_v2.validate_with_quote_repair("qwen", data, req, "url", tmp_path / "raw.json")
    assert result["occurrences"] == data["occurrences"]
    assert result["rationale"] == data["rationale"]
    assert result["label"] == "true_positive"
    assert result["quote_repair"]["original_model_quotes"] == data["evidence_quotes"]


def test_retry_feedback_preserves_core_input_and_does_not_force_a_label():
    req = {**request("tradeoff_audit"), "prompt": "Original evidence"}
    updated = run_v2.retry_request(req, "Supported labels require measured directions and IDs")
    assert updated["prompt"].startswith(req["prompt"])
    assert updated["prompt_sha256"] == req["prompt_sha256"]
    assert updated["occurrence_ids"] == req["occurrence_ids"]
    assert "do not invent support" in updated["prompt"]
    assert "unsupported or indeterminate" in updated["prompt"]


def test_tradeoffs_allow_unsupported_without_manufactured_directions():
    data = {**payload((True, False)), "label": "unsupported", "gain_direction": "improved",
            "memory_direction": "unknown", "gain_occurrence_ids": ["o0001"], "memory_occurrence_ids": []}
    assert v2.validate_result(data, request("tradeoff_audit"))["label"] == "unsupported"
    with pytest.raises(ValueError, match="consistent measured directions"):
        v2.validate_result({**data, "label": "tradeoff"}, request("tradeoff_audit"))


def test_tradeoff_support_must_reference_valid_matches_in_correct_dimensions():
    data = {**payload(), "label": "tradeoff", "gain_direction": "improved", "memory_direction": "increased",
            "gain_occurrence_ids": ["o0002"], "memory_occurrence_ids": ["o0001"]}
    with pytest.raises(ValueError, match="wrong dimension"):
        v2.validate_result(data, request("tradeoff_audit"))


def test_occurrence_consensus_is_not_majority_of_independent_pr_labels():
    req = request()
    values = {"openai": payload((True, False)), "gemini": payload((False, True)), "qwen": payload((False, False))}
    row, occurrences = consensus_for(req, {p: checkpoint(req, value) for p, value in values.items()})
    assert row["pr_label_majority"] == "true_positive"
    assert row["consensus_label"] == "false_positive"
    assert not any(o["consensus_valid"] for o in occurrences)


def test_tradeoff_without_majority_is_indeterminate():
    req = request("tradeoff_audit")
    values = {"openai": ("tradeoff", "increased"), "gemini": ("joint_improvement", "reduced"),
              "qwen": ("indeterminate", "mixed")}
    results = {p: checkpoint(req, {**payload(), "label": label, "gain_direction": "improved",
               "memory_direction": memory, "gain_occurrence_ids": ["o0001"], "memory_occurrence_ids": ["o0002"]})
               for p, (label, memory) in values.items()}
    row, _ = consensus_for(req, results)
    assert row["consensus_label"] == "indeterminate"
    assert row["consensus_status"] == "unresolved"


def test_regex_can_finalize_while_tradeoff_provider_is_blocked(tmp_path, monkeypatch):
    requests = [request(task) for task in ("regex_audit", "tradeoff_audit")]
    monkeypatch.setattr(v2, "load_requests", lambda output: requests)
    for req in requests:
        for provider in ("openai", "gemini", "qwen"):
            path = run_v2.checkpoint_path(tmp_path, provider, req)
            path.parent.mkdir(parents=True, exist_ok=True)
            status = "error" if provider == "gemini" and req["task"] == "tradeoff_audit" else "classified"
            path.write_text(json.dumps({"classification_status": status}))
    assert v2_consensus.ready_tasks(tmp_path) == ("regex_audit",)


def test_consensus_rejects_changed_prompt_hash():
    req = request()
    results = {p: checkpoint(req, payload()) for p in ("openai", "gemini", "qwen")}
    results["qwen"]["prompt_sha256"] = "different"
    with pytest.raises(ValueError, match="hashes"):
        consensus_for(req, results)


def test_large_response_schema_requires_every_id_without_an_output_match_cap():
    req = request()
    req["occurrence_ids"] = [f"o{i:04d}" for i in range(282)]
    schema = request_schema(req)
    assert schema["properties"]["occurrences"]["minItems"] == 282
    assert schema["properties"]["occurrences"]["maxItems"] == 282
    assert schema["$defs"]["OccurrenceVerdict"]["properties"]["id"]["enum"] == req["occurrence_ids"]


def test_gemini_uses_compact_schema_but_still_requires_full_occurrence_coverage():
    req = request()
    assert request_schema(req, "gemini") == v2.schema(req["task"])
    incomplete = payload()
    incomplete["occurrences"].pop()
    with pytest.raises(ValueError, match="Occurrence IDs"):
        v2.validate_result(incomplete, req)


def test_qwen_length_and_context_overflow_are_rejected():
    response = {"done": True, "done_reason": "length", "message": {"content": json.dumps(payload())}}
    with pytest.raises(ValueError, match="output limit"):
        parse_response("qwen", response)
    response.update(done_reason="stop", prompt_eval_count=v2.QWEN_CONTEXT - v2.MAX_OUTPUT_TOKENS)
    with pytest.raises(ValueError, match="possible truncation"):
        parse_response("qwen", response)


def test_prompt_policy_and_coverage_preserve_automated_evidence_and_late_matches():
    data = {"records": [{"record_id": "r0001", "source": "issue_comments", "table": "issue_comments",
                         "field": "body", "locator": "1", "metadata": {"author_type": "Bot"},
                         "text": "x" * 20000 + "\nBenchmark: 20ms -> 10ms"}],
            "occurrences": [{"occurrence_id": "o0001", "dimension": "D1", "source": "issue_comments",
                             "activations": [{"rule_id": "SELF_D1[0]", "cue_normalized_text": "20ms -> 10ms",
                                              "quant_normalized_text": "20ms -> 10ms",
                                              "cue_records": [{"record_id": "r0001", "start": 20012, "end": 20024}],
                                              "quant_records": [{"record_id": "r0001", "start": 20012, "end": 20024}]}]}]}
    text, quotes = v2.prompt_for("regex_audit", request(), data, [])
    assert "Benchmark: 20ms -> 10ms" in text
    assert data["records"][0]["text"] in text
    assert "Automated authorship is NEVER a reason to reject evidence" in text
    assert "routine tooling" in text
    assert "baselines" in text
    assert quotes == [data["records"][0]["text"]]
