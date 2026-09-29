import json
import hashlib
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from analysis.rq3_llm_validation import binary, run_binary, v2
from analysis.rq3_llm_validation import binary_validation
from analysis.rq3_llm_validation import experiment as v1


def request(number=2):
    return {"custom_id": f"tradeoff_audit:1:{number}", "repo_id": 1, "number": number,
            "task": "tradeoff_audit", "sample_arm": "agentic", "html_url": f"https://github.com/o/r/pull/{number}",
            "prompt": v2.POLICY + "\n\nTASK: tradeoff_audit. Do not force a binary choice.\n\nPR: url\nOriginal evidence",
            "prompt_sha256": f"prompt-{number}", "study_contract_sha256": "contract",
            "occurrence_ids": ["o0001", "o0002"], "occurrence_dimensions": {"o0001": "D1", "o0002": "D3"},
            "quote_corpus": ["Runtime fell from 20ms to 10ms. Memory rose from 2MB to 4MB."]}


def label(inferred=False, result="tradeoff"):
    return {"label": result, "choice_requires_inference": inferred,
            "gain_direction": "unknown" if inferred else "improved",
            "memory_direction": "unknown" if inferred else "increased" if result == "tradeoff" else "reduced",
            "gain_occurrence_ids": [] if inferred else ["o0001"],
            "memory_occurrence_ids": [] if inferred else ["o0002"],
            "occurrences": [{"id": f"o{i:04d}", "valid": not inferred,
                              "reason": "insufficient_context" if inferred else "reported_metric"} for i in (1, 2)],
            "evidence_quotes": ["Runtime fell from 20ms to 10ms."], "rationale": "Closest relationship supported by the artifacts."}


def openai_item(req, payload):
    return {"custom_id": req["custom_id"], "response": {"status_code": 200, "body": {
        "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(payload)}]}],
        "usage": {"input_tokens": 100, "output_tokens": 50}}}}


@pytest.mark.parametrize("value", ["unsupported", "indeterminate", "false_positive"])
def test_binary_schema_rejects_every_nonbinary_final_label(value):
    with pytest.raises(ValidationError):
        binary.validate_result(label(True, value), request())


def test_forced_choice_can_retain_missing_evidence_without_fabricated_directions():
    result = binary.validate_result(label(True), request())
    assert result["label"] == "tradeoff"
    assert result["choice_requires_inference"] is True
    assert result["gain_direction"] == result["memory_direction"] == "unknown"
    assert result["gain_occurrence_ids"] == result["memory_occurrence_ids"] == []
    assert not any(item["valid"] for item in result["occurrences"])


def test_observed_choice_still_requires_consistent_measured_support():
    values = label(True)
    values["choice_requires_inference"] = False
    with pytest.raises(ValueError, match="consistent measured directions"):
        binary.validate_result(values, request())


def test_retry_diagnostic_identifies_the_actual_conflicting_fields():
    values = label(False, "joint_improvement")
    values["memory_direction"] = "increased"
    with pytest.raises(ValueError, match='"memory_direction": "increased"'):
        binary.validate_result(values, request())
    values = label()
    values["occurrences"][0]["reason"] = "configuration"
    with pytest.raises(ValueError, match="o0001"):
        binary.validate_result(values, request())


def test_forced_choice_does_not_relax_occurrence_coverage_or_citation_checks():
    values = label(True)
    values["occurrences"].pop()
    with pytest.raises(ValueError, match="Occurrence IDs"):
        binary.validate_result(values, request())
    values = label(True)
    values["evidence_quotes"] = ["Invented 100x speedup"]
    with pytest.raises(ValueError, match="Quote is not"):
        binary.validate_result(values, request())


def test_forced_choice_does_not_allow_references_to_wrong_dimensions():
    values = label()
    values["choice_requires_inference"] = True
    values["gain_occurrence_ids"] = ["o0002"]
    with pytest.raises(ValueError, match="wrong dimension"):
        binary.validate_result(values, request())


def test_only_decision_instructions_change_and_source_evidence_is_byte_identical():
    original = request()
    updated = binary.binary_prompt(original)
    assert binary.evidence_suffix(updated) == binary.evidence_suffix(original["prompt"])
    assert "Do not force a binary choice." not in updated
    assert "FORCED BINARY" in updated
    assert updated.startswith(v2.POLICY)
    with pytest.raises(ValueError, match="only repeats"):
        binary.binary_prompt({**original, "task": "regex_audit"})


def test_cloud_payloads_use_native_batch_formats_and_unchanged_model_settings():
    first = run_binary.batch_item("openai", request())
    second = run_binary.batch_item("gemini", request())
    assert first["url"] == "/v1/responses"
    assert first["body"]["model"] == v2.MODELS["openai"]
    assert first["body"]["reasoning"] == {"effort": "medium"}
    assert first["body"]["max_output_tokens"] == v2.MAX_OUTPUT_TOKENS
    assert second["key"] == first["custom_id"]
    config = second["request"]["generation_config"]
    assert config["temperature"] == 0
    assert config["thinking_config"] == {"thinking_level": "MEDIUM"}
    assert config["response_json_schema"] == binary.schema()
    assert first["body"]["text"]["format"]["schema"]["properties"]["label"]["enum"] == ["tradeoff", "joint_improvement"]


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_submit_uses_only_batch_endpoints_and_is_idempotent(tmp_path, monkeypatch, provider):
    calls = []

    class Client:
        def __init__(self):
            self.files = SimpleNamespace(create=self.upload, upload=self.upload)
            self.batches = SimpleNamespace(create=self.create)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def upload(self, **kwargs):
            calls.append("upload")
            return SimpleNamespace(id="input-file", name="files/input-file")

        def create(self, **kwargs):
            calls.append("batch_create")
            if provider == "openai":
                assert kwargs["completion_window"] == "24h"
                return SimpleNamespace(id="batch-test", status="validating")
            assert kwargs["model"] == v2.MODELS[provider]
            return SimpleNamespace(name="batches/test", state=SimpleNamespace(value="JOB_STATE_PENDING"))

    directory = run_binary.prepare_batch(tmp_path, provider, [request()], 1)
    monkeypatch.setattr(run_binary, "_client", lambda _: Client())
    state = run_binary.submit(directory)
    assert run_binary.submit(directory)["batch_id"] == state["batch_id"]
    assert calls == ["upload", "batch_create"]


def test_uncertain_submission_cannot_create_duplicate_paid_batches(tmp_path, monkeypatch):
    class Client:
        def __init__(self):
            self.files = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(id="file-id"))
            self.batches = SimpleNamespace(create=self.create)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create(self, **kwargs):
            raise TimeoutError("Response lost after submission")

    directory = run_binary.prepare_batch(tmp_path, "openai", [request()], 1)
    monkeypatch.setattr(run_binary, "_client", lambda _: Client())
    with pytest.raises(TimeoutError):
        run_binary.submit(directory)
    with pytest.raises(RuntimeError, match="not resumable automatically"):
        run_binary.submit(directory)


def test_billing_rejection_pauses_submission_without_polling_paid_endpoints(tmp_path, monkeypatch):
    req = request()
    monkeypatch.setattr(binary, "load_requests", lambda _: [req])
    directory = run_binary.prepare_batch(tmp_path, "openai", [req], 1)
    state = run_binary._state(directory)
    state.update(status="submission_uncertain", input_file_id="uploaded-file",
                 error="BadRequestError: Billing hard limit has been reached (billing_hard_limit_reached)")
    v1.atomic_write_json(directory / "state.json", state)
    monkeypatch.setattr(run_binary, "_client", lambda _: pytest.fail("Billing rejection must not be resubmitted automatically"))
    result = run_binary.tick(tmp_path, "openai")
    assert result["status"] == "blocked_billing"
    assert run_binary._state(directory)["input_file_id"] == "uploaded-file"


def test_billing_resume_refuses_an_ambiguous_submission(tmp_path, monkeypatch):
    monkeypatch.setattr(binary, "load_requests", lambda _: [request()])
    directory = run_binary.prepare_batch(tmp_path, "openai", [request()], 1)
    state = run_binary._state(directory)
    state.update(status="submission_uncertain", error="Request timed out after upload")
    v1.atomic_write_json(directory / "state.json", state)
    with pytest.raises(ValueError, match="definite billing rejection"):
        run_binary.resume_submission(tmp_path, "openai")


def test_batch_collection_preserves_missing_cases_as_errors(tmp_path, monkeypatch):
    first, second = request(), request(3)
    monkeypatch.setattr(binary, "load_requests", lambda _: [first, second])
    directory = run_binary.prepare_batch(tmp_path, "openai", [first, second], 1)
    state = run_binary.collect_outputs(tmp_path, directory, [json.dumps(openai_item(first, label()))])
    assert state["classified"] == state["errors"] == 1
    assert state["checkpoints"] == 2
    assert binary.checked_checkpoint(tmp_path, "openai", second)["classification_status"] == "error"


def test_batch_retries_only_failed_identities_and_keeps_good_checkpoints(tmp_path, monkeypatch):
    first, second = request(), request(3)
    monkeypatch.setattr(binary, "load_requests", lambda _: [first, second])
    directory = run_binary.prepare_batch(tmp_path, "openai", [first, second], 1)
    run_binary.collect_outputs(tmp_path, directory, [json.dumps(openai_item(first, label()))])
    preserved = v1.sha256_file(binary.checkpoint_path(tmp_path, "openai", first))
    monkeypatch.setattr(run_binary, "submit", lambda path: {**run_binary._state(path), "status": "validating", "batch_id": "retry"})
    run_binary.tick(tmp_path, "openai")
    retried = run_binary._state(tmp_path / "openai/batches/attempt-02")
    assert retried["custom_ids"] == [second["custom_id"]]
    assert v1.sha256_file(binary.checkpoint_path(tmp_path, "openai", first)) == preserved


@pytest.mark.parametrize("unknown", [False, True])
def test_batch_rejects_duplicate_and_unknown_response_ids(tmp_path, monkeypatch, unknown):
    req = request()
    monkeypatch.setattr(binary, "load_requests", lambda _: [req])
    directory = run_binary.prepare_batch(tmp_path, "openai", [req], 1)
    item = openai_item(request(999) if unknown else req, label())
    text = json.dumps(item) if unknown else json.dumps(item) + "\n" + json.dumps(item)
    with pytest.raises(ValueError, match="Unknown or duplicate"):
        run_binary.collect_outputs(tmp_path, directory, [text])


def test_running_batch_cannot_be_collected_as_missing_results(tmp_path):
    directory = run_binary.prepare_batch(tmp_path, "openai", [request()], 1)
    with pytest.raises(RuntimeError, match="not terminal"):
        run_binary.collect_batch(tmp_path, directory)


def test_gemini_batch_parser_accepts_camelcase_usage_and_rejects_truncation():
    response = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(label())}]}}],
                "usageMetadata": {"promptTokenCount": 100}}
    payload, usage = run_binary.parse_response("gemini", response)
    assert payload["label"] == "tradeoff"
    assert usage == {"promptTokenCount": 100}
    response["candidates"][0]["finishReason"] = "MAX_TOKENS"
    with pytest.raises(ValueError, match="did not finish normally"):
        run_binary.parse_response("gemini", response)


def test_consensus_is_binary_majority_and_retains_inference_flags():
    req = request()
    votes = {provider: {**run_binary.result_base(req, provider, 1), "classification_status": "classified",
                         "label": label(True, "joint_improvement" if provider == "qwen" else "tradeoff")}
             for provider in binary.PROVIDERS}
    result = binary.consensus_record(req, votes)
    assert result["consensus_label"] == "tradeoff"
    assert result["consensus_status"] == "majority"
    assert result["inference_votes"] == 3
    votes["qwen"]["prompt_sha256"] = "different"
    with pytest.raises(ValueError, match="hashes"):
        binary.consensus_record(req, votes)


def test_consensus_refuses_missing_provider_results():
    with pytest.raises(ValueError, match="three valid"):
        binary.consensus_record(request(), {p: None for p in binary.PROVIDERS})


def test_html_to_markdown_quote_restores_original_text_and_keeps_judgments():
    original = "<ul><li>🟡 <code>loadNewAccount/FCP</code>: p75 1.8s</li><li>Next result</li></ul>"
    quoted = "<li>🟡 `loadNewAccount/FCP`: p75 1.8s</li>"
    req = {**request(), "quote_corpus": [original]}
    payload = {**label(True), "evidence_quotes": [quoted]}
    result = binary.validate_result(payload, req)
    assert result["evidence_quotes"][0] in original
    assert result["label"] == payload["label"]
    assert result["occurrences"] == payload["occurrences"]
    assert result["quote_alignments"][0]["model_quote"] == quoted


@pytest.mark.parametrize("quote", ["time: 1.9s", "time: 1.8ms", "time: -1.8s", "memory: 1.8s"])
def test_formatting_alignment_does_not_accept_changed_numbers_units_signs_or_words(quote):
    with pytest.raises(ValueError, match="not uniquely present"):
        binary_validation.align_formatted_quote(quote, ["<code>time</code>: 1.8s"])


def test_formatting_alignment_preserves_entities_and_contiguous_record_boundaries():
    original = "<li><code>A</code>: 1&nbsp;MB</li><li>B: 2 MB</li>"
    restored = binary_validation.align_formatted_quote("`A`: 1 MB\nB: 2 MB", [original])
    assert restored in original
    with pytest.raises(ValueError):
        binary_validation.align_formatted_quote("A: 1 MB B: 2 MB", ["A: 1 MB", "B: 2 MB"])
    with pytest.raises(ValueError):
        binary_validation.align_formatted_quote("A: 1 MB B: 2 MB", ["A: 1 MB OMITTED WORDS B: 2 MB"])


def corrected_request():
    req = request()
    text = "Memory benchmark: bundle@test used 2MB."
    header = {"record_id": "r0001", "locator": "comment-1"}
    occurrence = {"id": "o0002", "dimension": "D6", "activations": [
        {"quantity_at": [{"record_id": "r0001", "start": text.index("2MB"), "end": text.index("2MB") + 3}]}]}
    req["prompt"] += "\n\n" + json.dumps(occurrence, separators=(",", ":")) + "\n\n" + json.dumps(header) + "\n\n" + text + "\n\n[END RECORD]"
    req["quote_corpus"] = [*req["quote_corpus"], text]
    req["occurrence_dimensions"] = {"o0001": "D1", "o0002": "D6"}
    req["validation_dimension_corrections"] = [{
        "custom_id": req["custom_id"], "prompt_sha256": req["prompt_sha256"], "occurrence_id": "o0002",
        "from_dimension": "D6", "to_dimension": "D3", "record_id": "r0001", "record_locator": "comment-1",
        "record_sha256": hashlib.sha256(text.encode()).hexdigest(), "quantity_text": "2MB",
        "basis_quotes": [text], "review_type": "agent_audit", "rationale": "Explicit memory benchmark context."}]
    return req


def test_audited_dimension_fix_is_scoped_and_preserves_original_taxonomy_and_label():
    req = corrected_request()
    result = binary.validate_result(label(), req)
    assert result["label"] == "tradeoff"
    assert req["occurrence_dimensions"]["o0002"] == "D6"
    assert result["audited_dimension_corrections"][0]["to_dimension"] == "D3"
    req.pop("validation_dimension_corrections")
    with pytest.raises(ValueError, match="wrong dimension"):
        binary.validate_result(label(), req)


@pytest.mark.parametrize("field,value", [
    ("prompt_sha256", "different"), ("custom_id", "other"), ("record_sha256", "different"),
    ("record_locator", "different"), ("quantity_text", "20MB"), ("basis_quotes", ["invented evidence"]),
])
def test_audited_dimension_fix_rejects_wrong_identity_or_changed_evidence(field, value):
    req = corrected_request()
    req["validation_dimension_corrections"][0][field] = value
    with pytest.raises(ValueError, match="Audited"):
        binary.validate_result(label(), req)
