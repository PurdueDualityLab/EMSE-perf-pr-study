from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from analysis.rq3_llm_validation import experiment
from analysis.rq3_llm_validation import run_openai
from analysis.rq3_llm_validation import run_qwen
from analysis.rq3_llm_validation.build_consensus import build_consensus


def test_regex_sample_is_exact_balanced_deterministic_and_order_independent():
    rows = []
    for arm, count, repo_id in (("agentic", 437, 1), ("human_candidate", 441, 2)):
        rows.extend({"repo_id": repo_id, "number": number, "sample_arm": arm,
                     "validation_present": True, "n_dims": 1}
                    for number in range(1, count + 1))
    frame = pd.DataFrame(rows)

    first = experiment.select_regex_sample(frame)
    second = experiment.select_regex_sample(frame.sample(frac=1, random_state=7))

    assert len(first) == 88
    assert first.sample_arm.value_counts().to_dict() == {"agentic": 44, "human_candidate": 44}
    assert first[[*experiment.KEYS, "selection_hash"]].equals(
        second[[*experiment.KEYS, "selection_hash"]]
    )
    assert first.selection_hash.str.fullmatch(r"[0-9a-f]{64}").all()


def test_regex_sample_supports_explicit_test_population_controls():
    frame = pd.DataFrame([
        {"repo_id": arm_index, "number": number, "sample_arm": arm,
         "validation_present": True, "n_dims": 1}
        for arm_index, arm in enumerate(experiment.ARMS, 1) for number in range(3)
    ])
    result = experiment.select_regex_sample(
        frame, per_arm=2, expected_counts={"agentic": 3, "human_candidate": 3}
    )
    assert result.sample_arm.value_counts().to_dict() == {"agentic": 2, "human_candidate": 2}


def test_tradeoff_selector_returns_only_the_exact_29_eligible_cases():
    eligible = pd.DataFrame([
        {"repo_id": 1, "number": number, "sample_arm": "agentic",
         "validation_type": "benchmark", "sub_pattern": "Caching",
         "D1": True, "D2": False, "D3": True, "D5": number % 2 == 0}
        for number in range(29)
    ])
    ineligible = eligible.iloc[[0]].assign(number=99, D3=False)
    result = experiment.select_tradeoff_cases(pd.concat([eligible, ineligible], ignore_index=True))
    assert len(result) == 29
    assert result.D3.all()
    assert (result.D1 | result.D2 | result.D5).all()


def test_label_schemas_enforce_binary_semantics():
    experiment.RegexAuditLabel(
        label="true_positive", evidence_sources=["title_description"],
        evidence_quotes=["latency fell by 20%"], rationale="Explicit measured gain.",
    )
    with pytest.raises(ValidationError, match="requires evidence sources and quotes"):
        experiment.RegexAuditLabel(
            label="false_positive", evidence_sources=[],
            evidence_quotes=[], rationale="Incidental value.",
        )
    with pytest.raises(ValidationError, match="requires memory_direction=increased"):
        experiment.TradeoffAuditLabel(
            label="tradeoff", gain_direction="improved", memory_direction="reduced",
            evidence_sources=["title_description"], evidence_quotes=["faster but uses more memory"],
            rationale="Directions conflict with output.",
        )


def test_label_schemas_normalize_duplicate_evidence_sources():
    label = experiment.TradeoffAuditLabel(
        label="tradeoff", gain_direction="improved", memory_direction="increased",
        evidence_sources=["issue_comments", "issue_comments"],
        evidence_quotes=["20% faster", "uses 5% more memory"], rationale="Measured trade-off.",
    )
    assert label.evidence_sources == ["issue_comments"]
    parquet_label = experiment.TradeoffAuditLabel(
        label="joint_improvement", gain_direction="improved", memory_direction="reduced",
        evidence_sources=np.array(["code_diff", "code_diff"]),
        evidence_quotes=["RSS fell 2%"], rationale="Measured improvement.",
    )
    assert parquet_label.evidence_sources == ["code_diff"]


def test_prompts_include_full_corpus_matches_and_decision_boundaries():
    row = {"evidence_complete": True, "title": "Faster cache", "body": "Measured 20% faster",
           "issue_comments": "issue", "review_comments": "inline", "reviews": "review",
           "commit_messages": "perf commit", "ci": "benchmark check", "code_diff": "+ cache",
           "highlighted_matches": "dimension=D1 | quant=20%"}
    regex = experiment.prompt_for(row, "regex_audit")
    tradeoff = experiment.prompt_for(row, "tradeoff_audit")
    for heading in ("ISSUE COMMENTS", "REVIEW COMMENTS", "REVIEWS", "COMMIT MESSAGES",
                    "CI WORKFLOWS AND CHECKS", "CODE DIFF", "REGEX MATCHES TO AUDIT"):
        assert heading in regex
    assert "Incidental constants" in regex
    assert "prospective/expected claims" in regex
    assert "there is no eligibility or unknown class" in tradeoff
    assert "increased or worse memory" in tradeoff
    with pytest.raises(ValueError, match="incomplete evidence"):
        experiment.prompt_for({**row, "evidence_complete": False}, "regex_audit")


def test_prepare_hashes_are_provider_independent_except_provider_config(tmp_path, monkeypatch):
    sample = tmp_path / "sample.bin"
    matches = tmp_path / "matches.bin"
    sample.write_bytes(b"sample")
    matches.write_bytes(b"matches")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in experiment.EVIDENCE_FILES:
        (evidence / name).write_bytes(name.encode())
    frame = pd.DataFrame([{
        "repo_id": 1, "number": 2, "repo_full_name": "o/r", "html_url": "url",
        "sample_arm": "agentic", "title": "title", "body": "body",
        "issue_comments": "", "review_comments": "", "reviews": "",
        "commit_messages": "", "ci": "", "code_diff": "",
        "highlighted_matches": "dimension=D1 | quant=20%", "evidence_complete": True,
    }])
    monkeypatch.setattr(experiment, "build_input", lambda *args: frame.copy())

    _, first, _ = experiment.prepare_run(
        "regex_audit", sample, matches, evidence, tmp_path / "openai", {"provider": "openai"}
    )
    _, second, _ = experiment.prepare_run(
        "regex_audit", sample, matches, evidence, tmp_path / "gemini", {"provider": "gemini"}
    )
    for column in ("prompt_sha256", "input_row_sha256", "study_contract_sha256"):
        assert first[column].tolist() == second[column].tolist()
    assert first.provider_config_sha256.tolist() != second.provider_config_sha256.tolist()


def test_prepared_input_validation_rejects_changed_prompt(tmp_path, monkeypatch):
    sample = tmp_path / "sample.bin"
    matches = tmp_path / "matches.bin"
    sample.write_bytes(b"sample")
    matches.write_bytes(b"matches")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in experiment.EVIDENCE_FILES:
        (evidence / name).write_bytes(name.encode())
    frame = pd.DataFrame([{
        "repo_id": 1, "number": 2, "repo_full_name": "o/r", "html_url": "url",
        "sample_arm": "agentic", "title": "title", "body": "body",
        "issue_comments": "", "review_comments": "", "reviews": "",
        "commit_messages": "", "ci": "", "code_diff": "",
        "highlighted_matches": "dimension=D1 | quant=20%", "evidence_complete": True,
    }])
    monkeypatch.setattr(experiment, "build_input", lambda *args: frame.copy())
    _, manifest, metadata = experiment.prepare_run(
        "regex_audit", sample, matches, evidence, tmp_path / "run", {"provider": "qwen"}
    )

    experiment.validate_prepared_inputs(
        frame, manifest, metadata, "regex_audit", sample, matches, evidence
    )
    changed = frame.assign(title="changed")
    with pytest.raises(ValueError, match="Prepared prompt changed"):
        experiment.validate_prepared_inputs(
            changed, manifest, metadata, "regex_audit", sample, matches, evidence
        )


def test_openai_retry_merge_replaces_exact_error_population(tmp_path):
    columns = {
        "sample_arm": "agentic", "task": "regex_audit", "label": "true_positive",
        "study_contract_sha256": "contract", "provider_config_sha256": "provider",
        "input_row_sha256": "input", "prompt_sha256": "prompt",
    }
    source = pd.DataFrame([
        {**columns, "repo_id": 1, "number": 1, "custom_id": "1:1",
         "classification_status": "classified"},
        {**columns, "repo_id": 1, "number": 2, "custom_id": "1:2",
         "classification_status": "error"},
    ])
    retry = source.iloc[[1]].assign(classification_status="classified")
    source_dir = tmp_path / "source"
    retry_dir = tmp_path / "retry"
    source_dir.mkdir()
    retry_dir.mkdir()
    source.to_parquet(source_dir / "labels.parquet", index=False)
    retry.to_parquet(retry_dir / "labels.parquet", index=False)

    merged = run_openai.merge_retry(source_dir, retry_dir, tmp_path / "merged.parquet")

    assert len(merged) == 2
    assert merged.classification_status.eq("classified").all()


def test_qwen_legacy_ollama_fallback_remains_schema_validated(tmp_path, monkeypatch):
    sample = tmp_path / "sample.bin"
    matches = tmp_path / "matches.bin"
    sample.write_bytes(b"sample")
    matches.write_bytes(b"matches")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in experiment.EVIDENCE_FILES:
        (evidence / name).write_bytes(name.encode())
    frame = pd.DataFrame([{
        "repo_id": 1, "number": 2, "repo_full_name": "o/r", "html_url": "url",
        "sample_arm": "agentic", "title": "title", "body": "body",
        "issue_comments": "", "review_comments": "", "reviews": "",
        "commit_messages": "", "ci": "", "code_diff": "",
        "highlighted_matches": "dimension=D1 | quant=20%", "evidence_complete": True,
    }])
    monkeypatch.setattr(experiment, "build_input", lambda *args: frame.copy())
    calls = []

    def fake_chat(endpoint, model, system, prompt, schema, *args):
        calls.append((prompt, schema))
        if len(calls) == 1:
            raise RuntimeError("cannot unmarshal object into Go struct field ChatRequest.format of type string")
        return {"message": {"content": (
            '{"label":"true_positive","evidence_sources":["title_description"],'
            '"evidence_quotes":["20% faster"],"rationale":"Measured."}'
        )}, "done": True}

    monkeypatch.setattr(run_qwen.ollama, "ollama_chat", fake_chat)
    result = run_qwen.run(
        "regex_audit", sample, matches, evidence, tmp_path / "run", model="test-model"
    )

    assert result.iloc[0].classification_status == "classified"
    assert calls[1][1] == "json"
    assert "Return only JSON matching this schema" in calls[1][0]


def test_qwen_normalizes_direction_redundant_with_tradeoff_label():
    label, normalizations = run_qwen._validate_label(
        '{"label":"joint_improvement","gain_direction":"improved",'
        '"memory_direction":"increased","evidence_sources":["issue_comments"],'
        '"evidence_quotes":["RSS fell 2%"],"rationale":"Time and memory improved."}',
        "tradeoff_audit",
    )
    assert label.memory_direction == "reduced"
    assert normalizations == ["memory_direction_derived_from_label"]


def _vote(labels: list[str]) -> pd.DataFrame:
    rows = []
    for number, label in enumerate(labels, 1):
        positive = label == "true_positive"
        rows.append({
            "repo_id": 1, "number": number,
            "sample_arm": "agentic" if number % 2 else "human_candidate",
            "task": "regex_audit", "classification_status": "classified", "label": label,
            "evidence_sources": ["title_description"],
            "evidence_quotes": ["20% faster" if positive else "buffer size 1024"],
            "rationale": "Concise.",
            "input_row_sha256": f"input-{number}", "prompt_sha256": f"prompt-{number}",
            "study_contract_sha256": "contract",
        })
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("labels", "coalition"),
    [
        (("true_positive", "true_positive", "true_positive"), "openai+gemini+qwen"),
        (("true_positive", "true_positive", "false_positive"), "openai+gemini"),
        (("true_positive", "false_positive", "true_positive"), "openai+qwen"),
        (("false_positive", "true_positive", "true_positive"), "gemini+qwen"),
    ],
)
def test_consensus_resolves_all_coalitions(labels, coalition):
    votes = {provider: _vote([labels[index]]) for index, provider in enumerate(("openai", "gemini", "qwen"))}
    result, summary = build_consensus(votes, "regex_audit")
    assert result.iloc[0].coalition == coalition
    assert result.iloc[0].consensus_label == "true_positive"
    assert summary["agreement_counts"] == ({"unanimous": 1} if len(set(labels)) == 1 else {"majority": 1})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.assign(classification_status="error"), "must be classified"),
        (lambda frame: frame.assign(prompt_sha256="different"), "do not match"),
        (lambda frame: frame.iloc[0:0], "do not match"),
    ],
)
def test_consensus_rejects_errors_hash_mismatches_and_missing_rows(mutation, message):
    votes = {provider: _vote(["true_positive"]) for provider in ("openai", "gemini", "qwen")}
    votes["qwen"] = mutation(votes["qwen"])
    with pytest.raises(ValueError, match=message):
        build_consensus(votes, "regex_audit")
