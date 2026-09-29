"""Fresh Batch replication of the complete 88-PR regex-positive audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analysis.rq3_llm_validation import binary_validation
from analysis.rq3_llm_validation import experiment as v1
from analysis.rq3_llm_validation import v2

VERSION = "rq4-regex-positive-audit-v3"
DEFAULT_OUTPUT = v2.ROOT / "analysis/rq3_llm_validation/generated_regex_v3"
PROVIDERS = tuple(v2.MODELS)
ResultModel = v2.RegexResult
SCHEMA_NAME = "RQ4RegexOccurrencesV3"


def schema() -> dict:
    return v2.schema("regex_audit")


def retry_diagnostic(error: str) -> str:
    return (
        "\n\nVALIDATION DIAGNOSTIC (not additional PR evidence):\n"
        + json.dumps({"previous_error": error})
        + "\nReturn a corrected response under the same audit policy. Include every occurrence ID "
        "exactly once and in order. A valid occurrence must use reason=reported_metric; an invalid "
        "occurrence must use the most specific other reason. Quote only short contiguous original "
        "text copied exactly, preserving capitalization, punctuation, whitespace, and Markdown. "
        "Do not abbreviate table rows or combine separate spans. Do not invent measurements, citations, "
        "or identifiers. Do not emit a separate PR label."
    )


def validate_result(payload: dict, request: dict) -> dict:
    result = ResultModel.model_validate(payload).model_dump()
    alignments = []
    originals = [" ".join(text.split()) for text in request["quote_corpus"]]
    for index, quote in enumerate(result["evidence_quotes"]):
        if quote.strip() and not any(" ".join(quote.split()) in text for text in originals):
            try:
                restored = binary_validation.align_formatted_quote(quote, request["quote_corpus"])
            except ValueError:
                continue
            result["evidence_quotes"][index] = restored
            alignments.append({
                "model_quote": quote,
                "original_quote": restored,
                "normalization": "html_markdown_presentation_only",
            })
    checked = v2.validate_result(result, {**request, "task": "regex_audit"})
    if alignments:
        checked["quote_alignments"] = alignments + checked.get("quote_alignments", [])
    checked["validation_version"] = binary_validation.VERSION
    return checked


def prepare(output: Path = DEFAULT_OUTPUT, source: Path = v2.DEFAULT_OUTPUT) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Regex v3 output must be new; refusing to overwrite a run.")
    source_preflight = json.loads((source / "preflight.json").read_text())
    original = [row for row in v2.load_requests(source) if row["task"] == "regex_audit"]
    sample_path = source / "samples/regex_audit_sample.parquet"
    sample = pd.read_parquet(sample_path)
    expected = set(sample[v1.KEYS].itertuples(index=False, name=None))
    actual = {(row["repo_id"], row["number"]) for row in original}
    if len(original) != 88 or len(sample) != 88 or len(actual) != 88 or actual != expected:
        raise ValueError("Regex v3 must preserve exactly the frozen 88 unique PRs.")
    if v1.sha256_file(sample_path) != source_preflight["contract"]["sample_sha256"]["regex_audit"]:
        raise ValueError("The frozen regex sample changed.")

    contract = {
        "version": VERSION,
        "paper_question": "RQ4 quantitative metric extractor precision",
        "source_task": "regex_audit",
        "classification_unit": "pull_request",
        "occurrence_labels": ["valid", "invalid"],
        "pr_derivation": "true_positive if any occurrence has majority-valid consensus; otherwise false_positive",
        "system": v2.SYSTEM,
        "policy": v2.POLICY,
        "schema": schema(),
        "models": v2.MODELS,
        "source_v2_contract_sha256": source_preflight["study_contract_sha256"],
        "source_v2_requests_sha256": source_preflight["requests_sha256"],
        "sample_sha256": v1.sha256_file(sample_path),
        "cloud_transport": "batch_only_including_retries",
        "batch_completion_window": "24h",
        "max_output_tokens": v2.MAX_OUTPUT_TOKENS,
        "max_input_tokens_o200k": v2.MAX_INPUT_TOKENS,
        "qwen_num_ctx": v2.QWEN_CONTEXT,
        "qwen_think": True,
        "openai_reasoning": "medium",
        "gemini_thinking": "MEDIUM",
        "evidence_policy": "byte-identical v2 prompt and complete occurrence set",
        "prior_labels_in_model_inputs": False,
    }
    contract_hash = v1.sha256_json(contract)
    requests = []
    for source_row in original:
        prompt = source_row["prompt"]
        if hashlib.sha256(prompt.encode()).hexdigest() != source_row["prompt_sha256"]:
            raise ValueError("A source v2 prompt changed.")
        row = {
            **source_row,
            "custom_id": f"regex_audit_v3:{source_row['repo_id']}:{source_row['number']}",
            "source_custom_id": source_row["custom_id"],
            "source_prompt_sha256": source_row["prompt_sha256"],
            "study_contract_sha256": contract_hash,
        }
        requests.append(row)

    output.mkdir(parents=True)
    (output / "samples").mkdir()
    shutil.copy2(sample_path, output / "samples" / sample_path.name)
    (output / "reference").mkdir()
    reference = source / "consensus/regex_audit_consensus.parquet"
    shutil.copy2(reference, output / "reference/v2_consensus.parquet")
    with (output / "requests.jsonl").open("w") as handle:
        for row in requests:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_keys = {(row["repo_id"], row["number"]) for row in requests}
    occurrence_rows = [json.loads(line) for line in (source / "occurrences.jsonl").read_text().splitlines()]
    occurrence_rows = [row for row in occurrence_rows if (row["repo_id"], row["number"]) in selected_keys]
    if len(occurrence_rows) != 88:
        raise ValueError("The source occurrence artifact does not cover the frozen 88 PRs.")
    with (output / "occurrences.jsonl").open("w") as handle:
        for row in occurrence_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = [
        {key: value for key, value in row.items() if key not in {"prompt", "quote_corpus", "occurrence_dimensions"}}
        for row in requests
    ]
    pd.DataFrame(manifest).to_parquet(output / "manifest.parquet", index=False)
    summary = {
        "version": VERSION,
        "requests_per_provider": 88,
        "classification_requests": 264,
        "arms": sample.sample_arm.value_counts().to_dict(),
        "contract": contract,
        "study_contract_sha256": contract_hash,
        "requests_sha256": v1.sha256_file(output / "requests.jsonl"),
        "occurrences_sha256": v1.sha256_file(output / "occurrences.jsonl"),
        "reference_sha256": v1.sha256_file(output / "reference/v2_consensus.parquet"),
        "manifest_sha256": v1.sha256_file(output / "manifest.parquet"),
        "input_tokens_o200k": sum(row["input_tokens_o200k"] for row in requests),
        "max_input_tokens_o200k": max(row["input_tokens_o200k"] for row in requests),
        "occurrences": sum(len(row["occurrence_ids"]) for row in requests),
        "activations": sum(row["activation_count"] for row in requests),
        "identical_v2_prompts": len(requests),
        "truncations": 0,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    if summary["occurrences"] != 1_910 or summary["activations"] != 2_384:
        raise ValueError("Regex v3 occurrence or activation coverage differs from v2.")
    v1.atomic_write_json(output / "preflight.json", summary)
    snapshots = output / "source"
    snapshots.mkdir()
    for path in (Path(__file__), Path(v1.__file__), Path(v2.__file__), Path(binary_validation.__file__)):
        shutil.copy2(path, snapshots / path.name)
    return summary


def load_requests(output: Path) -> list[dict]:
    preflight = json.loads((output / "preflight.json").read_text())
    if v1.sha256_file(output / "requests.jsonl") != preflight["requests_sha256"]:
        raise ValueError("Prepared regex v3 requests changed.")
    contract = preflight["contract"]
    if (
        v1.sha256_json(contract) != preflight["study_contract_sha256"]
        or contract["schema"] != schema()
        or contract["system"] != v2.SYSTEM
        or contract["policy"] != v2.POLICY
        or contract["models"] != v2.MODELS
        or contract["max_output_tokens"] != v2.MAX_OUTPUT_TOKENS
        or contract["qwen_num_ctx"] != v2.QWEN_CONTEXT
    ):
        raise ValueError("Regex v3 study contract or runtime configuration changed.")
    rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    if len(rows) != 88 or len({row["custom_id"] for row in rows}) != 88:
        raise ValueError("Expected exactly 88 distinct regex v3 requests.")
    for row in rows:
        if (
            row["task"] != "regex_audit"
            or row["study_contract_sha256"] != preflight["study_contract_sha256"]
            or hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]
            or row["prompt_sha256"] != row["source_prompt_sha256"]
        ):
            raise ValueError("Regex v3 prompt, task, or contract hash mismatch.")
    return rows


def checkpoint_path(output: Path, provider: str, request: dict) -> Path:
    return output / provider / "responses" / f"{request['repo_id']}-{request['number']}.json"


def checked_checkpoint(output: Path, provider: str, request: dict) -> dict | None:
    path = checkpoint_path(output, provider, request)
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    for name in ("custom_id", "prompt_sha256", "study_contract_sha256"):
        if result[name] != request[name]:
            raise ValueError("Checkpoint belongs to another regex v3 request or contract.")
    if result["model"] != v2.MODELS[provider]:
        raise ValueError("Checkpoint belongs to another model.")
    if result["classification_status"] == "classified":
        validate_result({key: result["label"][key] for key in ResultModel.model_fields}, request)
    return result


def collect(output: Path, provider: str, requests: list[dict]) -> dict:
    rows = []
    for request in requests:
        checkpoint = checked_checkpoint(output, provider, request)
        if checkpoint:
            rows.append({
                **{key: value for key, value in checkpoint.items() if key not in {"label", "usage"}},
                **checkpoint.get("label", {}),
                "usage_json": json.dumps(checkpoint.get("usage", {})),
            })
    directory = output / provider
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    counts = frame.classification_status.value_counts().to_dict() if rows else {}
    if rows:
        temporary = directory / "labels.tmp.parquet"
        frame.to_parquet(temporary, index=False)
        temporary.replace(directory / "labels.parquet")
    state = {
        "provider": provider,
        "expected": len(requests),
        "classified": counts.get("classified", 0),
        "errors": counts.get("error", 0),
        "checkpoints": len(rows),
        "status": "completed" if counts.get("classified", 0) == len(requests) else "in_progress",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    v1.atomic_write_json(directory / "run_state.json", state)
    return state


def _wilson(successes: int, total: int) -> dict[str, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return {"estimate": proportion, "lower_95": centre - margin, "upper_95": centre + margin}


def consensus_record(request: dict, votes: dict[str, dict]) -> tuple[dict, list[dict]]:
    if set(votes) != set(PROVIDERS) or any(
        vote is None or vote["classification_status"] != "classified" for vote in votes.values()
    ):
        raise ValueError("Regex v3 consensus needs three valid classified votes.")
    validated = {}
    for provider, vote in votes.items():
        if any(vote[key] != request[key] for key in ("custom_id", "prompt_sha256", "study_contract_sha256")):
            raise ValueError("Consensus input hashes or identities differ.")
        validated[provider] = validate_result(
            {key: vote["label"][key] for key in ResultModel.model_fields}, request
        )
    base = {key: request[key] for key in ("repo_id", "number", "sample_arm", "html_url")}
    occurrences = []
    for index, occurrence_id in enumerate(request["occurrence_ids"]):
        occurrence_votes = {provider: validated[provider]["occurrences"][index] for provider in PROVIDERS}
        valid_votes = sum(vote["valid"] for vote in occurrence_votes.values())
        occurrences.append({
            **base,
            "occurrence_id": occurrence_id,
            "dimension": request["occurrence_dimensions"][occurrence_id],
            "consensus_valid": valid_votes >= 2,
            "valid_votes": valid_votes,
            "agreement": "unanimous" if valid_votes in {0, 3} else "majority",
            **{f"{provider}_valid": occurrence_votes[provider]["valid"] for provider in PROVIDERS},
            **{f"{provider}_reason": occurrence_votes[provider]["reason"] for provider in PROVIDERS},
        })
    labels = {provider: validated[provider]["label"] for provider in PROVIDERS}
    winner, count = Counter(labels.values()).most_common(1)[0]
    consensus_label = "true_positive" if any(row["consensus_valid"] for row in occurrences) else "false_positive"
    row = {
        **base,
        "consensus_label": consensus_label,
        "pr_label_majority": winner,
        "consensus_status": "unanimous" if count == 3 else "majority",
        "occurrence_aggregation_differs": consensus_label != winner,
        "valid_occurrences": sum(row["consensus_valid"] for row in occurrences),
        "occurrences": len(occurrences),
        "activation_count": request["activation_count"],
        **{f"{provider}_label": labels[provider] for provider in PROVIDERS},
        **{f"{provider}_rationale": validated[provider]["rationale"] for provider in PROVIDERS},
    }
    return row, occurrences


def build_consensus(output: Path) -> dict:
    requests = load_requests(output)
    records, occurrence_records = [], []
    for request in requests:
        row, occurrences = consensus_record(
            request, {provider: checked_checkpoint(output, provider, request) for provider in PROVIDERS}
        )
        records.append(row)
        occurrence_records.extend(occurrences)
    reference_path = output / "reference/v2_consensus.parquet"
    preflight = json.loads((output / "preflight.json").read_text())
    if v1.sha256_file(reference_path) != preflight["reference_sha256"]:
        raise ValueError("The frozen v2 comparison changed.")
    reference = pd.read_parquet(reference_path)[[*v1.KEYS, "consensus_label", "v1_label"]].rename(
        columns={"consensus_label": "v2_label"}
    )
    result = pd.DataFrame(records).merge(reference, on=v1.KEYS, validate="one_to_one")
    if len(result) != 88:
        raise ValueError("Regex v3/v2 comparison changed the frozen population.")
    result["changed_from_v2"] = result.consensus_label.ne(result.v2_label)
    details = pd.DataFrame(occurrence_records)
    directory = output / "consensus"
    directory.mkdir(exist_ok=True)
    result.to_parquet(directory / "regex_audit_consensus.parquet", index=False)
    result.to_csv(directory / "regex_audit_comparison.csv", index=False)
    details.to_csv(directory / "regex_audit_occurrence_votes.csv", index=False)
    review = result[
        result.consensus_label.eq("false_positive") | result.consensus_status.ne("unanimous")
        | result.changed_from_v2 | result.occurrence_aggregation_differs
    ]
    review.to_csv(directory / "review_candidates.csv", index=False)

    classes = result.consensus_label.value_counts().to_dict()
    by_arm = {arm: group.consensus_label.value_counts().to_dict() for arm, group in result.groupby("sample_arm")}
    precision = {"overall": _wilson(classes.get("true_positive", 0), len(result))}
    for arm, group in result.groupby("sample_arm"):
        precision[arm] = _wilson(int(group.consensus_label.eq("true_positive").sum()), len(group))
    summary = {
        "version": VERSION,
        "status": "completed",
        "prs": len(result),
        "classes": classes,
        "by_arm": by_arm,
        "precision_wilson": precision,
        "agreement": result.consensus_status.value_counts().to_dict(),
        "occurrence_agreement": details.agreement.value_counts().to_dict(),
        "valid_occurrences": int(details.consensus_valid.sum()),
        "invalid_occurrences": int((~details.consensus_valid).sum()),
        "changed_from_v2": int(result.changed_from_v2.sum()),
        "occurrence_vs_pr_majority_differences": int(result.occurrence_aggregation_differs.sum()),
        "review_candidates": len(review),
    }
    v1.atomic_write_json(directory / "summary.json", summary)
    (directory / "summary.md").write_text(
        "# RQ4 regex-positive audit v3\n\n"
        "Fresh Batch replication on the frozen 88-PR sample with byte-identical v2 prompts.\n\n"
        f"- Classes: {json.dumps(classes)}\n"
        f"- Agreement: {json.dumps(summary['agreement'])}\n"
        f"- Changed from v2: {summary['changed_from_v2']}\n"
        f"- Overall precision estimate: {precision['overall']['estimate']:.1%} "
        f"(95% Wilson CI {precision['overall']['lower_95']:.1%}--{precision['overall']['upper_95']:.1%})\n\n"
        "This three-model consensus is a sensitivity estimate, not human ground truth. Recall is not estimated.\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "consensus"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output_dir) if args.action == "prepare" else build_consensus(args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "contract"}, indent=2))
