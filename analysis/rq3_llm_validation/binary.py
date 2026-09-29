"""Frozen, forced-binary sensitivity experiment for the 29 RQ3 trade-off cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from analysis.rq3_llm_validation import experiment as v1
from analysis.rq3_llm_validation import v2
from analysis.rq3_llm_validation import binary_validation

VERSION = "rq3-tradeoff-forced-binary-v1"
DEFAULT_OUTPUT = v2.ROOT / "analysis/rq3_llm_validation/generated_binary_v1"
PROVIDERS = tuple(v2.MODELS)
INSTRUCTIONS = """TASK: tradeoff_audit, FORCED BINARY sensitivity experiment.
Validate EVERY listed occurrence using the evidence policy above. Then classify this PR as
EXACTLY one of tradeoff or joint_improvement. There is no third final label and no abstention.

tradeoff: the performance gain in time/throughput/I/O is accompanied by greater/worse memory use.
joint_improvement: the performance gain is accompanied by lower/better memory use.
Choose the closest overall interpretation even when a strict measured comparison is unavailable,
mixed, unchanged, ambiguous, or supports neither category conclusively. Prefer comparable
measurements for the primary workload and the relevant/latest PR revision. Consider all supplied
records and distinguish workload/revision differences; do not cherry-pick unrelated rows. When
measurements do not decide the choice, infer the most plausible relationship from the PR's
implementation and performance rationale. Do not default mechanically to either category.

Keep the occurrence verdicts and observed direction fields faithful to the evidence. Forced
classification does NOT make invalid quantities valid or turn predictions into measurements.
gain_direction and memory_direction describe the observed evidence, so unknown, mixed,
not_improved, and unchanged remain available in those fields even though the FINAL LABEL is binary.
gain_occurrence_ids must reference valid D1/D2/D5 occurrences; memory_occurrence_ids must reference
valid D3 occurrences. Use empty lists when no valid detected occurrence supports a dimension.

Set choice_requires_inference=false ONLY when a valid measured gain and a consistent measured
memory direction directly support the selected label, with nonempty supporting ID lists.
Otherwise set choice_requires_inference=true and explain the basis and limitation of the forced
choice in the rationale. This boolean is provenance, not an additional classification category.
Quote short contiguous original excerpts, preserving Markdown and punctuation. Never fabricate
measurements or identifiers to justify the required binary choice.
"""


class ForcedTradeoffResult(v2.TradeoffResult):
    label: Literal["tradeoff", "joint_improvement"]
    choice_requires_inference: bool


ResultModel = ForcedTradeoffResult
SCHEMA_NAME = "RQ3ForcedBinary"


def schema() -> dict:
    return ForcedTradeoffResult.model_json_schema()


def retry_diagnostic(error: str) -> str:
    return (
        "\n\nVALIDATION DIAGNOSTIC (not additional evidence):\n" + json.dumps({"previous_error": error})
        + "\nCorrect the structured response under the same policy. Return all occurrence IDs in order, "
        "short contiguous original-text quotes, and only tradeoff or joint_improvement as the final label. "
        "Use choice_requires_inference=true when measured gain/memory support is absent, ambiguous, or inconsistent. "
        "Never invent valid matches, citations, or observed directions to justify the forced choice."
    )


def validate_result(payload: dict, request: dict) -> dict:
    result = ForcedTradeoffResult.model_validate(payload).model_dump()
    quote_alignments = []
    originals = [" ".join(text.split()) for text in request["quote_corpus"]]
    for index, quote in enumerate(result["evidence_quotes"]):
        if quote.strip() and not any(" ".join(quote.split()) in text for text in originals):
            try:
                restored = binary_validation.align_formatted_quote(quote, request["quote_corpus"])
            except ValueError:
                # Retain the original validator's rejection for semantic changes.
                continue
            result["evidence_quotes"][index] = restored
            quote_alignments.append({"model_quote": quote, "original_quote": restored,
                                     "normalization": "html_markdown_presentation_only"})
    dimensions, corrections = binary_validation.corrected_dimensions(request)
    # Reuse all v2 occurrence, quotation, and reference checks. Only a measured
    # (non-inferred) choice asserts the strict gain/memory direction conditions.
    checked = {key: result[key] for key in v2.TradeoffResult.model_fields}
    if result["choice_requires_inference"]:
        checked["label"] = "indeterminate"
    try:
        checked = v2.validate_result(checked, {**request, "task": "tradeoff_audit", "occurrence_dimensions": dimensions})
    except ValueError as error:
        if "Occurrence validity and reason code disagree" in str(error):
            conflicts = [item for item in result["occurrences"]
                         if item["valid"] != (item["reason"] == "reported_metric")]
            raise ValueError(f"{error} Conflicting entries: {json.dumps(conflicts[:10])}. "
                             "A valid occurrence must have reason=reported_metric; an invalid one must have another reason. "
                             "Re-evaluate these fields against the supplied evidence.") from error
        if "Supported trade-off labels require consistent measured directions" in str(error):
            details = {name: result[name] for name in ("label", "gain_direction", "memory_direction", "choice_requires_inference")}
            raise ValueError(f"{error} Returned fields: {json.dumps(details)}. "
                             "If observed directions do not directly support the chosen label, mark choice_requires_inference=true. "
                             "Keep observed fields faithful to the records; do not invent measurements or change a label merely to pass validation.") from error
        raise
    checked.update(label=result["label"], choice_requires_inference=result["choice_requires_inference"])
    if quote_alignments:
        checked["quote_alignments"] = quote_alignments + checked.get("quote_alignments", [])
    checked["validation_version"] = binary_validation.VERSION
    if corrections:
        checked["audited_dimension_corrections"] = corrections
    return checked


def evidence_suffix(prompt: str) -> str:
    prefix, marker, evidence = prompt.partition("\n\nPR: ")
    if not marker or not prefix.startswith(v2.POLICY):
        raise ValueError("Source prompt does not have the expected v2 policy/evidence boundary.")
    return marker + evidence


def binary_prompt(source: dict) -> str:
    if source["task"] != "tradeoff_audit":
        raise ValueError("The binary experiment only repeats the 29 trade-off cases.")
    return v2.POLICY + "\n\n" + INSTRUCTIONS + evidence_suffix(source["prompt"])


def prepare(output: Path = DEFAULT_OUTPUT, source: Path = v2.DEFAULT_OUTPUT) -> dict:
    import tiktoken

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Binary output must be new; refusing to overwrite a run.")
    original = [r for r in v2.load_requests(source) if r["task"] == "tradeoff_audit"]
    sample_path = source / "samples/tradeoff_audit_sample.parquet"
    sample = pd.read_parquet(sample_path)
    expected = set(sample[v1.KEYS].itertuples(index=False, name=None))
    actual = {(r["repo_id"], r["number"]) for r in original}
    prior = json.loads((source / "preflight.json").read_text())
    if len(original) != 29 or len(sample) != 29 or len(actual) != 29 or actual != expected:
        raise ValueError("The binary run must preserve exactly the same 29 unique PRs.")
    if v1.sha256_file(sample_path) != prior["contract"]["sample_sha256"]["tradeoff_audit"]:
        raise ValueError("The source v2 sample changed.")
    contract = {
        "version": VERSION, "task": "tradeoff_audit", "classification_unit": "pull_request",
        "labels": ["tradeoff", "joint_improvement"], "forced_choice": True,
        "system": v2.SYSTEM, "instructions": INSTRUCTIONS, "schema": schema(), "models": v2.MODELS,
        "source_v2_contract_sha256": prior["study_contract_sha256"],
        "source_v2_requests_sha256": prior["requests_sha256"],
        "sample_sha256": v1.sha256_file(sample_path),
        "cloud_transport": "batch_only_including_retries", "batch_completion_window": "24h",
        "max_output_tokens": v2.MAX_OUTPUT_TOKENS, "max_input_tokens_o200k": v2.MAX_INPUT_TOKENS,
        "qwen_num_ctx": v2.QWEN_CONTEXT, "qwen_think": True,
        "openai_reasoning": "medium", "gemini_thinking": "MEDIUM",
        "evidence_policy": "byte-identical v2 evidence suffix and complete occurrence set",
        "prior_labels_in_model_inputs": False,
    }
    contract_hash = v1.sha256_json(contract)
    encoding = tiktoken.get_encoding("o200k_base")
    requests = []
    for row in original:
        prompt = binary_prompt(row)
        tokens = len(encoding.encode(v2.SYSTEM + prompt + json.dumps(schema())))
        if tokens > v2.MAX_INPUT_TOKENS:
            raise ValueError(f"Binary prompt exceeds the context budget: {row['custom_id']}")
        suffix_hash = hashlib.sha256(evidence_suffix(row["prompt"]).encode()).hexdigest()
        if hashlib.sha256(evidence_suffix(prompt).encode()).hexdigest() != suffix_hash:
            raise ValueError("Binary preparation changed source evidence.")
        requests.append({**row, "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                         "source_prompt_sha256": row["prompt_sha256"], "source_evidence_sha256": suffix_hash,
                         "study_contract_sha256": contract_hash, "input_tokens_o200k": tokens})
    output.mkdir(parents=True)
    (output / "samples").mkdir()
    shutil.copy2(sample_path, output / "samples" / sample_path.name)
    (output / "reference").mkdir()
    comparison = source / "consensus/tradeoff_audit_consensus.parquet"
    shutil.copy2(comparison, output / "reference/v2_consensus.parquet")
    with (output / "requests.jsonl").open("w") as handle:
        for row in requests:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = [{k: value for k, value in row.items() if k not in {"prompt", "quote_corpus", "occurrence_dimensions"}}
                for row in requests]
    pd.DataFrame(manifest).to_parquet(output / "manifest.parquet", index=False)
    summary = {
        "version": VERSION, "requests_per_provider": len(requests), "classification_requests": len(requests) * 3,
        "arms": sample.sample_arm.value_counts().to_dict(), "contract": contract,
        "study_contract_sha256": contract_hash, "requests_sha256": v1.sha256_file(output / "requests.jsonl"),
        "reference_sha256": v1.sha256_file(output / "reference/v2_consensus.parquet"),
        "manifest_sha256": v1.sha256_file(output / "manifest.parquet"),
        "input_tokens_o200k": sum(row["input_tokens_o200k"] for row in requests),
        "max_input_tokens_o200k": max(row["input_tokens_o200k"] for row in requests),
        "occurrences": sum(len(row["occurrence_ids"]) for row in requests),
        "activations": sum(row["activation_count"] for row in requests),
        "identical_evidence_suffixes": len(requests), "truncations": 0,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    v1.atomic_write_json(output / "preflight.json", summary)
    snapshots = output / "source"
    snapshots.mkdir()
    for path in [*Path(__file__).parent.glob("*binary*.py"), Path(v1.__file__), Path(v2.__file__)]:
        shutil.copy2(path, snapshots / path.name)
    return summary


def load_requests(output: Path) -> list[dict]:
    preflight = json.loads((output / "preflight.json").read_text())
    if v1.sha256_file(output / "requests.jsonl") != preflight["requests_sha256"]:
        raise ValueError("Prepared binary requests changed.")
    contract = preflight["contract"]
    if (v1.sha256_json(contract) != preflight["study_contract_sha256"] or contract["schema"] != schema()
            or contract["system"] != v2.SYSTEM or contract["instructions"] != INSTRUCTIONS
            or contract["models"] != v2.MODELS or contract["max_output_tokens"] != v2.MAX_OUTPUT_TOKENS
            or contract["qwen_num_ctx"] != v2.QWEN_CONTEXT):
        raise ValueError("Binary study contract or runtime configuration changed.")
    rows = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    if len(rows) != 29 or len({r["custom_id"] for r in rows}) != 29:
        raise ValueError("Expected exactly 29 distinct binary requests.")
    for row in rows:
        if (row["study_contract_sha256"] != preflight["study_contract_sha256"]
                or hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]
                or hashlib.sha256(evidence_suffix(row["prompt"]).encode()).hexdigest() != row["source_evidence_sha256"]):
            raise ValueError("Binary prompt, evidence, or contract hash mismatch.")
    correction_path = output / "validation_corrections.json"
    if correction_path.exists():
        ledger = json.loads(correction_path.read_text())
        if (ledger["validation_version"] != binary_validation.VERSION
                or ledger["study_contract_sha256"] != preflight["study_contract_sha256"]):
            raise ValueError("Validation correction ledger belongs to another study or validator.")
        indexed = {row["custom_id"]: row for row in rows}
        seen = set()
        for correction in ledger["dimension_corrections"]:
            key = (correction["custom_id"], correction["occurrence_id"])
            if key in seen or key[0] not in indexed:
                raise ValueError("Duplicate or unknown audited correction identity.")
            seen.add(key)
            indexed[key[0]].setdefault("validation_dimension_corrections", []).append(correction)
        for row in rows:
            binary_validation.corrected_dimensions(row)
            row["validation_corrections_sha256"] = v1.sha256_file(correction_path)
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
            raise ValueError("Checkpoint belongs to another binary request or contract.")
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
            rows.append({**{k: value for k, value in checkpoint.items() if k not in {"label", "usage"}},
                         **checkpoint.get("label", {}), "usage_json": json.dumps(checkpoint.get("usage", {}))})
    directory = output / provider
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    counts = frame.classification_status.value_counts().to_dict() if rows else {}
    if rows:
        temporary = directory / "labels.tmp.parquet"
        frame.to_parquet(temporary, index=False)
        temporary.replace(directory / "labels.parquet")
    state = {"provider": provider, "expected": len(requests), "classified": counts.get("classified", 0),
             "errors": counts.get("error", 0), "checkpoints": len(rows),
             "status": "completed" if counts.get("classified", 0) == len(requests) else "in_progress",
             "updated_at": datetime.now(timezone.utc).isoformat()}
    v1.atomic_write_json(directory / "run_state.json", state)
    return state


def consensus_record(request: dict, votes: dict[str, dict]) -> dict:
    if set(votes) != set(PROVIDERS) or any(v is None or v["classification_status"] != "classified" for v in votes.values()):
        raise ValueError("Binary consensus needs three valid classified votes.")
    for vote in votes.values():
        if any(vote[key] != request[key] for key in ("custom_id", "prompt_sha256", "study_contract_sha256")):
            raise ValueError("Consensus input hashes or identities differ.")
        validate_result({key: vote["label"][key] for key in ResultModel.model_fields}, request)
    labels = {p: votes[p]["label"]["label"] for p in PROVIDERS}
    winner, count = Counter(labels.values()).most_common(1)[0]
    if count < 2:
        raise ValueError("A binary three-model vote must have a majority.")
    return {**{key: request[key] for key in ("repo_id", "number", "sample_arm", "html_url")},
            "consensus_label": winner, "consensus_status": "unanimous" if count == 3 else "majority",
            "inference_votes": sum(votes[p]["label"]["choice_requires_inference"] for p in PROVIDERS),
            **{f"{p}_label": labels[p] for p in PROVIDERS},
            **{f"{p}_requires_inference": votes[p]["label"]["choice_requires_inference"] for p in PROVIDERS},
            **{f"{p}_rationale": votes[p]["label"]["rationale"] for p in PROVIDERS}}


def build_consensus(output: Path) -> dict:
    requests = load_requests(output)
    records = [consensus_record(request, {p: checked_checkpoint(output, p, request) for p in PROVIDERS})
               for request in requests]
    reference_path = output / "reference/v2_consensus.parquet"
    if v1.sha256_file(reference_path) != json.loads((output / "preflight.json").read_text())["reference_sha256"]:
        raise ValueError("The frozen v2 comparison changed.")
    old = pd.read_parquet(reference_path)[[*v1.KEYS, "consensus_label", "v1_label"]]
    result = pd.DataFrame(records).merge(old.rename(columns={"consensus_label": "v2_label"}),
                                       on=v1.KEYS, validate="one_to_one")
    if len(result) != len(requests):
        raise ValueError("Binary/v2 comparison changed the population.")
    directory = output / "consensus"
    directory.mkdir(exist_ok=True)
    result.to_parquet(directory / "tradeoff_binary_consensus.parquet", index=False)
    result.to_csv(directory / "tradeoff_binary_comparison.csv", index=False)
    summary = {"version": VERSION, "validation_version": binary_validation.VERSION,
               "validation_corrections_sha256": requests[0].get("validation_corrections_sha256"),
               "status": "completed", "prs": len(result),
               "classes": result.consensus_label.value_counts().to_dict(),
               "agreement": result.consensus_status.value_counts().to_dict(),
               "majority_requires_inference": int(result.inference_votes.ge(2).sum()),
               "by_arm": {arm: group.consensus_label.value_counts().to_dict() for arm, group in result.groupby("sample_arm")},
               "v2_transitions": [{"v2": old, "binary": new, "prs": int(n)} for (old, new), n in
                                  result.groupby(["v2_label", "consensus_label"]).size().items()]}
    v1.atomic_write_json(directory / "summary.json", summary)
    (directory / "summary.md").write_text(
        "# RQ3 forced-binary results\n\nThe same 29 trade-off candidates and complete evidence as v2.\n\n"
        + f"- Classes: {json.dumps(summary['classes'])}\n- Agreement: {json.dumps(summary['agreement'])}\n"
        + f"- Majority requires inference: {summary['majority_requires_inference']}\n\n"
        + "Final labels are forced choices. Observed directions and per-model inference flags remain available; "
        "a binary choice does not establish a measured trade-off where evidence is insufficient.\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "consensus"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = prepare(args.output_dir) if args.action == "prepare" else build_consensus(args.output_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "contract"}, indent=2))
