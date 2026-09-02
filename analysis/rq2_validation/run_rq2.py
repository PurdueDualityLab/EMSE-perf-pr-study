"""Prepare, submit, inspect, and collect the RQ2 OpenAI batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, model_validator

ENDPOINT = "/v1/responses"
PROMPT_VERSION = "rq2-performance-validation"
DEFAULT_MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 4_096
TEXT_CHARACTER_LIMIT = 15_000
PATCH_CHARACTER_LIMIT = 15_000
TRUNCATION_MARKER = "\n\n... [content truncated for length] ..."
VALIDATION_TYPES = ("benchmark", "profiling", "static-reasoning", "anecdotal")
EVIDENCE_SOURCES = ("description", "comments", "reviews", "code_diff", "ci")
SYSTEM_INSTRUCTION = (
    "You classify explicit, observable performance-validation evidence in GitHub pull "
    "request artifacts. Do not infer validation from performance intent or code changes alone."
)


class ValidationLabel(BaseModel):
    validation_present: bool
    validation_types: list[Literal["benchmark", "profiling", "static-reasoning", "anecdotal"]]
    primary_validation_type: Literal[
        "benchmark", "profiling", "static-reasoning", "anecdotal", "none"
    ]
    evidence_sources: list[Literal["description", "comments", "reviews", "code_diff", "ci"]]
    metrics: list[str]
    evidence_quotes: list[str]
    validation_description: str

    @model_validator(mode="after")
    def validate_semantics(self) -> "ValidationLabel":
        if len(set(self.validation_types)) != len(self.validation_types):
            raise ValueError("validation_types must not contain duplicates")
        if len(set(self.evidence_sources)) != len(self.evidence_sources):
            raise ValueError("evidence_sources must not contain duplicates")
        if not self.validation_present:
            if self.validation_types or self.primary_validation_type != "none":
                raise ValueError("absent validation requires no validation types")
            if self.evidence_sources or self.metrics or self.evidence_quotes:
                raise ValueError("absent validation requires no evidence")
        else:
            if not self.validation_types or self.primary_validation_type not in self.validation_types:
                raise ValueError("present validation requires a primary type in validation_types")
            if not self.evidence_sources or not self.evidence_quotes:
                raise ValueError("present validation requires sources and quotes")
        return self


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def semantic_schema() -> dict[str, Any]:
    schema = ValidationLabel.model_json_schema()
    schema["additionalProperties"] = False
    return schema


def _text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + TRUNCATION_MARKER


def _group_text(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    order: list[str],
) -> pd.DataFrame:
    keys = ["repo_id", "number"]
    if frame.empty:
        return pd.DataFrame(columns=[*keys, "text"])
    available_order = [column for column in order if column in frame.columns]
    rows = []
    for key, group in frame.sort_values([*keys, *available_order], kind="mergesort").groupby(
        keys, sort=False
    ):
        parts = []
        for row in group.to_dict("records"):
            values = [f"{column}={_text(row.get(column))}" for column in columns if _text(row.get(column))]
            if values:
                parts.append(" | ".join(values))
        rows.append({"repo_id": key[0], "number": key[1], "text": "\n\n".join(parts)})
    return pd.DataFrame(rows)


def build_input(sample_path: Path, evidence_dir: Path) -> pd.DataFrame:
    sample = pd.read_parquet(sample_path)
    if sample.empty or sample.duplicated(["repo_id", "number"]).any():
        raise ValueError("RQ2 requires a nonempty sample with unique PR identities.")
    arm_counts = sample["sample_arm"].value_counts().to_dict()
    if set(arm_counts) != {"agentic", "human_candidate"} or len(set(arm_counts.values())) != 1:
        raise ValueError("RQ2 requires the final 1:1 balanced sample.")

    pull_requests = pd.read_parquet(evidence_dir / "pull_requests.parquet")
    files = pd.read_parquet(evidence_dir / "pull_request_files.parquet")
    status = pd.read_parquet(evidence_dir / "collection_status.parquet")
    issue_comments = pd.read_parquet(evidence_dir / "issue_comments.parquet")
    review_comments = pd.read_parquet(evidence_dir / "review_comments.parquet")
    reviews = pd.read_parquet(evidence_dir / "reviews.parquet")
    workflows = pd.read_parquet(evidence_dir / "workflow_runs.parquet")
    checks = pd.read_parquet(evidence_dir / "check_runs.parquet")
    sample_ids = set(map(tuple, sample[["repo_id", "number"]].to_numpy()))
    status_ids = set(map(tuple, status[["repo_id", "number"]].to_numpy()))
    if status.duplicated(["repo_id", "number"]).any() or not sample_ids.issubset(status_ids):
        raise ValueError("RQ2 evidence status does not cover every analysis-sample identity.")
    status = status[status[["repo_id", "number"]].apply(tuple, axis=1).isin(sample_ids)].copy()
    required_statuses = [
        "pull_request_files_status", "commits_status", "issue_comments_status",
        "review_comments_status", "reviews_status",
        "workflow_runs_status", "check_runs_status",
    ]
    incomplete = ~status[required_statuses].eq("complete").all(axis=1)
    status["evidence_complete"] = ~incomplete

    patch = _group_text(files, columns=["filename", "patch"], order=["file_index"])
    comments = pd.concat(
        [
            issue_comments.assign(comment_kind="issue_comment"),
            review_comments.assign(comment_kind="review_comment"),
        ],
        ignore_index=True,
    )
    comment_text = _group_text(
        comments,
        columns=["comment_kind", "author_login", "created_at", "body"],
        order=["created_at", "comment_id"],
    )
    review_text = _group_text(
        reviews,
        columns=["author_login", "submitted_at", "state", "body"],
        order=["submitted_at", "review_id"],
    )
    workflow_text = _group_text(
        workflows,
        columns=["name", "display_title", "status", "conclusion", "created_at"],
        order=["created_at", "workflow_run_id"],
    )
    check_text = _group_text(
        checks,
        columns=["name", "app_slug", "status", "conclusion", "output_title", "output_summary"],
        order=["started_at", "check_run_id"],
    )
    ci = workflow_text.merge(check_text, on=["repo_id", "number"], how="outer", suffixes=("_workflow", "_check"))
    ci["ci"] = ci.get("text_workflow", "").fillna("") + "\n\n" + ci.get("text_check", "").fillna("")

    result = sample.merge(
        pull_requests[["repo_id", "number", "title", "body"]],
        on=["repo_id", "number"],
        how="left",
        suffixes=("", "_evidence"),
        validate="one_to_one",
    )
    if "title_evidence" in result:
        result["title"] = result["title_evidence"].combine_first(result["title"])
        result["body"] = result["body_evidence"].combine_first(result["body"])
        result = result.drop(columns=["title_evidence", "body_evidence"])
    for frame, name in (
        (patch, "code_diff"),
        (comment_text, "comments"),
        (review_text, "reviews"),
        (ci[["repo_id", "number", "ci"]], "ci"),
    ):
        value = frame.rename(columns={"text": name})
        result = result.merge(value, on=["repo_id", "number"], how="left", validate="one_to_one")
    result = result.merge(
        status[["repo_id", "number", "status", "evidence_complete"]].rename(
            columns={"status": "evidence_status"}
        ),
        on=["repo_id", "number"],
        how="left",
        validate="one_to_one",
    )
    for column in ("code_diff", "comments", "reviews", "ci"):
        result[column] = result[column].fillna("")
    return result


def prompt_for(row: dict[str, Any]) -> str:
    sections = {
        "TITLE": _text(row.get("title")),
        "DESCRIPTION": _truncate(_text(row.get("body")), TEXT_CHARACTER_LIMIT),
        "COMMENTS": _truncate(_text(row.get("comments")), TEXT_CHARACTER_LIMIT),
        "REVIEWS": _truncate(_text(row.get("reviews")), TEXT_CHARACTER_LIMIT),
        "CODE DIFF": _truncate(_text(row.get("code_diff")), PATCH_CHARACTER_LIMIT),
        "CI WORKFLOWS AND CHECKS": _truncate(_text(row.get("ci")), TEXT_CHARACTER_LIMIT),
    }
    if not bool(row.get("evidence_complete", False)):
        raise ValueError("RQ2 cannot classify a PR with incomplete observable evidence.")
    context = "\n\n".join(f"{name}:\n{value or '[NONE]'}" for name, value in sections.items())
    return f"""Classify the performance-validation evidence reported in this pull request.

Validation is present only when an artifact explicitly reports evidence supporting a performance claim. Do not infer validation merely because the patch looks faster, tests pass, or a workflow/check name contains words such as benchmark or performance.

Categories are non-exclusive:
- benchmark: quantitative runtime measurements such as latency, throughput, memory, CPU, allocations, or before/after benchmark results.
- profiling: profiler-derived evidence such as hotspots, traces, samples, flamegraphs, or function timings.
- static-reasoning: an explicit analytical performance argument, such as complexity, operation-count, allocation, copying, or I/O reasoning, without runtime measurements.
- anecdotal: an explicit claim of local/manual performance testing or observation without quantitative or analytical support.

Assign every category that is independently supported. Select primary_validation_type as the strongest central evidence, preferring benchmark, then profiling, then static-reasoning, then anecdotal when multiple types are equally central. If no explicit evidence exists, set validation_present=false, primary_validation_type=none, and all evidence lists empty.

evidence_quotes must contain short verbatim excerpts from the supplied artifacts. metrics must contain only explicitly reported metric names or values. CI success alone establishes correctness testing, not performance validation.

{context}
"""


def response_format() -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "ValidationLabel",
            "schema": semantic_schema(),
            "strict": True,
        }
    }


def study_contract(sample_sha256: str, evidence_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "study": "rq2_performance_validation_classification",
        "population": "complete-observability weekly-balanced RQ2 analysis sample",
        "prompt_version": PROMPT_VERSION,
        "system_instruction_sha256": hashlib.sha256(SYSTEM_INSTRUCTION.encode()).hexdigest(),
        "semantic_schema_sha256": sha256_json(semantic_schema()),
        "sample_sha256": sample_sha256,
        "evidence_sha256": evidence_hashes,
        "input_construction": {
            "identity_columns": ["repo_id", "number"],
            "independent_text_character_limit": TEXT_CHARACTER_LIMIT,
            "patch_character_limit": PATCH_CHARACTER_LIMIT,
            "truncation_marker": TRUNCATION_MARKER,
            "comments_order": ["created_at", "comment_id"],
            "reviews_order": ["submitted_at", "review_id"],
            "files_order": ["file_index"],
        },
    }


def openai_provider_config(model: str) -> dict[str, Any]:
    return {
        "provider_config_version": 1,
        "provider": "openai",
        "model": model,
        "endpoint": ENDPOINT,
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_wrapper_sha256": sha256_json(response_format()),
    }


def input_row_sha256(row: dict[str, Any]) -> str:
    return sha256_json(
        {
            "repo_id": int(row["repo_id"]),
            "number": int(row["number"]),
            "title": _text(row.get("title")),
            "body": _truncate(_text(row.get("body")), TEXT_CHARACTER_LIMIT),
            "comments": _truncate(_text(row.get("comments")), TEXT_CHARACTER_LIMIT),
            "reviews": _truncate(_text(row.get("reviews")), TEXT_CHARACTER_LIMIT),
            "code_diff": _truncate(_text(row.get("code_diff")), PATCH_CHARACTER_LIMIT),
            "ci": _truncate(_text(row.get("ci")), TEXT_CHARACTER_LIMIT),
        }
    )


def prepare_analysis_sample(
    source_sample_path: Path, evidence_dir: Path, output_dir: Path
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("RQ2 sample output directory is not empty.")
    sample = pd.read_parquet(source_sample_path)
    status = pd.read_parquet(evidence_dir / "collection_status.parquet")
    keys = ["repo_id", "number"]
    if sample.empty or sample.duplicated(keys).any() or status.duplicated(keys).any():
        raise ValueError("RQ2 sample preparation requires unique PR identities.")
    source_ids = set(map(tuple, sample[keys].to_numpy()))
    status_ids = set(map(tuple, status[keys].to_numpy()))
    if source_ids != status_ids:
        raise ValueError("Evidence status does not exactly match the source sample.")
    required_statuses = [
        "pull_request_files_status", "commits_status", "issue_comments_status",
        "review_comments_status", "reviews_status",
        "workflow_runs_status", "check_runs_status",
    ]
    complete = status[required_statuses].eq("complete").all(axis=1)
    unavailable = sample.merge(
        status.loc[~complete, [*keys, "status"]], on=keys, how="inner", validate="one_to_one"
    )
    exclusion_columns = [
        *keys, "sample_arm", "sampling_stratum", "selection_hash", "exclusion_reason",
        "paired_with_repo_id", "paired_with_number",
    ]
    if unavailable.empty:
        analysis_sample = sample.copy()
        exclusions = pd.DataFrame(columns=exclusion_columns)
    else:
        exclusions = []
        excluded_ids = set(map(tuple, unavailable[keys].to_numpy()))
        for row in unavailable.sort_values(keys).to_dict("records"):
            identity = (int(row["repo_id"]), int(row["number"]))
            exclusions.append(
                {
                    **{column: row[column] for column in keys},
                    "sample_arm": row["sample_arm"],
                    "sampling_stratum": row["sampling_stratum"],
                    "selection_hash": row["selection_hash"],
                    "exclusion_reason": "observable_evidence_incomplete",
                    "paired_with_repo_id": pd.NA,
                    "paired_with_number": pd.NA,
                }
            )
            opposite = "human_candidate" if row["sample_arm"] == "agentic" else "agentic"
            candidates = sample[
                sample["sample_arm"].eq(opposite)
                & sample["sampling_stratum"].eq(row["sampling_stratum"])
                & ~sample[keys].apply(tuple, axis=1).isin(excluded_ids)
            ].sort_values(["selection_hash", *keys], ascending=[False, True, True], kind="mergesort")
            if candidates.empty:
                raise ValueError(f"No balancing counterpart exists for {identity}.")
            paired = candidates.iloc[0]
            paired_identity = (int(paired["repo_id"]), int(paired["number"]))
            excluded_ids.add(paired_identity)
            exclusions[-1]["paired_with_repo_id"] = paired_identity[0]
            exclusions[-1]["paired_with_number"] = paired_identity[1]
            exclusions.append(
                {
                    "repo_id": paired_identity[0],
                    "number": paired_identity[1],
                    "sample_arm": opposite,
                    "sampling_stratum": paired["sampling_stratum"],
                    "selection_hash": paired["selection_hash"],
                    "exclusion_reason": "paired_balance_exclusion",
                    "paired_with_repo_id": identity[0],
                    "paired_with_number": identity[1],
                }
            )
        exclusions = pd.DataFrame(exclusions, columns=exclusion_columns)
        analysis_sample = sample[~sample[keys].apply(tuple, axis=1).isin(excluded_ids)].copy()
    arm_counts = analysis_sample["sample_arm"].value_counts().to_dict()
    weekly = analysis_sample.groupby(["sampling_stratum", "sample_arm"]).size().unstack(fill_value=0)
    if set(arm_counts) != {"agentic", "human_candidate"} or len(set(arm_counts.values())) != 1:
        raise ValueError("RQ2 analysis sample is not globally balanced.")
    if not weekly["agentic"].equals(weekly["human_candidate"]):
        raise ValueError("RQ2 analysis sample is not balanced within weekly strata.")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "balanced_sample.parquet"
    exclusions_path = output_dir / "evidence_exclusions.parquet"
    analysis_sample.sort_values(["created_at", "sample_arm", *keys], kind="mergesort").to_parquet(
        sample_path, index=False
    )
    exclusions.to_parquet(exclusions_path, index=False)
    metadata = {
        "method": "rq2_complete_observability_balanced_subset",
        "source_sample": str(source_sample_path),
        "source_sample_sha256": sha256_file(source_sample_path),
        "evidence_status_sha256": sha256_file(evidence_dir / "collection_status.parquet"),
        "rows": len(analysis_sample),
        "arm_counts": arm_counts,
        "exclusions": len(exclusions),
        "selection_rule": "Exclude incomplete rows and the opposite-arm row in the same weekly stratum with the greatest selection_hash.",
        "output_sha256": sha256_file(sample_path),
        "exclusions_sha256": sha256_file(exclusions_path),
    }
    atomic_write_json(output_dir / "prepare_metadata.json", metadata)
    return sample_path


def prepare_batch(
    sample_path: Path, evidence_dir: Path, output_dir: Path, model: str
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; refusing to overwrite a run.")
    frame = build_input(sample_path, evidence_dir).sort_values(["repo_id", "number"], kind="mergesort")
    incomplete = frame.loc[~frame["evidence_complete"], ["repo_id", "number"]]
    if not incomplete.empty:
        raise ValueError(
            "RQ2 cannot prepare until observable evidence is complete for: "
            + json.dumps(incomplete.to_dict("records"), sort_keys=True)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_files = (
        "pull_requests.parquet", "pull_request_files.parquet", "issue_comments.parquet",
        "review_comments.parquet", "reviews.parquet", "workflow_runs.parquet",
        "check_runs.parquet", "collection_status.parquet",
    )
    evidence_hashes = {name: sha256_file(evidence_dir / name) for name in evidence_files}
    contract = study_contract(sha256_file(sample_path), evidence_hashes)
    contract_hash = sha256_json(contract)
    provider = openai_provider_config(model)
    provider_hash = sha256_json(provider)
    payload_path = output_dir / "batch_input.jsonl"
    manifest_rows = []
    with payload_path.open("w", encoding="utf-8") as handle:
        for row in frame.to_dict("records"):
            custom_id = f"{int(row['repo_id'])}:{int(row['number'])}"
            prompt = prompt_for(row)
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": ENDPOINT,
                "body": {
                    "model": model,
                    "input": [
                        {"role": "developer", "content": SYSTEM_INSTRUCTION},
                        {"role": "user", "content": prompt},
                    ],
                    "reasoning": {"effort": REASONING_EFFORT},
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "text": response_format(),
                },
            }
            handle.write(json.dumps(request, ensure_ascii=True) + "\n")
            manifest_rows.append(
                {
                    "repo_id": int(row["repo_id"]), "number": int(row["number"]),
                    "custom_id": custom_id, "repo_full_name": row["repo_full_name"],
                    "html_url": row["html_url"], "sample_arm": row["sample_arm"],
                    "prompt_version": PROMPT_VERSION, "evidence_status": row["evidence_status"],
                    "evidence_complete": bool(row["evidence_complete"]),
                    "prompt_chars": len(prompt),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "input_row_sha256": input_row_sha256(row),
                    "study_contract_sha256": contract_hash,
                    "provider_config_sha256": provider_hash,
                }
            )
    manifest_path = output_dir / "batch_manifest.parquet"
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
    metadata = {
        "model": model, "endpoint": ENDPOINT, "prompt_version": PROMPT_VERSION,
        "requests": len(manifest_rows), "sample_sha256": sha256_file(sample_path),
        "evidence_sha256": evidence_hashes, "schema_sha256": sha256_json(response_format()),
        "batch_input_sha256": sha256_file(payload_path), "batch_input_bytes": payload_path.stat().st_size,
        "manifest_sha256": sha256_file(manifest_path), "study_contract": contract,
        "study_contract_sha256": contract_hash, "provider_config": provider,
        "provider_config_sha256": provider_hash,
    }
    (output_dir / "prepare_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload_path


def submit_batch(output_dir: Path) -> dict[str, Any]:
    payload_path = output_dir / "batch_input.jsonl"
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text())
    state_path = output_dir / "batch_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("batch_id"):
            raise FileExistsError("Batch was already submitted.")
    if sha256_file(payload_path) != metadata["batch_input_sha256"]:
        raise ValueError("Batch input changed after preparation.")
    if sha256_file(output_dir / "batch_manifest.parquet") != metadata["manifest_sha256"]:
        raise ValueError("Batch manifest changed after preparation.")
    else:
        state = {**metadata, "local_status": "prepared"}
        atomic_write_json(state_path, state)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not state.get("input_file_id"):
        with payload_path.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        state.update(input_file_id=uploaded.id, local_status="uploaded")
        atomic_write_json(state_path, state)
    batch = client.batches.create(
        input_file_id=state["input_file_id"], endpoint=ENDPOINT, completion_window="24h",
        metadata={"description": "RQ2 performance-validation classification"},
    )
    state.update(batch_id=batch.id, status=batch.status, local_status="submitted",
                 created_at=batch.created_at, expires_at=batch.expires_at)
    atomic_write_json(state_path, state)
    return state


def batch_status(output_dir: Path) -> dict[str, Any]:
    state_path = output_dir / "batch_state.json"
    state = json.loads(state_path.read_text())
    batch = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")).batches.retrieve(state["batch_id"])
    status = {
        "batch_id": batch.id, "status": batch.status,
        "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
        "output_file_id": batch.output_file_id, "error_file_id": batch.error_file_id,
        "completed_at": batch.completed_at, "expires_at": batch.expires_at,
    }
    state.update(status)
    atomic_write_json(state_path, state)
    return status


def _output_text(body: dict[str, Any]) -> str:
    if body.get("status") == "incomplete" or body.get("incomplete_details"):
        raise ValueError("Batch response is incomplete.")
    texts = [str(content.get("text") or "") for output in body.get("output", [])
             for content in output.get("content", []) if content.get("type") == "output_text"]
    if not texts:
        raise ValueError("Batch response has no output_text content.")
    return "".join(texts)


def write_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    summary = {
        "method": PROMPT_VERSION, "rows": len(frame),
        "status_counts": frame["classification_status"].value_counts().to_dict(),
        "arm_counts": frame["sample_arm"].value_counts().to_dict(),
        "validation_presence_counts": frame.get("validation_present", pd.Series(dtype=bool)).value_counts().to_dict(),
        "primary_type_counts": frame.get("primary_validation_type", pd.Series(dtype=str)).value_counts().to_dict(),
        "input_tokens": int(frame.get("input_tokens", pd.Series(dtype=float)).sum()),
        "output_tokens": int(frame.get("output_tokens", pd.Series(dtype=float)).sum()),
        "reasoning_tokens": int(frame.get("reasoning_tokens", pd.Series(dtype=float)).sum()),
        "total_tokens": int(frame.get("total_tokens", pd.Series(dtype=float)).sum()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def collect_batch(output_dir: Path) -> pd.DataFrame:
    status = batch_status(output_dir)
    if status["status"] != "completed" or not status["output_file_id"]:
        raise RuntimeError(f"Batch is not ready for collection: {status['status']}")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    output_text = client.files.content(status["output_file_id"]).text
    (output_dir / "batch_output.jsonl").write_text(output_text)
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text())
    manifest_path = output_dir / "batch_manifest.parquet"
    if sha256_file(manifest_path) != metadata["manifest_sha256"]:
        raise ValueError("Batch manifest changed after preparation.")
    manifest = pd.read_parquet(manifest_path)
    if manifest["custom_id"].isna().any() or not manifest["custom_id"].is_unique:
        raise ValueError("Batch manifest contains invalid or duplicate custom IDs.")
    by_id = manifest.set_index("custom_id").to_dict("index")
    rows, seen = [], set()
    for line in output_text.splitlines():
        result = json.loads(line)
        custom_id = result["custom_id"]
        if custom_id in seen or custom_id not in by_id:
            raise ValueError(f"Invalid or duplicate batch custom_id: {custom_id}")
        seen.add(custom_id)
        response = result.get("response") or {}
        body = response.get("body") or {}
        row = {**by_id[custom_id], "custom_id": custom_id,
               "model": body.get("model") or metadata["model"], "response_id": body.get("id")}
        if response.get("status_code") != 200:
            row.update(classification_status="error", error=json.dumps(result.get("error") or response)[:500])
        else:
            usage = body.get("usage") or {}
            row.update(
                input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
                cached_input_tokens=(usage.get("input_tokens_details") or {}).get("cached_tokens"),
                reasoning_tokens=(usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
            )
            try:
                label = ValidationLabel.model_validate_json(_output_text(body))
            except ValueError as error:
                row.update(classification_status="error", error=f"Invalid structured response: {error}"[:500])
            else:
                row.update(label.model_dump(), classification_status="classified")
        rows.append(row)
    for custom_id in sorted(set(by_id) - seen):
        rows.append({**by_id[custom_id], "custom_id": custom_id, "model": metadata["model"],
                     "classification_status": "error", "error": "Missing from batch output"})
    frame = pd.DataFrame(rows).sort_values(["repo_id", "number"]).reset_index(drop=True)
    if len(frame) != len(manifest) or frame["custom_id"].duplicated().any():
        raise ValueError("Collected rows do not exactly preserve the batch manifest.")
    frame.to_parquet(output_dir / "validation_labels.parquet", index=False)
    write_summary(frame, output_dir)
    return frame


def prepare_retry(source_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Retry output directory is not empty.")
    labels = pd.read_parquet(source_dir / "validation_labels.parquet")
    errors = labels[labels["classification_status"].eq("error")]
    if errors.empty:
        raise ValueError("Source run has no errors to retry.")
    metadata = json.loads((source_dir / "prepare_metadata.json").read_text())
    if sha256_file(source_dir / "batch_input.jsonl") != metadata["batch_input_sha256"]:
        raise ValueError("Source batch input changed after preparation.")
    if sha256_file(source_dir / "batch_manifest.parquet") != metadata["manifest_sha256"]:
        raise ValueError("Source batch manifest changed after preparation.")
    requests = {item["custom_id"]: item for item in map(json.loads, (source_dir / "batch_input.jsonl").read_text().splitlines())}
    retry_ids = errors["custom_id"].tolist()
    output_dir.mkdir(parents=True)
    payload = output_dir / "batch_input.jsonl"
    with payload.open("w") as handle:
        for custom_id in retry_ids:
            handle.write(json.dumps(requests[custom_id], ensure_ascii=True) + "\n")
    manifest_path = output_dir / "batch_manifest.parquet"
    pd.read_parquet(source_dir / "batch_manifest.parquet").query("custom_id in @retry_ids").to_parquet(manifest_path, index=False)
    retry_metadata = {**metadata, "requests": len(retry_ids), "retry_of": str(source_dir),
                      "batch_input_sha256": sha256_file(payload), "batch_input_bytes": payload.stat().st_size,
                      "manifest_sha256": sha256_file(manifest_path)}
    for key in ("input_file_id", "batch_id", "status", "created_at", "expires_at", "output_file_id", "error_file_id", "completed_at"):
        retry_metadata.pop(key, None)
    (output_dir / "prepare_metadata.json").write_text(json.dumps(retry_metadata, indent=2, sort_keys=True) + "\n")
    return payload


def merge_retry(source_dir: Path, retry_dir: Path, output_path: Path) -> pd.DataFrame:
    source = pd.read_parquet(source_dir / "validation_labels.parquet")
    retry = pd.read_parquet(retry_dir / "validation_labels.parquet")
    errors = source[source["classification_status"].eq("error")]
    if set(retry["custom_id"]) != set(errors["custom_id"]):
        raise ValueError("Retry identities do not exactly match source errors.")
    hashes = ["study_contract_sha256", "provider_config_sha256", "input_row_sha256", "prompt_sha256"]
    if not errors.set_index("custom_id")[hashes].sort_index().equals(retry.set_index("custom_id")[hashes].sort_index()):
        raise ValueError("Retry hashes do not match the source errors.")
    merged = pd.concat([source[~source["custom_id"].isin(retry["custom_id"])], retry], ignore_index=True).sort_values(["repo_id", "number"])
    if len(merged) != len(source) or merged.duplicated(["repo_id", "number"]).any():
        raise ValueError("Merged retry does not preserve source population.")
    if merged["classification_status"].eq("error").any():
        raise ValueError("Retry still contains errors; prepare another retry before finalizing.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    return merged.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--sample", type=Path, required=True)
    prepare.add_argument("--evidence-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    sample = commands.add_parser("prepare-sample")
    sample.add_argument("--source-sample", type=Path, required=True)
    sample.add_argument("--evidence-dir", type=Path, required=True)
    sample.add_argument("--output-dir", type=Path, required=True)
    for action in ("submit", "status", "collect"):
        command = commands.add_parser(action)
        command.add_argument("--output-dir", type=Path, required=True)
    retry = commands.add_parser("prepare-retry")
    retry.add_argument("--source-dir", type=Path, required=True)
    retry.add_argument("--output-dir", type=Path, required=True)
    merge = commands.add_parser("merge-retry")
    merge.add_argument("--source-dir", type=Path, required=True)
    merge.add_argument("--retry-dir", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        print(prepare_batch(args.sample, args.evidence_dir, args.output_dir, args.model))
    elif args.action == "prepare-sample":
        print(prepare_analysis_sample(args.source_sample, args.evidence_dir, args.output_dir))
    elif args.action == "submit":
        print(json.dumps(submit_batch(args.output_dir), sort_keys=True))
    elif args.action == "status":
        print(json.dumps(batch_status(args.output_dir), sort_keys=True))
    elif args.action == "collect":
        print(f"Collected {len(collect_batch(args.output_dir))} labels")
    elif args.action == "prepare-retry":
        print(prepare_retry(args.source_dir, args.output_dir))
    else:
        print(f"Merged {len(merge_retry(args.source_dir, args.retry_dir, args.output))} labels")


if __name__ == "__main__":
    main()
