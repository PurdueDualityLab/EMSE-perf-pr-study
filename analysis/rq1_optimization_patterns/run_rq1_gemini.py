"""Prepare, submit, inspect, collect, and retry one logical RQ1 Gemini run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from google import genai
from google.genai import types

from run_rq1 import (
    MAX_OUTPUT_TOKENS,
    PATCH_CHARACTER_LIMIT,
    PROMPT_VERSION,
    SYSTEM_INSTRUCTION,
    PatternLabel,
    build_input,
    input_row_sha256,
    load_taxonomy,
    prompt_for,
    semantic_schema,
    sha256_file,
    sha256_json,
    study_contract,
    taxonomy_labels,
    atomic_write_json,
)

DEFAULT_MODEL = "gemini-3.1-pro-preview"
THINKING_LEVEL = "MEDIUM"
MAX_REQUESTS_PER_JOB = 1_000
MAX_BYTES_PER_JOB = 1_500_000_000
MAX_ESTIMATED_TOKENS_PER_JOB = 4_500_000
ESTIMATED_CHARS_PER_TOKEN = 4
TERMINAL_STATES = {
    "ACTIVE",
    "FAILED",
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def response_schema() -> dict[str, Any]:
    return semantic_schema()


def request_for(prompt: str) -> dict[str, Any]:
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "generation_config": {
            "temperature": 0,
            "thinking_config": {"thinking_level": THINKING_LEVEL},
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json",
            "response_json_schema": response_schema(),
        },
    }


def gemini_provider_config(model: str) -> dict[str, Any]:
    return {
        "provider_config_version": 1,
        "provider": "google-gemini",
        "model": model,
        "temperature": 0,
        "thinking_level": THINKING_LEVEL,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
        "response_wrapper_sha256": sha256_json(request_for("")["generation_config"]),
    }


def split_jsonl_lines(
    lines: list[str],
    max_requests: int = MAX_REQUESTS_PER_JOB,
    max_bytes: int = MAX_BYTES_PER_JOB,
    max_estimated_tokens: int = MAX_ESTIMATED_TOKENS_PER_JOB,
) -> list[list[str]]:
    if max_requests < 1 or max_bytes < 1 or max_estimated_tokens < 1:
        raise ValueError("Gemini job limits must be positive.")
    jobs: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    current_tokens = 0
    for line in lines:
        line_bytes = len(line.encode("utf-8"))
        line_tokens = (line_bytes + ESTIMATED_CHARS_PER_TOKEN - 1) // ESTIMATED_CHARS_PER_TOKEN
        if line_bytes > max_bytes:
            raise ValueError("One Gemini request exceeds the per-job byte limit.")
        if line_tokens > max_estimated_tokens:
            raise ValueError("One Gemini request exceeds the per-job token budget.")
        if current and (
            len(current) >= max_requests
            or current_bytes + line_bytes > max_bytes
            or current_tokens + line_tokens > max_estimated_tokens
        ):
            jobs.append(current)
            current = []
            current_bytes = 0
            current_tokens = 0
        current.append(line)
        current_bytes += line_bytes
        current_tokens += line_tokens
    if current:
        jobs.append(current)
    return jobs


def prepare_batch(
    sample_path: Path,
    evidence_dir: Path,
    catalog_path: Path,
    output_dir: Path,
    model: str,
    limit: int | None = None,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; refusing to overwrite a run.")
    frame = build_input(sample_path, evidence_dir).sort_values(
        ["repo_id", "number"], kind="mergesort"
    )
    full_run_rows = len(frame)
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive.")
        frame = frame.head(limit)
    taxonomy = load_taxonomy(catalog_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider_dir = output_dir / ".provider"
    provider_dir.mkdir()
    catalog_snapshot = output_dir / "optimization_catalog.csv"
    shutil.copyfile(catalog_path, catalog_snapshot)
    common_contract = study_contract(
        sha256_file(sample_path),
        sha256_file(catalog_snapshot),
        sha256_file(evidence_dir / "pull_request_files.parquet"),
        sha256_file(evidence_dir / "collection_status.parquet"),
    )
    study_contract_sha256 = sha256_json(common_contract)
    provider_config = gemini_provider_config(model)
    provider_config_sha256 = sha256_json(provider_config)
    lines = []
    manifest_rows = []
    for row in frame.to_dict("records"):
        key = f"{int(row['repo_id'])}:{int(row['number'])}"
        prompt = prompt_for(row, taxonomy)
        lines.append(json.dumps({"key": key, "request": request_for(prompt)}, ensure_ascii=True) + "\n")
        patch_chars = len(str(row.get("patch") or ""))
        manifest_rows.append(
            {
                "repo_id": int(row["repo_id"]),
                "number": int(row["number"]),
                "key": key,
                "repo_full_name": row["repo_full_name"],
                "html_url": row["html_url"],
                "sample_arm": row["sample_arm"],
                "prompt_version": PROMPT_VERSION,
                "evidence_status": row["evidence_status"],
                "files_observed": int(row["files_observed"]),
                "patches_available": int(row["patches_available"]),
                "prompt_chars": len(prompt),
                "patch_chars_available": patch_chars,
                "patch_chars_used": min(patch_chars, PATCH_CHARACTER_LIMIT),
                "patch_truncated": patch_chars > PATCH_CHARACTER_LIMIT,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_row_sha256": input_row_sha256(row),
                "study_contract_sha256": study_contract_sha256,
                "provider_config_sha256": provider_config_sha256,
            }
        )
    jobs = split_jsonl_lines(lines)
    job_files = []
    for index, job_lines in enumerate(jobs, start=1):
        path = provider_dir / f"requests-{index:04d}.jsonl"
        path.write_text("".join(job_lines), encoding="utf-8")
        job_files.append(
            {
                "index": index,
                "input_file": str(path.relative_to(output_dir)),
                "requests": len(job_lines),
                "input_sha256": sha256_file(path),
                "estimated_tokens": sum(
                    (len(line.encode("utf-8")) + ESTIMATED_CHARS_PER_TOKEN - 1)
                    // ESTIMATED_CHARS_PER_TOKEN
                    for line in job_lines
                ),
                "status": "prepared",
                "attempt": 1,
            }
        )
    manifest_path = output_dir / "batch_manifest.parquet"
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
    metadata = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "requests": len(manifest_rows),
        "full_run_rows": full_run_rows,
        "smoke_run": limit is not None,
        "sample_sha256": sha256_file(sample_path),
        "catalog_file": catalog_snapshot.name,
        "catalog_sha256": sha256_file(catalog_snapshot),
        "evidence_files_sha256": sha256_file(evidence_dir / "pull_request_files.parquet"),
        "evidence_status_sha256": sha256_file(evidence_dir / "collection_status.parquet"),
        "study_contract": common_contract,
        "study_contract_sha256": study_contract_sha256,
        "provider_config": provider_config,
        "provider_config_sha256": provider_config_sha256,
        "manifest_sha256": sha256_file(manifest_path),
    }
    (output_dir / "prepare_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (provider_dir / "state.json").write_text(
        json.dumps({"jobs": job_files}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _load_state(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = output_dir / ".provider" / "state.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, state: dict[str, Any]) -> None:
    atomic_write_json(path, state)


def submit_batch(output_dir: Path) -> dict[str, Any]:
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text(encoding="utf-8"))
    if sha256_file(output_dir / "batch_manifest.parquet") != metadata["manifest_sha256"]:
        raise ValueError("Batch manifest changed after preparation.")
    state_path, state = _load_state(output_dir)
    active_states = {
        "submitted", "JOB_STATE_PENDING", "JOB_STATE_QUEUED", "JOB_STATE_RUNNING",
    }
    if any(job.get("batch_name") and job.get("status") in active_states for job in state["jobs"]):
        return run_status(output_dir)
    with _client() as client:
        for job in state["jobs"]:
            if job["status"] not in {"prepared", "uploaded"}:
                continue
            input_path = output_dir / job["input_file"]
            if sha256_file(input_path) != job["input_sha256"]:
                raise ValueError("Gemini job input changed after preparation.")
            if not job.get("input_file_name"):
                uploaded = client.files.upload(
                    file=input_path,
                    config=types.UploadFileConfig(
                        display_name=f"rq1-gemini-{job['index']:04d}", mime_type="application/jsonl"
                    ),
                )
                job.update(input_file_name=uploaded.name, status="uploaded")
                _write_state(state_path, state)
            batch = client.batches.create(
                model=metadata["model"],
                src=job["input_file_name"],
                config={"display_name": f"rq1-gemini-{job['index']:04d}"},
            )
            job.update(
                {
                    "batch_name": batch.name,
                    "status": batch.state.value if batch.state else "submitted",
                }
            )
            _write_state(state_path, state)
            break
    return run_status(output_dir)


def run_status(output_dir: Path) -> dict[str, Any]:
    state_path, state = _load_state(output_dir)
    with _client() as client:
        for job in state["jobs"]:
            if not job.get("batch_name") or job["status"] in TERMINAL_STATES:
                continue
            batch = client.batches.get(name=job["batch_name"])
            job.update(
                {
                    "status": batch.state.value if batch.state else None,
                    "output_file_name": batch.dest.file_name if batch.dest else None,
                    "error": batch.error.model_dump(mode="json") if batch.error else None,
                }
            )
    _write_state(state_path, state)
    statuses = [job["status"] for job in state["jobs"]]
    if statuses and all(status in TERMINAL_STATES for status in statuses):
        aggregate = "completed"
    elif any(status != "prepared" for status in statuses):
        aggregate = "running"
    else:
        aggregate = "prepared"
    return {"status": aggregate, "requests": sum(job["requests"] for job in state["jobs"])}


def _response_text(response: dict[str, Any]) -> str:
    return "".join(
        str(part.get("text") or "")
        for candidate in response.get("candidates", [])[:1]
        for part in (candidate.get("content") or {}).get("parts", [])
    )


def _parse_outputs(
    output_dir: Path, outputs: list[str], expected_keys: set[str] | None = None
) -> pd.DataFrame:
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text(encoding="utf-8"))
    manifest = pd.read_parquet(output_dir / "batch_manifest.parquet")
    if manifest["key"].isna().any() or not manifest["key"].is_unique:
        raise ValueError("Gemini manifest contains invalid or duplicate keys.")
    by_key = manifest.set_index("key").to_dict("index")
    expected = set(by_key) if expected_keys is None else expected_keys
    if not expected.issubset(by_key):
        raise ValueError("Gemini attempt contains keys outside the run manifest.")
    taxonomy = taxonomy_labels(load_taxonomy(output_dir / metadata["catalog_file"]))
    rows = []
    seen = set()
    for output in outputs:
        for line in output.splitlines():
            item = json.loads(line)
            key = item.get("key")
            if key in seen or key not in expected:
                raise ValueError(f"Invalid or duplicate Gemini key: {key}")
            seen.add(key)
            row = {**by_key[key], "key": key, "model": metadata["model"]}
            try:
                if item.get("error") or not item.get("response"):
                    raise ValueError(json.dumps(item.get("error") or "Missing response"))
                response = item["response"]
                finish_reason = ((response.get("candidates") or [{}])[0].get("finishReason"))
                if finish_reason and finish_reason != "STOP":
                    raise ValueError(f"Gemini response did not finish normally: {finish_reason}")
                label = PatternLabel.model_validate_json(_response_text(response))
                if label.high_level_pattern not in taxonomy or label.sub_pattern not in taxonomy[label.high_level_pattern]:
                    raise ValueError("Response label is not in the catalog.")
                usage = response.get("usageMetadata") or {}
                row.update(label.model_dump())
                row.update(
                    {
                        "classification_status": "classified",
                        "model": response.get("modelVersion") or metadata["model"],
                        "response_id": response.get("responseId"),
                        "prompt_tokens": usage.get("promptTokenCount"),
                        "response_tokens": usage.get("candidatesTokenCount") or usage.get("responseTokenCount"),
                        "thoughts_tokens": usage.get("thoughtsTokenCount"),
                        "total_tokens": usage.get("totalTokenCount"),
                    }
                )
            except ValueError as error:
                row.update({"classification_status": "error", "error": str(error)[:500]})
            rows.append(row)
    for key in sorted(expected - seen):
        rows.append({**by_key[key], "key": key, "model": metadata["model"], "classification_status": "error", "error": "Missing from batch output"})
    return pd.DataFrame(rows).sort_values(["repo_id", "number"]).reset_index(drop=True)


def collect_batch(output_dir: Path) -> pd.DataFrame:
    status = run_status(output_dir)
    if status["status"] != "completed":
        raise RuntimeError(f"Gemini run is not ready for collection: {status['status']}")
    state_path, state = _load_state(output_dir)
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text(encoding="utf-8"))
    if sha256_file(output_dir / metadata["catalog_file"]) != metadata["catalog_sha256"]:
        raise ValueError("Catalog snapshot changed after preparation.")
    if sha256_file(output_dir / "batch_manifest.parquet") != metadata["manifest_sha256"]:
        raise ValueError("Batch manifest changed after preparation.")
    outputs = []
    expected_keys = set()
    active_attempt = max(job["attempt"] for job in state["jobs"])
    with _client() as client:
        for job in state["jobs"]:
            if job["attempt"] != active_attempt:
                continue
            input_lines = (output_dir / job["input_file"]).read_text(encoding="utf-8").splitlines()
            job_keys = {json.loads(line)["key"] for line in input_lines}
            expected_keys.update(job_keys)
            if job["status"] not in {"ACTIVE", "JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"} or not job.get("output_file_name"):
                continue
            output = client.files.download(file=job["output_file_name"]).decode("utf-8")
            path = output_dir / ".provider" / f"output-{job['index']:04d}-attempt-{job['attempt']:02d}.jsonl"
            path.write_text(output, encoding="utf-8")
            outputs.append(output)
    frame = _parse_outputs(output_dir, outputs, expected_keys)
    prior_path = output_dir / "optimization_pattern_labels.parquet"
    if prior_path.exists():
        prior = pd.read_parquet(prior_path)
        untouched = prior[~prior["key"].isin(expected_keys)]
        attempted_prior = prior[prior["key"].isin(expected_keys)]
        prior_successes = attempted_prior[
            attempted_prior["classification_status"].eq("classified")
        ]
        frame = pd.concat(
            [untouched, frame[~frame["key"].isin(prior_successes["key"])], prior_successes],
            ignore_index=True,
        )
        frame = frame.sort_values(["repo_id", "number"]).reset_index(drop=True)
    manifest = pd.read_parquet(output_dir / "batch_manifest.parquet")
    if set(frame["key"]) != set(manifest["key"]) or len(frame) != len(manifest):
        raise ValueError("Gemini collection does not preserve the run manifest population.")
    frame.to_parquet(prior_path, index=False)
    summary = {"rows": len(frame), "status_counts": frame["classification_status"].value_counts().to_dict()}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frame


def prepare_retry(output_dir: Path) -> dict[str, Any]:
    labels = pd.read_parquet(output_dir / "optimization_pattern_labels.parquet")
    retry_keys = set(labels.loc[labels["classification_status"].eq("error"), "key"])
    if not retry_keys:
        raise ValueError("Run has no errors to retry.")
    requests = {}
    for path in sorted((output_dir / ".provider").glob("requests-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            requests[item["key"]] = json.dumps(item, ensure_ascii=True) + "\n"
    state_path, state = _load_state(output_dir)
    next_index = max(job["index"] for job in state["jobs"]) + 1
    attempt = max(job["attempt"] for job in state["jobs"]) + 1
    for lines in split_jsonl_lines([requests[key] for key in sorted(retry_keys)]):
        path = output_dir / ".provider" / f"requests-{next_index:04d}.jsonl"
        path.write_text("".join(lines), encoding="utf-8")
        state["jobs"].append({"index": next_index, "input_file": str(path.relative_to(output_dir)), "requests": len(lines), "input_sha256": sha256_file(path), "status": "prepared", "attempt": attempt})
        next_index += 1
    _write_state(state_path, state)
    return submit_batch(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--sample", type=Path, required=True)
    prepare.add_argument("--evidence-dir", type=Path, required=True)
    prepare.add_argument("--catalog", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--model", default=DEFAULT_MODEL)
    prepare.add_argument("--limit", type=int)
    for action in ("submit", "status", "collect", "retry"):
        command = commands.add_parser(action)
        command.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        print(prepare_batch(args.sample, args.evidence_dir, args.catalog, args.output_dir, args.model, args.limit))
    elif args.action == "submit":
        print(json.dumps(submit_batch(args.output_dir), sort_keys=True))
    elif args.action == "status":
        print(json.dumps(run_status(args.output_dir), sort_keys=True))
    elif args.action == "collect":
        print(f"Collected {len(collect_batch(args.output_dir))} labels")
    else:
        print(json.dumps(prepare_retry(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
