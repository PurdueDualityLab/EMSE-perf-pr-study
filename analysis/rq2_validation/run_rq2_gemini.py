"""Prepare, submit, inspect, collect, and retry one logical RQ2 Gemini run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from google import genai
from google.genai import types

from run_rq2 import (
    MAX_OUTPUT_TOKENS,
    PROMPT_VERSION,
    SYSTEM_INSTRUCTION,
    ValidationLabel,
    build_input,
    input_row_sha256,
    prompt_for,
    semantic_schema,
    sha256_file,
    sha256_json,
    study_contract,
    atomic_write_json,
)

DEFAULT_MODEL = "gemini-3.1-pro-preview"
THINKING_LEVEL = "MEDIUM"
MAX_REQUESTS_PER_JOB = 1_000
MAX_BYTES_PER_JOB = 1_500_000_000
MAX_ESTIMATED_TOKENS_PER_JOB = 4_500_000
ESTIMATED_CHARS_PER_TOKEN = 4
EVIDENCE_FILES = (
    "pull_requests.parquet", "pull_request_files.parquet", "issue_comments.parquet",
    "review_comments.parquet", "reviews.parquet", "workflow_runs.parquet",
    "check_runs.parquet", "collection_status.parquet",
)
TERMINAL_STATES = {
    "ACTIVE", "FAILED",
    "JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED", "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
}


def request_for(prompt: str) -> dict[str, Any]:
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "generation_config": {
            "temperature": 0,
            "thinking_config": {"thinking_level": THINKING_LEVEL},
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json",
            "response_json_schema": semantic_schema(),
        },
    }


def gemini_provider_config(model: str) -> dict[str, Any]:
    return {
        "provider_config_version": 1, "provider": "google-gemini", "model": model,
        "temperature": 0, "thinking_level": THINKING_LEVEL,
        "max_output_tokens": MAX_OUTPUT_TOKENS, "response_mime_type": "application/json",
        "response_wrapper_sha256": sha256_json(request_for("")["generation_config"]),
    }


def split_jsonl_lines(lines: list[str], max_requests: int = MAX_REQUESTS_PER_JOB,
                      max_bytes: int = MAX_BYTES_PER_JOB,
                      max_estimated_tokens: int = MAX_ESTIMATED_TOKENS_PER_JOB) -> list[list[str]]:
    if max_requests < 1 or max_bytes < 1 or max_estimated_tokens < 1:
        raise ValueError("Gemini job limits must be positive.")
    jobs, current, current_bytes, current_tokens = [], [], 0, 0
    for line in lines:
        line_bytes = len(line.encode())
        line_tokens = (line_bytes + ESTIMATED_CHARS_PER_TOKEN - 1) // ESTIMATED_CHARS_PER_TOKEN
        if line_bytes > max_bytes:
            raise ValueError("One Gemini request exceeds the per-job byte limit.")
        if line_tokens > max_estimated_tokens:
            raise ValueError("One Gemini request exceeds the per-job token budget.")
        if current and (len(current) >= max_requests or current_bytes + line_bytes > max_bytes
                        or current_tokens + line_tokens > max_estimated_tokens):
            jobs.append(current)
            current, current_bytes, current_tokens = [], 0, 0
        current.append(line)
        current_bytes += line_bytes
        current_tokens += line_tokens
    if current:
        jobs.append(current)
    return jobs


def prepare_batch(sample_path: Path, evidence_dir: Path, output_dir: Path, model: str,
                  limit: int | None = None) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; refusing to overwrite a run.")
    frame = build_input(sample_path, evidence_dir).sort_values(["repo_id", "number"], kind="mergesort")
    incomplete = frame.loc[~frame["evidence_complete"], ["repo_id", "number"]]
    if not incomplete.empty:
        raise ValueError(
            "RQ2 cannot prepare until observable evidence is complete for: "
            + json.dumps(incomplete.to_dict("records"), sort_keys=True)
        )
    full_run_rows = len(frame)
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive.")
        frame = frame.head(limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider_dir = output_dir / ".provider"
    provider_dir.mkdir()
    evidence_hashes = {name: sha256_file(evidence_dir / name) for name in EVIDENCE_FILES}
    contract = study_contract(sha256_file(sample_path), evidence_hashes)
    contract_hash = sha256_json(contract)
    provider = gemini_provider_config(model)
    provider_hash = sha256_json(provider)
    lines, manifest_rows = [], []
    for row in frame.to_dict("records"):
        key = f"{int(row['repo_id'])}:{int(row['number'])}"
        prompt = prompt_for(row)
        lines.append(json.dumps({"key": key, "request": request_for(prompt)}, ensure_ascii=True) + "\n")
        manifest_rows.append(
            {
                "repo_id": int(row["repo_id"]), "number": int(row["number"]), "key": key,
                "repo_full_name": row["repo_full_name"], "html_url": row["html_url"],
                "sample_arm": row["sample_arm"], "prompt_version": PROMPT_VERSION,
                "evidence_status": row["evidence_status"], "prompt_chars": len(prompt),
                "evidence_complete": bool(row["evidence_complete"]),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "input_row_sha256": input_row_sha256(row),
                "study_contract_sha256": contract_hash, "provider_config_sha256": provider_hash,
            }
        )
    jobs = []
    for index, job_lines in enumerate(split_jsonl_lines(lines), start=1):
        path = provider_dir / f"requests-{index:04d}.jsonl"
        path.write_text("".join(job_lines))
        jobs.append({"index": index, "input_file": str(path.relative_to(output_dir)),
                     "requests": len(job_lines), "input_sha256": sha256_file(path),
                     "estimated_tokens": sum((len(line.encode()) + ESTIMATED_CHARS_PER_TOKEN - 1)
                                             // ESTIMATED_CHARS_PER_TOKEN for line in job_lines),
                     "status": "prepared", "attempt": 1})
    manifest_path = output_dir / "batch_manifest.parquet"
    pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
    metadata = {
        "model": model, "prompt_version": PROMPT_VERSION, "requests": len(manifest_rows),
        "full_run_rows": full_run_rows, "smoke_run": limit is not None,
        "sample_sha256": sha256_file(sample_path), "evidence_sha256": evidence_hashes,
        "study_contract": contract, "study_contract_sha256": contract_hash,
        "provider_config": provider, "provider_config_sha256": provider_hash,
        "manifest_sha256": sha256_file(manifest_path),
    }
    (output_dir / "prepare_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (provider_dir / "state.json").write_text(json.dumps({"jobs": jobs}, indent=2, sort_keys=True) + "\n")
    return output_dir


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _load_state(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = output_dir / ".provider" / "state.json"
    return path, json.loads(path.read_text())


def _write_state(path: Path, state: dict[str, Any]) -> None:
    atomic_write_json(path, state)


def submit_batch(output_dir: Path) -> dict[str, Any]:
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text())
    if sha256_file(output_dir / "batch_manifest.parquet") != metadata["manifest_sha256"]:
        raise ValueError("Batch manifest changed after preparation.")
    state_path, state = _load_state(output_dir)
    active_states = {"submitted", "JOB_STATE_PENDING", "JOB_STATE_QUEUED", "JOB_STATE_RUNNING"}
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
                uploaded = client.files.upload(file=input_path, config=types.UploadFileConfig(
                    display_name=f"rq2-gemini-{job['index']:04d}", mime_type="application/jsonl"))
                job.update(input_file_name=uploaded.name, status="uploaded")
                _write_state(state_path, state)
            batch = client.batches.create(model=metadata["model"], src=job["input_file_name"],
                                          config={"display_name": f"rq2-gemini-{job['index']:04d}"})
            job.update(batch_name=batch.name, status=batch.state.value if batch.state else "submitted")
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
            job.update(status=batch.state.value if batch.state else None,
                       output_file_name=batch.dest.file_name if batch.dest else None,
                       error=batch.error.model_dump(mode="json") if batch.error else None)
    _write_state(state_path, state)
    statuses = [job["status"] for job in state["jobs"]]
    aggregate = "completed" if statuses and all(s in TERMINAL_STATES for s in statuses) else (
        "running" if any(s != "prepared" for s in statuses) else "prepared")
    return {"status": aggregate, "requests": sum(job["requests"] for job in state["jobs"])}


def _response_text(response: dict[str, Any]) -> str:
    return "".join(str(part.get("text") or "") for candidate in response.get("candidates", [])[:1]
                   for part in (candidate.get("content") or {}).get("parts", []))


def _parse_outputs(output_dir: Path, outputs: list[str],
                   expected_keys: set[str] | None = None) -> pd.DataFrame:
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text())
    manifest = pd.read_parquet(output_dir / "batch_manifest.parquet")
    if manifest["key"].isna().any() or not manifest["key"].is_unique:
        raise ValueError("Gemini manifest contains invalid or duplicate keys.")
    by_key = manifest.set_index("key").to_dict("index")
    expected = set(by_key) if expected_keys is None else expected_keys
    if not expected.issubset(by_key):
        raise ValueError("Gemini attempt contains keys outside the run manifest.")
    rows, seen = [], set()
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
                label = ValidationLabel.model_validate_json(_response_text(response))
                usage = response.get("usageMetadata") or {}
                row.update(label.model_dump())
                row.update(
                    classification_status="classified", model=response.get("modelVersion") or metadata["model"],
                    response_id=response.get("responseId"), prompt_tokens=usage.get("promptTokenCount"),
                    response_tokens=usage.get("candidatesTokenCount") or usage.get("responseTokenCount"),
                    thoughts_tokens=usage.get("thoughtsTokenCount"), total_tokens=usage.get("totalTokenCount"),
                )
            except ValueError as error:
                row.update(classification_status="error", error=str(error)[:500])
            rows.append(row)
    for key in sorted(expected - seen):
        rows.append({**by_key[key], "key": key, "model": metadata["model"],
                     "classification_status": "error", "error": "Missing from batch output"})
    return pd.DataFrame(rows).sort_values(["repo_id", "number"]).reset_index(drop=True)


def collect_batch(output_dir: Path) -> pd.DataFrame:
    status = run_status(output_dir)
    if status["status"] != "completed":
        raise RuntimeError(f"Gemini run is not ready for collection: {status['status']}")
    _, state = _load_state(output_dir)
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text())
    if sha256_file(output_dir / "batch_manifest.parquet") != metadata["manifest_sha256"]:
        raise ValueError("Batch manifest changed after preparation.")
    outputs, expected_keys = [], set()
    active_attempt = max(job["attempt"] for job in state["jobs"])
    with _client() as client:
        for job in state["jobs"]:
            if job["attempt"] != active_attempt:
                continue
            input_lines = (output_dir / job["input_file"]).read_text().splitlines()
            expected_keys.update(json.loads(line)["key"] for line in input_lines)
            if job["status"] not in {"ACTIVE", "JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"} or not job.get("output_file_name"):
                continue
            output = client.files.download(file=job["output_file_name"]).decode()
            (output_dir / ".provider" / f"output-{job['index']:04d}-attempt-{job['attempt']:02d}.jsonl").write_text(output)
            outputs.append(output)
    frame = _parse_outputs(output_dir, outputs, expected_keys)
    result_path = output_dir / "validation_labels.parquet"
    if result_path.exists():
        prior = pd.read_parquet(result_path)
        prior_successes = prior[prior["key"].isin(expected_keys) & prior["classification_status"].eq("classified")]
        frame = pd.concat([prior[~prior["key"].isin(expected_keys)],
                           frame[~frame["key"].isin(prior_successes["key"])], prior_successes], ignore_index=True)
        frame = frame.sort_values(["repo_id", "number"]).reset_index(drop=True)
    manifest = pd.read_parquet(output_dir / "batch_manifest.parquet")
    if set(frame["key"]) != set(manifest["key"]) or len(frame) != len(manifest):
        raise ValueError("Gemini collection does not preserve the run manifest population.")
    frame.to_parquet(result_path, index=False)
    summary = {"rows": len(frame), "status_counts": frame["classification_status"].value_counts().to_dict()}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return frame


def prepare_retry(output_dir: Path) -> dict[str, Any]:
    labels = pd.read_parquet(output_dir / "validation_labels.parquet")
    retry_keys = set(labels.loc[labels["classification_status"].eq("error"), "key"])
    if not retry_keys:
        raise ValueError("Run has no errors to retry.")
    requests = {}
    for path in sorted((output_dir / ".provider").glob("requests-*.jsonl")):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            requests[item["key"]] = json.dumps(item, ensure_ascii=True) + "\n"
    state_path, state = _load_state(output_dir)
    next_index = max(job["index"] for job in state["jobs"]) + 1
    attempt = max(job["attempt"] for job in state["jobs"]) + 1
    for lines in split_jsonl_lines([requests[key] for key in sorted(retry_keys)]):
        path = output_dir / ".provider" / f"requests-{next_index:04d}.jsonl"
        path.write_text("".join(lines))
        state["jobs"].append({"index": next_index, "input_file": str(path.relative_to(output_dir)),
                              "requests": len(lines), "input_sha256": sha256_file(path),
                              "status": "prepared", "attempt": attempt})
        next_index += 1
    _write_state(state_path, state)
    return submit_batch(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--sample", type=Path, required=True)
    prepare.add_argument("--evidence-dir", type=Path, required=True)
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
        print(prepare_batch(args.sample, args.evidence_dir, args.output_dir, args.model, args.limit))
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
