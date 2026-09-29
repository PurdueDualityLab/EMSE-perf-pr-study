"""Prepare, submit, inspect, collect, and retry one logical RQ3 Gemini run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from google import genai
from google.genai import types

from analysis.rq3_llm_validation import experiment

DEFAULT_MODEL = "gemini-3.1-pro-preview"
THINKING_LEVEL = "MEDIUM"
MAX_OUTPUT_TOKENS = 4_096
MAX_REQUESTS_PER_JOB = 1_000
MAX_BYTES_PER_JOB = 1_500_000_000
MAX_ESTIMATED_TOKENS_PER_JOB = 4_500_000
TERMINAL = {"ACTIVE", "FAILED", "JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED",
            "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


def provider_config(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    return {"provider_config_version": 1, "provider": "google-gemini", "model": model,
            "temperature": 0, "thinking_level": THINKING_LEVEL,
            "max_output_tokens": MAX_OUTPUT_TOKENS}


def request_for(prompt: str, task: str) -> dict[str, Any]:
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "system_instruction": {"parts": [{"text": experiment.SYSTEM_INSTRUCTION}]},
        "generation_config": {
            "temperature": 0, "thinking_config": {"thinking_level": THINKING_LEVEL},
            "max_output_tokens": MAX_OUTPUT_TOKENS, "response_mime_type": "application/json",
            "response_json_schema": experiment.semantic_schema(task),
        },
    }


def split_lines(lines: list[str], max_requests: int = MAX_REQUESTS_PER_JOB,
                max_bytes: int = MAX_BYTES_PER_JOB,
                max_tokens: int = MAX_ESTIMATED_TOKENS_PER_JOB) -> list[list[str]]:
    if min(max_requests, max_bytes, max_tokens) < 1:
        raise ValueError("Gemini job limits must be positive.")
    jobs, current, size, tokens = [], [], 0, 0
    for line in lines:
        line_size = len(line.encode())
        line_tokens = (line_size + 3) // 4
        if line_size > max_bytes or line_tokens > max_tokens:
            raise ValueError("One Gemini request exceeds a per-job limit.")
        if current and (len(current) >= max_requests or size + line_size > max_bytes
                        or tokens + line_tokens > max_tokens):
            jobs.append(current)
            current, size, tokens = [], 0, 0
        current.append(line)
        size += line_size
        tokens += line_tokens
    if current:
        jobs.append(current)
    return jobs


def prepare(task: str, sample: Path, matches: Path, evidence_dir: Path,
            output_dir: Path, model: str = DEFAULT_MODEL) -> Path:
    frame, manifest, metadata = experiment.prepare_run(
        task, sample, matches, evidence_dir, output_dir, provider_config(model)
    )
    provider_dir = output_dir / ".provider"
    provider_dir.mkdir()
    rows = frame.assign(key=lambda x: x.repo_id.astype(str) + ":" + x.number.astype(str)).set_index("key")
    lines = [json.dumps({"key": key, "request": request_for(
        experiment.prompt_for(rows.loc[key].to_dict(), task), task)}, ensure_ascii=True) + "\n"
        for key in manifest.custom_id]
    jobs = []
    for index, job_lines in enumerate(split_lines(lines), 1):
        path = provider_dir / f"requests-{index:04d}.jsonl"
        path.write_text("".join(job_lines), encoding="utf-8")
        jobs.append({"index": index, "input_file": str(path.relative_to(output_dir)),
                     "input_sha256": experiment.sha256_file(path), "requests": len(job_lines),
                     "attempt": 1, "status": "prepared"})
    experiment.atomic_write_json(provider_dir / "state.json", {"jobs": jobs})
    return output_dir


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _state(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = output_dir / ".provider/state.json"
    return path, json.loads(path.read_text())


def submit(output_dir: Path) -> dict[str, Any]:
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text())
    if experiment.sha256_file(output_dir / "batch_manifest.parquet") != metadata["manifest_sha256"]:
        raise ValueError("Batch manifest changed after preparation.")
    state_path, state = _state(output_dir)
    with _client() as client:
        for job in state["jobs"]:
            if job["status"] not in {"prepared", "uploaded"}:
                continue
            path = output_dir / job["input_file"]
            if experiment.sha256_file(path) != job["input_sha256"]:
                raise ValueError("Gemini input changed after preparation.")
            if not job.get("input_file_name"):
                uploaded = client.files.upload(file=path, config=types.UploadFileConfig(
                    display_name=f"rq3-{metadata['task']}-{job['index']:04d}", mime_type="application/jsonl"))
                job.update(input_file_name=uploaded.name, status="uploaded")
                experiment.atomic_write_json(state_path, state)
            batch = client.batches.create(model=metadata["provider_config"]["model"],
                                          src=job["input_file_name"],
                                          config={"display_name": f"rq3-{metadata['task']}-{job['index']:04d}"})
            job.update(batch_name=batch.name, status=batch.state.value if batch.state else "submitted")
            experiment.atomic_write_json(state_path, state)
            break
    return status(output_dir)


def status(output_dir: Path) -> dict[str, Any]:
    state_path, state = _state(output_dir)
    with _client() as client:
        for job in state["jobs"]:
            if not job.get("batch_name") or job["status"] in TERMINAL:
                continue
            batch = client.batches.get(name=job["batch_name"])
            job.update(status=batch.state.value if batch.state else None,
                       output_file_name=batch.dest.file_name if batch.dest else None,
                       error=batch.error.model_dump(mode="json") if batch.error else None)
    experiment.atomic_write_json(state_path, state)
    statuses = [job["status"] for job in state["jobs"]]
    aggregate = "completed" if statuses and all(value in TERMINAL for value in statuses) else "running"
    return {"status": aggregate, "requests": sum(job["requests"] for job in state["jobs"])}


def _response_text(response: dict[str, Any]) -> str:
    return "".join(str(part.get("text") or "") for candidate in response.get("candidates", [])[:1]
                   for part in (candidate.get("content") or {}).get("parts", []))


def parse_outputs(outputs: list[str], manifest: pd.DataFrame, task: str, model: str,
                  expected: set[str] | None = None) -> pd.DataFrame:
    by_id = manifest.set_index("custom_id").to_dict("index")
    expected = set(by_id) if expected is None else expected
    rows, seen = [], set()
    for output in outputs:
        for line in output.splitlines():
            item = json.loads(line)
            key = item.get("key")
            if key in seen or key not in expected:
                raise ValueError(f"Invalid or duplicate Gemini key: {key}")
            seen.add(key)
            row = {**by_id[key], "custom_id": key, "model": model}
            try:
                if item.get("error") or not item.get("response"):
                    raise ValueError(json.dumps(item.get("error") or "Missing response"))
                response = item["response"]
                finish = ((response.get("candidates") or [{}])[0].get("finishReason"))
                if finish and finish != "STOP":
                    raise ValueError(f"Gemini response did not finish normally: {finish}")
                label = experiment.label_model(task).model_validate_json(_response_text(response))
            except ValueError as error:
                row.update(classification_status="error", error=str(error)[:500])
            else:
                row.update(label.model_dump(), classification_status="classified")
            rows.append(row)
    for key in sorted(expected - seen):
        rows.append({**by_id[key], "custom_id": key, "model": model,
                     "classification_status": "error", "error": "Missing from batch output"})
    return pd.DataFrame(rows).sort_values(experiment.KEYS).reset_index(drop=True)


def collect(output_dir: Path) -> pd.DataFrame:
    if status(output_dir)["status"] != "completed":
        raise RuntimeError("Gemini run is not ready for collection.")
    _, state = _state(output_dir)
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text())
    if experiment.sha256_file(output_dir / "batch_manifest.parquet") != metadata["manifest_sha256"]:
        raise ValueError("Batch manifest changed after preparation.")
    manifest = pd.read_parquet(output_dir / "batch_manifest.parquet")
    attempt = max(job["attempt"] for job in state["jobs"])
    outputs, expected = [], set()
    with _client() as client:
        for job in state["jobs"]:
            if job["attempt"] != attempt:
                continue
            lines = (output_dir / job["input_file"]).read_text().splitlines()
            expected.update(json.loads(line)["key"] for line in lines)
            if job.get("output_file_name"):
                outputs.append(client.files.download(file=job["output_file_name"]).decode())
    result = parse_outputs(outputs, manifest, metadata["task"],
                           metadata["provider_config"]["model"], expected)
    path = output_dir / "labels.parquet"
    if path.exists():
        prior = pd.read_parquet(path)
        result = pd.concat([prior[~prior.custom_id.isin(expected)], result], ignore_index=True)
    result = result.sort_values(experiment.KEYS).reset_index(drop=True)
    if set(result.custom_id) != set(manifest.custom_id) or len(result) != len(manifest):
        raise ValueError("Gemini collection does not preserve the manifest population.")
    result.to_parquet(path, index=False)
    experiment.atomic_write_json(output_dir / "summary.json", {
        "rows": len(result), "status_counts": result.classification_status.value_counts().to_dict()
    })
    return result


def retry(output_dir: Path) -> dict[str, Any]:
    labels = pd.read_parquet(output_dir / "labels.parquet")
    retry_ids = set(labels.loc[labels.classification_status.eq("error"), "custom_id"])
    if not retry_ids:
        raise ValueError("Run has no errors to retry.")
    requests = {}
    for path in sorted((output_dir / ".provider").glob("requests-*.jsonl")):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            requests[item["key"]] = json.dumps(item, ensure_ascii=True) + "\n"
    state_path, state = _state(output_dir)
    attempt = max(job["attempt"] for job in state["jobs"]) + 1
    index = max(job["index"] for job in state["jobs"]) + 1
    for lines in split_lines([requests[key] for key in sorted(retry_ids)]):
        path = output_dir / ".provider" / f"requests-{index:04d}.jsonl"
        path.write_text("".join(lines))
        state["jobs"].append({"index": index, "input_file": str(path.relative_to(output_dir)),
                              "input_sha256": experiment.sha256_file(path), "requests": len(lines),
                              "attempt": attempt, "status": "prepared"})
        index += 1
    experiment.atomic_write_json(state_path, state)
    return submit(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--task", choices=experiment.TASKS, required=True)
    command.add_argument("--sample", type=Path, required=True)
    command.add_argument("--matches", type=Path, default=experiment.DEFAULT_MATCHES)
    command.add_argument("--evidence-dir", type=Path, default=experiment.DEFAULT_EVIDENCE)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--model", default=DEFAULT_MODEL)
    for action in ("submit", "status", "collect", "retry"):
        command = commands.add_parser(action)
        command.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        print(prepare(args.task, args.sample, args.matches, args.evidence_dir, args.output_dir, args.model))
    elif args.action == "submit":
        print(json.dumps(submit(args.output_dir), sort_keys=True))
    elif args.action == "status":
        print(json.dumps(status(args.output_dir), sort_keys=True))
    elif args.action == "collect":
        print(f"Collected {len(collect(args.output_dir))} labels")
    else:
        print(json.dumps(retry(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
