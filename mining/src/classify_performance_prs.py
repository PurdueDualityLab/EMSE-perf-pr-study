"""Classify pull-request task types with the AIDev-compatible Luna Batch cascade."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

import aidev_task_classifier as aidev
from schema import atomic_write_text, write_parquet


DEFAULT_BATCH_REQUESTS = 10_000
DEFAULT_BATCH_BYTES = 150 * 1024 * 1024
OPENAI_BATCH_REQUEST_LIMIT = 50_000
OPENAI_BATCH_BYTE_LIMIT = 200 * 1024 * 1024
TERMINAL_STATUSES = {"completed", "expired", "cancelled", "failed"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def batch_endpoint(endpoint: str) -> str:
    marker = "/v1/"
    if marker not in endpoint:
        raise ValueError("Chat Completions endpoint must include /v1/.")
    return endpoint.split(marker, 1)[0] + "/v1/batches"


def api_root(endpoint: str) -> str:
    marker = "/v1/"
    if marker not in endpoint:
        raise ValueError("Chat Completions endpoint must include /v1/.")
    return endpoint.split(marker, 1)[0] + "/v1"


def request_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(aidev.canonical_json(payload["messages"]).encode("utf-8")).hexdigest()


def automatic_record(task_type: str) -> dict[str, object]:
    return {
        "aidev_task_type": task_type,
        "aidev_task_type_reason": "title provides conventional commit label",
        "aidev_task_type_confidence": 10,
        "aidev_task_type_method": "title_regex",
        "aidev_task_type_status": "classified",
        "aidev_task_type_response_id": "",
        "aidev_task_type_input_tokens": None,
        "aidev_task_type_output_tokens": None,
        "aidev_task_type_request_sha256": "",
        "aidev_task_type_error": "",
    }


def state_signature(
    input_path: Path,
    model: str,
    endpoint: str,
    body_max_tokens: int,
    sample_size: int | None = None,
    sample_seed: int = 20260804,
    exclude_decisions: Path | None = None,
) -> dict[str, object]:
    value = aidev.signature(input_path, model, endpoint, body_max_tokens).copy()
    value["batch_runner"] = "openai_batch_v1"
    value["sample_size"] = sample_size
    value["sample_seed"] = sample_seed if sample_size is not None else None
    value["exclude_decisions_path"] = str(exclude_decisions.resolve()) if exclude_decisions else None
    value["exclude_decisions_sha256"] = aidev.sha256_file(exclude_decisions) if exclude_decisions else None
    return value


def load_state(path: Path, expected: dict[str, object], resume: bool) -> dict[str, object]:
    if not path.exists():
        return {"signature": expected, "batches": [], "completed": {}, "created_at": now()}
    if not resume:
        raise FileExistsError(f"State already exists at {path}; use --resume.")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("signature") != expected or not isinstance(state.get("batches"), list) or not isinstance(state.get("completed"), dict):
        raise ValueError("Cannot resume: incompatible or malformed checkpoint.")
    return state


def save_state(path: Path, state: dict[str, object]) -> None:
    state["updated_at"] = now()
    atomic_write_text(path, json.dumps(state, indent=2, sort_keys=True) + "\n")


def input_rows(input_path: Path) -> list[dict[str, object]]:
    frame = pd.read_parquet(input_path)
    if not set(aidev.IDENTITY_COLUMNS).issubset(frame.columns) or frame.duplicated(list(aidev.IDENTITY_COLUMNS)).any():
        raise ValueError("Input must contain unique immutable (repo_id, number) identities.")
    return frame.to_dict(orient="records")


def excluded_identities(path: Path | None) -> set[str]:
    if path is None:
        return set()
    frame = pd.read_parquet(path)
    required = {*aidev.IDENTITY_COLUMNS, "aidev_task_type_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Exclusion parquet is missing columns: {sorted(missing)}")
    if frame.duplicated(list(aidev.IDENTITY_COLUMNS)).any():
        raise ValueError("Exclusion parquet has duplicate immutable identities.")
    return {
        aidev.identity_key(row)
        for row in frame.loc[frame["aidev_task_type_status"].eq("classified")].to_dict(orient="records")
    }


def selected_rows(
    input_path: Path,
    sample_size: int | None,
    sample_seed: int,
    exclude_decisions: Path | None = None,
) -> list[dict[str, object]]:
    rows = input_rows(input_path)
    if sample_size is None:
        return rows
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    excluded = excluded_identities(exclude_decisions)
    unmatched = [
        row
        for row in rows
        if aidev.title_label(row.get("title")) is None and aidev.identity_key(row) not in excluded
    ]
    if sample_size > len(unmatched):
        raise ValueError("--sample-size exceeds the number of LLM-eligible PRs.")
    return random.Random(sample_seed).sample(unmatched, sample_size)


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines without logging secret values."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def partition_requests(
    rows: list[dict[str, object]], model: str, body_max_tokens: int, max_requests: int, max_bytes: int,
    max_batches: int | None = None,
) -> list[list[tuple[str, dict[str, object], str]]]:
    if not 0 < max_requests <= OPENAI_BATCH_REQUEST_LIMIT:
        raise ValueError(f"Batch requests must be between 1 and {OPENAI_BATCH_REQUEST_LIMIT}.")
    if not 0 < max_bytes <= OPENAI_BATCH_BYTE_LIMIT:
        raise ValueError(f"Batch bytes must be between 1 and {OPENAI_BATCH_BYTE_LIMIT}.")
    batches: list[list[tuple[str, dict[str, object], str]]] = []
    current: list[tuple[str, dict[str, object], str]] = []
    size = 0
    for row in rows:
        custom_id = aidev.identity_key(row)
        payload = aidev.request_payload(model, row, body_max_tokens)
        line = json.dumps({"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": payload}, separators=(",", ":"), ensure_ascii=True)
        line_bytes = len(line.encode("utf-8")) + 1
        if line_bytes > max_bytes:
            raise ValueError(f"Request {custom_id} exceeds the configured batch byte limit.")
        if current and (len(current) == max_requests or size + line_bytes > max_bytes):
            batches.append(current)
            if max_batches is not None and len(batches) >= max_batches:
                return batches
            current, size = [], 0
        current.append((custom_id, payload, line))
        size += line_bytes
    if current:
        batches.append(current)
    return batches


def upload_batch_file(endpoint: str, api_key: str, path: Path) -> str:
    with path.open("rb") as handle:
        response = requests.post(
            api_root(endpoint) + "/files",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"purpose": "batch"},
            files={"file": (path.name, handle, "application/jsonl")},
            timeout=120,
        )
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict) or not isinstance(value.get("id"), str):
        raise ValueError("OpenAI file upload returned no id.")
    return value["id"]


def create_batch(endpoint: str, api_key: str, input_file_id: str) -> dict[str, object]:
    response = requests.post(
        batch_endpoint(endpoint), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"input_file_id": input_file_id, "endpoint": "/v1/chat/completions", "completion_window": "24h"}, timeout=120,
    )
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict) or not isinstance(value.get("id"), str):
        raise ValueError("OpenAI batch creation returned no id.")
    return value


def retrieve_batch(endpoint: str, api_key: str, batch_id: str) -> dict[str, object]:
    response = requests.get(f"{batch_endpoint(endpoint)}/{batch_id}", headers={"Authorization": f"Bearer {api_key}"}, timeout=120)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("OpenAI batch retrieval returned a non-object JSON response.")
    return value


def download_batch_file(endpoint: str, api_key: str, file_id: str) -> str:
    last_error: requests.RequestException | None = None
    for attempt in range(5):
        try:
            response = requests.get(
                api_root(endpoint) + f"/files/{file_id}/content",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=120,
            )
            response.raise_for_status()
            return response.text
        except requests.RequestException as error:
            last_error = error
            if attempt + 1 < 5:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def submit(
    input_path: Path, output_dir: Path, model: str, endpoint: str, api_key: str, body_max_tokens: int,
    max_requests: int = DEFAULT_BATCH_REQUESTS, max_bytes: int = DEFAULT_BATCH_BYTES, max_batches: int | None = None, resume: bool = False,
    sample_size: int | None = None, sample_seed: int = 20260804, exclude_decisions: Path | None = None,
) -> dict[str, object]:
    if body_max_tokens <= 0 or (max_batches is not None and max_batches <= 0):
        raise ValueError("Body limit and max batches must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    state = load_state(state_path, state_signature(input_path, model, endpoint, body_max_tokens, sample_size, sample_seed, exclude_decisions), resume)
    rows = selected_rows(input_path, sample_size, sample_seed, exclude_decisions)
    target_llm_requests = sum(aidev.title_label(row.get("title")) is None for row in rows)
    existing_target = state.get("target_llm_requests")
    if existing_target is not None and int(existing_target) != target_llm_requests:
        raise ValueError("Cannot resume: the expected number of LLM requests changed.")
    state["target_llm_requests"] = target_llm_requests
    save_state(state_path, state)
    submitted = {custom_id for batch in state["batches"] if isinstance(batch, dict) for custom_id in batch.get("custom_ids", [])}
    pending = [row for row in rows if aidev.title_label(row.get("title")) is None and aidev.identity_key(row) not in submitted]
    planned = partition_requests(pending, model, body_max_tokens, max_requests, max_bytes, max_batches)
    inputs_dir = output_dir / "batch_inputs"
    for number, requests_ in enumerate(planned, start=len(state["batches"])):
        path = inputs_dir / f"batch-{number:05d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, "".join(item[2] + "\n" for item in requests_))
        batch: dict[str, object] = {"local_input_path": str(path.resolve()), "custom_ids": [item[0] for item in requests_], "request_count": len(requests_), "status": "prepared"}
        state["batches"].append(batch)
        save_state(state_path, state)
    created = 0
    for batch in state["batches"]:
        if not isinstance(batch, dict) or batch.get("batch_id"):
            continue
        path = Path(str(batch["local_input_path"]))
        if not batch.get("input_file_id"):
            batch["input_file_id"] = upload_batch_file(endpoint, api_key, path)
            save_state(state_path, state)
        remote = create_batch(endpoint, api_key, str(batch["input_file_id"]))
        batch.update(batch_id=remote["id"], status=str(remote.get("status") or "validating"))
        created += 1
        save_state(state_path, state)
        # Keep queued prompt tokens bounded by submitting one batch at a time.
        if max_batches == 1:
            break
    return {"submitted_batches": created, "pending_requests": len(pending), "total_batches": len(state["batches"])}


def status(input_path: Path, output_dir: Path, model: str, endpoint: str, api_key: str, body_max_tokens: int, resume: bool, sample_size: int | None = None, sample_seed: int = 20260804, exclude_decisions: Path | None = None) -> dict[str, object]:
    state_path = output_dir / "state.json"
    state = load_state(state_path, state_signature(input_path, model, endpoint, body_max_tokens, sample_size, sample_seed, exclude_decisions), resume)
    for batch in state["batches"]:
        if isinstance(batch, dict) and batch.get("batch_id"):
            remote = retrieve_batch(endpoint, api_key, str(batch["batch_id"]))
            batch.update(status=str(remote.get("status") or "unknown"), output_file_id=remote.get("output_file_id"), error_file_id=remote.get("error_file_id"), usage=remote.get("usage"), request_counts=remote.get("request_counts"), in_progress_at=remote.get("in_progress_at"), expires_at=remote.get("expires_at"))
    save_state(state_path, state)
    request_counts = Counter()
    for batch in state["batches"]:
        if isinstance(batch, dict) and isinstance(batch.get("request_counts"), dict):
            request_counts.update({key: int(value or 0) for key, value in batch["request_counts"].items()})
    return {
        "status_counts": dict(sorted(Counter(str(batch.get("status")) for batch in state["batches"] if isinstance(batch, dict)).items())),
        "request_counts": dict(sorted(request_counts.items())),
    }


def error_record(message: str, request_sha256: str = "") -> dict[str, object]:
    return aidev.llm_record(None, RuntimeError(message), request_sha256)


def collect_file(content: str, known_ids: set[str], completed: dict[str, object], is_error: bool) -> None:
    for line in content.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("custom_id"), str) or value["custom_id"] not in known_ids:
            raise ValueError("Batch result contains an unknown or invalid custom_id.")
        custom_id = value["custom_id"]
        response = value.get("response")
        if not is_error and isinstance(response, dict) and response.get("status_code") == 200 and isinstance(response.get("body"), dict):
            try:
                completed[custom_id] = aidev.llm_record(response["body"], None, "")
            except (ValueError, json.JSONDecodeError) as error:
                completed[custom_id] = error_record(str(error))
        else:
            detail = value.get("error") or (response.get("body") if isinstance(response, dict) else value)
            completed[custom_id] = error_record(json.dumps(detail, sort_keys=True) if isinstance(detail, (dict, list)) else str(detail))


def collect(input_path: Path, output_dir: Path, model: str, endpoint: str, api_key: str, body_max_tokens: int, resume: bool, sample_size: int | None = None, sample_seed: int = 20260804, exclude_decisions: Path | None = None) -> dict[str, object]:
    state_path = output_dir / "state.json"
    state = load_state(state_path, state_signature(input_path, model, endpoint, body_max_tokens, sample_size, sample_seed, exclude_decisions), resume)
    for batch in state["batches"]:
        if not isinstance(batch, dict) or not batch.get("batch_id"):
            continue
        remote = retrieve_batch(endpoint, api_key, str(batch["batch_id"]))
        batch.update(status=str(remote.get("status") or "unknown"), output_file_id=remote.get("output_file_id"), error_file_id=remote.get("error_file_id"), usage=remote.get("usage"), request_counts=remote.get("request_counts"), in_progress_at=remote.get("in_progress_at"), expires_at=remote.get("expires_at"))
        if batch["status"] not in TERMINAL_STATUSES:
            continue
        known = set(batch.get("custom_ids", []))
        for field, is_error in (("output_file_id", False), ("error_file_id", True)):
            file_id = batch.get(field)
            if isinstance(file_id, str) and file_id:
                collect_file(download_batch_file(endpoint, api_key, file_id), known, state["completed"], is_error)
        # Expired and cancelled batches may have neither an output nor an error line.
        # Keep an explicit record so finalization can account for every submitted PR.
        for custom_id in known:
            if custom_id not in state["completed"]:
                state["completed"][custom_id] = error_record(f"OpenAI batch ended with status {batch['status']} without a result.")
    save_state(state_path, state)
    return {"completed_records": len(state["completed"])}


def finalize(input_path: Path, output_dir: Path, model: str, endpoint: str, body_max_tokens: int, resume: bool, sample_size: int | None = None, sample_seed: int = 20260804, exclude_decisions: Path | None = None) -> dict[str, object]:
    state = load_state(output_dir / "state.json", state_signature(input_path, model, endpoint, body_max_tokens, sample_size, sample_seed, exclude_decisions), resume)
    rows = selected_rows(input_path, sample_size, sample_seed, exclude_decisions)
    decisions = []
    automatic = 0
    for row in rows:
        key = aidev.identity_key(row)
        label = aidev.title_label(row.get("title"))
        if label is not None:
            values = automatic_record(label)
            automatic += 1
        else:
            values = state["completed"].get(key)
            if not isinstance(values, dict):
                raise ValueError("No completed or explicit error record is available for an unmatched input PR.")
        decisions.append({**row, **values, "aidev_task_type_model": model, "aidev_task_type_classifier": aidev.METHOD_NAME})
    frame = pd.DataFrame(decisions)
    output_path = output_dir / "task_type_decisions.parquet"
    write_parquet(frame, output_path)
    input_tokens = int(pd.to_numeric(frame["aidev_task_type_input_tokens"], errors="coerce").fillna(0).sum())
    output_tokens = int(pd.to_numeric(frame["aidev_task_type_output_tokens"], errors="coerce").fillna(0).sum())
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "batch_input_cost_usd": input_tokens / 1_000_000 * 0.10,
        "batch_output_cost_usd": output_tokens / 1_000_000 * 0.60,
        "batch_total_cost_usd": input_tokens / 1_000_000 * 0.10 + output_tokens / 1_000_000 * 0.60,
    }
    summary = {**state["signature"], "rows": len(frame), "title_regex_rows": automatic, "llm_needed_rows": len(frame) - automatic, "status_counts": dict(sorted(Counter(frame["aidev_task_type_status"]).items())), "task_type_counts": dict(sorted(Counter(frame["aidev_task_type"]).items())), "usage": usage, "output_sha256": aidev.sha256_file(output_path)}
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_write_text(output_dir / "usage.json", json.dumps(usage, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIDev task classification through the OpenAI Batch API.")
    parser.add_argument("action", choices=("submit", "status", "collect", "finalize"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default=aidev.DEFAULT_ENDPOINT)
    parser.add_argument("--body-max-tokens", type=int, default=10_000)
    parser.add_argument("--max-requests", type=int, default=DEFAULT_BATCH_REQUESTS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_BATCH_BYTES)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int, default=20260804)
    parser.add_argument("--exclude-decisions", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env) if args.action != "finalize" else ""
    if args.action != "finalize" and not api_key:
        raise ValueError(f"Missing API key in environment variable {args.api_key_env}.")
    if args.action == "submit":
        result = submit(args.input, args.output_dir, args.model, args.endpoint, api_key, args.body_max_tokens, args.max_requests, args.max_bytes, args.max_batches, args.resume, args.sample_size, args.sample_seed, args.exclude_decisions)
    elif args.action == "status":
        result = status(args.input, args.output_dir, args.model, args.endpoint, api_key, args.body_max_tokens, args.resume, args.sample_size, args.sample_seed, args.exclude_decisions)
    elif args.action == "collect":
        result = collect(args.input, args.output_dir, args.model, args.endpoint, api_key, args.body_max_tokens, args.resume, args.sample_size, args.sample_seed, args.exclude_decisions)
    else:
        result = finalize(args.input, args.output_dir, args.model, args.endpoint, args.body_max_tokens, args.resume, args.sample_size, args.sample_seed, args.exclude_decisions)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
