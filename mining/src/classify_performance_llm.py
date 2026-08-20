"""Classify selected performance PR candidates through the OpenAI Responses API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from schema import atomic_write_text, write_parquet


METHOD_NAME = "openai_responses_performance_v1"
METHOD_VERSION = 1
IDENTITY_COLUMNS = ("repo_id", "number")
DEFAULT_ENDPOINT = "https://api.openai.com/v1/responses"
SYSTEM_PROMPT = """You classify pull requests as performance-improving or not.
Use only the supplied title and body. A performance-improving pull request makes
software measurably faster, use less memory/resources, reduce latency, improve
throughput, or adds a benchmark/profiling change whose purpose is performance.
Do not infer performance from unrelated refactors, correctness fixes, or vague
claims. Return uncertain only when the supplied metadata is insufficient."""
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "label": {"type": "string", "enum": ["performance", "non_performance", "uncertain"]},
        "rationale": {"type": "string", "maxLength": 500},
    },
    "required": ["label", "rationale"],
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def identity_key(row: dict[str, object]) -> str:
    try:
        return f"{int(row['repo_id'])}#{int(row['number'])}"
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Candidate row has an invalid (repo_id, number) identity.") from error


def text_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def request_input(row: dict[str, object]) -> dict[str, str]:
    return {"title": text_value(row.get("title")), "body": text_value(row.get("body"))}


def request_payload(model: str, row: dict[str, object]) -> dict[str, object]:
    values = request_input(row)
    user_text = f"Title:\n{values['title']}\n\nBody:\n{values['body']}"
    return {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "performance_classification",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
        "temperature": 0,
    }


def output_text(response: dict[str, object]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str):
        return direct
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("Responses API reply has no output text.")
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    return text
    raise ValueError("Responses API reply has no output_text content.")


def parse_response(response: dict[str, object]) -> tuple[str, str]:
    value = json.loads(output_text(response))
    if not isinstance(value, dict):
        raise ValueError("LLM response is not a JSON object.")
    label = value.get("label")
    rationale = value.get("rationale")
    if label not in {"performance", "non_performance", "uncertain"}:
        raise ValueError("LLM response contains an invalid label.")
    if not isinstance(rationale, str):
        raise ValueError("LLM response contains an invalid rationale.")
    return label, rationale


def post_response(endpoint: str, api_key: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("Responses API returned a non-object JSON response.")
    return value


def run_signature(input_path: Path, model: str, endpoint: str) -> dict[str, object]:
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "model": model,
        "endpoint": endpoint,
        "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "output_schema_sha256": sha256_bytes(canonical_json(OUTPUT_SCHEMA).encode("utf-8")),
    }


def load_state(path: Path, signature: dict[str, object], resume: bool) -> dict[str, object]:
    if not path.exists():
        return {"signature": signature, "completed": {}, "created_at": datetime.now(timezone.utc).isoformat()}
    if not resume:
        raise FileExistsError(f"State already exists at {path}; use --resume.")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("signature") != signature:
        raise ValueError("Cannot resume: LLM input, model, prompt, schema, or endpoint changed.")
    if not isinstance(state.get("completed"), dict):
        raise ValueError("Cannot resume: state has invalid completed records.")
    return state


def result_record(row: dict[str, object], response: dict[str, object] | None, error: Exception | None) -> dict[str, object]:
    payload = request_payload("", row)
    record: dict[str, object] = {
        "request_sha256": sha256_bytes(canonical_json(payload["input"]).encode("utf-8")),
        "llm_label": "",
        "llm_rationale": "",
        "llm_status": "api_error",
        "llm_response_id": "",
        "llm_input_tokens": None,
        "llm_output_tokens": None,
        "llm_error": "",
    }
    if error is not None:
        record["llm_error"] = f"{type(error).__name__}: {error}"
        return record
    assert response is not None
    label, rationale = parse_response(response)
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    record.update(
        llm_label=label,
        llm_rationale=rationale,
        llm_status="classified" if label != "uncertain" else "uncertain",
        llm_response_id=str(response.get("id") or ""),
        llm_input_tokens=usage.get("input_tokens"),
        llm_output_tokens=usage.get("output_tokens"),
    )
    return record


def classify(
    input_path: Path,
    output_dir: Path,
    model: str,
    endpoint: str,
    api_key: str,
    timeout: float,
    checkpoint_every: int,
    resume: bool,
    retry_errors: bool,
) -> dict[str, object]:
    if checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive.")
    candidates = pd.read_parquet(input_path)
    missing = set(IDENTITY_COLUMNS) - set(candidates.columns)
    if missing:
        raise ValueError(f"Input is missing identity columns: {sorted(missing)}")
    if candidates.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("Input has duplicate immutable identities.")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    signature = run_signature(input_path, model, endpoint)
    state = load_state(state_path, signature, resume)
    completed = state["completed"]
    dirty = 0
    for row in candidates.to_dict(orient="records"):
        key = identity_key(row)
        prior = completed.get(key)
        if isinstance(prior, dict) and (prior.get("llm_status") != "api_error" or not retry_errors):
            continue
        try:
            response = post_response(endpoint, api_key, request_payload(model, row), timeout)
            completed[key] = result_record(row, response, None)
        except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
            completed[key] = result_record(row, None, error)
        dirty += 1
        if dirty >= checkpoint_every:
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_text(state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")
            dirty = 0
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_text(state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")
    rows = []
    for row in candidates.to_dict(orient="records"):
        values = completed.get(identity_key(row))
        if not isinstance(values, dict):
            raise ValueError("State is missing a candidate result.")
        rows.append({**row, **values, "llm_model": model, "llm_method": METHOD_NAME})
    decisions = pd.DataFrame(rows)
    write_parquet(decisions, output_dir / "decisions.parquet")
    counts = Counter(decisions["llm_status"])
    labels = Counter(decisions.loc[decisions["llm_status"] != "api_error", "llm_label"])
    summary = {
        **signature,
        "rows": len(decisions),
        "status_counts": dict(sorted(counts.items())),
        "label_counts": dict(sorted(labels.items())),
        "output_sha256": sha256_file(output_dir / "decisions.parquet"),
    }
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify selected PRs through OpenAI Responses.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key in environment variable {args.api_key_env}.")
    summary = classify(
        args.input,
        args.output_dir,
        args.model,
        args.endpoint,
        api_key,
        args.timeout,
        args.checkpoint_every,
        args.resume,
        args.retry_errors,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
