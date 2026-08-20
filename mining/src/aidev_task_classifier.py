"""Internal AIDev-compatible task-type classifier used by the Batch runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from schema import atomic_write_text, write_parquet


METHOD_NAME = "aidev_public_task_classifier_luna_v1"
METHOD_VERSION = 3
DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
IDENTITY_COLUMNS = ("repo_id", "number")
TYPES = {
    "feat": "A new feature",
    "fix": "A bug fix",
    "docs": "Documentation only changes",
    "style": "Changes that do not affect the meaning of the code (white-space, formatting, etc)",
    "refactor": "A code change that neither fixes a bug nor adds a feature",
    "perf": "A code change that improves performance",
    "test": "Adding missing tests or correcting existing tests",
    "build": "Changes that affect the build system or external dependencies",
    "ci": "Changes to our CI configuration files and scripts",
    "chore": "Changes to the build process or auxiliary tools",
    "other": "Any other changes that do not fit the above categories",
    "revert": "Reverts a previous commit",
}
PATTERNS = {
    task_type: re.compile(rf"^{task_type}(\([^)]*\))?!?(?=\W|$)", flags=re.IGNORECASE)
    for task_type in TYPES
}
JSON_SCHEMA = {
    "name": "classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "A brief explanation for why this commit type was chosen",
            },
            "output": {
                "type": "string",
                "enum": list(TYPES),
                "description": "One of the allowed Conventional Commit types",
            },
            "confidence": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Confidence score (1-10)",
            },
        },
        "required": ["reason", "output", "confidence"],
        "additionalProperties": False,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def text_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def identity_key(row: dict[str, object]) -> str:
    try:
        return f"{int(row['repo_id'])}#{int(row['number'])}"
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Input row has an invalid (repo_id, number) identity.") from error


def title_label(title: object) -> str | None:
    """Return AIDev's first matching Conventional Commit label from title line one."""
    value = text_value(title)
    first_line = value.splitlines()[0] if value else ""
    for task_type, pattern in PATTERNS.items():
        if pattern.match(first_line):
            return task_type
    return None


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError("tiktoken is required to reproduce AIDev's GPT-4 body truncation.") from error
    encoding = tiktoken.encoding_for_model("gpt-4")
    # PR bodies can contain tokenizer control-token literals; they are ordinary source text here.
    tokens = encoding.encode(text, disallowed_special=())
    return text if len(tokens) <= max_tokens else encoding.decode(tokens[:max_tokens])


def system_prompt() -> str:
    definitions = "\n".join(f"{task_type}: {description}" for task_type, description in TYPES.items())
    return (
        "You are a Conventional Commit classifier. "
        "Given a PR title and body, pick **exactly one** label from:\n"
        f"{definitions}\n"
        "Respond in JSON with schema: {reason, output, confidence (1-10)}."
    )


def request_payload(model: str, row: dict[str, object], body_max_tokens: int) -> dict[str, object]:
    title = text_value(row.get("title"))
    body = truncate_to_tokens(text_value(row.get("body")), body_max_tokens)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": f"Title:\n{title}\n\nBody:\n{body}"},
        ],
        # GPT-5.6 Luna replaces the legacy Chat Completions parameter name.
        "max_completion_tokens": 4096,
        "response_format": {"type": "json_schema", "json_schema": JSON_SCHEMA},
    }


def parse_response(response: dict[str, object]) -> tuple[str, str, int]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Chat Completions response has no choice.")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Chat Completions response has no message content.")
    value = json.loads(message["content"])
    if not isinstance(value, dict) or value.get("output") not in TYPES:
        raise ValueError("LLM response contains an invalid Conventional Commit type.")
    if not isinstance(value.get("reason"), str):
        raise ValueError("LLM response contains an invalid reason.")
    confidence = value.get("confidence")
    if not isinstance(confidence, int) or not 1 <= confidence <= 10:
        raise ValueError("LLM response contains an invalid confidence.")
    return value["output"], value["reason"], confidence


def post_with_retry(
    endpoint: str,
    api_key: str,
    payload: dict[str, object],
    timeout: float,
    max_retries: int,
) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            if response.status_code == 429 or 500 <= response.status_code < 600:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Chat Completions returned a non-object JSON response.")
            return result
        except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < max_retries:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def signature(input_path: Path, model: str, endpoint: str, body_max_tokens: int) -> dict[str, object]:
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "aidev_source": "SAILResearch/AI_Teammates_in_SE3@be60e52962ef867b300aac433a05f93440cbb653/analysis/classify_pr.py",
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "model": model,
        "endpoint": endpoint,
        "body_tokenizer": "gpt-4",
        "body_max_tokens": body_max_tokens,
        "system_prompt_sha256": hashlib.sha256(system_prompt().encode("utf-8")).hexdigest(),
        "output_schema_sha256": hashlib.sha256(canonical_json(JSON_SCHEMA).encode("utf-8")).hexdigest(),
    }


def load_state(path: Path, expected_signature: dict[str, object], resume: bool) -> dict[str, object]:
    if not path.exists():
        return {"signature": expected_signature, "completed": {}, "created_at": datetime.now(timezone.utc).isoformat()}
    if not resume:
        raise FileExistsError(f"State already exists at {path}; use --resume.")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("signature") != expected_signature or not isinstance(state.get("completed"), dict):
        raise ValueError("Cannot resume: incompatible or malformed checkpoint.")
    return state


def llm_record(response: dict[str, object] | None, error: Exception | None, request_sha256: str) -> dict[str, object]:
    record: dict[str, object] = {
        "aidev_task_type": "",
        "aidev_task_type_reason": "",
        "aidev_task_type_confidence": None,
        "aidev_task_type_method": "llm",
        "aidev_task_type_status": "api_error",
        "aidev_task_type_response_id": "",
        "aidev_task_type_input_tokens": None,
        "aidev_task_type_output_tokens": None,
        "aidev_task_type_request_sha256": request_sha256,
        "aidev_task_type_error": "",
    }
    if error is not None:
        record["aidev_task_type_error"] = f"{type(error).__name__}: {error}"
        return record
    assert response is not None
    task_type, reason, confidence = parse_response(response)
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    record.update(
        aidev_task_type=task_type,
        aidev_task_type_reason=reason,
        aidev_task_type_confidence=confidence,
        aidev_task_type_status="classified",
        aidev_task_type_response_id=str(response.get("id") or ""),
        aidev_task_type_input_tokens=usage.get("prompt_tokens"),
        aidev_task_type_output_tokens=usage.get("completion_tokens"),
    )
    return record


def classify_row(
    endpoint: str,
    api_key: str,
    model: str,
    row: dict[str, object],
    body_max_tokens: int,
    timeout: float,
    max_retries: int,
) -> tuple[dict[str, object], str]:
    payload = request_payload(model, row, body_max_tokens)
    request_sha256 = hashlib.sha256(canonical_json(payload["messages"]).encode("utf-8")).hexdigest()
    return post_with_retry(endpoint, api_key, payload, timeout, max_retries), request_sha256


def classify(
    input_path: Path,
    output_dir: Path,
    model: str,
    endpoint: str,
    api_key: str,
    body_max_tokens: int,
    workers: int,
    timeout: float,
    max_retries: int,
    checkpoint_every: int,
    resume: bool,
    retry_errors: bool,
) -> dict[str, object]:
    if min(body_max_tokens, workers, max_retries, checkpoint_every) <= 0:
        raise ValueError("Body limit, workers, retries, and checkpoint interval must be positive.")
    frame = pd.read_parquet(input_path)
    if not set(IDENTITY_COLUMNS).issubset(frame.columns) or frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("Input must contain unique immutable (repo_id, number) identities.")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    expected_signature = signature(input_path, model, endpoint, body_max_tokens)
    state = load_state(state_path, expected_signature, resume)
    automatic: dict[str, dict[str, object]] = {}
    pending: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        key = identity_key(row)
        task_type = title_label(row.get("title"))
        if task_type is not None:
            automatic[key] = {
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
            continue
        previous = state["completed"].get(key)
        if not isinstance(previous, dict) or (previous.get("aidev_task_type_status") == "api_error" and retry_errors):
            pending.append(row)
    dirty = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(pending), workers):
            batch = pending[start : start + workers]
            futures = {
                executor.submit(
                    classify_row,
                    endpoint,
                    api_key,
                    model,
                    row,
                    body_max_tokens,
                    timeout,
                    max_retries,
                ): row
                for row in batch
            }
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                key = identity_key(row)
                try:
                    response, request_sha256 = future.result()
                    state["completed"][key] = llm_record(response, None, request_sha256)
                except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
                    state["completed"][key] = llm_record(None, error, "")
                dirty += 1
                if dirty >= checkpoint_every:
                    state["updated_at"] = datetime.now(timezone.utc).isoformat()
                    atomic_write_text(state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")
                    dirty = 0
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_text(state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")
    rows = []
    for row in frame.to_dict(orient="records"):
        values = automatic.get(identity_key(row)) or state["completed"].get(identity_key(row))
        if not isinstance(values, dict):
            raise ValueError("No task-type decision is available for an input PR.")
        rows.append({**row, **values, "aidev_task_type_model": model, "aidev_task_type_classifier": METHOD_NAME})
    decisions = pd.DataFrame(rows)
    output_path = output_dir / "task_type_decisions.parquet"
    write_parquet(decisions, output_path)
    summary = {
        **expected_signature,
        "rows": len(decisions),
        "title_regex_rows": len(automatic),
        "llm_needed_rows": len(frame) - len(automatic),
        "status_counts": dict(sorted(Counter(decisions["aidev_task_type_status"]).items())),
        "task_type_counts": dict(sorted(Counter(decisions["aidev_task_type"]).items())),
        "output_sha256": sha256_file(output_path),
    }
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIDev's public task-type cascade with a replacement LLM.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--body-max-tokens", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=50)
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
        args.body_max_tokens,
        args.workers,
        args.timeout,
        args.max_retries,
        args.checkpoint_every,
        args.resume,
        args.retry_errors,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
