"""Native cloud Batch execution and resumable local Qwen for an RQ3 study module."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from analysis.rq3_llm_validation import binary, v2
from analysis.rq3_llm_validation import experiment as v1

MAX_BATCH_ATTEMPTS = 5
OPENAI_TERMINAL = {"completed", "failed", "expired", "cancelled"}
GEMINI_TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED", "JOB_STATE_FAILED",
                   "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
STOP = False


def is_billing_error(error: object) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in (
        "billing_hard_limit_reached", "billing hard limit", "prepayment credits are depleted",
        "insufficient_quota", "exceeded your current quota",
    ))


def effective_prompt(request: dict, error: str | None) -> str:
    if not error:
        return request["prompt"]
    return request["prompt"] + binary.retry_diagnostic(error)


def response_schema(request: dict, provider: str) -> dict:
    result = binary.schema()
    if provider != "gemini":
        result["$defs"]["OccurrenceVerdict"]["properties"]["id"]["enum"] = request["occurrence_ids"]
        result["properties"]["occurrences"].update(minItems=len(request["occurrence_ids"]),
                                                  maxItems=len(request["occurrence_ids"]))
    return result


def qwen_transport(request: dict, attempt: int) -> tuple[dict, object, str]:
    """Relax only transport grammar after repeated failures; validation stays exact."""
    if attempt >= 10:
        return binary.schema(), "json", "json_with_prompt_schema"
    if attempt >= 7:
        return binary.schema(), binary.schema(), "compact_semantic_schema"
    schema = response_schema(request, "qwen")
    return schema, schema, "bounded_id_schema"


def batch_item(provider: str, request: dict, error: str | None = None) -> dict:
    prompt = effective_prompt(request, error)
    schema = response_schema(request, provider)
    if provider == "openai":
        return {"custom_id": request["custom_id"], "method": "POST", "url": "/v1/responses", "body": {
            "model": v2.MODELS[provider], "input": [{"role": "developer", "content": v2.SYSTEM},
                                                    {"role": "user", "content": prompt}],
            "reasoning": {"effort": "medium"}, "max_output_tokens": v2.MAX_OUTPUT_TOKENS,
            "text": {"format": {"type": "json_schema", "name": binary.SCHEMA_NAME, "schema": schema, "strict": True}}}}
    if provider == "gemini":
        return {"key": request["custom_id"], "request": {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "system_instruction": {"parts": [{"text": v2.SYSTEM}]},
            "generation_config": {"temperature": 0, "thinking_config": {"thinking_level": "MEDIUM"},
                                  "max_output_tokens": v2.MAX_OUTPUT_TOKENS,
                                  "response_mime_type": "application/json", "response_json_schema": schema}}}
    raise ValueError("Cloud payloads support OpenAI and Gemini only.")


def prepare_batch(output: Path, provider: str, requests: list[dict], attempt: int) -> Path:
    directory = output / provider / "batches" / f"attempt-{attempt:02d}"
    if directory.exists():
        raise FileExistsError("Refusing to overwrite a prepared Batch attempt.")
    directory.mkdir(parents=True)
    rows, hashes = [], {}
    for request in requests:
        prior = binary.checked_checkpoint(output, provider, request)
        if prior and prior["classification_status"] == "classified":
            raise ValueError("A Batch retry may not reclassify a successful checkpoint.")
        error = prior.get("error") if prior else None
        rows.append(batch_item(provider, request, error))
        hashes[request["custom_id"]] = hashlib.sha256(effective_prompt(request, error).encode()).hexdigest()
    payload = directory / "requests.jsonl"
    payload.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    state = {"provider": provider, "attempt": attempt, "custom_ids": [r["custom_id"] for r in requests],
             "requests": len(requests), "payload_sha256": v1.sha256_file(payload),
             "effective_prompt_sha256": hashes, "study_contract_sha256": requests[0]["study_contract_sha256"],
              "run_id": f"{binary.VERSION}-{requests[0]['study_contract_sha256'][:12]}-{provider}-{attempt}",
             "transport": "batch", "status": "prepared"}
    v1.atomic_write_json(directory / "state.json", state)
    return directory


def _client(provider: str):
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(timeout=120, max_retries=0)
    from google import genai
    from google.genai import types
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),
                        http_options=types.HttpOptions(timeout=120_000,
                            retry_options=types.HttpRetryOptions(attempts=1)))


def _state(directory: Path) -> dict:
    state = json.loads((directory / "state.json").read_text())
    if v1.sha256_file(directory / "requests.jsonl") != state["payload_sha256"]:
        raise ValueError("Prepared Batch payload changed.")
    key = "custom_id" if state["provider"] == "openai" else "key"
    ids = [json.loads(line)[key] for line in (directory / "requests.jsonl").read_text().splitlines()]
    if ids != state["custom_ids"] or len(ids) != len(set(ids)):
        raise ValueError("Batch state identities differ from the prepared payload.")
    return state


def submit(directory: Path) -> dict:
    state = _state(directory)
    if state.get("batch_id"):
        return state
    if state["status"] not in {"prepared", "uploaded"}:
        raise RuntimeError(f"Submission is not resumable automatically: {state['status']}")
    provider = state["provider"]
    with _client(provider) as client:
        if not state.get("input_file_id"):
            if provider == "openai":
                with (directory / "requests.jsonl").open("rb") as handle:
                    uploaded = client.files.create(file=handle, purpose="batch")
                state["input_file_id"] = uploaded.id
            else:
                from google.genai import types
                uploaded = client.files.upload(file=directory / "requests.jsonl", config=types.UploadFileConfig(
                    display_name=state["run_id"], mime_type="application/jsonl"))
                state["input_file_id"] = uploaded.name
            state["status"] = "uploaded"
            v1.atomic_write_json(directory / "state.json", state)
        state["status"] = "submitting"
        v1.atomic_write_json(directory / "state.json", state)
        try:
            if provider == "openai":
                batch = client.batches.create(input_file_id=state["input_file_id"], endpoint="/v1/responses",
                                              completion_window="24h", metadata={"run_id": state["run_id"]})
                state.update(batch_id=batch.id, status=batch.status)
            else:
                batch = client.batches.create(model=v2.MODELS[provider], src=state["input_file_id"],
                                              config={"display_name": state["run_id"]})
                state.update(batch_id=batch.name, status=batch.state.value if batch.state else "submitted")
        except Exception as error:
            state.update(status="blocked_billing" if is_billing_error(error) else "submission_uncertain",
                         error=f"{type(error).__name__}: {error}"[:2000])
            v1.atomic_write_json(directory / "state.json", state)
            raise
    state["submitted_at"] = datetime.now(timezone.utc).isoformat()
    v1.atomic_write_json(directory / "state.json", state)
    return state


def poll(directory: Path) -> dict:
    state = _state(directory)
    if not state.get("batch_id"):
        return state
    with _client(state["provider"]) as client:
        if state["provider"] == "openai":
            batch = client.batches.retrieve(state["batch_id"])
            state.update(status=batch.status, output_file_id=batch.output_file_id, error_file_id=batch.error_file_id)
        else:
            batch = client.batches.get(name=state["batch_id"])
            state.update(status=batch.state.value if batch.state else "unknown",
                         output_file_id=batch.dest.file_name if batch.dest else None)
        state["provider_status"] = batch.model_dump(mode="json")
    state["checked_at"] = datetime.now(timezone.utc).isoformat()
    v1.atomic_write_json(directory / "state.json", state)
    return state


def parse_response(provider: str, response: dict) -> tuple[dict, dict]:
    if provider == "gemini":
        candidate = (response.get("candidates") or [{}])[0]
        finish = candidate.get("finishReason", candidate.get("finish_reason"))
        if finish != "STOP":
            raise ValueError(f"Gemini did not finish normally: {finish}")
        text = "".join(part.get("text", "") for part in (candidate.get("content") or {}).get("parts", [])
                       if not part.get("thought"))
        return json.loads(text), response.get("usageMetadata", response.get("usage_metadata")) or {}
    from analysis.rq3_llm_validation.run_v2 import parse_response as parse_v2_response
    return parse_v2_response(provider, response)


def result_base(request: dict, provider: str, attempt: int) -> dict:
    return {**{key: request[key] for key in ("custom_id", "repo_id", "number", "sample_arm", "html_url",
                                            "prompt_sha256", "study_contract_sha256")},
            "provider": provider, "model": v2.MODELS[provider], "attempts": attempt,
            "transport": "ollama" if provider == "qwen" else "batch"}


def collect_outputs(output: Path, directory: Path, texts: list[str]) -> dict:
    state = _state(directory)
    provider = state["provider"]
    requests = {r["custom_id"]: r for r in binary.load_requests(output)}
    expected = set(state["custom_ids"])
    if not expected <= set(requests) or any(requests[key]["study_contract_sha256"] != state["study_contract_sha256"] for key in expected):
        raise ValueError("Batch manifest identities or contract changed.")
    items = {}
    for text in texts:
        for line in text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = item.get("custom_id") if provider == "openai" else item.get("key")
            if key not in expected or key in items:
                raise ValueError(f"Unknown or duplicate Batch identity: {key}")
            items[key] = item
    for key in state["custom_ids"]:
        request = requests[key]
        item = items.get(key, {})
        result = result_base(request, provider, state["attempt"])
        result.update(batch_id=state.get("batch_id"), effective_prompt_sha256=state["effective_prompt_sha256"][key])
        raw = output / provider / "raw" / f"{request['repo_id']}-{request['number']}-attempt-{state['attempt']}.json"
        try:
            if not item or item.get("error"):
                raise ValueError(json.dumps(item.get("error") or "Missing from Batch output"))
            response = item.get("response") or {}
            if provider == "openai":
                if response.get("status_code") != 200:
                    raise ValueError(json.dumps(response))
                response = response.get("body") or {}
            v1.atomic_write_json(raw, response)
            payload, usage = parse_response(provider, response)
            result["usage"] = usage
            v1.atomic_write_json(raw.with_suffix(".parsed.json"), payload)
            result.update(label=binary.validate_result(payload, request), classification_status="classified", error=None)
        except ValueError as error:
            result.update(classification_status="error", error=str(error)[:2000])
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        v1.atomic_write_json(raw.with_suffix(".status.json"), result)
        v1.atomic_write_json(binary.checkpoint_path(output, provider, request), result)
    state["collected"] = True
    v1.atomic_write_json(directory / "state.json", state)
    return binary.collect(output, provider, list(requests.values()))


def collect_batch(output: Path, directory: Path) -> dict:
    state = _state(directory)
    terminal = OPENAI_TERMINAL if state["provider"] == "openai" else GEMINI_TERMINAL
    if state["status"] not in terminal:
        raise RuntimeError("Batch is not terminal; collection would mark unfinished requests as missing.")
    texts = []
    with _client(state["provider"]) as client:
        for name in ("output_file_id", "error_file_id"):
            if not state.get(name):
                continue
            if state["provider"] == "openai":
                text = client.files.content(state[name]).text
            else:
                text = client.files.download(file=state[name]).decode()
            (directory / f"{name}.jsonl").write_text(text)
            texts.append(text)
    return collect_outputs(output, directory, texts)


def tick(output: Path, provider: str) -> dict:
    if provider not in {"openai", "gemini"}:
        raise ValueError("Batch controller only supports cloud providers.")
    requests = binary.load_requests(output)
    provider_dir = output / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    with (provider_dir / "batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = binary.collect(output, provider, requests)
        if current["status"] == "completed":
            return current
        attempts = sorted((provider_dir / "batches").glob("attempt-*/state.json"))
        if not attempts:
            directory = prepare_batch(output, provider, requests, 1)
        else:
            directory = attempts[-1].parent
        state = _state(directory)
        if not state.get("batch_id") and state["status"] in {"blocked_billing", "submission_uncertain", "submitting"}:
            if is_billing_error(state.get("error")):
                state["status"] = "blocked_billing"
                v1.atomic_write_json(directory / "state.json", state)
            current.update(status=state["status"], error=state.get("error", "Submission acknowledgement is missing."))
            v1.atomic_write_json(provider_dir / "run_state.json", current)
            return current
        if state.get("collected"):
            if state["attempt"] >= MAX_BATCH_ATTEMPTS:
                current.update(status="failed", error="Batch retry limit reached; inspect invalid checkpoints.")
                v1.atomic_write_json(provider_dir / "run_state.json", current)
                return current
            pending = [r for r in requests if (binary.checked_checkpoint(output, provider, r) or {}).get("classification_status") != "classified"]
            directory = prepare_batch(output, provider, pending, state["attempt"] + 1)
            state = _state(directory)
        if not state.get("batch_id"):
            state = submit(directory)
        else:
            state = poll(directory)
        terminal = OPENAI_TERMINAL if provider == "openai" else GEMINI_TERMINAL
        if state["status"] in terminal:
            current = collect_batch(output, directory)
        current.update(batch_id=state.get("batch_id"), batch_status=state["status"], batch_attempt=state["attempt"])
        v1.atomic_write_json(provider_dir / "run_state.json", current)
        return current


def resume_submission(output: Path, provider: str) -> dict:
    """Explicitly resume a definitively rejected submission after billing is fixed."""
    binary.load_requests(output)
    directory = output / provider
    with (directory / "batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempts = sorted((directory / "batches").glob("attempt-*/state.json"))
        if not attempts:
            raise ValueError("No rejected submission to resume.")
        path = attempts[-1]
        state = _state(path.parent)
        if state.get("batch_id") or not is_billing_error(state.get("error")):
            raise ValueError("Only a definite billing rejection without a Batch ID may be resumed.")
        state["previous_submission_error"] = state.pop("error")
        state["status"] = "uploaded" if state.get("input_file_id") else "prepared"
        v1.atomic_write_json(path, state)
    return tick(output, provider)


def run_qwen(output: Path, url: str = "http://127.0.0.1:11434", max_attempts: int = 6) -> dict:
    from analysis.run_qwen import ollama_chat

    requests = binary.load_requests(output)
    directory = output / "qwen"
    directory.mkdir(parents=True, exist_ok=True)
    # Validate the longest case early, then retain deterministic PR ordering.
    requests = sorted(requests, key=lambda r: (-len(r["occurrence_ids"]), r["repo_id"], r["number"]))
    with (directory / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for request in requests:
            prior = binary.checked_checkpoint(output, "qwen", request) or {}
            if prior.get("classification_status") == "classified":
                continue
            for attempt in range(prior.get("attempts", 0) + 1, max_attempts + 1):
                if STOP:
                    binary.collect(output, "qwen", requests)
                    raise SystemExit(75)
                prompt = effective_prompt(request, prior.get("error"))
                schema, response_format, schema_mode = qwen_transport(request, attempt)
                sent_prompt = prompt
                if response_format == "json":
                    sent_prompt += "\nReturn JSON matching this schema:\n" + json.dumps(schema)
                started = time.monotonic()
                result = result_base(request, "qwen", attempt)
                raw = directory / "raw" / f"{request['repo_id']}-{request['number']}-attempt-{attempt}.json"
                result["effective_prompt_sha256"] = hashlib.sha256(sent_prompt.encode()).hexdigest()
                result["transport_schema_mode"] = schema_mode
                v1.atomic_write_json(raw.with_suffix(".request.json"), {
                    "prompt": sent_prompt, "schema": schema, "response_format": response_format,
                    "transport_schema_mode": schema_mode, "system": v2.SYSTEM})
                try:
                    try:
                        response = ollama_chat(url.rstrip("/") + "/api/chat", v2.MODELS["qwen"], v2.SYSTEM,
                            sent_prompt, response_format, True, v2.QWEN_CONTEXT, v2.MAX_OUTPUT_TOKENS, 1800)
                    except RuntimeError as error:
                        if (response_format == "json" or "cannot unmarshal object" not in str(error)
                                or "ChatRequest.format" not in str(error)):
                            raise
                        sent_prompt = prompt + "\nReturn JSON matching this schema:\n" + json.dumps(schema)
                        result["effective_prompt_sha256"] = hashlib.sha256(sent_prompt.encode()).hexdigest()
                        result["transport_schema_mode"] += "+legacy_json_fallback"
                        v1.atomic_write_json(raw.with_suffix(".request.json"), {
                            "prompt": sent_prompt, "schema": schema, "response_format": "json",
                            "transport_schema_mode": result["transport_schema_mode"], "system": v2.SYSTEM})
                        response = ollama_chat(url.rstrip("/") + "/api/chat", v2.MODELS["qwen"], v2.SYSTEM,
                            sent_prompt, "json", True,
                            v2.QWEN_CONTEXT, v2.MAX_OUTPUT_TOKENS, 1800)
                        response["legacy_schema_fallback"] = True
                    v1.atomic_write_json(raw, response)
                    payload, usage = parse_response("qwen", response)
                    result["usage"] = usage
                    v1.atomic_write_json(raw.with_suffix(".parsed.json"), payload)
                    result.update(label=binary.validate_result(payload, request), classification_status="classified", error=None)
                except Exception as error:
                    result.update(classification_status="error", error=f"{type(error).__name__}: {error}"[:2000])
                result.update(completed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
                v1.atomic_write_json(raw.with_suffix(".status.json"), result)
                v1.atomic_write_json(binary.checkpoint_path(output, "qwen", request), result)
                binary.collect(output, "qwen", requests)
                print(json.dumps({key: result[key] for key in ("custom_id", "attempts", "classification_status", "error")}), flush=True)
                prior = result
                if result["classification_status"] == "classified":
                    break
        state = binary.collect(output, "qwen", requests)
        if STOP:
            raise SystemExit(75)
        if state["status"] != "completed":
            state["status"] = "failed"
            v1.atomic_write_json(directory / "run_state.json", state)
            raise RuntimeError("Qwen exhausted retries with invalid responses.")
        return state


def stop_handler(signum, frame):
    global STOP
    STOP = True


def main(study=binary, default_output: Path | None = None) -> None:
    global binary
    binary = study
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("tick", "qwen", "resume-submission"))
    parser.add_argument("--provider", choices=("openai", "gemini"))
    parser.add_argument("--output-dir", type=Path, default=default_output or binary.DEFAULT_OUTPUT)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-attempts", type=int, default=6)
    args = parser.parse_args()
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop_handler)
    if args.action == "qwen":
        result = run_qwen(args.output_dir, args.ollama_url, args.max_attempts)
    else:
        if not args.provider:
            parser.error("--provider is required for tick")
        result = (resume_submission(args.output_dir, args.provider) if args.action == "resume-submission"
                  else tick(args.output_dir, args.provider))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
