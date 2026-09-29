"""Run resumable RQ3 v2 requests, with automatic smoke checks and bounded retries."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from analysis.rq3_llm_validation import experiment as v1
from analysis.rq3_llm_validation import v2

STOP = False


def request_schema(request: dict, provider: str | None = None) -> dict:
    if "quote_repair_schema" in request:
        return request["quote_repair_schema"]
    schema = v2.schema(request["task"])
    # Gemini rejects large bounded arrays combined with occurrence-ID enums.
    # Keep the same semantic schema and enforce exact ID coverage after parsing.
    if provider == "gemini":
        return schema
    schema["$defs"]["OccurrenceVerdict"]["properties"]["id"]["enum"] = request["occurrence_ids"]
    schema["properties"]["occurrences"].update(minItems=len(request["occurrence_ids"]),
                                              maxItems=len(request["occurrence_ids"]))
    return schema


def call_provider(provider: str, request: dict, url: str) -> dict:
    model = v2.MODELS[provider]
    schema = request_schema(request, provider)
    if provider == "openai":
        from openai import OpenAI
        with OpenAI(timeout=1800, max_retries=2) as client:
            response = client.responses.create(
                model=model, input=[{"role": "developer", "content": v2.SYSTEM},
                                    {"role": "user", "content": request["prompt"]}],
                reasoning={"effort": "medium"}, max_output_tokens=v2.MAX_OUTPUT_TOKENS,
                text={"format": {"type": "json_schema", "name": "RQ3Occurrences", "schema": schema, "strict": True}})
            return response.model_dump(mode="json")
    if provider == "gemini":
        from google import genai
        from google.genai import types
        with genai.Client(api_key=os.environ.get("GEMINI_API_KEY"),
                          http_options=types.HttpOptions(timeout=1_800_000)) as client:
            response = client.models.generate_content(model=model, contents=request["prompt"],
                config=types.GenerateContentConfig(
                    system_instruction=v2.SYSTEM, temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
                    max_output_tokens=v2.MAX_OUTPUT_TOKENS, response_mime_type="application/json",
                    response_json_schema=schema))
            return response.model_dump(mode="json")
    from analysis.run_qwen import ollama_chat
    try:
        return ollama_chat(url.rstrip("/") + "/api/chat", model, v2.SYSTEM, request["prompt"],
                           schema, True, v2.QWEN_CONTEXT, v2.MAX_OUTPUT_TOKENS, 1800)
    except RuntimeError as error:
        if "cannot unmarshal object" not in str(error) or "ChatRequest.format" not in str(error):
            raise
        response = ollama_chat(url.rstrip("/") + "/api/chat", model, v2.SYSTEM,
                              request["prompt"] + "\nReturn JSON matching this schema:\n" + json.dumps(schema),
                              "json", True, v2.QWEN_CONTEXT, v2.MAX_OUTPUT_TOKENS, 1800)
        response["legacy_schema_fallback"] = True
        return response


def parse_response(provider: str, response: dict) -> tuple[dict, dict]:
    if provider == "openai":
        from analysis.rq3_llm_validation.run_openai import _output_text
        content = _output_text(response)
        usage = response.get("usage") or {}
    elif provider == "gemini":
        candidate = (response.get("candidates") or [{}])[0]
        if candidate.get("finish_reason") != "STOP":
            raise ValueError(f"Gemini response did not finish normally: {candidate.get('finish_reason')}")
        content = "".join(p.get("text", "") for p in (candidate.get("content") or {}).get("parts", [])
                          if not p.get("thought"))
        usage = response.get("usage_metadata") or {}
    else:
        if not response.get("done") or response.get("done_reason") == "length":
            raise ValueError("Qwen response is incomplete or reached its output limit.")
        if int(response.get("prompt_eval_count", 0)) + v2.MAX_OUTPUT_TOKENS >= v2.QWEN_CONTEXT:
            raise ValueError("Qwen prompt lacks the reserved output context; possible truncation.")
        content = (response.get("message") or {}).get("content", "")
        usage = {name: response.get(name) for name in
                 ("prompt_eval_count", "eval_count", "total_duration", "prompt_eval_duration", "eval_duration")}
    return json.loads(content), usage


def checkpoint_path(output: Path, provider: str, request: dict) -> Path:
    return output / provider / request["task"] / "responses" / f"{request['repo_id']}-{request['number']}.json"


def retry_request(request: dict, previous_error: str | None) -> dict:
    if not previous_error:
        return request
    prompt = request["prompt"] + (
        "\n\nOUTPUT VALIDATION DIAGNOSTIC (not additional PR evidence):\n"
        + json.dumps({"previous_error": previous_error})
        + "\nProduce a corrected complete response under the SAME audit policy. "
        "Include every occurrence exactly once in order. Quote only short contiguous original text. "
        "For tradeoff/joint_improvement, both measured directions and valid gain/memory occurrence IDs "
        "are mandatory; do not invent support to meet this condition. Use unsupported or indeterminate "
        "as defined when the evidence does not support a binary trade-off conclusion."
    )
    return {**request, "prompt": prompt}


def validate_with_quote_repair(provider: str, payload: dict, request: dict, url: str, raw_path: Path) -> dict:
    """Repair citations only; keep every occurrence verdict and PR decision fixed."""
    try:
        return v2.validate_result(payload, request)
    except ValueError as error:
        if "Quote is not" not in str(error):
            raise
    repair_path = raw_path.with_suffix(".quote-repair.json")
    repair_prompt = (
        "Repair source quotations only. The audit decisions are fixed and must not be reclassified. "
        "For the following citations, locate the corresponding content in the original records and "
        "return short contiguous VERBATIM excerpts, preserving Markdown, case, punctuation, and numbers. "
        "Do not combine separate lines into one invented sentence or omit text inside an excerpt. "
        "Several short excerpts may replace a combined citation. Do not substitute an unrelated metric. "
        "If the cited content cannot be found, return an empty evidence_quotes list.\n\nCITATIONS:\n"
        + json.dumps(payload["evidence_quotes"], ensure_ascii=False)
        + "\n\nORIGINAL RECORDS:\n" + "\n\n[RECORD BOUNDARY]\n\n".join(request["quote_corpus"])
    )
    repair = {**request, "prompt": repair_prompt, "quote_repair_schema": {
        "type": "object", "properties": {"evidence_quotes": {"type": "array", "items": {"type": "string"}}},
        "required": ["evidence_quotes"], "additionalProperties": False}}
    if repair_path.exists():
        response = json.loads(repair_path.read_text())
    else:
        v1.atomic_write_json(raw_path.with_suffix(".quote-repair-request.json"), {
            "original_prompt_sha256": request["prompt_sha256"], "prompt": repair_prompt,
            "scope": "quotes_only_no_label_changes", "schema": repair["quote_repair_schema"]})
        response = call_provider(provider, repair, url)
        v1.atomic_write_json(repair_path, response)
    corrected, usage = parse_response(provider, response)
    if set(corrected) != {"evidence_quotes"}:
        raise ValueError("Quote-only repair returned unexpected fields.")
    # Only this field can change; decisions are copied from the original response.
    result = v2.validate_result({**payload, "evidence_quotes": corrected["evidence_quotes"]}, request)
    result["quote_repair"] = {"original_model_quotes": payload["evidence_quotes"], "usage": usage,
                              "scope": "quotes_only_no_label_changes"}
    return result


def classify(output: Path, provider: str, request: dict, url: str, max_attempts: int) -> dict:
    path = checkpoint_path(output, provider, request)
    prior = json.loads(path.read_text()) if path.exists() else {}
    if prior:
        for name in ("prompt_sha256", "study_contract_sha256"):
            if prior[name] != request[name]:
                raise ValueError("Checkpoint belongs to a different prompt/study contract.")
        if prior["model"] != v2.MODELS[provider]:
            raise ValueError("Checkpoint belongs to a different model.")
        if prior["classification_status"] == "classified":
            # Reject stale or manually corrupted successes before resuming.
            fields = (v2.RegexResult if request["task"] == "regex_audit" else v2.TradeoffResult).model_fields
            v2.validate_result({k: prior["label"][k] for k in fields}, request)
            return prior
        raw_path = path.parent.parent / "raw" / f"{request['repo_id']}-{request['number']}-attempt-{prior['attempts']}.json"
        if raw_path.exists():
            try:
                payload, usage = parse_response(provider, json.loads(raw_path.read_text()))
                recovered = validate_with_quote_repair(provider, payload, request, url, raw_path)
            except ValueError:
                pass
            else:
                prior.update(label=recovered, usage=usage, classification_status="classified", error=None,
                             revalidated=True, revalidation="citation_validation_without_label_changes")
                v1.atomic_write_json(path, prior)
                return prior
    for attempt in range(prior.get("attempts", 0) + 1, max_attempts + 1):
        if STOP:
            return prior
        started = time.monotonic()
        result = {name: request[name] for name in ("custom_id", "task", "repo_id", "number", "sample_arm",
                                                  "html_url", "prompt_sha256", "study_contract_sha256")}
        result.update(provider=provider, model=v2.MODELS[provider], attempts=attempt,
                      schema_sha256=v1.sha256_json(request_schema(request, provider)))
        raw_path = path.parent.parent / "raw" / f"{request['repo_id']}-{request['number']}-attempt-{attempt}.json"
        effective = retry_request(request, prior.get("error"))
        result["effective_prompt_sha256"] = hashlib.sha256(effective["prompt"].encode()).hexdigest()
        v1.atomic_write_json(raw_path.with_suffix(".request.json"), {
            "prompt": effective["prompt"], "core_prompt_sha256": request["prompt_sha256"],
            "effective_prompt_sha256": result["effective_prompt_sha256"],
            "previous_validation_error": prior.get("error"), "schema": request_schema(effective, provider)})
        try:
            response = call_provider(provider, effective, url)
            v1.atomic_write_json(raw_path, response)
            payload, usage = parse_response(provider, response)
            v1.atomic_write_json(raw_path.with_suffix(".parsed.json"), payload)
            result.update(label=validate_with_quote_repair(provider, payload, request, url, raw_path), usage=usage,
                          classification_status="classified", error=None)
        except Exception as error:
            result.update(classification_status="error", error=f"{type(error).__name__}: {error}"[:2000])
        result.update(elapsed_seconds=round(time.monotonic() - started, 3),
                      completed_at=datetime.now(timezone.utc).isoformat())
        v1.atomic_write_json(raw_path.with_suffix(".status.json"), result)
        v1.atomic_write_json(path, result)
        print(json.dumps({key: result[key] for key in ("custom_id", "attempts", "classification_status", "elapsed_seconds", "error")}), flush=True)
        if result["classification_status"] == "classified":
            return result
        prior = result
        if attempt < max_attempts:
            time.sleep(min(120, 10 * 2 ** (attempt - 1)))
    return prior


def collect(output: Path, provider: str, requests: list[dict]) -> dict:
    rows = []
    for request in requests:
        path = checkpoint_path(output, provider, request)
        if path.exists():
            value = json.loads(path.read_text())
            rows.append({**{k: v for k, v in value.items() if k not in {"label", "usage"}},
                         **value.get("label", {}), "usage_json": json.dumps(value.get("usage", {}))})
    frame = pd.DataFrame(rows)
    tasks = {}
    for task in v1.TASKS:
        subset = frame[frame.task.eq(task)] if not frame.empty else frame
        directory = output / provider / task
        directory.mkdir(parents=True, exist_ok=True)
        if not subset.empty:
            temporary = directory / "labels.tmp.parquet"
            subset.to_parquet(temporary, index=False)
            os.replace(temporary, directory / "labels.parquet")
        expected = sum(r["task"] == task for r in requests)
        tasks[task] = {"expected": expected, "checkpoints": len(subset),
                       "classified": int(subset.classification_status.eq("classified").sum()) if not subset.empty else 0,
                       "errors": int(subset.classification_status.eq("error").sum()) if not subset.empty else 0}
        v1.atomic_write_json(directory / "summary.json", tasks[task])
    state = {"provider": provider, "tasks": tasks,
             "status": "completed" if all(t["expected"] == t["classified"] for t in tasks.values()) else "in_progress",
             "updated_at": datetime.now(timezone.utc).isoformat()}
    v1.atomic_write_json(output / provider / "run_state.json", state)
    return state


def smoke_requests(requests: list[dict]) -> list[dict]:
    regex = [r for r in requests if r["task"] == "regex_audit"]
    tradeoff = [r for r in requests if r["task"] == "tradeoff_audit"]
    chosen = [next(r for r in regex if r["repo_id"] == repo and r["number"] == number)
              for repo, number in ((476642602, 14013), (121814210, 7848))]
    chosen.extend([max(regex, key=lambda r: len(r["occurrence_ids"])),
                   max(tradeoff, key=lambda r: len(r["occurrence_ids"]))])
    return list({r["custom_id"]: r for r in chosen}.values())


def run(output: Path, provider: str, workers: int = 2, url: str = "http://127.0.0.1:11434",
        max_attempts: int = 6, smoke_only: bool = False) -> dict:
    requests = v2.load_requests(output)
    directory = output / provider
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for request in smoke_requests(requests):
            result = classify(output, provider, request, url, max_attempts)
            collect(output, provider, requests)
            if STOP:
                raise SystemExit(75)
            if result.get("classification_status") != "classified":
                state = collect(output, provider, requests)
                state.update(status="failed", error=f"Smoke failed: {request['custom_id']}")
                v1.atomic_write_json(directory / "run_state.json", state)
                raise RuntimeError(f"Smoke failed: {request['custom_id']}: {result.get('error')}")
        v1.atomic_write_json(directory / "smoke_summary.json", {"status": "passed",
                             "custom_ids": [r["custom_id"] for r in smoke_requests(requests)],
                             "checks": ["schema", "all_occurrence_ids", "original_quotes", "complete_response"]})
        if not smoke_only:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1 if provider == "qwen" else workers) as pool:
                futures = [pool.submit(classify, output, provider, request, url, max_attempts) for request in requests]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    collect(output, provider, requests)
        state = collect(output, provider, requests)
        if STOP:
            raise SystemExit(75)
        if not smoke_only and state["status"] != "completed":
            state["status"] = "failed"
            v1.atomic_write_json(directory / "run_state.json", state)
            raise RuntimeError("Provider finished with unresolved errors; inspect raw responses and checkpoints.")
        if provider == "qwen" and state["status"] == "completed":
            from analysis.rq3_llm_validation.v2_consensus import build, ready_tasks
            ready = ready_tasks(output)
            if ready:
                build(output, ready)
        return state


def stop_handler(signum, frame):
    global STOP
    STOP = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=v2.DEFAULT_OUTPUT)
    parser.add_argument("--provider", choices=tuple(v2.MODELS), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop_handler)
    print(json.dumps(run(args.output_dir, args.provider, args.workers, args.ollama_url,
                         args.max_attempts, args.smoke_only), indent=2), flush=True)
