"""Prepare, submit, inspect, collect, and retry RQ3 OpenAI batches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from analysis.rq3_llm_validation import experiment

DEFAULT_MODEL = "gpt-5.6-sol"
ENDPOINT = "/v1/responses"
REASONING_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 4_096


def provider_config(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    return {
        "provider_config_version": 1, "provider": "openai", "model": model,
        "endpoint": ENDPOINT, "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }


def response_format(task: str) -> dict[str, Any]:
    return {"format": {"type": "json_schema", "name": "RQ3Label",
                       "schema": experiment.semantic_schema(task), "strict": True}}


def prepare(task: str, sample: Path, matches: Path, evidence_dir: Path,
            output_dir: Path, model: str = DEFAULT_MODEL) -> Path:
    frame, manifest, metadata = experiment.prepare_run(
        task, sample, matches, evidence_dir, output_dir, provider_config(model)
    )
    by_id = frame.assign(custom_id=lambda x: x.repo_id.astype(str) + ":" + x.number.astype(str)).set_index("custom_id")
    payload = output_dir / "batch_input.jsonl"
    with payload.open("w", encoding="utf-8") as handle:
        for custom_id in manifest["custom_id"]:
            prompt = experiment.prompt_for(by_id.loc[custom_id].to_dict(), task)
            request = {
                "custom_id": custom_id, "method": "POST", "url": ENDPOINT,
                "body": {
                    "model": model,
                    "input": [{"role": "developer", "content": experiment.SYSTEM_INSTRUCTION},
                              {"role": "user", "content": prompt}],
                    "reasoning": {"effort": REASONING_EFFORT},
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "text": response_format(task),
                },
            }
            handle.write(json.dumps(request, ensure_ascii=True) + "\n")
    metadata["batch_input_sha256"] = experiment.sha256_file(payload)
    experiment.atomic_write_json(output_dir / "prepare_metadata.json", metadata)
    return payload


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def submit(output_dir: Path) -> dict[str, Any]:
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text())
    payload = output_dir / "batch_input.jsonl"
    if experiment.sha256_file(payload) != metadata["batch_input_sha256"]:
        raise ValueError("Batch input changed after preparation.")
    if experiment.sha256_file(output_dir / "batch_manifest.parquet") != metadata["manifest_sha256"]:
        raise ValueError("Batch manifest changed after preparation.")
    state_path = output_dir / "batch_state.json"
    if state_path.exists() and json.loads(state_path.read_text()).get("batch_id"):
        raise FileExistsError("Batch was already submitted.")
    client = _client()
    with payload.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint=ENDPOINT,
                                  completion_window="24h",
                                  metadata={"description": f"RQ3 {metadata['task']}"})
    state = {"input_file_id": uploaded.id, "batch_id": batch.id, "status": batch.status}
    experiment.atomic_write_json(state_path, state)
    return state


def status(output_dir: Path) -> dict[str, Any]:
    state_path = output_dir / "batch_state.json"
    state = json.loads(state_path.read_text())
    batch = _client().batches.retrieve(state["batch_id"])
    value = {"batch_id": batch.id, "status": batch.status,
             "output_file_id": batch.output_file_id, "error_file_id": batch.error_file_id}
    experiment.atomic_write_json(state_path, {**state, **value})
    return value


def _output_text(body: dict[str, Any]) -> str:
    if body.get("status") == "incomplete" or body.get("incomplete_details"):
        raise ValueError("OpenAI response is incomplete.")
    texts = [str(content.get("text") or "") for output in body.get("output", [])
             for content in output.get("content", []) if content.get("type") == "output_text"]
    if not texts:
        raise ValueError("OpenAI response has no output text.")
    return "".join(texts)


def parse_output(output_text: str, manifest: pd.DataFrame, task: str, model: str) -> pd.DataFrame:
    by_id = manifest.set_index("custom_id").to_dict("index")
    rows, seen = [], set()
    for line in output_text.splitlines():
        item = json.loads(line)
        custom_id = item.get("custom_id")
        if custom_id in seen or custom_id not in by_id:
            raise ValueError(f"Invalid or duplicate custom_id: {custom_id}")
        seen.add(custom_id)
        response = item.get("response") or {}
        body = response.get("body") or {}
        row = {**by_id[custom_id], "custom_id": custom_id, "model": body.get("model") or model}
        try:
            if response.get("status_code") != 200:
                raise ValueError(json.dumps(item.get("error") or response))
            label = experiment.label_model(task).model_validate_json(_output_text(body))
        except ValueError as error:
            row.update(classification_status="error", error=str(error)[:500])
        else:
            usage = body.get("usage") or {}
            row.update(label.model_dump(), classification_status="classified",
                       input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))
        rows.append(row)
    for custom_id in sorted(set(by_id) - seen):
        rows.append({**by_id[custom_id], "custom_id": custom_id, "model": model,
                     "classification_status": "error", "error": "Missing from batch output"})
    return pd.DataFrame(rows).sort_values(experiment.KEYS).reset_index(drop=True)


def collect(output_dir: Path) -> pd.DataFrame:
    current = status(output_dir)
    if current["status"] != "completed" or not current["output_file_id"]:
        raise RuntimeError(f"Batch is not ready: {current['status']}")
    output = _client().files.content(current["output_file_id"]).text
    (output_dir / "batch_output.jsonl").write_text(output, encoding="utf-8")
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text())
    if experiment.sha256_file(output_dir / "batch_manifest.parquet") != metadata["manifest_sha256"]:
        raise ValueError("Batch manifest changed after preparation.")
    manifest = pd.read_parquet(output_dir / "batch_manifest.parquet")
    result = parse_output(output, manifest, metadata["task"], metadata["provider_config"]["model"])
    result.to_parquet(output_dir / "labels.parquet", index=False)
    experiment.atomic_write_json(output_dir / "summary.json", {
        "rows": len(result), "status_counts": result.classification_status.value_counts().to_dict()
    })
    return result


def prepare_retry(source_dir: Path, output_dir: Path) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Retry output directory is not empty.")
    labels = pd.read_parquet(source_dir / "labels.parquet")
    retry_ids = set(labels.loc[labels.classification_status.eq("error"), "custom_id"])
    if not retry_ids:
        raise ValueError("Source run has no errors to retry.")
    metadata = json.loads((source_dir / "prepare_metadata.json").read_text())
    output_dir.mkdir(parents=True)
    requests = [json.loads(line) for line in (source_dir / "batch_input.jsonl").read_text().splitlines()]
    payload = output_dir / "batch_input.jsonl"
    payload.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in requests if row["custom_id"] in retry_ids))
    manifest = pd.read_parquet(source_dir / "batch_manifest.parquet").query("custom_id in @retry_ids")
    manifest.to_parquet(output_dir / "batch_manifest.parquet", index=False)
    metadata.update(requests=len(manifest), retry_of=str(source_dir),
                    batch_input_sha256=experiment.sha256_file(payload),
                    manifest_sha256=experiment.sha256_file(output_dir / "batch_manifest.parquet"))
    experiment.atomic_write_json(output_dir / "prepare_metadata.json", metadata)
    return payload


def merge_retry(source_dir: Path, retry_dir: Path, output_path: Path) -> pd.DataFrame:
    source = pd.read_parquet(source_dir / "labels.parquet")
    retry = pd.read_parquet(retry_dir / "labels.parquet")
    errors = source[source["classification_status"].eq("error")]
    if set(retry["custom_id"]) != set(errors["custom_id"]):
        raise ValueError("Retry identities do not exactly match source errors.")
    hashes = [
        "study_contract_sha256", "provider_config_sha256",
        "input_row_sha256", "prompt_sha256",
    ]
    if not errors.set_index("custom_id")[hashes].sort_index().equals(
        retry.set_index("custom_id")[hashes].sort_index()
    ):
        raise ValueError("Retry hashes do not match source errors.")
    merged = pd.concat(
        [source[~source["custom_id"].isin(retry["custom_id"])], retry],
        ignore_index=True,
    ).sort_values(experiment.KEYS, kind="mergesort")
    if len(merged) != len(source) or merged.duplicated(experiment.KEYS).any():
        raise ValueError("Merged retry does not preserve the source population.")
    if not merged["classification_status"].eq("classified").all():
        raise ValueError("Retry still contains errors.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    return merged.reset_index(drop=True)


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
    for action in ("submit", "status", "collect"):
        command = commands.add_parser(action)
        command.add_argument("--output-dir", type=Path, required=True)
    command = commands.add_parser("prepare-retry")
    command.add_argument("--source-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command = commands.add_parser("merge-retry")
    command.add_argument("--source-dir", type=Path, required=True)
    command.add_argument("--retry-dir", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
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
    elif args.action == "prepare-retry":
        print(prepare_retry(args.source_dir, args.output_dir))
    else:
        print(f"Merged {len(merge_retry(args.source_dir, args.retry_dir, args.output))} labels")


if __name__ == "__main__":
    main()
