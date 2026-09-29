"""Run resumable per-PR RQ3 classification through a local Ollama server."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from analysis import run_qwen as ollama
from analysis.rq3_llm_validation import experiment

DEFAULT_MODEL = "qwen3.8:27b"
DEFAULT_URL = "http://127.0.0.1:11434"
NUM_CTX = 65_536
NUM_PREDICT = 4_096


def provider_config(model: str = DEFAULT_MODEL, url: str = DEFAULT_URL) -> dict[str, Any]:
    return {
        "provider_config_version": 1, "provider": "ollama", "model": model,
        "endpoint": url.rstrip("/") + "/api/chat", "stream": False, "think": True,
        "response_format": "json_schema_with_legacy_json_fallback",
        "options": {"temperature": 0, "seed": 42, "num_ctx": NUM_CTX,
                    "num_predict": NUM_PREDICT},
    }


def prepare(task: str, sample: Path, matches: Path, evidence_dir: Path,
            output_dir: Path, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL) -> Path:
    experiment.prepare_run(task, sample, matches, evidence_dir, output_dir,
                           provider_config(model, url))
    return output_dir


def _checkpoint(output_dir: Path, custom_id: str) -> Path:
    return output_dir / "responses" / f"{custom_id.replace(':', '-')}.json"


def _validate_label(content: str, task: str) -> tuple[Any, list[str]]:
    payload = json.loads(content)
    normalizations = []
    if task == "tradeoff_audit" and payload.get("label") in {"tradeoff", "joint_improvement"}:
        expected = "increased" if payload["label"] == "tradeoff" else "reduced"
        if payload.get("memory_direction") != expected:
            payload["memory_direction"] = expected
            normalizations.append("memory_direction_derived_from_label")
    return experiment.label_model(task).model_validate(payload), normalizations


def _collect(output_dir: Path, manifest: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in manifest.to_dict("records"):
        path = _checkpoint(output_dir, item["custom_id"])
        if not path.exists():
            continue
        checkpoint = json.loads(path.read_text())
        rows.append({**item, "model": metadata["provider_config"]["model"],
                     "classification_status": checkpoint["classification_status"],
                     "attempts": checkpoint["attempts"], "error": checkpoint.get("error"),
                     **(checkpoint.get("label") or {})})
    return pd.DataFrame(rows)


def run(task: str, sample: Path, matches: Path, evidence_dir: Path, output_dir: Path,
        model: str = DEFAULT_MODEL, url: str = DEFAULT_URL, max_attempts: int = 3,
        timeout: float = 900) -> pd.DataFrame:
    config = provider_config(model, url)
    if not (output_dir / "batch_manifest.parquet").exists():
        prepare(task, sample, matches, evidence_dir, output_dir, model, url)
    frame = experiment.build_input(sample, matches, evidence_dir, task).sort_values(experiment.KEYS)
    manifest = pd.read_parquet(output_dir / "batch_manifest.parquet")
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text())
    if experiment.sha256_file(output_dir / "batch_manifest.parquet") != metadata["manifest_sha256"]:
        raise ValueError("Prepared Qwen manifest changed.")
    if metadata["study_contract_sha256"] != manifest.study_contract_sha256.iloc[0]:
        raise ValueError("Prepared Qwen contract and manifest do not match.")
    if experiment.sha256_json(config) != metadata["provider_config_sha256"]:
        raise ValueError("Prepared Qwen provider configuration changed.")
    experiment.validate_prepared_inputs(
        frame, manifest, metadata, task, sample, matches, evidence_dir
    )
    rows = frame.assign(custom_id=lambda x: x.repo_id.astype(str) + ":" + x.number.astype(str)).set_index("custom_id")
    for item in manifest.to_dict("records"):
        path = _checkpoint(output_dir, item["custom_id"])
        prior = json.loads(path.read_text()) if path.exists() else {}
        if prior.get("classification_status") == "classified" or int(prior.get("attempts", 0)) >= max_attempts:
            continue
        prior_content = ((prior.get("response") or {}).get("message") or {}).get("content", "")
        if prior_content:
            try:
                recovered, normalizations = _validate_label(prior_content, task)
            except (ValueError, json.JSONDecodeError):
                pass
            else:
                prior.update(classification_status="classified", label=recovered.model_dump(),
                             error=None, revalidated=True, normalizations=normalizations)
                ollama.atomic_write_json(path, prior)
                continue
        attempts = int(prior.get("attempts", 0)) + 1
        response = None
        started = time.monotonic()
        try:
            prompt = experiment.prompt_for(rows.loc[item["custom_id"]].to_dict(), task)
            schema = experiment.semantic_schema(task)
            try:
                response = ollama.ollama_chat(
                    config["endpoint"], model, experiment.SYSTEM_INSTRUCTION, prompt,
                    schema, True, NUM_CTX, NUM_PREDICT, timeout,
                )
            except RuntimeError as error:
                if "cannot unmarshal object" not in str(error) or "ChatRequest.format" not in str(error):
                    raise
                legacy_prompt = prompt + "\n\nReturn only JSON matching this schema:\n" + json.dumps(schema)
                response = ollama.ollama_chat(
                    config["endpoint"], model, experiment.SYSTEM_INSTRUCTION, legacy_prompt,
                    "json", True, NUM_CTX, NUM_PREDICT, timeout,
                )
            label, normalizations = _validate_label(
                (response.get("message") or {}).get("content", ""), task
            )
        except Exception as error:
            value = {"custom_id": item["custom_id"], "classification_status": "error",
                     "attempts": attempts, "error": f"{type(error).__name__}: {error}"[:1000],
                     "response": response, "elapsed_seconds": time.monotonic() - started,
                     "completed_at": datetime.now(timezone.utc).isoformat()}
        else:
            value = {"custom_id": item["custom_id"], "classification_status": "classified",
                      "attempts": attempts, "label": label.model_dump(), "response": response,
                      "normalizations": normalizations,
                      "elapsed_seconds": time.monotonic() - started,
                      "completed_at": datetime.now(timezone.utc).isoformat()}
        ollama.atomic_write_json(path, value)
        experiment.atomic_write_json(output_dir / "run_state.json", {
            "completed_checkpoints": len(_collect(output_dir, manifest, metadata)),
            "requests": len(manifest),
        })
    result = _collect(output_dir, manifest, metadata).sort_values(experiment.KEYS).reset_index(drop=True)
    if len(result) != len(manifest):
        raise ValueError("Cannot finalize incomplete Qwen checkpoints.")
    result.to_parquet(output_dir / "labels.parquet", index=False)
    experiment.atomic_write_json(output_dir / "summary.json", {
        "rows": len(result), "status_counts": result.classification_status.value_counts().to_dict()
    })
    return result


def revalidate(task: str, output_dir: Path) -> pd.DataFrame:
    manifest = pd.read_parquet(output_dir / "batch_manifest.parquet")
    metadata = json.loads((output_dir / "prepare_metadata.json").read_text())
    for item in manifest.to_dict("records"):
        path = _checkpoint(output_dir, item["custom_id"])
        checkpoint = json.loads(path.read_text())
        if checkpoint.get("classification_status") == "classified":
            continue
        content = ((checkpoint.get("response") or {}).get("message") or {}).get("content", "")
        try:
            label, normalizations = _validate_label(content, task)
        except (ValueError, json.JSONDecodeError):
            continue
        checkpoint.update(classification_status="classified", label=label.model_dump(), error=None,
                          revalidated=True, normalizations=normalizations)
        ollama.atomic_write_json(path, checkpoint)
    result = _collect(output_dir, manifest, metadata).sort_values(experiment.KEYS).reset_index(drop=True)
    result.to_parquet(output_dir / "labels.parquet", index=False)
    experiment.atomic_write_json(output_dir / "summary.json", {
        "rows": len(result), "status_counts": result.classification_status.value_counts().to_dict()
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--task", choices=experiment.TASKS, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--matches", type=Path, default=experiment.DEFAULT_MATCHES)
    parser.add_argument("--evidence-dir", type=Path, default=experiment.DEFAULT_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_URL)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = (args.task, args.sample, args.matches, args.evidence_dir, args.output_dir,
              args.model, args.ollama_url)
    if args.action == "prepare":
        print(prepare(*values))
    else:
        print(f"Finalized {len(run(*values, args.max_attempts, args.timeout))} labels")


if __name__ == "__main__":
    main()
