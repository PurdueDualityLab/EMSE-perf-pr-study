"""Run resumable RQ1 and RQ2 classification through a local Ollama server."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import signal
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.rq1_optimization_patterns import run_rq1  # noqa: E402
from analysis.rq2_validation import run_rq2  # noqa: E402

DEFAULT_MODEL = "qwen3.8:27b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
STOP_REQUESTED = False


@dataclass(frozen=True)
class Study:
    name: str
    module: Any
    sample_path: Path
    evidence_dir: Path
    output_dir: Path
    catalog_path: Path | None = None


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def provider_config(
    model: str,
    ollama_url: str,
    schema: dict[str, Any],
    think: bool,
    num_ctx: int,
    num_predict: int = 4_096,
) -> dict[str, Any]:
    return {
        "provider_config_version": 1,
        "provider": "ollama",
        "model": model,
        "endpoint": ollama_url.rstrip("/") + "/api/chat",
        "stream": False,
        "think": think,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "response_schema_sha256": run_rq1.sha256_json(schema),
    }


def bounded_rq2_retry_schema(schema: dict[str, Any]) -> dict[str, Any]:
    bounded = copy.deepcopy(schema)
    properties = bounded["properties"]
    properties["validation_types"]["maxItems"] = len(run_rq2.VALIDATION_TYPES)
    properties["evidence_sources"]["maxItems"] = len(run_rq2.EVIDENCE_SOURCES)
    return bounded


def ollama_chat(
    endpoint: str,
    model: str,
    system_instruction: str,
    prompt: str,
    schema: dict[str, Any],
    think: bool,
    num_ctx: int,
    num_predict: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        "format": schema,
        "stream": False,
        "think": think,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {error.code}: {detail[:500]}") from error
    if body.get("error"):
        raise RuntimeError(f"Ollama error: {body['error']}")
    if not body.get("done"):
        raise RuntimeError("Ollama returned an incomplete response.")
    content = (body.get("message") or {}).get("content")
    if not content:
        raise RuntimeError("Ollama response has no assistant content.")
    return body


def _prepare_rq1(study: Study, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    assert study.catalog_path is not None
    catalog_snapshot = study.output_dir / "optimization_catalog.csv"
    if not catalog_snapshot.exists():
        shutil.copyfile(study.catalog_path, catalog_snapshot)
    taxonomy = run_rq1.load_taxonomy(catalog_snapshot)
    frame = run_rq1.build_input(study.sample_path, study.evidence_dir).sort_values(
        ["repo_id", "number"], kind="mergesort"
    )
    contract = run_rq1.study_contract(
        run_rq1.sha256_file(study.sample_path),
        run_rq1.sha256_file(catalog_snapshot),
        run_rq1.sha256_file(study.evidence_dir / "pull_request_files.parquet"),
        run_rq1.sha256_file(study.evidence_dir / "collection_status.parquet"),
    )
    contract_hash = run_rq1.sha256_json(contract)
    manifest_rows = []
    for row in frame.to_dict("records"):
        prompt = run_rq1.prompt_for(row, taxonomy)
        manifest_rows.append(
            {
                "repo_id": int(row["repo_id"]),
                "number": int(row["number"]),
                "custom_id": f"{int(row['repo_id'])}:{int(row['number'])}",
                "repo_full_name": row["repo_full_name"],
                "html_url": row["html_url"],
                "sample_arm": row["sample_arm"],
                "prompt_version": run_rq1.PROMPT_VERSION,
                "evidence_status": row["evidence_status"],
                "files_observed": int(row["files_observed"]),
                "patches_available": int(row["patches_available"]),
                "prompt_chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_row_sha256": run_rq1.input_row_sha256(row),
                "study_contract_sha256": contract_hash,
                "provider_config_sha256": run_rq1.sha256_json(config),
            }
        )
    metadata = {
        "study": study.name,
        "model": config["model"],
        "prompt_version": run_rq1.PROMPT_VERSION,
        "requests": len(manifest_rows),
        "sample_sha256": run_rq1.sha256_file(study.sample_path),
        "catalog_file": catalog_snapshot.name,
        "catalog_sha256": run_rq1.sha256_file(catalog_snapshot),
        "evidence_files_sha256": run_rq1.sha256_file(study.evidence_dir / "pull_request_files.parquet"),
        "evidence_status_sha256": run_rq1.sha256_file(study.evidence_dir / "collection_status.parquet"),
        "study_contract": contract,
        "study_contract_sha256": contract_hash,
        "provider_config": config,
        "provider_config_sha256": run_rq1.sha256_json(config),
    }
    return frame, pd.DataFrame(manifest_rows), metadata


def _prepare_rq2(study: Study, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = run_rq2.build_input(study.sample_path, study.evidence_dir).sort_values(
        ["repo_id", "number"], kind="mergesort"
    )
    incomplete = frame.loc[~frame["evidence_complete"], ["repo_id", "number"]]
    if not incomplete.empty:
        raise ValueError(f"RQ2 evidence is incomplete for {len(incomplete)} sample rows.")
    evidence_files = (
        "pull_requests.parquet", "pull_request_files.parquet", "issue_comments.parquet",
        "review_comments.parquet", "reviews.parquet", "workflow_runs.parquet",
        "check_runs.parquet", "collection_status.parquet",
    )
    evidence_hashes = {
        name: run_rq2.sha256_file(study.evidence_dir / name) for name in evidence_files
    }
    contract = run_rq2.study_contract(run_rq2.sha256_file(study.sample_path), evidence_hashes)
    contract_hash = run_rq2.sha256_json(contract)
    manifest_rows = []
    for row in frame.to_dict("records"):
        prompt = run_rq2.prompt_for(row)
        manifest_rows.append(
            {
                "repo_id": int(row["repo_id"]),
                "number": int(row["number"]),
                "custom_id": f"{int(row['repo_id'])}:{int(row['number'])}",
                "repo_full_name": row["repo_full_name"],
                "html_url": row["html_url"],
                "sample_arm": row["sample_arm"],
                "prompt_version": run_rq2.PROMPT_VERSION,
                "evidence_status": row["evidence_status"],
                "evidence_complete": bool(row["evidence_complete"]),
                "prompt_chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_row_sha256": run_rq2.input_row_sha256(row),
                "study_contract_sha256": contract_hash,
                "provider_config_sha256": run_rq2.sha256_json(config),
            }
        )
    metadata = {
        "study": study.name,
        "model": config["model"],
        "prompt_version": run_rq2.PROMPT_VERSION,
        "requests": len(manifest_rows),
        "sample_sha256": run_rq2.sha256_file(study.sample_path),
        "evidence_sha256": evidence_hashes,
        "study_contract": contract,
        "study_contract_sha256": contract_hash,
        "provider_config": config,
        "provider_config_sha256": run_rq2.sha256_json(config),
    }
    return frame, pd.DataFrame(manifest_rows), metadata


def prepare_study(study: Study, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    study.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_rq1(study, config) if study.name == "rq1" else _prepare_rq2(study, config)
    frame, expected_manifest, metadata = prepared
    manifest_path = study.output_dir / "batch_manifest.parquet"
    metadata_path = study.output_dir / "prepare_metadata.json"
    if manifest_path.exists() != metadata_path.exists():
        raise ValueError(f"Incomplete preparation artifacts in {study.output_dir}.")
    if manifest_path.exists():
        manifest = pd.read_parquet(manifest_path)
        columns = [
            "custom_id", "prompt_sha256", "input_row_sha256",
            "study_contract_sha256", "provider_config_sha256",
        ]
        if manifest[columns].to_dict("records") != expected_manifest[columns].to_dict("records"):
            raise ValueError(f"Prepared {study.name} manifest does not match current inputs or configuration.")
        stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored["study_contract_sha256"] != metadata["study_contract_sha256"]:
            raise ValueError(f"Prepared {study.name} study contract changed.")
        metadata = stored
    else:
        atomic_write_parquet(manifest_path, expected_manifest)
        metadata["manifest_sha256"] = run_rq1.sha256_file(manifest_path)
        atomic_write_json(metadata_path, metadata)
        manifest = expected_manifest
    if run_rq1.sha256_file(manifest_path) != metadata["manifest_sha256"]:
        raise ValueError(f"Prepared {study.name} manifest hash mismatch.")
    return frame, manifest, metadata


def _validated_label(study: Study, content: str, taxonomy: dict[str, list[str]] | None) -> BaseModel:
    if study.name == "rq1":
        assert taxonomy is not None
        return pattern_label_with_adjudication(content, taxonomy)[0]
    return run_rq2.ValidationLabel.model_validate_json(content)


def pattern_label_with_adjudication(
    content: str, taxonomy: dict[str, list[str]]
) -> tuple[run_rq1.PatternLabel, dict[str, str] | None]:
    label = run_rq1.PatternLabel.model_validate_json(content)
    if (
        label.high_level_pattern in taxonomy
        and label.sub_pattern in taxonomy[label.high_level_pattern]
    ):
        return label, None
    candidates = [
        (parent, sub_pattern)
        for parent, sub_patterns in taxonomy.items()
        for sub_pattern in sub_patterns
        if sub_pattern == label.sub_pattern
        or sub_pattern.rstrip(")") == label.sub_pattern.rstrip(")")
        or sub_pattern.startswith(label.sub_pattern)
    ]
    if len(candidates) != 1:
        if label.high_level_pattern not in taxonomy:
            raise ValueError(f"Unknown high-level pattern: {label.high_level_pattern}")
        raise ValueError(
            f"Invalid sub-pattern for {label.high_level_pattern}: {label.sub_pattern}"
        )
    parent, sub_pattern = candidates[0]
    corrected = label.model_copy(
        update={"high_level_pattern": parent, "sub_pattern": sub_pattern}
    )
    return corrected, {
        "original_high_level_pattern": label.high_level_pattern,
        "original_sub_pattern": label.sub_pattern,
        "corrected_high_level_pattern": parent,
        "corrected_sub_pattern": sub_pattern,
        "adjudication_rule": "Preserve the model sub-pattern and use its unique catalog parent.",
        "authorization": "Study owner approved deterministic unique-parent adjudication.",
    }


def validation_label_with_normalization(
    content: str,
) -> tuple[run_rq2.ValidationLabel, dict[str, Any] | None]:
    value = json.loads(content)
    changes = {}
    for field in ("validation_types", "evidence_sources"):
        original = value.get(field)
        if not isinstance(original, list):
            continue
        normalized = list(dict.fromkeys(original))
        if normalized != original:
            changes[field] = {"original": original, "normalized": normalized}
            value[field] = normalized
    label = run_rq2.ValidationLabel.model_validate(value)
    if not changes:
        return label, None
    return label, {
        "changes": changes,
        "normalization_rule": "Remove duplicate set-like labels while preserving first-seen order.",
        "authorization": "Study owner approved semantics-preserving duplicate normalization.",
    }


def _checkpoint_path(study: Study, custom_id: str) -> Path:
    return study.output_dir / "responses" / f"{custom_id.replace(':', '-')}.json"


def record_adjudication(output_dir: Path, custom_id: str, value: dict[str, str]) -> None:
    path = output_dir / "taxonomy_adjudications.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    by_id = {row["custom_id"]: row for row in rows}
    by_id[custom_id] = {"custom_id": custom_id, **value}
    atomic_write_json(path, [by_id[key] for key in sorted(by_id)])


def record_normalization(output_dir: Path, custom_id: str, value: dict[str, Any]) -> None:
    path = output_dir / "label_normalizations.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    by_id = {row["custom_id"]: row for row in rows}
    by_id[custom_id] = {"custom_id": custom_id, **value}
    atomic_write_json(path, [by_id[key] for key in sorted(by_id)])


def adjudicate_error_checkpoints(
    study: Study,
    manifest: pd.DataFrame,
    taxonomy: dict[str, list[str]],
) -> int:
    corrected = 0
    for custom_id in manifest["custom_id"]:
        path = _checkpoint_path(study, custom_id)
        if not path.exists():
            continue
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("classification_status") != "error" or not checkpoint.get("response"):
            continue
        content = (checkpoint["response"].get("message") or {}).get("content", "")
        try:
            label, adjudication = pattern_label_with_adjudication(content, taxonomy)
        except ValueError:
            continue
        if adjudication is None:
            continue
        checkpoint["original_error"] = checkpoint.get("error")
        checkpoint["error"] = None
        checkpoint["classification_status"] = "classified"
        checkpoint["label"] = label.model_dump()
        checkpoint["taxonomy_adjudication"] = adjudication
        atomic_write_json(path, checkpoint)
        record_adjudication(study.output_dir, custom_id, adjudication)
        corrected += 1
    return corrected


def normalize_rq2_error_checkpoints(study: Study, manifest: pd.DataFrame) -> int:
    corrected = 0
    for custom_id in manifest["custom_id"]:
        path = _checkpoint_path(study, custom_id)
        if not path.exists():
            continue
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("classification_status") != "error" or not checkpoint.get("response"):
            continue
        content = (checkpoint["response"].get("message") or {}).get("content", "")
        try:
            label, normalization = validation_label_with_normalization(content)
        except (ValueError, json.JSONDecodeError):
            continue
        if normalization is None:
            continue
        checkpoint["original_error"] = checkpoint.get("error")
        checkpoint["error"] = None
        checkpoint["classification_status"] = "classified"
        checkpoint["label"] = label.model_dump()
        checkpoint["label_normalization"] = normalization
        atomic_write_json(path, checkpoint)
        record_normalization(study.output_dir, custom_id, normalization)
        corrected += 1
    return corrected


def collect_checkpoints(study: Study, manifest: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in manifest.to_dict("records"):
        path = _checkpoint_path(study, item["custom_id"])
        if not path.exists():
            continue
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        row = {
            **item,
            "model": metadata["model"],
            "response_id": None,
            "classification_status": checkpoint["classification_status"],
            "attempts": checkpoint["attempts"],
            "error": checkpoint.get("error"),
            "response_provider_config_sha256": checkpoint.get(
                "response_provider_config_sha256", item["provider_config_sha256"]
            ),
        }
        response = checkpoint.get("response") or {}
        row.update(
            input_tokens=response.get("prompt_eval_count"),
            output_tokens=response.get("eval_count"),
            total_tokens=(response.get("prompt_eval_count") or 0) + (response.get("eval_count") or 0),
            cached_input_tokens=None,
            reasoning_tokens=None,
            total_duration_ns=response.get("total_duration"),
            load_duration_ns=response.get("load_duration"),
            prompt_eval_duration_ns=response.get("prompt_eval_duration"),
            eval_duration_ns=response.get("eval_duration"),
        )
        row.update(checkpoint.get("label") or {})
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(study: Study, manifest: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    result = collect_checkpoints(study, manifest, metadata).sort_values(
        ["repo_id", "number"], kind="mergesort"
    ).reset_index(drop=True)
    if len(result) != len(manifest) or result["custom_id"].duplicated().any():
        raise ValueError(f"Cannot finalize incomplete or duplicate {study.name} checkpoints.")
    output_name = "optimization_pattern_labels.parquet" if study.name == "rq1" else "validation_labels.parquet"
    atomic_write_parquet(study.output_dir / output_name, result)
    summary = {
        "method": metadata["prompt_version"],
        "model": metadata["model"],
        "rows": len(result),
        "status_counts": result["classification_status"].value_counts().to_dict(),
        "arm_counts": result["sample_arm"].value_counts().to_dict(),
        "input_tokens": int(result["input_tokens"].fillna(0).sum()),
        "output_tokens": int(result["output_tokens"].fillna(0).sum()),
        "total_tokens": int(result["total_tokens"].fillna(0).sum()),
    }
    atomic_write_json(study.output_dir / "summary.json", summary)
    return result


def select_manifest(manifest: pd.DataFrame, limit_per_arm: int | None) -> pd.DataFrame:
    if limit_per_arm is None:
        return manifest
    selected = (
        manifest.sort_values(["sample_arm", "repo_id", "number"], kind="mergesort")
        .groupby("sample_arm", sort=True, group_keys=False)
        .head(limit_per_arm)
    )
    expected = manifest["sample_arm"].nunique() * limit_per_arm
    if len(selected) != expected:
        raise ValueError("Smoke selection does not contain the requested rows for every sample arm.")
    return selected.sort_values(["repo_id", "number"], kind="mergesort").reset_index(drop=True)


def run_study(study: Study, args: argparse.Namespace) -> bool:
    schema = study.module.semantic_schema()
    config = provider_config(
        args.model, args.ollama_url, schema, args.think, args.num_ctx, args.num_predict
    )
    frame, manifest, metadata = prepare_study(study, config)
    run_manifest = select_manifest(manifest, args.limit_per_arm)
    retry_config = None
    request_schema = schema
    if args.retry_errors_num_predict is not None:
        error_ids = []
        for custom_id in manifest["custom_id"]:
            path = _checkpoint_path(study, custom_id)
            if not path.exists():
                continue
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            if checkpoint.get("classification_status") == "error":
                error_ids.append(custom_id)
        run_manifest = manifest[manifest["custom_id"].isin(error_ids)].copy()
        if study.name == "rq2" and args.bound_rq2_retry_lists:
            request_schema = bounded_rq2_retry_schema(schema)
        retry_config = provider_config(
            args.model,
            args.ollama_url,
            request_schema,
            args.think,
            args.num_ctx,
            args.retry_errors_num_predict,
        )
        atomic_write_json(
            study.output_dir / "retry_provider_config.json",
            {
                "errors_selected": len(run_manifest),
                "bounded_rq2_lists": request_schema != schema,
                "provider_config": retry_config,
                "provider_config_sha256": run_rq1.sha256_json(retry_config),
            },
        )
    if args.limit_per_arm is not None:
        atomic_write_json(
            study.output_dir / "smoke_metadata.json",
            {
                "limit_per_arm": args.limit_per_arm,
                "requests": len(run_manifest),
                "custom_ids": run_manifest["custom_id"].tolist(),
                "full_manifest_requests": len(manifest),
            },
        )
    rows = frame.assign(
        custom_id=lambda value: value["repo_id"].astype(str) + ":" + value["number"].astype(str)
    ).set_index("custom_id").to_dict("index")
    taxonomy = None
    if study.name == "rq1":
        taxonomy_frame = run_rq1.load_taxonomy(study.output_dir / metadata["catalog_file"])
        taxonomy = run_rq1.taxonomy_labels(taxonomy_frame)
        if args.adjudicate_unique_parents:
            adjudicate_error_checkpoints(study, manifest, taxonomy)
    elif args.normalize_duplicate_labels:
        normalize_rq2_error_checkpoints(study, manifest)
    consecutive_errors = 0
    for item in run_manifest.to_dict("records"):
        if STOP_REQUESTED:
            return False
        checkpoint_path = _checkpoint_path(study, item["custom_id"])
        checkpoint = {}
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("classification_status") == "classified":
                continue
            if (
                args.defer_length_errors
                and args.retry_errors_num_predict is None
                and (checkpoint.get("response") or {}).get("done_reason") == "length"
            ):
                continue
            if int(checkpoint.get("attempts", 0)) >= args.max_attempts:
                continue
        row = rows[item["custom_id"]]
        prompt = study.module.prompt_for(row, taxonomy_frame) if study.name == "rq1" else study.module.prompt_for(row)
        attempts = int(checkpoint.get("attempts", 0)) + 1
        if checkpoint and args.retry_errors_num_predict is not None:
            history_path = (
                study.output_dir / "response_history"
                / f"{item['custom_id'].replace(':', '-')}-attempt-{attempts - 1}.json"
            )
            if not history_path.exists():
                atomic_write_json(history_path, checkpoint)
        started = time.monotonic()
        response = None
        request_config = retry_config or config
        try:
            response = ollama_chat(
                config["endpoint"], args.model, study.module.SYSTEM_INSTRUCTION,
                prompt,
                request_schema,
                args.think,
                args.num_ctx,
                request_config["options"]["num_predict"],
                args.timeout,
            )
            content = (response.get("message") or {}).get("content", "")
            if study.name == "rq1" and args.adjudicate_unique_parents:
                assert taxonomy is not None
                label, adjudication = pattern_label_with_adjudication(content, taxonomy)
                normalization = None
            elif study.name == "rq2" and args.normalize_duplicate_labels:
                label, normalization = validation_label_with_normalization(content)
                adjudication = None
            else:
                label = _validated_label(study, content, taxonomy)
                adjudication = None
                normalization = None
        except Exception as error:
            atomic_write_json(
                checkpoint_path,
                {
                    "custom_id": item["custom_id"],
                    "classification_status": "error",
                    "attempts": attempts,
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(error).__name__}: {error}"[:1_000],
                    "response": response,
                    "response_provider_config_sha256": run_rq1.sha256_json(request_config),
                },
            )
            consecutive_errors += 1
            if consecutive_errors >= args.max_consecutive_errors:
                raise RuntimeError(f"Stopping after {consecutive_errors} consecutive Ollama errors.") from error
        else:
            checkpoint_value = {
                    "custom_id": item["custom_id"],
                    "classification_status": "classified",
                    "attempts": attempts,
                    "elapsed_seconds": time.monotonic() - started,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "label": label.model_dump(),
                    "response": response,
                    "response_provider_config_sha256": run_rq1.sha256_json(request_config),
                }
            if adjudication is not None:
                checkpoint_value["taxonomy_adjudication"] = adjudication
                record_adjudication(study.output_dir, item["custom_id"], adjudication)
            if normalization is not None:
                checkpoint_value["label_normalization"] = normalization
                record_normalization(study.output_dir, item["custom_id"], normalization)
            atomic_write_json(checkpoint_path, checkpoint_value)
            consecutive_errors = 0
        completed = len(collect_checkpoints(study, run_manifest, metadata))
        atomic_write_json(
            study.output_dir / "run_state.json",
            {"study": study.name, "completed_checkpoints": completed, "requests": len(run_manifest)},
        )
    write_outputs(study, manifest if args.retry_errors_num_predict is not None else run_manifest, metadata)
    return True


def _handle_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; stopping after the current request.", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("rq1", "rq2"), nargs="+", default=["rq1", "rq2"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "mining/sample_evidence/final")
    parser.add_argument("--rq1-sample", type=Path, default=ROOT / "data/data/sample/balanced_sample.parquet")
    parser.add_argument("--rq1-catalog", type=Path, default=ROOT / "analysis/rq1_optimization_patterns/catalog/updated_optimization_catalog.csv")
    parser.add_argument("--rq2-sample", type=Path, default=ROOT / "analysis/rq2_validation/sample/balanced_sample.parquet")
    parser.add_argument("--num-ctx", type=int, default=65_536)
    parser.add_argument("--num-predict", type=int, default=4_096)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-consecutive-errors", type=int, default=5)
    parser.add_argument("--limit-per-arm", type=int)
    parser.add_argument("--adjudicate-unique-parents", action="store_true")
    parser.add_argument("--normalize-duplicate-labels", action="store_true")
    parser.add_argument("--defer-length-errors", action="store_true")
    parser.add_argument("--retry-errors-num-predict", type=int)
    parser.add_argument("--bound-rq2-retry-lists", action="store_true")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if (
        args.max_attempts < 1
        or args.max_consecutive_errors < 1
        or args.num_ctx < 1
        or args.num_predict < 1
        or (args.retry_errors_num_predict is not None and args.retry_errors_num_predict < 1)
        or args.timeout <= 0
        or (args.limit_per_arm is not None and args.limit_per_arm < 1)
    ):
        parser.error("attempt limits, num-ctx, and timeout must be positive")
    if args.bound_rq2_retry_lists and args.retry_errors_num_predict is None:
        parser.error("--bound-rq2-retry-lists requires --retry-errors-num-predict")
    return args


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGUSR1, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)
    studies = {
        "rq1": Study(
            "rq1", run_rq1, args.rq1_sample, args.evidence_dir,
            args.output_root / "rq1", args.rq1_catalog,
        ),
        "rq2": Study(
            "rq2", run_rq2, args.rq2_sample, args.evidence_dir,
            args.output_root / "rq2",
        ),
    }
    for name in args.study:
        if not run_study(studies[name], args):
            return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
