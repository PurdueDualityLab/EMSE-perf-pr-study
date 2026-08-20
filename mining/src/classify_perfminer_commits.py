from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parquet_parts import (
    finalize_parts,
    part_path,
    schema_sha256,
    sha256_file,
    validate_part,
    write_json,
    write_table_part,
)
from perfannotator_common import (
    configure_reproducibility,
    load_model,
    model_artifact_sha256,
)
from perfminer_reproduction import (
    PERFMINER_MAX_LENGTH,
    PERFMINER_MODEL_ARTIFACT_SHA256,
    PERFMINER_MODEL_WEIGHTS_SHA256,
    PERFMINER_REVISION,
    PERFMINER_THRESHOLD,
    PERFMINER_TRUNCATION,
    classify_loaded_pairs,
    pair_token_metadata,
    verify_figshare_model,
)


METHOD_NAME = "perfannotator_mini_ease_operational_v1"
METHOD_VERSION = 1

PREDICTION_FIELDS = (
    pa.field("perfminer_classification_status", pa.string()),
    pa.field("perfminer_classification_error", pa.string()),
    pa.field("perfminer_label_id", pa.int64()),
    pa.field("perfminer_is_performance", pa.bool_()),
    pa.field("perfminer_score", pa.float64()),
    pa.field("perfminer_input_tokens", pa.int64()),
    pa.field("perfminer_context_fits", pa.bool_()),
    pa.field("perfminer_message_fits", pa.bool_()),
    pa.field("perfminer_model_weights_sha256", pa.string()),
    pa.field("perfminer_model_artifact_sha256", pa.string()),
    pa.field("perfminer_device", pa.string()),
    pa.field("perfminer_inference_batch_size", pa.int64()),
    pa.field("perfminer_max_length", pa.int64()),
    pa.field("perfminer_truncation", pa.string()),
    pa.field("perfminer_threshold", pa.float64()),
)
PREDICTION_COLUMNS = frozenset(field.name for field in PREDICTION_FIELDS)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def seed_integer(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed < 2**32:
        raise argparse.ArgumentTypeError(f"seed must be between 0 and {2**32 - 1}")
    return parsed


def prediction_schema(input_schema: pa.Schema) -> pa.Schema:
    required = {"commit_message", "code_diff", "perfminer_evidence_status"}
    missing = sorted(required - set(input_schema.names))
    if missing:
        raise ValueError("Commit evidence is missing columns: " + ", ".join(missing))
    collisions = sorted(PREDICTION_COLUMNS.intersection(input_schema.names))
    if collisions:
        raise ValueError("Input already contains PerfMiner prediction columns: " + ", ".join(collisions))
    return pa.schema(list(input_schema) + list(PREDICTION_FIELDS), metadata=input_schema.metadata)


def runtime_fingerprint() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "torch", "transformers", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "packages": packages,
    }


def expected_part_rows(total_rows: int, row_batch_size: int) -> dict[int, int]:
    return {
        index: min(row_batch_size, total_rows - (index - 1) * row_batch_size)
        for index in range(1, (total_rows + row_batch_size - 1) // row_batch_size + 1)
    }


def add_predictions(
    frame: pd.DataFrame,
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    batch_size: int,
    model_weights_sha256: str,
    model_sha256: str,
) -> pd.DataFrame:
    rows = len(frame)
    statuses = ["not_eligible"] * rows
    errors = [""] * rows
    labels: list[int | None] = [None] * rows
    is_performance: list[bool | None] = [None] * rows
    scores: list[float | None] = [None] * rows
    input_tokens: list[int | None] = [None] * rows
    context_fits: list[bool | None] = [None] * rows
    message_fits: list[bool | None] = [None] * rows
    valid_positions: list[int] = []
    valid_messages: list[str] = []
    valid_diffs: list[str] = []

    for position, (_, row) in enumerate(frame.iterrows()):
        if str(row.get("perfminer_evidence_status") or "") != "eligible":
            continue
        commit_message = str(row.get("commit_message") or "")
        code_diff = str(row.get("code_diff") or "")
        try:
            metadata = pair_token_metadata(tokenizer, commit_message, code_diff)
        except Exception as error:
            statuses[position] = "tokenization_error"
            errors[position] = f"{type(error).__name__}: {error}"[:1000]
            continue
        input_tokens[position] = int(metadata["input_tokens"])
        context_fits[position] = bool(metadata["context_fits"])
        message_fits[position] = bool(metadata["message_fits"])
        if not metadata["message_fits"]:
            statuses[position] = "message_exceeds_pair_budget"
            continue
        valid_positions.append(position)
        valid_messages.append(commit_message)
        valid_diffs.append(code_diff)

    if valid_positions:
        valid_labels, valid_scores = classify_loaded_pairs(
            valid_messages,
            valid_diffs,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
        )
        if len(valid_labels) != len(valid_positions) or len(valid_scores) != len(valid_positions):
            raise ValueError("PerfAnnotator returned a different number of predictions than inputs.")
        for position, label, score in zip(
            valid_positions, valid_labels, valid_scores, strict=True
        ):
            statuses[position] = "classified"
            labels[position] = int(label)
            is_performance[position] = int(label) == 1
            scores[position] = float(score)

    result = frame.copy()
    result["perfminer_classification_status"] = statuses
    result["perfminer_classification_error"] = errors
    result["perfminer_label_id"] = labels
    result["perfminer_is_performance"] = is_performance
    result["perfminer_score"] = scores
    result["perfminer_input_tokens"] = input_tokens
    result["perfminer_context_fits"] = context_fits
    result["perfminer_message_fits"] = message_fits
    result["perfminer_model_weights_sha256"] = model_weights_sha256
    result["perfminer_model_artifact_sha256"] = model_sha256
    result["perfminer_device"] = device
    result["perfminer_inference_batch_size"] = 1
    result["perfminer_max_length"] = PERFMINER_MAX_LENGTH
    result["perfminer_truncation"] = PERFMINER_TRUNCATION
    result["perfminer_threshold"] = PERFMINER_THRESHOLD
    return result


def build_run_signature(
    args: argparse.Namespace,
    source: pq.ParquetFile,
    weights_sha256: str,
    model_sha256: str,
) -> dict[str, object]:
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "perfminer_revision": PERFMINER_REVISION,
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "input_row_count": int(source.metadata.num_rows),
        "input_schema_sha256": schema_sha256(source.schema_arrow),
        "model_dir": str(args.model_dir.resolve()),
        "model_weights_sha256": weights_sha256,
        "expected_model_weights_sha256": PERFMINER_MODEL_WEIGHTS_SHA256,
        "model_artifact_sha256": model_sha256,
        "device": args.device,
        "batch_size": args.batch_size,
        "row_batch_size": args.row_batch_size,
        "seed": args.seed,
        "max_length": PERFMINER_MAX_LENGTH,
        "truncation": PERFMINER_TRUNCATION,
        "threshold": PERFMINER_THRESHOLD,
        "runtime": runtime_fingerprint(),
        "code_sha256": {
            Path(__file__).name: sha256_file(Path(__file__)),
            "perfminer_reproduction.py": sha256_file(
                Path(__file__).with_name("perfminer_reproduction.py")
            ),
            "perfannotator_common.py": sha256_file(
                Path(__file__).with_name("perfannotator_common.py")
            ),
            "parquet_parts.py": sha256_file(Path(__file__).with_name("parquet_parts.py")),
        },
    }


def validate_recorded_part(
    metadata: Mapping[str, object],
    schema: pa.Schema,
    expected_rows: int,
) -> Path:
    path = Path(str(metadata.get("path") or ""))
    validate_part(
        path,
        schema,
        expected_rows=expected_rows,
        expected_sha256=str(metadata.get("sha256") or ""),
    )
    return path


def summarize(output: Path) -> dict[str, object]:
    table = pq.read_table(
        output,
        columns=["perfminer_classification_status", "perfminer_is_performance"],
    )
    rows = table.to_pylist()
    statuses = Counter(row["perfminer_classification_status"] for row in rows)
    positives = sum(row["perfminer_is_performance"] is True for row in rows)
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "row_count": len(rows),
        "classification_status_counts": dict(sorted(statuses.items())),
        "positive_commit_count": positives,
        "output": {"path": str(output.resolve()), "sha256": sha256_file(output)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify eligible commits with the published EASE PerfAnnotator model."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=positive_integer, default=1)
    parser.add_argument("--row-batch-size", type=positive_integer, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=seed_integer, default=20260720)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("the operational PerfMiner reproduction requires --batch-size 1")

    source = pq.ParquetFile(args.input)
    output_schema = prediction_schema(source.schema_arrow)
    total_rows = int(source.metadata.num_rows)
    weights_sha256 = verify_figshare_model(args.model_dir)
    model_sha256 = model_artifact_sha256(args.model_dir)
    if model_sha256 != PERFMINER_MODEL_ARTIFACT_SHA256:
        raise ValueError(
            "PerfAnnotator model directory does not match the complete EASE 2026 Figshare artifact."
        )
    signature = build_run_signature(args, source, weights_sha256, model_sha256)
    expected_parts = expected_part_rows(total_rows, args.row_batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output_dir / "parts" / "commit_predictions"
    state_file = args.output_dir / "commit_predictions.state.json"
    summary_file = args.output_dir / "commit_predictions.summary.json"
    output = args.output_dir / "commit_predictions.parquet"
    has_artifacts = any(path.exists() for path in (state_file, summary_file, output)) or parts_dir.exists()
    if has_artifacts and not args.resume:
        raise FileExistsError(f"Commit-prediction artifacts already exist in {args.output_dir}; use --resume.")
    if args.resume and not state_file.exists() and has_artifacts:
        raise ValueError("Cannot resume commit-prediction artifacts without matching state.")

    if state_file.exists():
        state: dict[str, Any] = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("run_signature") != signature:
            raise ValueError("Existing commit-prediction run signature does not match the request.")
    else:
        state = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "run_signature": signature,
            "completed_parts": {},
            "finalized": False,
        }
        write_json(state_file, state)

    completed_parts = state.setdefault("completed_parts", {})
    for index, expected_rows in expected_parts.items():
        metadata = completed_parts.get(str(index))
        if isinstance(metadata, Mapping):
            validate_recorded_part(metadata, output_schema, expected_rows)

    tokenizer = classifier_model = None
    resolved_device = args.device
    remaining = total_rows
    for index, batch in enumerate(source.iter_batches(batch_size=args.row_batch_size), start=1):
        if remaining <= 0:
            break
        expected_rows = expected_parts[index]
        if batch.num_rows != expected_rows:
            raise ValueError(f"Input batch {index} has {batch.num_rows} rows; expected {expected_rows}.")
        remaining -= batch.num_rows
        if str(index) in completed_parts:
            continue
        frame = pa.Table.from_batches([batch]).to_pandas()
        if frame["perfminer_evidence_status"].eq("eligible").any() and tokenizer is None:
            configure_reproducibility(args.seed, True)
            tokenizer, classifier_model, resolved_device = load_model(
                str(args.model_dir), None, args.device
            )
        if tokenizer is None:
            # No eligible row needs tokenization or model inference in this part.
            result = frame.copy()
            for field in PREDICTION_FIELDS:
                if field.name == "perfminer_classification_status":
                    result[field.name] = "not_eligible"
                elif field.name == "perfminer_classification_error":
                    result[field.name] = ""
                elif field.name == "perfminer_model_weights_sha256":
                    result[field.name] = weights_sha256
                elif field.name == "perfminer_model_artifact_sha256":
                    result[field.name] = model_sha256
                elif field.name == "perfminer_device":
                    result[field.name] = resolved_device
                elif field.name == "perfminer_inference_batch_size":
                    result[field.name] = 1
                elif field.name == "perfminer_max_length":
                    result[field.name] = PERFMINER_MAX_LENGTH
                elif field.name == "perfminer_truncation":
                    result[field.name] = PERFMINER_TRUNCATION
                elif field.name == "perfminer_threshold":
                    result[field.name] = PERFMINER_THRESHOLD
                else:
                    result[field.name] = None
        else:
            result = add_predictions(
                frame,
                tokenizer=tokenizer,
                model=classifier_model,
                device=resolved_device,
                batch_size=args.batch_size,
                model_weights_sha256=weights_sha256,
                model_sha256=model_sha256,
            )
        table = pa.Table.from_pandas(result, schema=output_schema, preserve_index=False)
        metadata = write_table_part(part_path(parts_dir, index), table, output_schema)
        completed_parts[str(index)] = metadata
        state["device"] = resolved_device
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(state_file, state)
        print(
            f"[commit-classification {len(completed_parts)}/{len(expected_parts)}] "
            f"rows={sum(int(item['row_count']) for item in completed_parts.values())}/{total_rows}",
            flush=True,
        )
    if remaining != 0 or set(completed_parts) != {str(index) for index in expected_parts}:
        raise ValueError("Completed prediction parts do not cover the input rows.")

    ordered_parts = [
        validate_recorded_part(completed_parts[str(index)], output_schema, expected_parts[index])
        for index in expected_parts
    ]
    state["output"] = finalize_parts(ordered_parts, output, output_schema)
    summary = summarize(output)
    write_json(summary_file, summary)
    state["summary"] = {"path": str(summary_file.resolve()), "sha256": sha256_file(summary_file)}
    state["finalized"] = True
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(state_file, state)
    print(
        f"finalized_commit_predictions={summary['row_count']} "
        f"positive_commits={summary['positive_commit_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
