from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from perfannotator_common import (
    MODEL_ID,
    MODEL_ARTIFACT_SHA256,
    MODEL_REVISION,
    as_list,
    classify_loaded_model,
    configure_reproducibility,
    load_model,
    metadata_text,
    model_artifact_sha256,
    resolve_model_path,
)


PREDICTION_FIELDS = (
    pa.field("perfannotator_metadata_label_id", pa.int64()),
    pa.field("perfannotator_metadata_is_performance_improving", pa.bool_()),
    pa.field("perfannotator_metadata_score", pa.float64()),
    pa.field("perfannotator_metadata_model", pa.string()),
    pa.field("perfannotator_metadata_model_revision", pa.string()),
    pa.field("perfannotator_metadata_device", pa.string()),
    pa.field("perfannotator_metadata_input_chars", pa.int64()),
    pa.field("perfannotator_metadata_truncated", pa.bool_()),
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


def part_directory(output: Path) -> Path:
    return output.with_suffix("").with_name(f"{output.stem}_parts")


def state_path(output: Path) -> Path:
    return output.with_suffix(".state.json")


def part_path(parts_dir: Path, batch_index: int) -> Path:
    return parts_dir / f"part-{batch_index:06d}.parquet"


def write_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def build_run_signature(
    args: argparse.Namespace,
    source: pq.ParquetFile,
    input_sha256: str,
    model_sha256: str,
) -> dict:
    return {
        "input_path": str(args.input.resolve()),
        "input_sha256": input_sha256,
        "input_schema": str(source.schema_arrow),
        "input_schema_sha256": schema_sha256(source.schema_arrow),
        "input_row_count": int(source.metadata.num_rows),
        "model": args.model,
        "model_revision": args.model_revision or "",
        "model_artifact_sha256": model_sha256,
        "device": args.device,
        "batch_size": args.batch_size,
        "row_batch_size": args.row_batch_size,
        "limit": args.limit,
        "max_input_chars": args.max_input_chars,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "runtime": runtime_fingerprint(),
        "code_sha256": {
            Path(__file__).name: sha256_file(Path(__file__)),
            "perfannotator_common.py": sha256_file(Path(__file__).with_name("perfannotator_common.py")),
        },
    }


def expected_part_row_counts(total_rows: int, row_batch_size: int) -> dict[str, int]:
    total_parts = (total_rows + row_batch_size - 1) // row_batch_size
    return {
        f"part-{batch_index:06d}.parquet": min(
            row_batch_size, total_rows - (batch_index - 1) * row_batch_size
        )
        for batch_index in range(1, total_parts + 1)
    }


def checkpoint_parts(parts_dir: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in parts_dir.glob("part-*.parquet")
        if not path.name.endswith(".tmp.parquet")
    }


def resolved_model_sha256(model_id: str, revision: str | None) -> str:
    artifact_hash = model_artifact_sha256(resolve_model_path(model_id, revision))
    if (
        model_id == MODEL_ID
        and revision == MODEL_REVISION
        and artifact_hash != MODEL_ARTIFACT_SHA256
    ):
        raise ValueError(
            "Pinned PerfAnnotator artifact does not match the expected SHA-256: "
            f"{artifact_hash}"
        )
    return artifact_hash


def runtime_fingerprint() -> dict:
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


def string_list(value: object) -> list[str]:
    return as_list(value)


def prediction_schema(input_schema: pa.Schema) -> pa.Schema:
    collisions = sorted(PREDICTION_COLUMNS.intersection(input_schema.names))
    if collisions:
        raise ValueError(
            "Input already contains PerfAnnotator metadata prediction columns: "
            + ", ".join(collisions)
        )
    fields = []
    for field in input_schema:
        if field.name in {"filenames", "commit_messages"}:
            fields.append(pa.field(field.name, pa.list_(pa.string())))
        elif field.name == "aidev_source_pr_number":
            fields.append(pa.field(field.name, pa.int64()))
        else:
            fields.append(field)
    return pa.schema(fields + list(PREDICTION_FIELDS))


def normalized_table(table: pa.Table, schema: pa.Schema) -> pa.Table:
    frame = table.to_pandas()
    for column in ("filenames", "commit_messages"):
        if column in frame.columns:
            frame[column] = frame[column].map(string_list)
    if "aidev_source_pr_number" in frame.columns:
        frame["aidev_source_pr_number"] = pd.to_numeric(
            frame["aidev_source_pr_number"], errors="coerce"
        ).astype("Int64")
    return pa.Table.from_pandas(frame, schema=schema, preserve_index=False)


def add_predictions(
    df: pd.DataFrame,
    *,
    model: str,
    revision: str | None,
    device: str,
    max_input_chars: int,
    batch_size: int,
    tokenizer,
    classifier_model,
) -> pd.DataFrame:
    collisions = sorted(PREDICTION_COLUMNS.intersection(df.columns))
    if collisions:
        raise ValueError(
            "Input already contains PerfAnnotator metadata prediction columns: "
            + ", ".join(collisions)
        )
    built = [metadata_text(row, max_input_chars) for _, row in df.iterrows()]
    texts = [item[0] for item in built]
    truncated = [item[1] for item in built]
    labels, scores = classify_loaded_model(
        texts,
        tokenizer=tokenizer,
        model=classifier_model,
        device=device,
        batch_size=batch_size,
    )
    if len(labels) != len(df) or len(scores) != len(df):
        raise ValueError("PerfAnnotator returned a different number of predictions than input rows.")

    result = df.copy()
    result["perfannotator_metadata_label_id"] = labels
    result["perfannotator_metadata_is_performance_improving"] = [label == 1 for label in labels]
    result["perfannotator_metadata_score"] = scores
    result["perfannotator_metadata_model"] = model
    result["perfannotator_metadata_model_revision"] = revision or ""
    result["perfannotator_metadata_device"] = device
    result["perfannotator_metadata_input_chars"] = [len(text) for text in texts]
    result["perfannotator_metadata_truncated"] = truncated
    return result


def validate_part(path: Path, schema: pa.Schema, expected_rows: int) -> None:
    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as error:
        raise ValueError(f"Could not read checkpoint part: {path}") from error
    if parquet_file.metadata.num_rows != expected_rows:
        raise ValueError(
            f"Checkpoint part {path.name} has {parquet_file.metadata.num_rows} rows; "
            f"expected {expected_rows}."
        )
    if not parquet_file.schema_arrow.equals(schema):
        raise ValueError(f"Checkpoint part {path.name} does not have the expected schema.")


def finalize_parts(
    parts_dir: Path,
    output: Path,
    schema: pa.Schema,
    expected_parts: dict[str, int],
) -> int:
    parts = checkpoint_parts(parts_dir)
    actual_names = set(parts)
    expected_names = set(expected_parts)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError(
            "Classification checkpoint parts do not match the expected set ("
            + "; ".join(details)
            + ")."
        )

    expected_rows = sum(expected_parts.values())
    temporary = output.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    row_count = 0
    try:
        if not expected_parts:
            pq.write_table(pa.Table.from_batches([], schema=schema), temporary, compression="snappy")
        else:
            with pq.ParquetWriter(temporary, schema, compression="snappy") as writer:
                for name in sorted(expected_parts):
                    part = parts[name]
                    validate_part(part, schema, expected_parts[name])
                    table = pq.read_table(part)
                    writer.write_table(table)
                    row_count += table.num_rows
        if row_count != expected_rows:
            raise ValueError(f"Finalized row count is {row_count}; expected {expected_rows}.")
        finalized = pq.ParquetFile(temporary)
        if finalized.metadata.num_rows != expected_rows:
            raise ValueError(
                f"Final parquet has {finalized.metadata.num_rows} rows; expected {expected_rows}."
            )
        if not finalized.schema_arrow.equals(schema):
            raise ValueError("Final parquet does not have the expected schema.")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return row_count


def classification_state(
    args: argparse.Namespace,
    run_signature: dict,
    output_schema: pa.Schema,
    expected_parts: dict[str, int],
    completed_parts: dict[str, int],
    device: str,
    *,
    finalized: bool,
) -> dict:
    part_schema_hash = schema_sha256(output_schema)
    return {
        "version": 3,
        "run_signature": run_signature,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "model": args.model,
        "model_revision": args.model_revision or "",
        "device": device,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "total_rows": int(sum(expected_parts.values())),
        "completed_rows": int(sum(completed_parts.values())),
        "completed_batches": len(completed_parts),
        "total_batches": len(expected_parts),
        "completed_parts": {
            name: {
                "row_count": completed_parts[name],
                "schema_sha256": part_schema_hash,
                "sha256": sha256_file(part_directory(args.output) / name),
            }
            for name in sorted(completed_parts)
        },
        "finalized": finalized,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify PRs with PerfAnnotator-mini using PR metadata.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--batch-size", type=positive_integer, default=16)
    parser.add_argument("--row-batch-size", type=positive_integer, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-input-chars", type=positive_integer, default=20000)
    parser.add_argument("--limit", type=positive_integer)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--seed", type=seed_integer)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    if args.deterministic and args.seed is None:
        parser.error("--deterministic requires --seed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = part_directory(args.output)
    parts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = state_path(args.output)
    source = pq.ParquetFile(args.input)
    output_schema = prediction_schema(source.schema_arrow)
    total_rows = min(source.metadata.num_rows, args.limit) if args.limit is not None else source.metadata.num_rows
    expected_parts = expected_part_row_counts(total_rows, args.row_batch_size)
    existing_parts = checkpoint_parts(parts_dir)
    continuing = args.resume or args.finalize_only

    if args.output.exists() and not continuing:
        raise FileExistsError(f"Output already exists: {args.output}. Use --resume or choose another output path.")
    if existing_parts and not continuing:
        raise FileExistsError(f"Checkpoint parts already exist: {parts_dir}. Use --resume or remove them.")
    if checkpoint_path.exists() and not continuing:
        raise FileExistsError(
            f"Checkpoint state already exists: {checkpoint_path}. Use --resume or remove it."
        )

    input_hash = sha256_file(args.input)
    model_sha256 = (
        resolved_model_sha256(args.model, args.model_revision)
        if total_rows
        else "not-used-empty-input"
    )
    run_signature = build_run_signature(args, source, input_hash, model_sha256)
    existing_state = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists()
        else {}
    )
    if existing_state and existing_state.get("run_signature") != run_signature:
        raise ValueError("Existing checkpoint run signature does not match the requested run.")
    if continuing and not existing_state and (existing_parts or args.output.exists()):
        raise ValueError("Cannot resume classification artifacts without matching checkpoint state.")

    unexpected_parts = sorted(set(existing_parts) - set(expected_parts))
    if unexpected_parts:
        raise ValueError("Unexpected classification checkpoint parts: " + ", ".join(unexpected_parts))
    recorded_parts = existing_state.get("completed_parts", {}) if existing_state else {}
    unexpected_recorded = sorted(set(recorded_parts) - set(expected_parts))
    if unexpected_recorded:
        raise ValueError(
            "Unexpected parts recorded in classification state: "
            + ", ".join(unexpected_recorded)
        )
    if args.finalize_only and set(recorded_parts) != set(existing_parts):
        raise ValueError("Finalize-only requires checkpoint state and files for every part.")
    completed_parts: dict[str, int] = {}
    for name in sorted(set(existing_parts).intersection(recorded_parts)):
        path = existing_parts[name]
        validate_part(path, output_schema, expected_parts[name])
        recorded = recorded_parts.get(name, {})
        if not isinstance(recorded, dict) or recorded.get("sha256") != sha256_file(path):
            raise ValueError(f"Checkpoint part {name} does not match its recorded SHA-256.")
        completed_parts[name] = expected_parts[name]

    device = str(existing_state.get("device", args.device))
    if args.finalize_only:
        final_rows = finalize_parts(parts_dir, args.output, output_schema, expected_parts)
        if final_rows != total_rows:
            raise ValueError(f"Finalized row count is {final_rows}; expected {total_rows}.")
        write_state(
            checkpoint_path,
            classification_state(
                args,
                run_signature,
                output_schema,
                expected_parts,
                expected_parts,
                device,
                finalized=True,
            ),
        )
        print(f"finalized_rows={final_rows}", flush=True)
        return

    write_state(
        checkpoint_path,
        classification_state(
            args,
            run_signature,
            output_schema,
            expected_parts,
            completed_parts,
            device,
            finalized=False,
        ),
    )

    tokenizer = classifier_model = None
    if len(completed_parts) < len(expected_parts):
        if args.seed is not None:
            configure_reproducibility(args.seed, args.deterministic)
        tokenizer, classifier_model, device = load_model(
            args.model, args.model_revision, device
        )

    remaining = total_rows
    for batch_index, batch in enumerate(source.iter_batches(batch_size=args.row_batch_size), start=1):
        if remaining <= 0:
            break
        table = batch if batch.num_rows <= remaining else batch.slice(0, remaining)
        remaining -= table.num_rows
        output_part = part_path(parts_dir, batch_index)
        expected_rows = expected_parts.get(output_part.name)
        if expected_rows is None or table.num_rows != expected_rows:
            raise ValueError(
                f"Input batch {batch_index} has {table.num_rows} rows; expected {expected_rows}."
            )
        if output_part.name in completed_parts:
            continue

        result = add_predictions(
            table.to_pandas(),
            model=args.model,
            revision=args.model_revision,
            device=device,
            max_input_chars=args.max_input_chars,
            batch_size=args.batch_size,
            tokenizer=tokenizer,
            classifier_model=classifier_model,
        )
        temporary_part = output_part.with_suffix(".tmp.parquet")
        result_table = normalized_table(
            pa.Table.from_pandas(result, preserve_index=False), output_schema
        )
        if result_table.num_rows != expected_rows:
            raise ValueError(
                f"Prediction part {output_part.name} has {result_table.num_rows} rows; "
                f"expected {expected_rows}."
            )
        try:
            pq.write_table(result_table, temporary_part, compression="snappy")
            validate_part(temporary_part, output_schema, expected_rows)
            os.replace(temporary_part, output_part)
        except Exception:
            temporary_part.unlink(missing_ok=True)
            raise
        completed_parts[output_part.name] = expected_rows
        write_state(
            checkpoint_path,
            classification_state(
                args,
                run_signature,
                output_schema,
                expected_parts,
                completed_parts,
                device,
                finalized=False,
            ),
        )
        print(
            f"completed_batches={len(completed_parts)}/{len(expected_parts)} "
            f"completed_rows={sum(completed_parts.values())}/{total_rows}",
            flush=True,
        )

    if remaining != 0:
        raise ValueError(f"Input ended with {remaining} expected rows unprocessed.")
    if set(completed_parts) != set(expected_parts) or sum(completed_parts.values()) != total_rows:
        raise ValueError("Completed classification parts do not cover the expected input rows.")

    final_rows = finalize_parts(parts_dir, args.output, output_schema, expected_parts)
    if final_rows != total_rows:
        raise ValueError(f"Finalized row count is {final_rows}; expected {total_rows}.")
    write_state(
        checkpoint_path,
        classification_state(
            args,
            run_signature,
            output_schema,
            expected_parts,
            expected_parts,
            device,
            finalized=True,
        ),
    )
    print(f"finalized_rows={final_rows}", flush=True)


if __name__ == "__main__":
    main()
