from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from perfannotator_common import (
    MODEL_ID,
    classify_loaded_model,
    configure_reproducibility,
    load_model,
    metadata_text,
)


def part_directory(output: Path) -> Path:
    return output.with_suffix("").with_name(f"{output.stem}_parts")


def state_path(output: Path) -> Path:
    return output.with_suffix(".state.json")


def part_path(parts_dir: Path, batch_index: int) -> Path:
    return parts_dir / f"part-{batch_index:06d}.parquet"


def write_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    if hasattr(value, "tolist"):
        return string_list(value.tolist())
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass
    return [str(value)]


def prediction_schema(input_schema: pa.Schema) -> pa.Schema:
    fields = []
    for field in input_schema:
        if field.name in {"filenames", "commit_messages"}:
            fields.append(pa.field(field.name, pa.list_(pa.string())))
        elif field.name == "aidev_source_pr_number":
            fields.append(pa.field(field.name, pa.int64()))
        else:
            fields.append(field)
    return pa.schema(
        fields
        + [
            pa.field("perfannotator_metadata_label_id", pa.int64()),
            pa.field("perfannotator_metadata_is_performance_improving", pa.bool_()),
            pa.field("perfannotator_metadata_score", pa.float64()),
            pa.field("perfannotator_metadata_model", pa.string()),
            pa.field("perfannotator_metadata_model_revision", pa.string()),
            pa.field("perfannotator_metadata_device", pa.string()),
            pa.field("perfannotator_metadata_input_chars", pa.int64()),
            pa.field("perfannotator_metadata_truncated", pa.bool_()),
        ]
    )


def normalized_table(table: pa.Table, schema: pa.Schema) -> pa.Table:
    frame = table.to_pandas()
    for column in ("filenames", "commit_messages"):
        if column in frame.columns:
            frame[column] = frame[column].map(string_list)
    if "aidev_source_pr_number" in frame.columns:
        frame["aidev_source_pr_number"] = pd.to_numeric(
            frame["aidev_source_pr_number"], errors="coerce"
        ).astype("Int64")
    return pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=False)


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


def finalize_parts(parts_dir: Path, output: Path, schema: pa.Schema) -> int:
    parts = sorted(parts_dir.glob("part-*.parquet"))
    if not parts:
        raise ValueError(f"No completed classification parts found under {parts_dir}")

    temporary = output.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    writer = None
    row_count = 0
    try:
        for part in parts:
            table = normalized_table(pq.read_table(part), schema)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="snappy")
            writer.write_table(table)
            row_count += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    os.replace(temporary, output)
    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify PRs with PerfAnnotator-mini using PR metadata.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--row-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-input-chars", type=int, default=20000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = part_directory(args.output)
    parts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = state_path(args.output)
    source = pq.ParquetFile(args.input)
    output_schema = prediction_schema(source.schema_arrow)
    total_rows = min(source.metadata.num_rows, args.limit) if args.limit is not None else source.metadata.num_rows
    total_batches = math.ceil(total_rows / args.row_batch_size)
    existing_state = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}

    if args.finalize_only:
        final_rows = finalize_parts(parts_dir, args.output, output_schema)
        existing_state.update(
            {
                "input": str(args.input),
                "output": str(args.output),
                "total_rows": int(total_rows),
                "completed_rows": int(final_rows),
                "completed_batches": int(total_batches),
                "total_batches": int(total_batches),
                "finalized": True,
            }
        )
        write_state(checkpoint_path, existing_state)
        print(f"finalized_rows={final_rows}", flush=True)
        return

    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Output already exists: {args.output}. Use --resume or choose another output path.")

    existing_parts = {path.name for path in parts_dir.glob("part-*.parquet")} if args.resume else set()
    if not args.resume and existing_parts:
        raise FileExistsError(f"Checkpoint parts already exist: {parts_dir}. Use --resume or remove them.")

    if args.seed is not None:
        configure_reproducibility(args.seed, args.deterministic)
    tokenizer, classifier_model, device = load_model(args.model, args.model_revision, args.device)
    processed_rows = 0
    remaining = args.limit
    for batch_index, batch in enumerate(source.iter_batches(batch_size=args.row_batch_size), start=1):
        if remaining is not None and remaining <= 0:
            break
        table = batch if remaining is None or batch.num_rows <= remaining else batch.slice(0, remaining)
        if remaining is not None:
            remaining -= table.num_rows
        output_part = part_path(parts_dir, batch_index)
        if output_part.name in existing_parts:
            processed_rows += pq.ParquetFile(output_part).metadata.num_rows
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
        pq.write_table(normalized_table(pa.Table.from_pandas(result, preserve_index=False), output_schema), temporary_part, compression="snappy")
        os.replace(temporary_part, output_part)
        processed_rows += len(result)
        write_state(
            checkpoint_path,
            {
                "input": str(args.input),
                "output": str(args.output),
                "model": args.model,
                "model_revision": args.model_revision or "",
                "device": device,
                "seed": args.seed,
                "deterministic": args.deterministic,
                "total_rows": int(total_rows),
                "completed_rows": int(processed_rows),
                "completed_batches": batch_index,
                "total_batches": int(total_batches),
            },
        )
        print(f"completed_batches={batch_index}/{total_batches} completed_rows={processed_rows}/{total_rows}", flush=True)

    final_rows = finalize_parts(parts_dir, args.output, output_schema)
    write_state(
        checkpoint_path,
        {
            "input": str(args.input),
            "output": str(args.output),
            "model": args.model,
            "model_revision": args.model_revision or "",
            "device": device,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "total_rows": int(total_rows),
            "completed_rows": int(final_rows),
            "completed_batches": int(total_batches),
            "total_batches": int(total_batches),
            "finalized": True,
        },
    )
    print(f"finalized_rows={final_rows}", flush=True)


if __name__ == "__main__":
    main()
