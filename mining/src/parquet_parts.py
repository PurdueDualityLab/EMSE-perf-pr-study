from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from schema import atomic_write_text


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def write_json(path: Path, value: Mapping[str, object]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def part_path(directory: Path, index: int) -> Path:
    return directory / f"part-{index:06d}.parquet"


def write_part(path: Path, rows: list[dict], schema: pa.Schema) -> dict[str, object]:
    return write_table_part(path, pa.Table.from_pylist(rows, schema=schema), schema)


def write_table_part(path: Path, table: pa.Table, schema: pa.Schema) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    temporary.unlink(missing_ok=True)
    if not table.schema.equals(schema, check_metadata=True):
        table = table.cast(schema)
    try:
        pq.write_table(table, temporary, compression="snappy")
        validate_part(temporary, schema, expected_rows=table.num_rows)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(path.resolve()),
        "row_count": table.num_rows,
        "schema_sha256": schema_sha256(schema),
        "sha256": sha256_file(path),
    }


def validate_part(
    path: Path,
    schema: pa.Schema,
    *,
    expected_rows: int | None = None,
    expected_sha256: str | None = None,
) -> int:
    parquet_file = pq.ParquetFile(path)
    rows = int(parquet_file.metadata.num_rows)
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"Part {path.name} has {rows} rows; expected {expected_rows}.")
    if not parquet_file.schema_arrow.equals(schema, check_metadata=True):
        raise ValueError(f"Part {path.name} does not have the expected schema.")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"Part {path.name} does not match its recorded SHA-256.")
    return rows


def finalize_parts(
    parts: Iterable[Path],
    output: Path,
    schema: pa.Schema,
) -> dict[str, object]:
    ordered = sorted(Path(path) for path in parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    temporary.unlink(missing_ok=True)
    rows = 0
    try:
        if not ordered:
            pq.write_table(pa.Table.from_pylist([], schema=schema), temporary, compression="snappy")
        else:
            with pq.ParquetWriter(temporary, schema, compression="snappy") as writer:
                for part in ordered:
                    validate_part(part, schema)
                    table = pq.read_table(part)
                    writer.write_table(table)
                    rows += table.num_rows
        finalized = pq.ParquetFile(temporary)
        if finalized.metadata.num_rows != rows or not finalized.schema_arrow.equals(
            schema, check_metadata=True
        ):
            raise ValueError(f"Finalized parquet validation failed: {output}")
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(output.resolve()),
        "row_count": rows,
        "schema_sha256": schema_sha256(schema),
        "sha256": sha256_file(output),
    }
