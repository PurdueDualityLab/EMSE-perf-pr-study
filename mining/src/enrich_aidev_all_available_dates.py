from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from enrich_aidev_attribution import (
    ATTRIBUTION_FIELDS,
    AIDEV_PAPER_URL,
    METHOD_VERSION,
    created_date,
    output_schema,
    parse_positive_number,
    row_identity,
    row_repository_id,
    sha256_file,
)
from schema import atomic_write_text


METHOD_NAME = "aidev_all_available_dates_incremental_v1"
PENDING_FILENAME = "pending.parquet"
PENDING_OUTPUT_FILENAME = "pending_attribution.parquet"
PLAN_FILENAME = "plan.json"
SUMMARY_FILENAME = "summary.json"
STATE_FILENAME = "state.json"
DEFAULT_AIDEV_AGENTIC = (
    "hf://datasets/dysavepeople/AIDev@b6d1b8af952053f20fd9f9aa03e66bddd12228fe/"
    "pull_request.parquet"
)
AGENT_NAMES = {
    "OpenAI_Codex": "openai_codex",
    "Devin": "devin",
    "Copilot": "github_copilot",
    "Cursor": "cursor",
    "Claude_Code": "claude_code",
}


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def iso_date(value: str) -> date:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise argparse.ArgumentTypeError(f"expected an ISO date, got {value!r}")
    return parsed.date()


def write_json(path: Path, value: Mapping[str, object]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def identity_key(row: Mapping[str, object]) -> str | None:
    repository_id = row_repository_id(row)
    identity = row_identity(row)
    if repository_id is None or identity is None:
        return None
    return f"{repository_id}:{identity.number}"


def required_attribution_columns(schema: pa.Schema) -> list[str]:
    columns = [field.name for field in ATTRIBUTION_FIELDS]
    missing = [column for column in columns if column not in schema.names]
    if missing:
        raise ValueError("Existing enrichment is missing attribution columns: " + ", ".join(missing))
    return columns


def validate_existing_enrichment(raw_path: Path, existing_path: Path, existing_state_path: Path) -> dict[str, Any]:
    if not raw_path.is_file() or not existing_path.is_file() or not existing_state_path.is_file():
        raise FileNotFoundError("Raw input, existing enrichment, and existing state must all exist.")
    state = json.loads(existing_state_path.read_text(encoding="utf-8"))
    if not state.get("finalized"):
        raise ValueError("Existing enrichment state is not finalized.")
    expected_output_hash = str(state.get("output_sha256") or "")
    if not expected_output_hash or sha256_file(existing_path) != expected_output_hash:
        raise ValueError("Existing enrichment does not match its recorded output SHA-256.")
    signature = state.get("run_signature")
    if not isinstance(signature, Mapping):
        raise ValueError("Existing enrichment state has no run signature.")
    expected_input_hash = str(signature.get("input_sha256") or "")
    if not expected_input_hash:
        raise ValueError("Existing enrichment state has no input SHA-256.")
    existing = pq.ParquetFile(existing_path)
    required_attribution_columns(existing.schema_arrow)
    return {
        "existing_input_sha256": expected_input_hash,
        "existing_output_sha256": expected_output_hash,
        "existing_rows": int(existing.metadata.num_rows),
        "existing_end_date": str(signature.get("end_date") or ""),
        "existing_code_sha256": signature.get("code_sha256", {}),
    }


def existing_identity_keys(existing_path: Path, batch_size: int) -> set[str]:
    source = pq.ParquetFile(existing_path)
    columns = [column for column in ("repo_id", "repository_id", "number", "pr_number", "html_url", "aidev_source_html_url", "repo_full_name") if column in source.schema_arrow.names]
    keys: set[str] = set()
    for batch in source.iter_batches(batch_size=batch_size, columns=columns):
        for row in pa.Table.from_batches([batch]).to_pylist():
            key = identity_key(row)
            if key is None:
                raise ValueError("Existing enrichment contains a row without a canonical PR identity.")
            if key in keys:
                raise ValueError(f"Existing enrichment contains duplicate identity {key}.")
            keys.add(key)
    return keys


def prepare_pending(raw_path: Path, existing_path: Path, pending_path: Path, batch_size: int) -> dict[str, int]:
    existing_keys = existing_identity_keys(existing_path, batch_size)
    source = pq.ParquetFile(raw_path)
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=pending_path.parent,
        prefix=f".{pending_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    counts: Counter[str] = Counter()
    seen_raw_keys: set[str] = set()
    try:
        with pq.ParquetWriter(temporary_path, source.schema_arrow, compression="snappy") as writer:
            for batch in source.iter_batches(batch_size=batch_size):
                table = pa.Table.from_batches([batch])
                rows = table.to_pylist()
                pending_indexes = []
                for index, row in enumerate(rows):
                    counts["raw_rows"] += 1
                    if created_date(row.get("created_at")) is None:
                        counts["invalid_created_at_rows"] += 1
                    key = identity_key(row)
                    if key is None:
                        counts["invalid_identity_rows"] += 1
                        pending_indexes.append(index)
                        continue
                    if key in seen_raw_keys:
                        raise ValueError(f"Raw input contains duplicate identity {key}.")
                    seen_raw_keys.add(key)
                    if key in existing_keys:
                        counts["reused_rows"] += 1
                    else:
                        counts["pending_rows"] += 1
                        pending_indexes.append(index)
                if pending_indexes:
                    writer.write_table(table.take(pa.array(pending_indexes, type=pa.int64())))
        os.replace(temporary_path, pending_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    counts["existing_identity_count"] = len(existing_keys)
    counts["raw_identity_count"] = len(seen_raw_keys)
    return {key: int(value) for key, value in counts.items()}


def build_plan(
    raw_path: Path,
    existing_path: Path,
    existing_state_path: Path,
    end_date: date,
    pending_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    existing = validate_existing_enrichment(raw_path, existing_path, existing_state_path)
    counts = prepare_pending(raw_path, existing_path, pending_path, batch_size)
    source = pq.ParquetFile(raw_path)
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw": {
            "path": str(raw_path.resolve()),
            "sha256": sha256_file(raw_path),
            "row_count": int(source.metadata.num_rows),
            "schema": str(source.schema_arrow),
        },
        "existing_enrichment": {
            "path": str(existing_path.resolve()),
            "state_path": str(existing_state_path.resolve()),
            **existing,
        },
        "pending": {
            "path": str(pending_path.resolve()),
            "sha256": sha256_file(pending_path),
            "row_count": int(pq.ParquetFile(pending_path).metadata.num_rows),
        },
        "scope": {
            "end_date": end_date.isoformat(),
            "description": "All available dates in the fixed mined raw snapshot.",
        },
        "counts": counts,
    }


def validate_plan(plan: Mapping[str, object], raw_path: Path, pending_path: Path, end_date: date) -> None:
    raw = plan.get("raw")
    pending = plan.get("pending")
    scope = plan.get("scope")
    if not isinstance(raw, Mapping) or not isinstance(pending, Mapping) or not isinstance(scope, Mapping):
        raise ValueError("Invalid incremental attribution plan.")
    if raw.get("sha256") != sha256_file(raw_path):
        raise ValueError("Raw input changed since the incremental plan was prepared.")
    if pending.get("sha256") != sha256_file(pending_path):
        raise ValueError("Pending input changed since the incremental plan was prepared.")
    if scope.get("end_date") != end_date.isoformat():
        raise ValueError("Requested end date does not match the prepared plan.")


def validate_existing_against_plan(
    plan: Mapping[str, object],
    raw_path: Path,
    existing_path: Path,
    existing_state_path: Path,
) -> None:
    planned = plan.get("existing_enrichment")
    if not isinstance(planned, Mapping):
        raise ValueError("Prepared plan has no existing enrichment signature.")
    current = validate_existing_enrichment(raw_path, existing_path, existing_state_path)
    fields = (
        "existing_input_sha256",
        "existing_output_sha256",
        "existing_rows",
        "existing_end_date",
        "existing_code_sha256",
    )
    if any(planned.get(field) != current.get(field) for field in fields):
        raise ValueError("Existing enrichment changed since the incremental plan was prepared.")


def run_pending_enrichment(args: argparse.Namespace, pending_path: Path, pending_output_path: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("enrich_aidev_attribution.py")),
        "--input",
        str(pending_path),
        "--output",
        str(pending_output_path),
        "--end-date",
        args.end_date.isoformat(),
        "--token-file",
        str(args.token_file),
        "--workers",
        str(args.workers),
        "--row-batch-size",
        str(args.row_batch_size),
        "--checkpoint-every",
        str(args.checkpoint_every),
    ]
    if pending_output_path.with_suffix(".state.json").exists():
        command.append("--resume")
    subprocess.run(command, check=True)


def historical_aidev_agents(path_or_url: str) -> tuple[dict[str, str], str]:
    frame = pd.read_parquet(path_or_url, columns=["repo_id", "number", "agent"])
    result: dict[str, str] = {}
    for row in frame.to_dict("records"):
        repository_id = parse_positive_number(row.get("repo_id"))
        number = parse_positive_number(row.get("number"))
        agent = AGENT_NAMES.get(str(row.get("agent") or ""))
        if repository_id is None or number is None or agent is None:
            raise ValueError("Pinned AIDev agentic table contains an invalid identity or agent.")
        key = f"{repository_id}:{number}"
        prior = result.setdefault(key, agent)
        if prior != agent:
            raise ValueError(f"Pinned AIDev identity {key} has conflicting agents.")
    digest = hashlib.sha256()
    for key, agent in sorted(result.items()):
        digest.update(f"{key}\t{agent}\n".encode("utf-8"))
    return result, digest.hexdigest()


def sqlite_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE attribution (identity TEXT PRIMARY KEY, values_json TEXT NOT NULL) WITHOUT ROWID"
    )
    return connection


def load_attributions(connection: sqlite3.Connection, path: Path, batch_size: int) -> int:
    source = pq.ParquetFile(path)
    attributes = [field.name for field in ATTRIBUTION_FIELDS]
    identity_columns = [column for column in ("repo_id", "repository_id", "number", "pr_number", "html_url", "aidev_source_html_url", "repo_full_name") if column in source.schema_arrow.names]
    count = 0
    for batch in source.iter_batches(batch_size=batch_size, columns=[*identity_columns, *attributes]):
        records = []
        for row in pa.Table.from_batches([batch]).to_pylist():
            key = identity_key(row)
            if key is None:
                raise ValueError(f"Attribution source {path} contains an invalid identity.")
            values = {name: str(row.get(name) or "") for name in attributes}
            records.append((key, json.dumps(values, sort_keys=True)))
        try:
            connection.executemany("INSERT INTO attribution VALUES (?, ?)", records)
        except sqlite3.IntegrityError as error:
            raise ValueError(f"Attribution sources overlap or contain duplicate identities: {path}") from error
        count += len(records)
    connection.commit()
    return count


def historical_override(values: dict[str, str], agent: str) -> dict[str, str]:
    if values.get("aidev_attribution_label") == "agentic":
        return values
    result = dict(values)
    result.update(
        aidev_attribution_label="agentic",
        aidev_attribution_agent=agent,
        aidev_attribution_rule="aidev_pinned_historical",
        aidev_attribution_evidence="dysavepeople/AIDev@b6d1b8af952053f20fd9f9aa03e66bddd12228fe",
        aidev_attribution_status="historical_aidev_precedence",
        aidev_attribution_method="aidev_pinned_plus_live_v1",
        aidev_attribution_detail_status="historical_snapshot",
    )
    return result


def query_attributions(connection: sqlite3.Connection, keys: list[str]) -> dict[str, dict[str, str]]:
    if not keys:
        return {}
    placeholders = ", ".join("?" for _ in keys)
    rows = connection.execute(
        f"SELECT identity, values_json FROM attribution WHERE identity IN ({placeholders})", keys
    ).fetchall()
    return {key: json.loads(values) for key, values in rows}


def merge_output(
    raw_path: Path,
    existing_path: Path,
    pending_output_path: Path,
    output_path: Path,
    historical_agents: Mapping[str, str],
    batch_size: int,
) -> dict[str, int]:
    raw = pq.ParquetFile(raw_path)
    existing = pq.ParquetFile(existing_path)
    required_attribution_columns(existing.schema_arrow)
    if not pending_output_path.is_file():
        raise FileNotFoundError(f"Pending attribution output does not exist: {pending_output_path}")
    pending = pq.ParquetFile(pending_output_path)
    required_attribution_columns(pending.schema_arrow)
    schema = output_schema(raw.schema_arrow)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    database_path = output_path.with_suffix(".attribution.sqlite")
    database_path.unlink(missing_ok=True)
    counts: Counter[str] = Counter()
    try:
        connection = sqlite_connection(database_path)
        counts["reused_attribution_rows"] = load_attributions(connection, existing_path, batch_size)
        counts["pending_attribution_rows"] = load_attributions(connection, pending_output_path, batch_size)
        with pq.ParquetWriter(temporary_path, schema, compression="snappy") as writer:
            for batch in raw.iter_batches(batch_size=min(batch_size, 500)):
                table = pa.Table.from_batches([batch])
                rows = table.to_pylist()
                keys = [identity_key(row) for row in rows]
                valid_keys = [key for key in keys if key is not None]
                attributes = query_attributions(connection, valid_keys)
                arrays = list(batch.columns)
                appended = {field.name: [] for field in ATTRIBUTION_FIELDS}
                for key in keys:
                    if key is None:
                        values = {field.name: "" for field in ATTRIBUTION_FIELDS}
                        values.update(
                            aidev_attribution_label="unresolved",
                            aidev_attribution_status="invalid_identity",
                            aidev_attribution_method=METHOD_NAME,
                            aidev_attribution_detail_status="not_requested",
                        )
                        counts["invalid_identity_rows"] += 1
                    else:
                        values = attributes.get(key)
                        if values is None:
                            raise ValueError(f"No attribution was available for raw identity {key}.")
                        if key in historical_agents:
                            overridden = historical_override(values, historical_agents[key])
                            if overridden != values:
                                counts["historical_overrides"] += 1
                            values = overridden
                        counts[values["aidev_attribution_label"]] += 1
                    for field in ATTRIBUTION_FIELDS:
                        appended[field.name].append(values[field.name])
                arrays.extend(pa.array(appended[field.name], type=field.type) for field in ATTRIBUTION_FIELDS)
                writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
        connection.close()
        final = pq.ParquetFile(temporary_path)
        if final.metadata.num_rows != raw.metadata.num_rows or not final.schema_arrow.equals(schema):
            raise ValueError("Merged output does not match the raw input row count or expected schema.")
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        database_path.unlink(missing_ok=True)
        database_path.with_name(database_path.name + "-wal").unlink(missing_ok=True)
        database_path.with_name(database_path.name + "-shm").unlink(missing_ok=True)
    return {key: int(value) for key, value in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incrementally attribute every PR in a fixed mined raw snapshot without re-enriching existing rows."
    )
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--existing", required=True, type=Path)
    parser.add_argument("--existing-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--end-date", required=True, type=iso_date)
    parser.add_argument("--token-file", type=Path, default=Path("mining/github_tokens.txt"))
    parser.add_argument("--aidev-agentic", default=DEFAULT_AIDEV_AGENTIC)
    parser.add_argument("--allow-custom-aidev-agentic", action="store_true")
    parser.add_argument("--workers", type=positive_integer, default=5)
    parser.add_argument("--row-batch-size", type=positive_integer, default=10_000)
    parser.add_argument("--checkpoint-every", type=positive_integer, default=25)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not args.raw.is_file():
        parser.error(f"raw input does not exist: {args.raw}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending_path = args.output_dir / PENDING_FILENAME
    plan_path = args.output_dir / PLAN_FILENAME
    pending_output_path = args.output_dir / PENDING_OUTPUT_FILENAME
    output_path = args.output_dir / "aidev_all_available_dates.parquet"
    state_path = args.output_dir / STATE_FILENAME
    summary_path = args.output_dir / SUMMARY_FILENAME

    if not args.execute:
        plan = build_plan(
            args.raw,
            args.existing,
            args.existing_state,
            args.end_date,
            pending_path,
            args.row_batch_size,
        )
        write_json(plan_path, plan)
        print(json.dumps(plan["counts"], sort_keys=True), flush=True)
        print(f"Prepared {pending_path}; rerun with --execute to use GitHub.", flush=True)
        return

    if not plan_path.is_file() or not pending_path.is_file():
        raise ValueError("Run without --execute first to prepare and review the pending PR partition.")
    if any(path.exists() for path in (output_path, state_path, summary_path)):
        raise FileExistsError("Final all-available-dates artifacts already exist; choose a new output directory.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan, args.raw, pending_path, args.end_date)
    validate_existing_against_plan(plan, args.raw, args.existing, args.existing_state)
    if args.aidev_agentic != DEFAULT_AIDEV_AGENTIC and not args.allow_custom_aidev_agentic:
        raise ValueError("Custom AIDev agentic input requires --allow-custom-aidev-agentic.")
    counts = plan.get("counts", {})
    if int(counts.get("pending_rows", 0)):
        run_pending_enrichment(args, pending_path, pending_output_path)
    else:
        raw_schema = pq.ParquetFile(args.raw).schema_arrow
        pq.write_table(pa.Table.from_pylist([], schema=output_schema(raw_schema)), pending_output_path)
    historical_agents, historical_sha256 = historical_aidev_agents(args.aidev_agentic)
    merged_counts = merge_output(
        args.raw,
        args.existing,
        pending_output_path,
        output_path,
        historical_agents,
        args.row_batch_size,
    )
    summary = {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "paper": AIDEV_PAPER_URL,
        "plan": {"path": str(plan_path.resolve()), "sha256": sha256_file(plan_path)},
        "output": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path)},
        "counts": {**counts, **merged_counts},
        "historical_aidev_agentic": {
            "source": args.aidev_agentic,
            "identity_count": len(historical_agents),
            "identity_agent_sha256": historical_sha256,
        },
    }
    write_json(summary_path, summary)
    write_json(
        state_path,
        {
            "finalized": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": sha256_file(plan_path),
            "output_sha256": summary["output"]["sha256"],
            "summary_path": str(summary_path.resolve()),
        },
    )
    print(f"Wrote {output_path} with {merged_counts.get('agentic', 0)} agentic PRs.", flush=True)


if __name__ == "__main__":
    main()
