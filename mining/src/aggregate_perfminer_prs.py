from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from parquet_parts import (
    schema_sha256,
    sha256_file,
    validate_part,
    write_json,
    write_table_part,
)
from perfminer_reproduction import (
    PERFMINER_MAX_LENGTH,
    PERFMINER_MODEL_ARTIFACT_SHA256,
    PERFMINER_MODEL_WEIGHTS_SHA256,
    PERFMINER_REVISION,
    PERFMINER_THRESHOLD,
    PERFMINER_TRUNCATION,
    commit_manifest_sha256,
)


METHOD_NAME = "perfminer_any_positive_commit_v1"
METHOD_VERSION = 1

AGGREGATE_FIELDS = (
    pa.field("perfminer_pr_status", pa.string()),
    pa.field("perfminer_pr_is_performance", pa.bool_()),
    pa.field("perfminer_pr_evidence_complete", pa.bool_()),
    pa.field("perfminer_expected_commit_count", pa.int64()),
    pa.field("perfminer_listed_commit_count", pa.int64()),
    pa.field("perfminer_eligible_commit_count", pa.int64()),
    pa.field("perfminer_classified_commit_count", pa.int64()),
    pa.field("perfminer_excluded_commit_count", pa.int64()),
    pa.field("perfminer_unavailable_commit_count", pa.int64()),
    pa.field("perfminer_positive_commit_count", pa.int64()),
    pa.field(
        "perfminer_positive_commit_shas",
        pa.list_(pa.field("element", pa.string())),
    ),
    pa.field("perfminer_positive_source_pair_count", pa.int64()),
    pa.field("perfminer_max_commit_score", pa.float64()),
    pa.field("perfminer_pr_aggregation", pa.string()),
    pa.field("perfminer_model_weights_sha256", pa.string()),
)
AGGREGATE_COLUMNS = frozenset(field.name for field in AGGREGATE_FIELDS)


def output_schema(input_schema: pa.Schema) -> pa.Schema:
    required = {
        "repo_id",
        "resolved_repo_full_name",
        "pr_number",
        "pr_base_sha",
        "pr_head_sha",
        "commit_manifest_sha256",
        "expected_commit_count",
        "listed_commit_count",
        "commit_manifest_status",
    }
    missing = sorted(required - set(input_schema.names))
    if missing:
        raise ValueError("PR manifest is missing columns: " + ", ".join(missing))
    collisions = sorted(AGGREGATE_COLUMNS.intersection(input_schema.names))
    if collisions:
        raise ValueError("PR manifest already contains aggregate columns: " + ", ".join(collisions))
    return pa.schema(list(input_schema) + list(AGGREGATE_FIELDS), metadata=input_schema.metadata)


def commit_groups(path: Path) -> dict[tuple[int, int], list[dict]]:
    required = {
        "repo_id",
        "resolved_repo_full_name",
        "pr_number",
        "pr_base_sha",
        "pr_head_sha",
        "commit_manifest_sha256",
        "pr_manifest_complete",
        "commit_index",
        "commit_sha",
        "source_pair_md5",
        "perfminer_evidence_status",
        "perfminer_classification_status",
        "perfminer_label_id",
        "perfminer_is_performance",
        "perfminer_score",
        "perfminer_model_weights_sha256",
        "perfminer_model_artifact_sha256",
        "perfminer_inference_batch_size",
        "perfminer_max_length",
        "perfminer_truncation",
        "perfminer_threshold",
    }
    source = pq.ParquetFile(path)
    missing = sorted(required - set(source.schema_arrow.names))
    if missing:
        raise ValueError("Commit predictions are missing columns: " + ", ".join(missing))
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    seen: set[tuple[int, int, str]] = set()
    artifact_hashes: set[str] = set()
    for batch in source.iter_batches(batch_size=50_000, columns=sorted(required)):
        for row in pa.Table.from_batches([batch]).to_pylist():
            evidence_status = str(row["perfminer_evidence_status"] or "")
            classification_status = str(row["perfminer_classification_status"] or "")
            if evidence_status not in {"eligible", "excluded", "unavailable"}:
                raise ValueError(f"Unknown PerfMiner evidence status: {evidence_status!r}")
            if classification_status not in {
                "classified",
                "not_eligible",
                "message_exceeds_pair_budget",
                "tokenization_error",
            }:
                raise ValueError(
                    f"Unknown PerfMiner classification status: {classification_status!r}"
                )
            if evidence_status == "eligible" and classification_status == "not_eligible":
                raise ValueError("Eligible commit was incorrectly marked not_eligible.")
            if evidence_status != "eligible" and classification_status != "not_eligible":
                raise ValueError("Ineligible commit has an incompatible classification status.")
            if str(row["perfminer_model_weights_sha256"] or "") != PERFMINER_MODEL_WEIGHTS_SHA256:
                raise ValueError("Commit predictions do not use the EASE 2026 Figshare weights.")
            artifact_hash = str(row["perfminer_model_artifact_sha256"] or "")
            if artifact_hash != PERFMINER_MODEL_ARTIFACT_SHA256:
                raise ValueError("Commit prediction does not use the complete Figshare model bundle.")
            artifact_hashes.add(artifact_hash)
            if int(row["perfminer_max_length"] or 0) != PERFMINER_MAX_LENGTH:
                raise ValueError("Commit prediction uses an unexpected maximum input length.")
            if int(row["perfminer_inference_batch_size"] or 0) != 1:
                raise ValueError("Commit prediction does not use one inference at a time.")
            if str(row["perfminer_truncation"] or "") != PERFMINER_TRUNCATION:
                raise ValueError("Commit prediction uses an unexpected truncation strategy.")
            if float(row["perfminer_threshold"]) != PERFMINER_THRESHOLD:
                raise ValueError("Commit prediction uses an unexpected classification threshold.")
            if classification_status == "classified":
                label = row["perfminer_label_id"]
                score = row["perfminer_score"]
                predicted = row["perfminer_is_performance"]
                if label not in {0, 1} or score is None or not 0.0 <= float(score) <= 1.0:
                    raise ValueError("Classified commit has invalid label or score semantics.")
                expected_positive = float(score) >= PERFMINER_THRESHOLD
                if bool(predicted) != expected_positive or int(label) != int(expected_positive):
                    raise ValueError("Commit label, score, and threshold are inconsistent.")
            elif any(
                row[column] is not None
                for column in (
                    "perfminer_label_id",
                    "perfminer_is_performance",
                    "perfminer_score",
                )
            ):
                raise ValueError("Unclassified commit unexpectedly contains a prediction.")
            key = (int(row["repo_id"]), int(row["pr_number"]))
            commit_key = (*key, str(row["commit_sha"] or ""))
            if commit_key in seen:
                raise ValueError(f"Duplicate commit prediction row: {commit_key}")
            seen.add(commit_key)
            groups[key].append(row)
    if len(artifact_hashes) > 1:
        raise ValueError("Commit predictions mix multiple model artifacts.")
    return groups


def aggregate_row(pr: Mapping[str, object], commits: list[dict]) -> dict[str, object]:
    expected = pr.get("expected_commit_count")
    expected_count = int(expected) if expected is not None else None
    listed_count = int(pr.get("listed_commit_count") or 0)
    manifest_complete = str(pr.get("commit_manifest_status") or "") == "complete"
    if not manifest_complete and commits:
        raise ValueError("Incomplete PR manifest unexpectedly has commit predictions.")
    if commits:
        for row in commits:
            for column in (
                "resolved_repo_full_name",
                "pr_base_sha",
                "pr_head_sha",
                "commit_manifest_sha256",
            ):
                if str(row[column] or "") != str(pr.get(column) or ""):
                    raise ValueError(f"PR and commit predictions disagree on {column}.")
            if not bool(row["pr_manifest_complete"]):
                raise ValueError("Commit prediction is not bound to a complete PR manifest.")
        ordered_pairs = sorted(
            (int(row["commit_index"]), str(row["commit_sha"])) for row in commits
        )
        if [index for index, _ in ordered_pairs] != list(range(1, listed_count + 1)):
            raise ValueError("Commit prediction indices do not cover the frozen PR manifest.")
        if commit_manifest_sha256(ordered_pairs) != str(pr.get("commit_manifest_sha256") or ""):
            raise ValueError("Commit predictions do not match the frozen PR manifest digest.")
        if ordered_pairs[-1][1] != str(pr.get("pr_head_sha") or ""):
            raise ValueError("Commit predictions do not end at the frozen PR head.")
    eligible = [row for row in commits if row["perfminer_evidence_status"] == "eligible"]
    classified = [row for row in eligible if row["perfminer_classification_status"] == "classified"]
    excluded = [row for row in commits if row["perfminer_evidence_status"] == "excluded"]
    unavailable = [row for row in commits if row["perfminer_evidence_status"] == "unavailable"]
    if len(eligible) + len(excluded) + len(unavailable) != len(commits):
        raise ValueError("PerfMiner evidence statuses do not partition all commit rows.")
    positives = [row for row in classified if row["perfminer_is_performance"] is True]
    positive_shas = sorted({str(row["commit_sha"]) for row in positives})
    positive_sources = {
        str(row["source_pair_md5"])
        for row in positives
        if str(row.get("source_pair_md5") or "")
    }
    scores = [float(row["perfminer_score"]) for row in classified if row["perfminer_score"] is not None]
    row_coverage_complete = len(commits) == listed_count
    classification_complete = len(classified) == len(eligible)
    evidence_complete = (
        manifest_complete
        and row_coverage_complete
        and classification_complete
        and not unavailable
        and expected_count == listed_count
    )

    if positives:
        status = "performance_positive"
        is_performance: bool | None = True
    elif not evidence_complete:
        status = "unresolved"
        is_performance = None
    elif not eligible:
        status = "out_of_scope_no_eligible_commit"
        is_performance = None
    else:
        status = "no_positive_eligible_commit"
        is_performance = False

    model_hashes = {
        str(row["perfminer_model_weights_sha256"])
        for row in classified
        if str(row.get("perfminer_model_weights_sha256") or "")
    }
    if len(model_hashes) > 1:
        raise ValueError("A PR contains predictions from multiple model weight artifacts.")
    return {
        "perfminer_pr_status": status,
        "perfminer_pr_is_performance": is_performance,
        "perfminer_pr_evidence_complete": evidence_complete,
        "perfminer_expected_commit_count": expected_count,
        "perfminer_listed_commit_count": listed_count,
        "perfminer_eligible_commit_count": len(eligible),
        "perfminer_classified_commit_count": len(classified),
        "perfminer_excluded_commit_count": len(excluded),
        "perfminer_unavailable_commit_count": len(unavailable),
        "perfminer_positive_commit_count": len(positives),
        "perfminer_positive_commit_shas": positive_shas,
        "perfminer_positive_source_pair_count": len(positive_sources),
        "perfminer_max_commit_score": max(scores) if scores else None,
        "perfminer_pr_aggregation": "any_positive_commit",
        "perfminer_model_weights_sha256": next(iter(model_hashes), ""),
    }


def aggregate(pull_requests: Path, commit_predictions: Path) -> tuple[pa.Table, dict[str, int]]:
    source = pq.ParquetFile(pull_requests)
    schema = output_schema(source.schema_arrow)
    groups = commit_groups(commit_predictions)
    rows = []
    seen_prs: set[tuple[int, int]] = set()
    status_counts: Counter[str] = Counter()
    for batch in source.iter_batches(batch_size=50_000):
        for pr in pa.Table.from_batches([batch]).to_pylist():
            key = (int(pr["repo_id"]), int(pr["pr_number"]))
            if key in seen_prs:
                raise ValueError(f"Duplicate PR manifest row: {key}")
            seen_prs.add(key)
            aggregate_values = aggregate_row(pr, groups.pop(key, []))
            status_counts[aggregate_values["perfminer_pr_status"]] += 1
            rows.append({**pr, **aggregate_values})
    if groups:
        examples = ", ".join(f"{repo_id}#{number}" for repo_id, number in sorted(groups)[:5])
        raise ValueError(f"Commit predictions reference PRs absent from the manifest: {examples}")
    return pa.Table.from_pylist(rows, schema=schema), dict(sorted(status_counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate commit-level PerfAnnotator predictions to one conservative PR result."
    )
    parser.add_argument("--pull-requests", required=True, type=Path)
    parser.add_argument("--commit-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "pr_predictions.parquet"
    summary_file = args.output_dir / "pr_predictions.summary.json"
    state_file = args.output_dir / "pr_predictions.state.json"
    has_artifacts = any(path.exists() for path in (output, summary_file, state_file))
    if has_artifacts and not args.resume:
        raise FileExistsError(f"PR prediction artifacts already exist in {args.output_dir}; use --resume.")
    if args.resume and not state_file.exists() and has_artifacts:
        raise ValueError("Cannot resume PR prediction artifacts without matching state.")

    pr_source = pq.ParquetFile(args.pull_requests)
    expected_schema = output_schema(pr_source.schema_arrow)
    signature = {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "perfminer_revision": PERFMINER_REVISION,
        "pull_requests_path": str(args.pull_requests.resolve()),
        "pull_requests_sha256": sha256_file(args.pull_requests),
        "pull_requests_schema_sha256": schema_sha256(pr_source.schema_arrow),
        "commit_predictions_path": str(args.commit_predictions.resolve()),
        "commit_predictions_sha256": sha256_file(args.commit_predictions),
        "commit_predictions_schema_sha256": schema_sha256(
            pq.ParquetFile(args.commit_predictions).schema_arrow
        ),
        "code_sha256": {
            Path(__file__).name: sha256_file(Path(__file__)),
            "perfminer_reproduction.py": sha256_file(
                Path(__file__).with_name("perfminer_reproduction.py")
            ),
            "parquet_parts.py": sha256_file(Path(__file__).with_name("parquet_parts.py")),
        },
    }
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("run_signature") != signature:
            raise ValueError("Existing PR prediction run signature does not match the request.")
    else:
        state = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "run_signature": signature,
            "finalized": False,
        }
        write_json(state_file, state)

    if output.exists() and isinstance(state.get("output"), Mapping):
        output_metadata = state.get("output")
        validate_part(
            output,
            expected_schema,
            expected_rows=int(output_metadata.get("row_count", -1)),
            expected_sha256=str(output_metadata.get("sha256") or ""),
        )
        table = pq.read_table(output)
        status_counts = dict(
            sorted(Counter(table["perfminer_pr_status"].to_pylist()).items())
        )
    else:
        table, status_counts = aggregate(args.pull_requests, args.commit_predictions)
        output_metadata = write_table_part(output, table, table.schema)
        state["output"] = output_metadata
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(state_file, state)

    summary = {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "perfminer_revision": PERFMINER_REVISION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "pull_requests": {
                "path": str(args.pull_requests.resolve()),
                "sha256": sha256_file(args.pull_requests),
                "schema_sha256": schema_sha256(pr_source.schema_arrow),
            },
            "commit_predictions": {
                "path": str(args.commit_predictions.resolve()),
                "sha256": sha256_file(args.commit_predictions),
                "schema_sha256": schema_sha256(
                    pq.ParquetFile(args.commit_predictions).schema_arrow
                ),
            },
        },
        "row_count": table.num_rows,
        "status_counts": status_counts,
        "aggregation_rule": "any_positive_commit",
        "negative_requires_complete_evidence": True,
        "output": output_metadata,
        "code_sha256": signature["code_sha256"],
    }
    write_json(summary_file, summary)
    state["summary"] = {"path": str(summary_file.resolve()), "sha256": sha256_file(summary_file)}
    state["finalized"] = True
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(state_file, state)
    print(
        f"finalized_pr_predictions={table.num_rows} "
        f"positive_prs={status_counts.get('performance_positive', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
