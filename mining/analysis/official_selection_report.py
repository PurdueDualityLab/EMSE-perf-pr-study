"""Report and validate completed official-selection outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "official_selection"
PARQUET_PATHS = {
    "candidate_decisions": Path("candidate_decisions.parquet"),
    "official_population": Path("official_population.parquet"),
    "official_agentic": Path("official_agentic_prs.parquet"),
    "official_human_candidates": Path("official_human_non_agentic_candidates.parquet"),
    "date_filtered": Path("intermediate/date_filtered.parquet"),
    "heuristic_candidates": Path("intermediate/heuristic_candidates.parquet"),
    "quality_passed": Path("intermediate/quality_passed.parquet"),
    "model_predictions": Path("intermediate/model_predictions.parquet"),
}
SUMMARY_COUNT_KEYS = {
    "candidate_decisions": "after_heuristic",
    "official_population": "after_model",
    "official_agentic": "agentic",
    "official_human_candidates": "human_non_agentic_candidate",
    "date_filtered": "after_date",
    "heuristic_candidates": "after_heuristic",
    "quality_passed": "after_quality",
    "model_predictions": "after_quality",
}
REQUIRED_COLUMNS = {
    "candidate_decisions": {"html_url", "quality_passed", "selection_status"},
    "official_population": {"html_url", "arm", "arm_source"},
    "official_agentic": {"html_url", "arm", "arm_source"},
    "official_human_candidates": {"html_url", "arm", "arm_source"},
    "date_filtered": {"html_url", "created_at"},
    "heuristic_candidates": {"html_url", "heuristic_match", "heuristic_reason"},
    "quality_passed": {"html_url", "quality_passed", "quality_reason"},
    "model_predictions": {
        "html_url",
        "perfannotator_metadata_label_id",
        "perfannotator_metadata_is_performance_improving",
    },
}
STAGE_COUNT_KEYS = (
    "input",
    "after_date",
    "after_heuristic",
    "after_quality",
    "after_model",
    "agentic",
    "human_non_agentic_candidate",
)


def load_required_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"Missing required file: {path}")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Could not read required JSON file {path}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"Required JSON file must contain an object: {path}")
        return {}
    return payload


def parquet_row_count(path: Path, errors: list[str]) -> int | None:
    if not path.is_file():
        errors.append(f"Missing required file: {path}")
        return None
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        errors.append(f"Could not read parquet metadata for {path}: {exc}")
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def schema_sha256(path: Path) -> str:
    schema = pq.ParquetFile(path).schema_arrow
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def artifact_checksum_lock(output_dir: Path) -> dict[Path, str]:
    checksums: dict[Path, str] = {}
    candidates = (
        output_dir.parent / "artifacts.sha256",
        Path(f"{output_dir}.sha256"),
    )
    for checksum_path in candidates:
        if not checksum_path.is_file():
            continue
        path_base = (
            checksum_path.parent.parent
            if checksum_path.name == "artifacts.sha256"
            else checksum_path.parent
        )
        for raw_line in checksum_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            checksum, stored_path = parts
            parsed_path = Path(stored_path)
            resolved = (
                parsed_path.resolve()
                if parsed_path.is_absolute()
                else (path_base / parsed_path).resolve()
            )
            checksums[resolved] = checksum
    return checksums


def validate_artifact_integrity(
    output_dir: Path,
    manifest: dict[str, Any],
    files: dict[str, Path],
    errors: list[str],
) -> dict[str, Any]:
    locked = artifact_checksum_lock(output_dir)
    manifest_outputs = manifest.get("outputs", {})
    if not isinstance(manifest_outputs, dict):
        manifest_outputs = {}
    results: dict[str, Any] = {}
    for name, path in files.items():
        if not path.is_file():
            continue
        expected_hash = locked.get(path.resolve())
        recorded = (
            manifest.get("summary_artifact")
            if name == "selection_summary"
            else manifest_outputs.get(name)
        )
        if expected_hash is None and isinstance(recorded, dict):
            expected_hash = recorded.get("sha256")
        if not expected_hash:
            errors.append(f"No SHA-256 integrity metadata is available for {path}.")
            results[name] = {"verified": False, "source": None}
            continue
        actual_hash = sha256_file(path)
        source = "checksum_lock" if path.resolve() in locked else "run_manifest"
        verified = actual_hash == expected_hash
        results[name] = {"verified": verified, "source": source, "sha256": actual_hash}
        if not verified:
            errors.append(f"SHA-256 mismatch for {path}.")
            continue
        if path.suffix == ".parquet" and isinstance(recorded, dict):
            expected_rows = recorded.get("output_rows")
            actual_rows = int(pq.ParquetFile(path).metadata.num_rows)
            if expected_rows != actual_rows:
                errors.append(
                    f"Manifest row-count mismatch for {path}: expected {expected_rows}, found {actual_rows}."
                )
            expected_schema = recorded.get("schema_sha256")
            if not expected_schema:
                errors.append(f"No schema hash is recorded for {path}.")
            elif schema_sha256(path) != expected_schema:
                errors.append(f"Schema hash mismatch for {path}.")
    return results


def validated_stage_counts(summary: dict[str, Any], errors: list[str]) -> dict[str, int]:
    raw_counts = summary.get("stage_counts")
    if not isinstance(raw_counts, dict):
        errors.append("selection_summary.json must contain a stage_counts object.")
        return {}

    counts: dict[str, int] = {}
    for key in STAGE_COUNT_KEYS:
        value = raw_counts.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(
                f"selection_summary.json stage_counts.{key} must be a non-negative integer."
            )
            continue
        counts[key] = value
    return counts


def validate_manifest_metadata(manifest: dict[str, Any], errors: list[str]) -> None:
    signature = manifest.get("run_signature")
    if not isinstance(signature, dict):
        errors.append("run_manifest.json must contain a run_signature object.")
        return
    if signature.get("method_version") != 2:
        return
    for key in (
        "input_sha256",
        "input_schema_sha256",
        "model_artifact_sha256",
        "runtime",
        "code_sha256",
    ):
        if not signature.get(key):
            errors.append(f"V2 run signature is missing {key}.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        errors.append("V2 run manifest must contain output artifact metadata.")
        return
    missing_outputs = sorted(set(PARQUET_PATHS) - set(outputs))
    if missing_outputs:
        errors.append(
            "V2 run manifest is missing output metadata for: "
            + ", ".join(missing_outputs)
        )


def validate_stage_order(stage_counts: dict[str, int], errors: list[str]) -> None:
    ordered_keys = ("input", "after_date", "after_heuristic", "after_quality", "after_model")
    if not all(key in stage_counts for key in ordered_keys):
        return
    for previous, current in zip(ordered_keys, ordered_keys[1:]):
        if stage_counts[current] > stage_counts[previous]:
            errors.append(
                "Selection stage counts must not increase: "
                f"{current}={stage_counts[current]} exceeds {previous}={stage_counts[previous]}."
            )


def validate_required_columns(
    parquet_paths: dict[str, Path], errors: list[str]
) -> None:
    for name, path in parquet_paths.items():
        if not path.is_file():
            continue
        try:
            available = set(pq.ParquetFile(path).schema_arrow.names)
        except Exception:
            continue
        missing = sorted(REQUIRED_COLUMNS[name] - available)
        if missing:
            errors.append(
                f"Required columns missing from {path}: " + ", ".join(missing)
            )


def partition_details(
    total: int | None,
    agentic: int | None,
    human: int | None,
) -> dict[str, int | bool | None]:
    matches = None
    if total is not None and agentic is not None and human is not None:
        matches = total == agentic + human
    return {
        "total": total,
        "agentic": agentic,
        "human_non_agentic_candidate": human,
        "matches": matches,
    }


def build_report(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "selection_summary.json"
    manifest_path = output_dir / "run_manifest.json"
    parquet_paths = {
        name: output_dir / relative_path for name, relative_path in PARQUET_PATHS.items()
    }
    files = {
        "selection_summary": summary_path,
        "run_manifest": manifest_path,
        **parquet_paths,
    }
    errors: list[str] = []

    summary = load_required_json(summary_path, errors)
    manifest = load_required_json(manifest_path, errors)
    validate_manifest_metadata(manifest, errors)
    stage_counts = validated_stage_counts(summary, errors)
    validate_stage_order(stage_counts, errors)

    manifest_status = manifest.get("status")
    if manifest_status != "completed":
        errors.append(
            "run_manifest.json status must be 'completed'; "
            f"found {manifest_status!r}."
        )

    parquet_counts = {
        name: parquet_row_count(path, errors) for name, path in parquet_paths.items()
    }
    validate_required_columns(parquet_paths, errors)
    for parquet_name, summary_key in SUMMARY_COUNT_KEYS.items():
        actual = parquet_counts[parquet_name]
        expected = stage_counts.get(summary_key)
        if actual is not None and expected is not None and actual != expected:
            errors.append(
                f"Count mismatch for {PARQUET_PATHS[parquet_name]}: "
                f"parquet metadata={actual}, stage_counts.{summary_key}={expected}."
            )

    summary_partition = partition_details(
        stage_counts.get("after_model"),
        stage_counts.get("agentic"),
        stage_counts.get("human_non_agentic_candidate"),
    )
    if summary_partition["matches"] is False:
        errors.append(
            "Summary count partition is invalid: after_model must equal "
            "agentic + human_non_agentic_candidate."
        )

    parquet_partition = partition_details(
        parquet_counts["official_population"],
        parquet_counts["official_agentic"],
        parquet_counts["official_human_candidates"],
    )
    if parquet_partition["matches"] is False:
        errors.append(
            "Parquet count partition is invalid: official_population must equal "
            "official_agentic + official_human_non_agentic_candidates."
        )

    integrity = validate_artifact_integrity(
        output_dir,
        manifest,
        {
            "selection_summary": summary_path,
            "run_manifest": manifest_path,
            **parquet_paths,
        },
        errors,
    )

    return {
        "output_dir": str(output_dir),
        "run": {
            "status": manifest_status,
            "started_at": manifest.get("started_at"),
            "resumed_at": manifest.get("resumed_at"),
            "completed_at": manifest.get("completed_at"),
            "run_signature": manifest.get("run_signature"),
            "reproducibility": manifest.get("reproducibility"),
        },
        "window": summary.get("window"),
        "sampling": summary.get("sampling"),
        "model": summary.get("model"),
        "stage_counts": stage_counts,
        "parquet_row_counts": parquet_counts,
        "partition": {
            "summary": summary_partition,
            "parquet": parquet_partition,
        },
        "integrity": integrity,
        "validation": {
            "valid": not errors,
            "errors": errors,
        },
        "files": {name: str(path) for name, path in files.items()},
    }


def display_count(value: object) -> object:
    return "missing" if value is None else value


def print_report(report: dict[str, Any]) -> None:
    print("# Official Selection Report")
    print()
    print(f"Output directory: {report['output_dir']}")
    print(f"Run status: {report.get('run', {}).get('status') or 'missing'}")
    validation = report.get("validation", {})
    print(f"Validation: {'passed' if validation.get('valid') else 'failed'}")
    print()

    errors = validation.get("errors", [])
    if errors:
        print("Validation errors")
        for error in errors:
            print(f"- {error}")
        print()

    window = report.get("window")
    if isinstance(window, dict):
        suffix = " (inclusive)" if window.get("inclusive") else ""
        print(f"Window: {window.get('start')} -> {window.get('end')}{suffix}")
    sampling = report.get("sampling")
    if isinstance(sampling, dict):
        print(f"Sampling performed: {sampling.get('performed')}")
    if isinstance(window, dict) or isinstance(sampling, dict):
        print()

    print("Stage counts")
    stage_counts = report.get("stage_counts", {})
    for key in STAGE_COUNT_KEYS:
        print(f"- {key}: {display_count(stage_counts.get(key))}")
    print()

    print("Parquet row counts")
    for key, value in report.get("parquet_row_counts", {}).items():
        print(f"- {key}: {display_count(value)}")
    print()

    print("Count partitions")
    for source, partition in report.get("partition", {}).items():
        print(
            f"- {source}: total={display_count(partition.get('total'))}, "
            f"agentic={display_count(partition.get('agentic'))}, "
            "human_non_agentic_candidate="
            f"{display_count(partition.get('human_non_agentic_candidate'))}, "
            f"matches={display_count(partition.get('matches'))}"
        )
    print()

    print("Files")
    for label, path in report.get("files", {}).items():
        print(f"- {label}: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate official-selection outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory containing official-selection outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as formatted JSON instead of text.",
    )
    args = parser.parse_args(argv)

    report = build_report(args.output_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0 if report["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
