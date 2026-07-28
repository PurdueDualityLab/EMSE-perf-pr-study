"""Build the reproducible official PR population without sampling.

The input is the full mined PR parquet. The pipeline applies, in order:
date filtering, the project heuristic, quality filters, PerfAnnotator-mini,
and author-arm assignment. Sampling is deliberately not part of this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from author_filter import is_agent_authored
from classify_task_type import CONVENTIONAL_TYPES, PERF_TERMS, compatible_task_type
from quality_filters import quality_filter_flags


DEFAULT_START = "2024-12-24T00:23:09+00:00"
DEFAULT_END = "2025-07-30T19:36:13+00:00"
DEFAULT_MODEL = "annon-123/PerfAnnotator-mini"
DEFAULT_MODEL_REVISION = "7d7ba362c257c3ea8c69d52b4a736ae4f182e68c"
DEFAULT_AIDEV_DATASET = "dysavepeople/AIDev"
DEFAULT_AIDEV_REVISION = "b6d1b8af952053f20fd9f9aa03e66bddd12228fe"
DEFAULT_SEED = 20260720
QUALITY_REASONS = ("empty_filename", "config_only", "deleted_repo", "merge_only")


def log(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def write_dataframe_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    if temporary.exists():
        temporary.unlink()
    frame.to_parquet(temporary, index=False, compression="snappy")
    os.replace(temporary, path)


def row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def text_value(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def normalized_url(value: object) -> str:
    return text_value(value).strip().rstrip("/").lower()


def inclusive_date_mask(values: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    return timestamps.notna() & timestamps.between(start, end, inclusive="both")


def heuristic_result(title: object, body: object) -> tuple[bool, str, float]:
    """Apply the project heuristic to a PR title and body."""
    task_type, confidence, reason = compatible_task_type(
        {"title": text_value(title), "body": text_value(body)}
    )
    return task_type == "perf", reason, confidence


def add_heuristic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    results = [
        heuristic_result(title, body)
        for title, body in zip(frame["title"], frame["body"], strict=True)
    ]
    result = frame.copy()
    result["heuristic_match"] = [item[0] for item in results]
    result["heuristic_reason"] = [item[1] for item in results]
    result["heuristic_confidence"] = [item[2] for item in results]
    return result


def add_quality_decisions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    flags = quality_filter_flags(frame)
    result = frame.copy()
    reason = pd.Series(pd.NA, index=result.index, dtype="string")
    for name in QUALITY_REASONS:
        result[f"quality_{name}"] = flags[name]
        reason.loc[reason.isna() & flags[name]] = name
    result["quality_passed"] = reason.isna()
    result["quality_reason"] = reason.fillna("passed")
    removed_counts = {name: int((result["quality_reason"] == name).sum()) for name in QUALITY_REASONS}
    flag_counts = {name: int(flags[name].sum()) for name in QUALITY_REASONS}
    return result, removed_counts, flag_counts


def filter_dates(input_path: Path, output_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, int]:
    source = pq.ParquetFile(input_path)
    if "created_at" not in source.schema_arrow.names:
        raise ValueError(f"Input parquet has no created_at column: {input_path}")
    temporary = temporary_path(output_path)
    if temporary.exists():
        temporary.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_rows = int(source.metadata.num_rows)
    valid_dates = 0
    kept_rows = 0
    writer = pq.ParquetWriter(temporary, source.schema_arrow, compression="snappy")
    try:
        for batch in source.iter_batches(batch_size=50_000):
            created_at = pd.Series(batch.column("created_at").to_pylist())
            mask = inclusive_date_mask(created_at, start, end)
            valid_dates += int(pd.to_datetime(created_at, utc=True, errors="coerce").notna().sum())
            kept_rows += int(mask.sum())
            if mask.any():
                table = pa.Table.from_batches([batch]).filter(pa.array(mask.to_numpy(dtype=bool)))
                writer.write_table(table)
    finally:
        writer.close()
    os.replace(temporary, output_path)
    return {
        "input_rows": input_rows,
        "valid_created_at_rows": valid_dates,
        "invalid_created_at_rows": input_rows - valid_dates,
        "output_rows": kept_rows,
    }


def validate_pr_urls(path: Path) -> dict[str, int]:
    source = pq.ParquetFile(path)
    if "html_url" not in source.schema_arrow.names:
        raise ValueError(f"Input parquet has no html_url column: {path}")
    seen: set[str] = set()
    missing = 0
    duplicate = 0
    for batch in source.iter_batches(columns=["html_url"], batch_size=50_000):
        for value in batch.column("html_url").to_pylist():
            url = normalized_url(value)
            if not url:
                missing += 1
            elif url in seen:
                duplicate += 1
            else:
                seen.add(url)
    if missing or duplicate:
        raise ValueError(
            f"Date-filtered input must have unique html_url values; missing={missing}, duplicate={duplicate}."
        )
    return {"missing_html_url_rows": missing, "duplicate_html_url_rows": duplicate, "unique_html_url_rows": len(seen)}


def filter_heuristic(input_path: Path, output_path: Path) -> dict[str, Any]:
    source = pq.ParquetFile(input_path)
    required = {"title", "body"}
    missing = required - set(source.schema_arrow.names)
    if missing:
        raise ValueError(f"Input parquet is missing heuristic columns: {sorted(missing)}")
    schema = source.schema_arrow.append(pa.field("heuristic_match", pa.bool_()))
    schema = schema.append(pa.field("heuristic_reason", pa.string()))
    schema = schema.append(pa.field("heuristic_confidence", pa.float64()))
    temporary = temporary_path(output_path)
    if temporary.exists():
        temporary.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    matched = 0
    reasons: dict[str, int] = {}
    writer = pq.ParquetWriter(temporary, schema, compression="snappy")
    try:
        for batch in source.iter_batches(batch_size=25_000):
            titles = batch.column("title").to_pylist()
            bodies = batch.column("body").to_pylist()
            results = [heuristic_result(title, body) for title, body in zip(titles, bodies, strict=True)]
            matches = [item[0] for item in results]
            processed += len(results)
            matched += sum(matches)
            for keep, reason, _ in results:
                if keep:
                    reasons[reason] = reasons.get(reason, 0) + 1
            if not any(matches):
                continue
            table = pa.Table.from_batches([batch]).filter(pa.array(matches))
            selected = [item for item in results if item[0]]
            table = table.append_column("heuristic_match", pa.array([True] * len(selected), type=pa.bool_()))
            table = table.append_column("heuristic_reason", pa.array([item[1] for item in selected], type=pa.string()))
            table = table.append_column(
                "heuristic_confidence",
                pa.array([item[2] for item in selected], type=pa.float64()),
            )
            writer.write_table(table)
    finally:
        writer.close()
    os.replace(temporary, output_path)
    return {"input_rows": processed, "output_rows": matched, "matched_reasons": reasons}


def run_quality_filter(input_path: Path, output_path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(input_path)
    evaluated, removed_counts, flag_counts = add_quality_decisions(frame)
    passed = evaluated.loc[evaluated["quality_passed"]].copy()
    write_dataframe_atomic(passed, output_path)
    return {
        "input_rows": int(len(frame)),
        "output_rows": int(len(passed)),
        "removed_counts": removed_counts,
        "flag_counts": flag_counts,
    }


def run_model(
    quality_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    classifier = Path(__file__).with_name("classify_perfannotator_metadata.py")
    command = [
        sys.executable,
        str(classifier),
        "--input",
        str(quality_path),
        "--output",
        str(output_path),
        "--model",
        args.model,
        "--model-revision",
        args.model_revision,
        "--batch-size",
        str(args.model_batch_size),
        "--row-batch-size",
        str(args.model_row_batch_size),
        "--device",
        args.device,
        "--max-input-chars",
        str(args.max_input_chars),
        "--seed",
        str(args.seed),
        "--deterministic",
    ]
    if args.resume:
        command.append("--resume")
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(args.seed)
    log("[4/6] Running PerfAnnotator-mini on quality-passed PRs")
    subprocess.run(command, check=True, env=environment)


def aidev_table_url(dataset: str, revision: str, filename: str) -> str:
    return f"hf://datasets/{dataset}@{revision}/{filename}"


def aidev_url_sets(dataset: str, revision: str) -> tuple[set[str], set[str], int]:
    agent = pd.read_parquet(aidev_table_url(dataset, revision, "pull_request.parquet"), columns=["html_url"])
    human = pd.read_parquet(
        aidev_table_url(dataset, revision, "human_pull_request.parquet"), columns=["html_url"]
    )
    agent_urls = {normalized_url(value) for value in agent["html_url"] if normalized_url(value)}
    human_urls = {normalized_url(value) for value in human["html_url"] if normalized_url(value)}
    overlap = agent_urls & human_urls
    return agent_urls, human_urls, len(overlap)


def assign_author_arms(
    frame: pd.DataFrame,
    agent_urls: set[str],
    human_urls: set[str],
) -> pd.DataFrame:
    arms: list[str] = []
    sources: list[str] = []
    source_urls = frame.get("aidev_source_html_url", pd.Series("", index=frame.index))
    for (_, row), source_url in zip(frame.iterrows(), source_urls, strict=True):
        urls = {normalized_url(row.get("html_url")), normalized_url(source_url)} - {""}
        if urls & agent_urls:
            arms.append("agentic")
            sources.append("aidev_pull_request")
        elif urls & human_urls:
            arms.append("human_non_agentic_candidate")
            sources.append("aidev_human_pull_request")
        elif is_agent_authored(row.to_dict()):
            arms.append("agentic")
            sources.append("author_heuristic_agentic")
        else:
            arms.append("human_non_agentic_candidate")
            sources.append("author_heuristic_non_agentic")
    result = frame.copy()
    result["arm"] = arms
    result["arm_source"] = sources
    return result


def build_decisions(heuristic_path: Path, model_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidates = pd.read_parquet(heuristic_path)
    decisions, removed_counts, flag_counts = add_quality_decisions(candidates)
    predictions = pd.read_parquet(model_path)
    prediction_columns = [
        column for column in predictions.columns if column.startswith("perfannotator_metadata_")
    ]
    expected = decisions.loc[decisions["quality_passed"], "html_url"]
    if len(predictions) != len(expected) or predictions["html_url"].duplicated().any():
        raise ValueError("PerfAnnotator output does not match the quality-passed input population.")
    if set(predictions["html_url"]) != set(expected):
        raise ValueError("PerfAnnotator output URLs do not match the quality-passed input population.")
    decisions = decisions.merge(
        predictions[["html_url", *prediction_columns]],
        on="html_url",
        how="left",
        validate="one_to_one",
    )
    positive = decisions["quality_passed"] & decisions[
        "perfannotator_metadata_is_performance_improving"
    ].fillna(False)
    decisions["model_status"] = "not_run_quality_rejected"
    decisions.loc[decisions["quality_passed"], "model_status"] = "not_performance_improving"
    decisions.loc[positive, "model_status"] = "performance_improving"
    decisions["selection_status"] = "quality_rejected"
    decisions.loc[decisions["quality_passed"], "selection_status"] = "model_rejected"
    decisions.loc[positive, "selection_status"] = "selected"
    selected = decisions.loc[positive].copy()
    summary = {
        "input_rows": int(len(candidates)),
        "quality_passed_rows": int(decisions["quality_passed"].sum()),
        "model_positive_rows": int(positive.sum()),
        "model_negative_rows": int(
            (decisions["quality_passed"] & ~positive).sum()
        ),
        "quality_removed_counts": removed_counts,
        "quality_flag_counts": flag_counts,
    }
    return decisions, selected, summary


def output_paths(output_dir: Path) -> dict[str, Path]:
    intermediate = output_dir / "intermediate"
    return {
        "date_filtered": intermediate / "date_filtered.parquet",
        "heuristic_candidates": intermediate / "heuristic_candidates.parquet",
        "quality_passed": intermediate / "quality_passed.parquet",
        "model_predictions": intermediate / "model_predictions.parquet",
        "candidate_decisions": output_dir / "candidate_decisions.parquet",
        "official_population": output_dir / "official_population.parquet",
        "official_agentic": output_dir / "official_agentic_prs.parquet",
        "official_human_candidates": output_dir / "official_human_non_agentic_candidates.parquet",
        "summary": output_dir / "selection_summary.json",
        "manifest": output_dir / "run_manifest.json",
    }


def selection_signature(args: argparse.Namespace, input_hash: str) -> dict[str, Any]:
    return {
        "input_path": str(args.input.resolve()),
        "input_sha256": input_hash,
        "window": {"start": args.start, "end": args.end},
        "model": args.model,
        "model_revision": args.model_revision,
        "device": args.device,
        "seed": args.seed,
        "model_batch_size": args.model_batch_size,
        "model_row_batch_size": args.model_row_batch_size,
        "max_input_chars": args.max_input_chars,
        "aidev_dataset": args.aidev_dataset,
        "aidev_revision": args.aidev_revision,
        "script_sha256": sha256_file(Path(__file__)),
    }


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("pandas", "pyarrow", "torch", "transformers", "huggingface_hub")
        },
    }


def load_or_create_manifest(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, Any]:
    log("Calculating SHA-256 for the input parquet")
    signature = selection_signature(args, sha256_file(args.input))
    manifest_path = paths["manifest"]
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"Official-selection output already exists: {manifest_path.parent}. Use --resume to continue it."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_signature = manifest.get("run_signature", {})
        if existing_signature != signature:
            existing_without_script = dict(existing_signature)
            requested_without_script = dict(signature)
            previous_script_hash = existing_without_script.pop("script_sha256", None)
            requested_script_hash = requested_without_script.pop("script_sha256", None)
            if (
                existing_without_script != requested_without_script
                or not args.accept_script_update
            ):
                raise ValueError(
                    "Existing official-selection manifest does not match the requested run. "
                    "Use --accept-script-update only when the parameters are unchanged and a code fix "
                    "must resume the same selection."
                )
            manifest.setdefault("implementation_updates", []).append(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "previous_script_sha256": previous_script_hash,
                    "current_script_sha256": requested_script_hash,
                }
            )
            manifest["run_signature"] = signature
        manifest["status"] = "running"
        manifest["resumed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        return manifest

    if any(paths["manifest"].parent.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {paths['manifest'].parent}. Choose an empty directory or use --resume."
        )
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_signature": signature,
        "reproducibility": {
            "inference_device": "cpu",
            "seed": args.seed,
            "deterministic_algorithms": True,
            "model_eval_mode": True,
            "model_revision": args.model_revision,
            "aidev_revision": args.aidev_revision,
        },
        "runtime": runtime_metadata(),
        "stages": {},
    }
    write_json(manifest_path, manifest)
    return manifest


def stage_stats_from_existing(path: Path) -> dict[str, int]:
    return {"output_rows": row_count(path), "resumed": True}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the official PR population from a full mined parquet without sampling."
    )
    parser.add_argument("--input", required=True, type=Path, help="Full mined PR parquet.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("mining/official_selection"), help="Empty output directory."
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Inclusive UTC start timestamp.")
    parser.add_argument("--end", default=DEFAULT_END, help="Inclusive UTC end timestamp.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-batch-size", type=int, default=16)
    parser.add_argument("--model-row-batch-size", type=int, default=512)
    parser.add_argument("--max-input-chars", type=int, default=20_000)
    parser.add_argument("--aidev-dataset", default=DEFAULT_AIDEV_DATASET)
    parser.add_argument("--aidev-revision", default=DEFAULT_AIDEV_REVISION)
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run with the same manifest.")
    parser.add_argument(
        "--accept-script-update",
        action="store_true",
        help="Explicitly resume after a code-only correction while preserving all selection parameters.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input parquet not found: {args.input}")
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("--start and --end must include an explicit UTC offset.")
    if end < start:
        raise ValueError("--end must be after --start.")
    args.start = start.isoformat()
    args.end = end.isoformat()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.output_dir)
    manifest = load_or_create_manifest(paths, args)

    if paths["date_filtered"].exists() and args.resume:
        date_stats = manifest["stages"].get("date", stage_stats_from_existing(paths["date_filtered"]))
        log(f"[1/6] Reusing date-filtered parquet ({row_count(paths['date_filtered'])} rows)")
    else:
        log("[1/6] Filtering the full mined parquet by the inclusive UTC window")
        date_stats = filter_dates(args.input, paths["date_filtered"], start, end)
        date_stats["url_validation"] = validate_pr_urls(paths["date_filtered"])
    manifest["stages"]["date"] = date_stats
    write_json(paths["manifest"], manifest)

    if paths["heuristic_candidates"].exists() and args.resume:
        heuristic_stats = manifest["stages"].get(
            "heuristic", stage_stats_from_existing(paths["heuristic_candidates"])
        )
        log(f"[2/6] Reusing heuristic candidates ({row_count(paths['heuristic_candidates'])} rows)")
    else:
        log("[2/6] Applying the heuristic to title and body")
        heuristic_stats = filter_heuristic(paths["date_filtered"], paths["heuristic_candidates"])
    manifest["stages"]["heuristic"] = heuristic_stats
    write_json(paths["manifest"], manifest)

    if paths["quality_passed"].exists() and args.resume:
        quality_stats = manifest["stages"].get("quality", stage_stats_from_existing(paths["quality_passed"]))
        log(f"[3/6] Reusing quality-passed PRs ({row_count(paths['quality_passed'])} rows)")
    else:
        log("[3/6] Applying PR-level quality filters")
        quality_stats = run_quality_filter(paths["heuristic_candidates"], paths["quality_passed"])
    manifest["stages"]["quality"] = quality_stats
    write_json(paths["manifest"], manifest)

    if paths["model_predictions"].exists() and args.resume:
        log(f"[4/6] Reusing PerfAnnotator predictions ({row_count(paths['model_predictions'])} rows)")
    else:
        run_model(paths["quality_passed"], paths["model_predictions"], args)
    manifest["stages"]["model"] = {"output_rows": row_count(paths["model_predictions"])}
    write_json(paths["manifest"], manifest)

    log("[5/6] Building candidate decisions and separating author arms")
    decisions, selected, decision_stats = build_decisions(
        paths["heuristic_candidates"], paths["model_predictions"]
    )
    agent_urls, human_urls, aidev_url_overlap_count = aidev_url_sets(
        args.aidev_dataset, args.aidev_revision
    )
    population = assign_author_arms(selected, agent_urls, human_urls)
    population = population.sort_values(["created_at", "html_url"], kind="stable").reset_index(drop=True)
    agentic = population.loc[population["arm"] == "agentic"].copy()
    human_candidates = population.loc[
        population["arm"] == "human_non_agentic_candidate"
    ].copy()
    decisions = decisions.merge(
        population[["html_url", "arm", "arm_source"]],
        on="html_url",
        how="left",
        validate="one_to_one",
    )

    write_dataframe_atomic(decisions, paths["candidate_decisions"])
    write_dataframe_atomic(population, paths["official_population"])
    write_dataframe_atomic(agentic, paths["official_agentic"])
    write_dataframe_atomic(human_candidates, paths["official_human_candidates"])

    summary = {
        "window": {"start": args.start, "end": args.end, "inclusive": True},
        "unit_of_analysis": "pull_request",
        "sampling": {"performed": False, "reason": "Sampling parameters have not been selected."},
        "heuristic": {
            "conventional_commit_types": sorted(CONVENTIONAL_TYPES),
            "performance_terms": list(PERF_TERMS),
            "rule": "A recognized Conventional Commit prefix takes precedence; otherwise title and body are checked for performance terms.",
        },
        "quality_filter_order": list(QUALITY_REASONS),
        "model": {
            "id": args.model,
            "revision": args.model_revision,
            "device": args.device,
            "seed": args.seed,
            "deterministic_algorithms": True,
            "positive_label": "performance_improving",
        },
        "author_arms": {
            "agentic": "AIDev pull_request URL match, then local agent signals.",
            "human_non_agentic_candidate": "AIDev human_pull_request URL match, then no local agent signal.",
            "aidev_agent_url_count": len(agent_urls),
            "aidev_human_url_count": len(human_urls),
            "aidev_url_overlap_count": aidev_url_overlap_count,
            "overlap_policy": "agentic precedence",
        },
        "stage_counts": {
            "input": int(pq.ParquetFile(args.input).metadata.num_rows),
            "after_date": row_count(paths["date_filtered"]),
            "after_heuristic": row_count(paths["heuristic_candidates"]),
            "after_quality": row_count(paths["quality_passed"]),
            "after_model": int(len(population)),
            "agentic": int(len(agentic)),
            "human_non_agentic_candidate": int(len(human_candidates)),
        },
        "quality": decision_stats,
        "arm_sources": {
            str(key): int(value)
            for key, value in population["arm_source"].value_counts(dropna=False).to_dict().items()
        },
        "outputs": {name: str(path) for name, path in paths.items() if name not in {"summary", "manifest"}},
    }
    write_json(paths["summary"], summary)

    manifest["stages"]["decisions"] = decision_stats
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["summary"] = str(paths["summary"])
    write_json(paths["manifest"], manifest)
    log("[6/6] Official population written without sampling")
    log(
        f"  after model={len(population)} agentic={len(agentic)} "
        f"human/non-agentic candidates={len(human_candidates)}"
    )


if __name__ == "__main__":
    main()
