from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


class ReportInputError(RuntimeError):
    pass


def load_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        parquet = pq.ParquetFile(path)
        available = [column for column in columns if column in parquet.schema_arrow.names]
        if not available:
            return pd.DataFrame()
        return parquet.read(columns=available).to_pandas()
    except Exception as exc:
        raise ReportInputError(f"Could not read parquet file {path}: {exc}") from exc


def parquet_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        raise ReportInputError(f"Could not read parquet metadata for {path}: {exc}") from exc


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportInputError(f"Expected a JSON object in {path}.")
    return payload


def safe_unique_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns or df.empty:
        return 0
    return int(df[column].dropna().nunique())


def safe_date_range(df: pd.DataFrame, column: str) -> dict[str, str] | None:
    if column not in df.columns or df.empty:
        return None
    series = pd.to_datetime(df[column], errors="coerce", utc=True).dropna()
    if series.empty:
        return None
    return {
        "start": series.min().isoformat(),
        "end": series.max().isoformat(),
    }


def value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns or df.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in df[column].fillna("missing").value_counts().to_dict().items()
    }


def build_report(outputs_dir: Path) -> dict[str, Any]:
    final_dir = outputs_dir / "final"
    intermediate_dir = outputs_dir / "intermediate"
    raw_dir = outputs_dir / "raw"

    summary_path = final_dir / "rebalancing_summary.json"
    detailed_path = final_dir / "rebalancing_detailed_report.json"
    agent_path = final_dir / "agent_perf_prs_matched_criteria.parquet"
    human_path = final_dir / "human_perf_prs_matched_criteria.parquet"
    human_typed_path = intermediate_dir / "human_prs_task_typed.parquet"
    human_filtered_path = intermediate_dir / "human_prs_filtered.parquet"
    raw_github_path = raw_dir / "github_human_prs.parquet"
    expected_paths = (
        summary_path,
        detailed_path,
        agent_path,
        human_path,
        human_typed_path,
        human_filtered_path,
        raw_github_path,
    )
    if not any(path.is_file() for path in expected_paths):
        expected = ", ".join(str(path.relative_to(outputs_dir)) for path in expected_paths)
        raise FileNotFoundError(
            f"No rebalancing outputs found in {outputs_dir}. Expected at least one of: {expected}"
        )
    missing_paths = [path for path in expected_paths if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path.relative_to(outputs_dir)) for path in missing_paths)
        raise FileNotFoundError(f"Incomplete rebalancing outputs in {outputs_dir}; missing: {missing}")

    summary = load_json(summary_path)
    detailed = load_json(detailed_path)
    summary_stage_counts = summary.get("stage_counts") or {}
    detailed_stage_counts = detailed.get("stage_counts") or {}
    if set(summary_stage_counts) != set(detailed_stage_counts):
        missing_from_detailed = sorted(
            set(summary_stage_counts) - set(detailed_stage_counts)
        )
        extra_in_detailed = sorted(
            set(detailed_stage_counts) - set(summary_stage_counts)
        )
        details = []
        if missing_from_detailed:
            details.append("missing from detailed: " + ", ".join(missing_from_detailed))
        if extra_in_detailed:
            details.append("extra in detailed: " + ", ".join(extra_in_detailed))
        raise ReportInputError("Stage count keys differ (" + "; ".join(details) + ").")
    for key in sorted(set(summary_stage_counts).intersection(detailed_stage_counts)):
        if summary_stage_counts[key] != detailed_stage_counts[key]:
            raise ReportInputError(
                f"Stage count {key} differs between summary "
                f"({summary_stage_counts[key]}) and detailed report "
                f"({detailed_stage_counts[key]})."
            )
    agent_rows = parquet_row_count(agent_path)
    human_rows = parquet_row_count(human_path)
    human_typed_rows = parquet_row_count(human_typed_path)
    human_filtered_rows = parquet_row_count(human_filtered_path)
    raw_github_rows = parquet_row_count(raw_github_path)
    expected_counts = {
        "agent final": summary.get("agent_perf_prs_after_quality_filters"),
        "human final": summary.get("human_perf_prs_after_quality_filters"),
        "raw GitHub": summary_stage_counts.get("raw_github_human_prs"),
    }
    actual_counts = {
        "agent final": agent_rows,
        "human final": human_rows,
        "raw GitHub": raw_github_rows,
    }
    for label, expected in expected_counts.items():
        if expected is None or isinstance(expected, bool) or not isinstance(expected, int):
            raise ReportInputError(f"Summary is missing a valid {label} row count.")
        if expected != actual_counts[label]:
            raise ReportInputError(
                f"{label} row-count mismatch: summary={expected}, parquet={actual_counts[label]}."
            )
    if human_filtered_rows != human_rows:
        raise ReportInputError(
            "Human filtered/final row-count mismatch: "
            f"intermediate={human_filtered_rows}, final={human_rows}."
        )
    agent = load_parquet_columns(agent_path, ["repo_id", "user", "created_at"])
    human = load_parquet_columns(human_path, ["repo_id", "user", "created_at"])
    human_typed = load_parquet_columns(human_typed_path, ["task_type_source", "task_type"])

    source_counts = detailed.get("source_counts") or summary.get("source_counts", {})
    stage_counts = detailed.get("stage_counts") or summary.get("stage_counts", {})
    repo_mining = detailed.get("repo_mining", {})
    enrichment_reports = detailed.get("enrichment_reports", {})
    author_filter = detailed.get("author_filter", {})
    quality_filter_removed_records = detailed.get("quality_filter_removed_records", {})
    repo_failures = [item for item in repo_mining.get("reports", []) if item.get("status") != "completed"]
    agent_enrichment = enrichment_reports.get("agent", {})
    human_enrichment = enrichment_reports.get("human", {})
    author_removed_counts = author_filter.get("removed_counts", {})
    quality_removed_counts = summary.get("quality_filter_removed_counts", {})

    report = {
        "methodological_note": summary.get("methodological_note"),
        "time_window": detailed.get("time_window") or summary.get("time_window"),
        "repositories_at_or_above_star_floor": detailed.get("repositories_at_or_above_star_floor", summary.get("repositories_at_or_above_star_floor", 0)),
        "source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "stage_counts": {str(key): int(value) for key, value in stage_counts.items()},
        "counts": {
            "agent_perf_prs": agent_rows,
            "human_perf_prs": human_rows,
            "total_perf_prs": agent_rows + human_rows,
            "agent_before_quality_filters": int(
                summary.get("agent_perf_prs_before_quality_filters", agent_rows)
            ),
            "agent_after_quality_filters": int(
                summary.get("agent_perf_prs_after_quality_filters", agent_rows)
            ),
            "human_before_quality_filters": int(
                summary.get("human_perf_prs_before_quality_filters", human_typed_rows)
            ),
            "human_after_quality_filters": human_filtered_rows,
            "raw_github_human_prs": raw_github_rows,
            "repo_mining_failures": int(len(repo_failures)),
            "pr_fetch_failures": int(len(detailed.get("pr_fetch_failures", []))),
            "agent_enrichment_failures": int(agent_enrichment.get("failed", 0)),
            "human_enrichment_failures": int(human_enrichment.get("failed", 0)),
            "agent_author_filter_removed": int(author_removed_counts.get("agent", 0)),
            "human_author_filter_removed": int(author_removed_counts.get("human", 0)),
            "agent_quality_filter_removed": int(
                sum(int(value) for value in summary.get("agent_quality_filter_removed_counts", {}).values())
            ),
            "human_quality_filter_removed": int(
                sum(int(value) for value in summary.get("human_quality_filter_removed_counts", {}).values())
            ),
        },
        "unique_counts": {
            "agent_repos": safe_unique_count(agent, "repo_id"),
            "human_repos": safe_unique_count(human, "repo_id"),
            "agent_authors": safe_unique_count(agent, "user"),
            "human_authors": safe_unique_count(human, "user"),
        },
        "date_ranges": {
            "agent_created_at": safe_date_range(agent, "created_at"),
            "human_created_at": safe_date_range(human, "created_at"),
        },
        "breakdowns": {
            "human_task_type_sources": value_counts(human_typed, "task_type_source"),
            "human_task_types": value_counts(human_typed, "task_type"),
            "quality_filter_removed_counts": {
                str(key): int(value) for key, value in quality_removed_counts.items()
            },
            "author_filter_removed_counts": {
                str(key): int(value) for key, value in author_removed_counts.items()
            },
            "agent_quality_filter_removed_counts": {
                str(key): int(value) for key, value in summary.get("agent_quality_filter_removed_counts", {}).items()
            },
            "human_quality_filter_removed_counts": {
                str(key): int(value) for key, value in summary.get("human_quality_filter_removed_counts", {}).items()
            },
            "source_counts": source_counts,
            "stage_counts": stage_counts,
        },
        "failures": {
            "repo_mining": repo_failures,
            "pr_fetch": detailed.get("pr_fetch_failures", []),
            "agent_enrichment": agent_enrichment.get("failures", []),
            "human_enrichment": human_enrichment.get("failures", []),
            "author_filter_removed_records": author_filter.get("removed_records", {}),
            "quality_filter_removed_records": quality_filter_removed_records,
        },
        "files": {
            "agent_perf_prs": str(final_dir / "agent_perf_prs_matched_criteria.parquet"),
            "human_perf_prs": str(final_dir / "human_perf_prs_matched_criteria.parquet"),
            "human_task_typed": str(intermediate_dir / "human_prs_task_typed.parquet"),
            "human_filtered": str(intermediate_dir / "human_prs_filtered.parquet"),
            "raw_github_human_prs": str(raw_dir / "github_human_prs.parquet"),
            "summary_json": str(final_dir / "rebalancing_summary.json"),
            "summary_md": str(final_dir / "rebalancing_summary.md"),
            "detailed_report_json": str(final_dir / "rebalancing_detailed_report.json"),
            "detailed_report_md": str(final_dir / "rebalancing_detailed_report.md"),
        },
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    print("# Rebalancing Report")
    print()
    if report.get("methodological_note"):
        print(f"Methodological note: {report['methodological_note']}")
        print()

    time_window = report.get("time_window") or {}
    if time_window:
        print(f"Time window: {time_window.get('start')} -> {time_window.get('end')}")
    print(f"Repositories >= star floor: {report.get('repositories_at_or_above_star_floor', 0)}")
    print()

    counts = report.get("counts", {})
    print("Counts")
    print(f"- AI perf PRs: {counts.get('agent_perf_prs', 0)}")
    print(f"- Human perf PRs: {counts.get('human_perf_prs', 0)}")
    print(f"- Total perf PRs: {counts.get('total_perf_prs', 0)}")
    print(f"- AI before quality filters: {counts.get('agent_before_quality_filters', 0)}")
    print(f"- AI after quality filters: {counts.get('agent_after_quality_filters', 0)}")
    print(f"- Human before quality filters: {counts.get('human_before_quality_filters', 0)}")
    print(f"- Human after quality filters: {counts.get('human_after_quality_filters', 0)}")
    print(f"- Raw GitHub human PRs: {counts.get('raw_github_human_prs', 0)}")
    print(f"- Repo mining failures: {counts.get('repo_mining_failures', 0)}")
    print(f"- PR fetch failures: {counts.get('pr_fetch_failures', 0)}")
    print(f"- Agent enrichment failures: {counts.get('agent_enrichment_failures', 0)}")
    print(f"- Human enrichment failures: {counts.get('human_enrichment_failures', 0)}")
    print(f"- Agent author-filter removals: {counts.get('agent_author_filter_removed', 0)}")
    print(f"- Human author-filter removals: {counts.get('human_author_filter_removed', 0)}")
    print(f"- Agent quality-filter removals: {counts.get('agent_quality_filter_removed', 0)}")
    print(f"- Human quality-filter removals: {counts.get('human_quality_filter_removed', 0)}")
    print()

    source_counts = report.get("source_counts", {})
    if source_counts:
        print("Source counts")
        for key, value in source_counts.items():
            print(f"- {key}: {value}")
        print()

    stage_counts = report.get("stage_counts", {})
    if stage_counts:
        print("Stage counts")
        for key, value in stage_counts.items():
            print(f"- {key}: {value}")
        print()

    uniques = report.get("unique_counts", {})
    print("Unique counts")
    print(f"- AI repos: {uniques.get('agent_repos', 0)}")
    print(f"- Human repos: {uniques.get('human_repos', 0)}")
    print(f"- AI authors: {uniques.get('agent_authors', 0)}")
    print(f"- Human authors: {uniques.get('human_authors', 0)}")
    print()

    date_ranges = report.get("date_ranges", {})
    for label, rng in date_ranges.items():
        if rng:
            print(f"{label}: {rng['start']} -> {rng['end']}")
    print()

    breakdowns = report.get("breakdowns", {})
    print("Human task type sources")
    for key, value in breakdowns.get("human_task_type_sources", {}).items():
        print(f"- {key}: {value}")
    print()

    print("Human task types")
    for key, value in breakdowns.get("human_task_types", {}).items():
        print(f"- {key}: {value}")
    print()

    print("Author filter removals")
    for key, value in breakdowns.get("author_filter_removed_counts", {}).items():
        print(f"- {key}: {value}")
    print()

    print("Removed by quality filter")
    for key, value in breakdowns.get("quality_filter_removed_counts", {}).items():
        print(f"- {key}: {value}")
    print()

    print("Agent quality filter removals")
    for key, value in breakdowns.get("agent_quality_filter_removed_counts", {}).items():
        print(f"- {key}: {value}")
    print()

    print("Human quality filter removals")
    for key, value in breakdowns.get("human_quality_filter_removed_counts", {}).items():
        print(f"- {key}: {value}")
    print()

    failures = report.get("failures", {})
    if failures.get("repo_mining"):
        print("Repo mining failures")
        for item in failures["repo_mining"]:
            repo_name = item.get("repo_full_name") or "missing-repo-name"
            status = item.get("status")
            error = item.get("error")
            suffix = f" - {error}" if error else ""
            print(f"- {repo_name}: {status}{suffix}")
        print()

    if failures.get("pr_fetch"):
        print("PR fetch failures")
        for item in failures["pr_fetch"]:
            print(f"- {item.get('repo_full_name')}#{item.get('number')}: {item.get('error')}")
        print()

    if failures.get("agent_enrichment") or failures.get("human_enrichment"):
        print("Enrichment failures")
        for arm in ("agent_enrichment", "human_enrichment"):
            items = failures.get(arm, [])
            if not items:
                continue
            print(f"- {arm}")
            for item in items:
                identifier = f"{item.get('repo_full_name')}#{item.get('number')}"
                print(f"  - {identifier}: {item.get('status')}")
        print()

    if failures.get("quality_filter_removed_records"):
        print("Quality filter removed records")
        for arm, reasons in failures["quality_filter_removed_records"].items():
            print(f"- {arm}")
            for reason, records in reasons.items():
                print(f"  - {reason}: {len(records)}")
        print()

    print("Files")
    for label, path in report.get("files", {}).items():
        print(f"- {label}: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect mined perf PR outputs.")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help=f"Directory containing the mining outputs (default: {DEFAULT_OUTPUTS_DIR}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as formatted JSON instead of text.",
    )
    args = parser.parse_args(argv)

    try:
        report = build_report(args.outputs_dir)
    except (FileNotFoundError, ReportInputError, TypeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
