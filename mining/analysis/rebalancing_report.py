from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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

    summary = load_json(final_dir / "rebalancing_summary.json")
    detailed = load_json(final_dir / "rebalancing_detailed_report.json")
    agent = load_parquet(final_dir / "agent_perf_prs_matched_criteria.parquet")
    human = load_parquet(final_dir / "human_perf_prs_matched_criteria.parquet")
    human_typed = load_parquet(intermediate_dir / "human_prs_task_typed.parquet")
    human_filtered = load_parquet(intermediate_dir / "human_prs_filtered.parquet")
    raw_github = load_parquet(raw_dir / "github_human_prs.parquet")

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
            "agent_perf_prs": int(len(agent)),
            "human_perf_prs": int(len(human)),
            "total_perf_prs": int(len(agent) + len(human)),
            "agent_before_quality_filters": int(
                summary.get("agent_perf_prs_before_quality_filters", len(agent))
            ),
            "agent_after_quality_filters": int(
                summary.get("agent_perf_prs_after_quality_filters", len(agent))
            ),
            "human_before_quality_filters": int(
                summary.get("human_perf_prs_before_quality_filters", len(human_typed))
            ),
            "human_after_quality_filters": int(len(human_filtered)),
            "raw_github_human_prs": int(len(raw_github)),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect mined perf PR outputs.")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("mining/outputs"),
        help="Directory containing the mining outputs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as formatted JSON instead of text.",
    )
    args = parser.parse_args()

    report = build_report(args.outputs_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
