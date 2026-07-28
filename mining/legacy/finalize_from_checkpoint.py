from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from author_filter import filter_by_author_with_details
from build_rebalanced_dataset import (
    build_detailed_report,
    build_summary,
    describe_removed_rows,
    filter_perf,
    filter_prs_to_repos_and_window,
    filter_repositories,
    load_config,
    write_detailed_report_markdown,
    write_summary_markdown,
)
from classify_task_type import attach_task_type
from load_aidev import load_aidev_tables
from quality_filters import apply_quality_filters_with_details
from resume import load_checkpoint_dataframe, load_resume_state, resume_state_path
from schema import TimeWindow, ensure_datetime, ensure_output_dirs, write_parquet


def log(message: str) -> None:
    print(message, flush=True)


def dedupe_human_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "html_url" in df.columns:
        return df.drop_duplicates(subset=["html_url"], keep="first").copy()
    if "url" in df.columns:
        return df.drop_duplicates(subset=["url"], keep="first").copy()
    if {"repo_full_name", "number"}.issubset(df.columns):
        return df.drop_duplicates(subset=["repo_full_name", "number"], keep="first").copy()
    return df.drop_duplicates(keep="first").copy()


def json_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts().to_dict().items()}


def final_output_paths(output_dirs: dict[str, Path], raw_path: Path) -> dict[str, str]:
    return {
        "agent_perf_prs": str(output_dirs["final"] / "agent_perf_prs_matched_criteria.parquet"),
        "human_perf_prs": str(output_dirs["final"] / "human_perf_prs_matched_criteria.parquet"),
        "human_task_typed": str(output_dirs["intermediate"] / "human_prs_task_typed.parquet"),
        "human_filtered": str(output_dirs["intermediate"] / "human_prs_filtered.parquet"),
        "raw_github_human_prs": str(raw_path),
        "summary_json": str(output_dirs["final"] / "rebalancing_summary.json"),
        "summary_md": str(output_dirs["final"] / "rebalancing_summary.md"),
        "detailed_report_json": str(output_dirs["final"] / "rebalancing_detailed_report.json"),
        "detailed_report_md": str(output_dirs["final"] / "rebalancing_detailed_report.md"),
    }


def load_existing_agent_final() -> pd.DataFrame | None:
    path = Path("mining/outputs/final/agent_perf_prs_matched_criteria.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


def main() -> None:
    log(
        "DEPRECATED: this historical finalizer is retained for audit only; "
        "use build_rebalanced_dataset.py for supported runs."
    )
    parser = argparse.ArgumentParser(description="Finalize rebalanced outputs from completed GitHub checkpoint.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("outputs", {}).get("dir", "mining/outputs"))
    output_dirs = ensure_output_dirs(output_dir)
    raw_path = output_dirs["raw"] / "github_human_prs.parquet"
    state = load_resume_state(resume_state_path(output_dir))
    if not state:
        raise ValueError(f"Missing resume state under {output_dir}")
    signature = state.get("signature", {})
    window_data = signature.get("window", {})
    window = TimeWindow(
        start=ensure_datetime(pd.Series([window_data["start"]])).iloc[0],
        end=ensure_datetime(pd.Series([window_data["end"]])).iloc[0],
    )
    target_type = str(config.get("criteria", {}).get("task_type", "perf"))
    star_floor = int(config.get("criteria", {}).get("star_floor", 100))

    log("[finalize 1/5] Loading AIDev tables and raw checkpoint")
    tables = load_aidev_tables(config)
    github_human = load_checkpoint_dataframe(raw_path)
    log(f"  raw GitHub human PRs: {len(github_human)}")

    log("[finalize 2/5] Preparing agent arm")
    repos = filter_repositories(tables["repository"], star_floor)
    agent_prs = filter_prs_to_repos_and_window(tables["pull_request"], repos, window)
    agent_typed = attach_task_type(agent_prs, tables.get("pr_task_type"))
    agent_authored, agent_author_removed = filter_by_author_with_details(agent_typed, "agent")
    agent_perf_before_quality = filter_perf(agent_authored, target_type)
    existing_agent_perf = load_existing_agent_final()
    if existing_agent_perf is not None:
        agent_perf = existing_agent_perf
        agent_quality_removed_counts = {
            "empty_filename": max(int(len(agent_perf_before_quality) - len(agent_perf)), 0),
            "config_only": 0,
            "deleted_repo": 0,
            "merge_only": 0,
        }
        agent_quality_removed_records = {
            "empty_filename": [],
            "config_only": [],
            "deleted_repo": [],
            "merge_only": [],
        }
    else:
        agent_perf, agent_quality_removed_counts, agent_quality_removed_records = apply_quality_filters_with_details(agent_perf_before_quality)
    write_parquet(agent_perf, output_dirs["final"] / "agent_perf_prs_matched_criteria.parquet")
    log(f"  agent perf after quality: {len(agent_perf)}")

    log("[finalize 3/5] Preparing human arm")
    existing_human = tables.get("human_pull_request", pd.DataFrame())
    human_candidates = dedupe_human_candidates(pd.concat([existing_human, github_human], ignore_index=True, sort=False))
    human_candidates = filter_prs_to_repos_and_window(human_candidates, repos, window)
    human_candidates_before_author = len(human_candidates)
    human_candidates, human_author_removed = filter_by_author_with_details(human_candidates, "human")
    human_typed = attach_task_type(human_candidates, tables.get("human_pr_task_type"))
    log(f"  human typed rows: {len(human_typed)}")
    human_perf_before_quality = filter_perf(human_typed, target_type)
    log(f"  human perf before quality: {len(human_perf_before_quality)}")
    human_perf, human_quality_removed_counts, human_quality_removed_records = apply_quality_filters_with_details(human_perf_before_quality)
    log(f"  human perf after quality: {len(human_perf)}")
    write_parquet(human_typed, output_dirs["intermediate"] / "human_prs_task_typed.parquet")
    write_parquet(human_perf, output_dirs["intermediate"] / "human_prs_filtered.parquet")
    write_parquet(human_perf, output_dirs["final"] / "human_perf_prs_matched_criteria.parquet")

    log("[finalize 4/5] Building reports")
    source_counts = {
        "pull_request": int(len(tables.get("pull_request", pd.DataFrame()))),
        "repository": int(len(tables.get("repository", pd.DataFrame()))),
        "human_pull_request": int(len(tables.get("human_pull_request", pd.DataFrame()))),
        "pr_task_type": int(len(tables.get("pr_task_type", pd.DataFrame()))),
        "human_pr_task_type": int(len(tables.get("human_pr_task_type", pd.DataFrame()))),
    }
    stage_counts = {
        "repositories_after_star_floor": int(len(repos)),
        "agent_typed": int(len(agent_typed)),
        "agent_author_kept": int(len(agent_authored)),
        "agent_author_removed": int(len(agent_author_removed)),
        "agent_perf_before_enrichment": int(len(agent_perf_before_quality)),
        "agent_perf_after_enrichment": int(len(agent_perf_before_quality)),
        "agent_perf_after_quality": int(len(agent_perf)),
        "human_candidates_before_author": int(human_candidates_before_author),
        "human_author_kept": int(len(human_candidates)),
        "human_author_removed": int(len(human_author_removed)),
        "human_typed": int(len(human_typed)),
        "human_perf_before_enrichment": int(len(human_perf_before_quality)),
        "human_perf_after_enrichment": int(len(human_perf_before_quality)),
        "human_perf_after_quality": int(len(human_perf)),
        "raw_github_human_prs": int(len(github_human)),
    }
    quality_filter_removed_counts = {
        reason: int(agent_quality_removed_counts.get(reason, 0) + human_quality_removed_counts.get(reason, 0))
        for reason in sorted(set(agent_quality_removed_counts) | set(human_quality_removed_counts))
    }
    task_type_sources = json_counts(human_typed.get("task_type_source", pd.Series(dtype=str)).fillna("missing"))
    summary = build_summary(
        window=window,
        repo_count=len(repos),
        agent_before_quality_count=len(agent_perf_before_quality),
        agent_after_quality_count=len(agent_perf),
        agent_perf_count=len(agent_perf),
        human_before_quality_count=len(human_perf_before_quality),
        human_after_quality_count=len(human_perf),
        task_type_sources=task_type_sources,
        removed_counts={str(key): int(value) for key, value in quality_filter_removed_counts.items()},
    )
    repo_reports = state.get("repo_mining_reports", [])
    pr_failures = state.get("pr_failures", [])
    repo_mining_report: dict[str, Any] = {"repo_reports": repo_reports, "pr_failures": pr_failures}
    output_paths = final_output_paths(output_dirs, raw_path)
    detailed_report = build_detailed_report(
        summary=summary,
        source_counts=source_counts,
        stage_counts=stage_counts,
        author_filter_removed_counts={"agent": int(len(agent_author_removed)), "human": int(len(human_author_removed))},
        author_filter_removed_records={
            "agent": describe_removed_rows(agent_author_removed, "non_agent_authored"),
            "human": describe_removed_rows(human_author_removed, "agent_authored"),
        },
        repo_mining_report=repo_mining_report,
        enrichment_reports={"agent": {"finalized_from_checkpoint": True}, "human": {"finalized_from_checkpoint": True}},
        quality_filter_removed_records={"agent": agent_quality_removed_records, "human": human_quality_removed_records},
        output_paths=output_paths,
    )
    summary.update(
        {
            "source_counts": source_counts,
            "stage_counts": stage_counts,
            "author_filter_removed_counts": {"agent": int(len(agent_author_removed)), "human": int(len(human_author_removed))},
            "agent_quality_filter_removed_counts": {str(k): int(v) for k, v in agent_quality_removed_counts.items()},
            "human_quality_filter_removed_counts": {str(k): int(v) for k, v in human_quality_removed_counts.items()},
            "repo_mining_failures": int(detailed_report["repo_mining"]["failed"]),
            "pr_fetch_failures": int(len(pr_failures)),
            "detailed_report_json": output_paths["detailed_report_json"],
            "detailed_report_md": output_paths["detailed_report_md"],
        }
    )

    log("[finalize 5/5] Writing reports")
    (output_dirs["final"] / "rebalancing_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary_markdown(summary, output_dirs["final"] / "rebalancing_summary.md")
    (output_dirs["final"] / "rebalancing_detailed_report.json").write_text(json.dumps(detailed_report, indent=2) + "\n", encoding="utf-8")
    write_detailed_report_markdown(detailed_report, output_dirs["final"] / "rebalancing_detailed_report.md")
    log("Done. Finalized outputs from checkpoint.")


if __name__ == "__main__":
    main()
