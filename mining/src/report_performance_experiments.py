"""Join performance-experiment decisions into one PR-level comparison artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from schema import atomic_write_text, write_parquet


IDENTITY = ("repo_id", "number")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_columns(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    result = frame.copy()
    if "number" not in result.columns and "pr_number" in result.columns:
        result = result.rename(columns={"pr_number": "number"})
    missing = set(IDENTITY) - set(result.columns)
    if missing:
        raise ValueError(f"{label} is missing immutable identity columns: {sorted(missing)}")
    if result[list(IDENTITY)].isna().any().any() or result.duplicated(list(IDENTITY)).any():
        raise ValueError(f"{label} has invalid or duplicate immutable identities.")
    return result


def add_columns(base: pd.DataFrame, source: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    available = [column for column in columns if column in source.columns]
    subset = source[[*IDENTITY, *available]].copy()
    subset = subset.rename(columns={column: f"{prefix}{column}" for column in available})
    return base.merge(subset, on=list(IDENTITY), how="left", validate="one_to_one", sort=False)


def build_comparison(
    project_decisions: pd.DataFrame,
    prefix_decisions: pd.DataFrame | None = None,
    perfminer: pd.DataFrame | None = None,
    llm: pd.DataFrame | None = None,
    aidev_task_type: pd.DataFrame | None = None,
) -> pd.DataFrame:
    project = identity_columns(project_decisions, "Project decisions")
    project_keys = set(map(tuple, project[list(IDENTITY)].itertuples(index=False, name=None)))
    prefix = None
    if prefix_decisions is not None:
        prefix = identity_columns(prefix_decisions, "Prefix decisions")
        prefix_keys = set(map(tuple, prefix[list(IDENTITY)].itertuples(index=False, name=None)))
        if project_keys != prefix_keys:
            raise ValueError("Project and prefix decisions have nonmatching identities.")
    base_columns = [
        column
        for column in (
            "repo_full_name",
            "html_url",
            "created_at",
            "aidev_attribution_label",
            "aidev_attribution_status",
        )
        if column in project.columns
    ]
    result = project[[*IDENTITY, *base_columns]].copy()
    result = add_columns(
        result,
        project,
        ["selected", "heuristic_match", "selection_reason", "date_eligible", "date_reason"],
        "project_",
    )
    if prefix is not None:
        result = add_columns(
            result,
            prefix,
            ["selected", "heuristic_match", "selection_reason", "date_eligible", "date_reason"],
            "prefix_",
        )
    if perfminer is not None:
        result = add_columns(
            result,
            identity_columns(perfminer, "PerfMiner predictions"),
            ["perfminer_pr_status", "perfminer_pr_is_performance", "perfminer_pr_evidence_complete", "perfminer_positive_commit_count", "perfminer_max_commit_score"],
            "",
        )
    if llm is not None:
        result = add_columns(
            result,
            identity_columns(llm, "LLM decisions"),
            ["llm_label", "llm_status", "llm_model", "llm_response_id", "llm_input_tokens", "llm_output_tokens", "llm_error"],
            "",
        )
    if aidev_task_type is not None:
        result = add_columns(
            result,
            identity_columns(aidev_task_type, "AIDev task-type decisions"),
            [
                "aidev_task_type",
                "aidev_task_type_reason",
                "aidev_task_type_confidence",
                "aidev_task_type_method",
                "aidev_task_type_status",
                "aidev_task_type_model",
                "aidev_task_type_input_tokens",
                "aidev_task_type_output_tokens",
            ],
            "",
        )
    return result


def report(
    project_path: Path,
    prefix_path: Path | None,
    output_dir: Path,
    perfminer_path: Path | None = None,
    llm_path: Path | None = None,
    aidev_task_type_path: Path | None = None,
) -> dict[str, object]:
    project = pd.read_parquet(project_path)
    prefix = pd.read_parquet(prefix_path) if prefix_path else None
    perfminer = pd.read_parquet(perfminer_path) if perfminer_path else None
    llm = pd.read_parquet(llm_path) if llm_path else None
    aidev_task_type = pd.read_parquet(aidev_task_type_path) if aidev_task_type_path else None
    comparison = build_comparison(project, prefix, perfminer, llm, aidev_task_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison.parquet"
    write_parquet(comparison, output_path)
    summary = {
        "rows": len(comparison),
        "project_selected": int(comparison["project_selected"].fillna(False).sum()),
        "prefix_selected": int(comparison["prefix_selected"].fillna(False).sum()) if "prefix_selected" in comparison else None,
        "aidev_attribution_labels": dict(sorted(Counter(comparison.get("aidev_attribution_label", [])).items())),
        "perfminer_statuses": dict(sorted(Counter(comparison.get("perfminer_pr_status", []).dropna()).items())) if "perfminer_pr_status" in comparison else {},
        "llm_statuses": dict(sorted(Counter(comparison.get("llm_status", []).dropna()).items())) if "llm_status" in comparison else {},
        "llm_labels": dict(sorted(Counter(comparison.get("llm_label", []).dropna()).items())) if "llm_label" in comparison else {},
        "aidev_task_type_statuses": dict(sorted(Counter(comparison.get("aidev_task_type_status", []).dropna()).items())) if "aidev_task_type_status" in comparison else {},
        "aidev_task_type_labels": dict(sorted(Counter(comparison.get("aidev_task_type", []).dropna()).items())) if "aidev_task_type" in comparison else {},
        "inputs": {
            "project_decisions": {"path": str(project_path), "sha256": sha256_file(project_path)},
            "prefix_decisions": {"path": str(prefix_path), "sha256": sha256_file(prefix_path)} if prefix_path else None,
            "perfminer_predictions": {"path": str(perfminer_path), "sha256": sha256_file(perfminer_path)} if perfminer_path else None,
            "llm_decisions": {"path": str(llm_path), "sha256": sha256_file(llm_path)} if llm_path else None,
            "aidev_task_type_decisions": {"path": str(aidev_task_type_path), "sha256": sha256_file(aidev_task_type_path)} if aidev_task_type_path else None,
        },
        "output_sha256": sha256_file(output_path),
    }
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report a PR-level comparison of performance experiments.")
    parser.add_argument("--project-decisions", required=True, type=Path)
    parser.add_argument("--prefix-decisions", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--perfminer-prs", type=Path)
    parser.add_argument("--llm-decisions", type=Path)
    parser.add_argument("--aidev-task-type", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = report(
        args.project_decisions,
        args.prefix_decisions,
        args.output_dir,
        args.perfminer_prs,
        args.llm_decisions,
        args.aidev_task_type,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
