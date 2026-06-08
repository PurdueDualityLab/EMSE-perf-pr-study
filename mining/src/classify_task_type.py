from __future__ import annotations

import re
from typing import Mapping

import pandas as pd

from schema import PR_ID_COLUMNS, TASK_TYPE_COLUMNS, first_existing_column, normalize_type


CONVENTIONAL_TYPES = {
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
}

PERF_TERMS = (
    "perf",
    "performance",
    "optimize",
    "optimise",
    "optimization",
    "optimisation",
    "speed",
    "faster",
    "latency",
    "throughput",
    "memory",
    "cache",
    "benchmark",
    "profiling",
)


def compatible_task_type(row: Mapping[str, object]) -> tuple[str, float, str]:
    """Conventional-Commit-compatible task classifier for PRs missing AIDev labels."""
    title = str(row.get("title") or "")
    body = str(row.get("body") or "")
    text = f"{title}\n{body}".lower()

    match = re.match(r"^\s*([a-z]+)(\([^)]+\))?!?:", title.lower())
    if match and match.group(1) in CONVENTIONAL_TYPES:
        task_type = match.group(1)
        return task_type, 1.0, "conventional_commit_prefix"

    if any(term in text for term in PERF_TERMS):
        return "perf", 0.65, "performance_keyword_heuristic"

    return "other", 0.3, "fallback_other"


def assign_missing_task_types(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "task_type" not in result.columns:
        result["task_type"] = pd.NA
    if "task_type_source" not in result.columns:
        result["task_type_source"] = pd.NA
    if "task_type_confidence" not in result.columns:
        result["task_type_confidence"] = pd.NA
    if "task_type_reason" not in result.columns:
        result["task_type_reason"] = pd.NA

    missing = result["task_type"].isna() | (result["task_type"].astype(str).str.len() == 0)
    for idx, row in result.loc[missing].iterrows():
        task_type, confidence, reason = compatible_task_type(row)
        result.at[idx, "task_type"] = task_type
        result.at[idx, "task_type_source"] = "compatible_classifier"
        result.at[idx, "task_type_confidence"] = confidence
        result.at[idx, "task_type_reason"] = reason

    result.loc[~missing & result["task_type_source"].isna(), "task_type_source"] = "aidev_parquet"
    result["task_type"] = result["task_type"].map(normalize_type)
    return result


def attach_task_type(pr_df: pd.DataFrame, task_df: pd.DataFrame | None) -> pd.DataFrame:
    result = pr_df.copy()
    if task_df is None or task_df.empty:
        return assign_missing_task_types(result)

    task_type_col = first_existing_column(task_df, TASK_TYPE_COLUMNS)
    if task_type_col is None:
        return assign_missing_task_types(result)

    task_id_col = first_existing_column(task_df, PR_ID_COLUMNS)
    pr_id_col = first_existing_column(result, PR_ID_COLUMNS)
    if task_id_col is None or pr_id_col is None:
        return assign_missing_task_types(result)

    task_subset = task_df[[task_id_col, task_type_col]].copy()
    task_subset = task_subset.rename(
        columns={task_id_col: pr_id_col, task_type_col: "_aidev_task_type"}
    )
    result = result.merge(task_subset.drop_duplicates(pr_id_col), on=pr_id_col, how="left")

    if "task_type" not in result.columns:
        result["task_type"] = pd.NA
    result["task_type"] = result["task_type"].fillna(result["_aidev_task_type"])
    result["task_type_source"] = result["_aidev_task_type"].notna().map(
        {True: "aidev_parquet", False: pd.NA}
    )
    result = result.drop(columns=["_aidev_task_type"])
    return assign_missing_task_types(result)
