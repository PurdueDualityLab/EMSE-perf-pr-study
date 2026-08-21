"""Build a compact PR-level table of the study's derived labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from schema import write_parquet


IDENTITY_COLUMNS = ("repo_id", "number")
ATTRIBUTION_COLUMNS = (
    "repo_id",
    "repo_full_name",
    "number",
    "html_url",
    "created_at",
    "aidev_attribution_label",
    "aidev_attribution_agent",
    "aidev_attribution_rule",
    "aidev_attribution_evidence",
    "aidev_attribution_status",
    "aidev_attribution_method",
)
TASK_COLUMNS = (
    "repo_id",
    "number",
    "aidev_task_type",
    "aidev_task_type_reason",
    "aidev_task_type_confidence",
    "aidev_task_type_method",
    "aidev_task_type_status",
    "aidev_task_type_model",
    "aidev_task_type_classifier",
)
HUMAN_FILTER_COLUMNS = (
    "repo_id",
    "number",
    "human_filter_label",
    "human_filter_reasons",
    "human_filter_primary_reason",
    "human_filter_method",
)
SAMPLING_COLUMNS = (
    "repo_id",
    "number",
    "sample_arm",
    "sampling_iso_year",
    "sampling_iso_week",
    "sampling_stratum",
    "stratum_population",
    "stratum_quota",
    "inclusion_probability",
    "selected",
    "sampling_seed",
    "sampling_method",
)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {', '.join(missing)}")


def _require_unique_identities(frame: pd.DataFrame, label: str) -> None:
    _require_columns(frame, IDENTITY_COLUMNS, label)
    if frame[list(IDENTITY_COLUMNS)].isna().any(axis=None):
        raise ValueError(f"{label} contains missing identities.")
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{label} contains duplicate identities.")


def build_curated_labels_frame(
    attribution: pd.DataFrame,
    task_types: pd.DataFrame,
    human_decisions: pd.DataFrame,
    sampling_manifest: pd.DataFrame,
) -> pd.DataFrame:
    inputs = (
        (attribution, ATTRIBUTION_COLUMNS, "AIDev attribution"),
        (task_types, TASK_COLUMNS, "Task-type decisions"),
        (human_decisions, HUMAN_FILTER_COLUMNS, "Human-filter decisions"),
        (sampling_manifest, SAMPLING_COLUMNS, "Sampling manifest"),
    )
    for frame, columns, label in inputs:
        _require_columns(frame, columns, label)
        _require_unique_identities(frame, label)

    attribution_keys = attribution[list(IDENTITY_COLUMNS)].sort_values(
        list(IDENTITY_COLUMNS), ignore_index=True
    )
    task_keys = task_types[list(IDENTITY_COLUMNS)].sort_values(
        list(IDENTITY_COLUMNS), ignore_index=True
    )
    if not attribution_keys.equals(task_keys):
        raise ValueError("AIDev attribution and task-type decisions have nonmatching identities.")

    result = attribution[list(ATTRIBUTION_COLUMNS)].merge(
        task_types[list(TASK_COLUMNS)],
        on=list(IDENTITY_COLUMNS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    result["is_performance"] = result["aidev_task_type"].eq("perf")
    result = result.merge(
        human_decisions[list(HUMAN_FILTER_COLUMNS)],
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    result = result.merge(
        sampling_manifest[list(SAMPLING_COLUMNS)],
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(result) != len(attribution):
        raise ValueError("Curated labels do not reconcile to the attribution population.")
    return result.sort_values(list(IDENTITY_COLUMNS), kind="mergesort").reset_index(drop=True)


def build_curated_labels(
    attribution_path: Path,
    task_type_path: Path,
    human_decisions_path: Path,
    sampling_manifest_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    for path in (
        attribution_path,
        task_type_path,
        human_decisions_path,
        sampling_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    result = build_curated_labels_frame(
        pd.read_parquet(attribution_path, columns=list(ATTRIBUTION_COLUMNS)),
        pd.read_parquet(task_type_path, columns=list(TASK_COLUMNS)),
        pd.read_parquet(human_decisions_path, columns=list(HUMAN_FILTER_COLUMNS)),
        pd.read_parquet(sampling_manifest_path, columns=list(SAMPLING_COLUMNS)),
    )
    write_parquet(result, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--task-types", type=Path, required=True)
    parser.add_argument("--human-decisions", type=Path, required=True)
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_curated_labels(
        args.attribution,
        args.task_types,
        args.human_decisions,
        args.sampling_manifest,
        args.output,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(result):,} rows to {args.output}")


if __name__ == "__main__":
    main()
