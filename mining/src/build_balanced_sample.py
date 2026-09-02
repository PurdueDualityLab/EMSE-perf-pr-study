from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from select_human_candidates import (
    ATTRIBUTION_COLUMNS,
    IDENTITY_COLUMNS,
    matching_attribution_rows,
    performance_rows,
    require_unique_identities,
    sha256_file,
)
from quality_filters import quality_filter_flags
from schema import atomic_write_text, write_parquet


METHOD_NAME = "aidev_performance_weekly_balanced"
METHOD_VERSION = 1
DEFAULT_SEED = "emse-primary-human-sample-v1"
QUALITY_FILTER_ORDER = ("empty_filename", "config_only", "deleted_repo", "merge_only")
MANIFEST_COLUMNS = (
    *IDENTITY_COLUMNS,
    "html_url",
    "created_at",
    "sample_arm",
    "sampling_iso_year",
    "sampling_iso_week",
    "sampling_stratum",
    "stratum_population",
    "stratum_quota",
    "inclusion_probability",
    "selection_hash",
    "selected",
    "sampling_seed",
    "sampling_method",
)


def selection_hash(seed: str, repo_id: object, number: object) -> str:
    value = f"{seed}\0{int(repo_id)}\0{int(number)}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def add_weekly_strata(frame: pd.DataFrame) -> pd.DataFrame:
    if "created_at" not in frame.columns:
        raise ValueError("Weekly sampling requires a created_at column.")
    result = frame.copy()
    timestamps = pd.to_datetime(result["created_at"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Weekly sampling input contains invalid created_at values.")
    calendar = timestamps.dt.isocalendar()
    result["sampling_iso_year"] = calendar["year"].astype("int32")
    result["sampling_iso_week"] = calendar["week"].astype("int16")
    result["sampling_stratum"] = (
        result["sampling_iso_year"].astype(str)
        + "-W"
        + result["sampling_iso_week"].astype(str).str.zfill(2)
    )
    return result


def apply_sampling_quality_filters(
    frame: pd.DataFrame, arm: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flags = quality_filter_flags(frame)
    keep = pd.Series(True, index=frame.index)
    exclusions = []
    for reason in QUALITY_FILTER_ORDER:
        removed = keep & flags[reason]
        if removed.any():
            rows = frame.loc[removed, [*IDENTITY_COLUMNS, "html_url", "created_at"]].copy()
            rows["sample_arm"] = arm
            rows["exclusion_reason"] = reason
            exclusions.append(rows)
        keep &= ~flags[reason]
    excluded = pd.concat(exclusions, ignore_index=True) if exclusions else pd.DataFrame(
        columns=[*IDENTITY_COLUMNS, "html_url", "created_at", "sample_arm", "exclusion_reason"]
    )
    return frame.loc[keep].copy(), excluded


def _manifest_for_arm(
    frame: pd.DataFrame,
    arm: str,
    seed: str,
    populations: pd.Series,
    quotas: pd.Series,
    selected: pd.Series,
) -> pd.DataFrame:
    result = frame[[*IDENTITY_COLUMNS, "html_url", "created_at"]].copy()
    result["sample_arm"] = arm
    result["sampling_iso_year"] = frame["sampling_iso_year"].to_numpy()
    result["sampling_iso_week"] = frame["sampling_iso_week"].to_numpy()
    result["sampling_stratum"] = frame["sampling_stratum"].to_numpy()
    result["stratum_population"] = frame["sampling_stratum"].map(populations).astype("int64")
    result["stratum_quota"] = frame["sampling_stratum"].map(quotas).fillna(0).astype("int64")
    result["inclusion_probability"] = (
        result["stratum_quota"] / result["stratum_population"]
    ).astype(float)
    result["selection_hash"] = [
        selection_hash(seed, repo_id, number)
        for repo_id, number in result[list(IDENTITY_COLUMNS)].itertuples(index=False, name=None)
    ]
    result["selected"] = selected.to_numpy(dtype=bool)
    result["sampling_seed"] = seed
    result["sampling_method"] = METHOD_NAME
    return result[list(MANIFEST_COLUMNS)]


def build_balanced_sample_frames(
    agentic: pd.DataFrame,
    humans: pd.DataFrame,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    if not seed:
        raise ValueError("Sampling seed must not be empty.")
    require_unique_identities(agentic, "Agentic population")
    require_unique_identities(humans, "Human-candidate population")
    input_agent_keys = set(
        map(tuple, agentic[list(IDENTITY_COLUMNS)].itertuples(index=False, name=None))
    )
    input_human_keys = set(
        map(tuple, humans[list(IDENTITY_COLUMNS)].itertuples(index=False, name=None))
    )
    if input_agent_keys & input_human_keys:
        raise ValueError("Agentic and human-candidate populations overlap.")
    agentic_input_rows = len(agentic)
    human_input_rows = len(humans)
    agentic, agentic_excluded = apply_sampling_quality_filters(agentic, "agentic")
    humans, human_excluded = apply_sampling_quality_filters(humans, "human_candidate")

    agentic_strata = add_weekly_strata(agentic)
    human_strata = add_weekly_strata(humans)
    agent_counts = agentic_strata["sampling_stratum"].value_counts().sort_index()
    human_counts = human_strata["sampling_stratum"].value_counts().sort_index()
    deficient = agent_counts[agent_counts > human_counts.reindex(agent_counts.index, fill_value=0)]
    if not deficient.empty:
        details = ", ".join(
            f"{stratum}: agentic={int(count)}, human={int(human_counts.get(stratum, 0))}"
            for stratum, count in deficient.items()
        )
        raise ValueError("Insufficient human candidates for weekly quotas: " + details)

    human_ranked = human_strata.copy()
    human_ranked["selection_hash"] = [
        selection_hash(seed, repo_id, number)
        for repo_id, number in human_ranked[list(IDENTITY_COLUMNS)].itertuples(index=False, name=None)
    ]
    human_ranked = human_ranked.sort_values(
        ["sampling_stratum", "selection_hash", *IDENTITY_COLUMNS], kind="mergesort"
    )
    ranks = human_ranked.groupby("sampling_stratum", sort=False).cumcount()
    quotas_for_rows = human_ranked["sampling_stratum"].map(agent_counts).fillna(0).astype("int64")
    human_ranked["selected"] = ranks.lt(quotas_for_rows)
    human_selected_keys = set(
        map(
            tuple,
            human_ranked.loc[human_ranked["selected"], list(IDENTITY_COLUMNS)].itertuples(
                index=False, name=None
            ),
        )
    )
    human_selected = human_strata[list(IDENTITY_COLUMNS)].apply(tuple, axis=1).isin(
        human_selected_keys
    )
    agent_selected = pd.Series(True, index=agentic_strata.index)

    agent_manifest = _manifest_for_arm(
        agentic_strata,
        "agentic",
        seed,
        agent_counts,
        agent_counts,
        agent_selected,
    )
    human_manifest = _manifest_for_arm(
        human_strata,
        "human_candidate",
        seed,
        human_counts,
        agent_counts,
        human_selected,
    )
    manifest = pd.concat([agent_manifest, human_manifest], ignore_index=True)

    selected_metadata = [
        column
        for column in MANIFEST_COLUMNS
        if column not in {*IDENTITY_COLUMNS, "html_url", "created_at"}
    ]
    agentic_sample = agentic.merge(
        agent_manifest.loc[agent_manifest["selected"], [*IDENTITY_COLUMNS, *selected_metadata]],
        on=list(IDENTITY_COLUMNS),
        validate="one_to_one",
        how="inner",
        sort=False,
    )
    human_sample = humans.merge(
        human_manifest.loc[human_manifest["selected"], [*IDENTITY_COLUMNS, *selected_metadata]],
        on=list(IDENTITY_COLUMNS),
        validate="one_to_one",
        how="inner",
        sort=False,
    )
    balanced_sample = pd.concat([agentic_sample, human_sample], ignore_index=True, sort=False)
    balanced_sample = balanced_sample.sort_values(
        ["created_at", "sample_arm", *IDENTITY_COLUMNS], kind="mergesort"
    ).reset_index(drop=True)

    selected_weekly = manifest.loc[manifest["selected"]].groupby(
        ["sampling_stratum", "sample_arm"]
    ).size().unstack(fill_value=0)
    if not selected_weekly["agentic"].equals(selected_weekly["human_candidate"]):
        raise ValueError("Selected agentic and human weekly quotas are not balanced.")
    if len(agentic_sample) != len(agentic) or len(human_sample) != len(agentic):
        raise ValueError("Balanced sample does not contain every agentic PR and a 1:1 human arm.")

    weekly = pd.DataFrame(
        {
            "sampling_stratum": agent_counts.index,
            "agentic_population": agent_counts.to_numpy(dtype="int64"),
            "human_population": human_counts.reindex(agent_counts.index).to_numpy(dtype="int64"),
            "human_quota": agent_counts.to_numpy(dtype="int64"),
        }
    )
    weekly["human_inclusion_probability"] = (
        weekly["human_quota"] / weekly["human_population"]
    )
    return {
        "manifest": manifest,
        "agentic_sample": agentic_sample,
        "human_sample": human_sample,
        "balanced_sample": balanced_sample,
        "weekly": weekly,
        "quality_exclusions": pd.concat(
            [agentic_excluded, human_excluded], ignore_index=True
        ),
        "population_counts": {
            "agentic_before_quality_filters": agentic_input_rows,
            "agentic_after_quality_filters": len(agentic),
            "human_before_quality_filters": human_input_rows,
            "human_after_quality_filters": len(humans),
        },
    }


def _load_populations(
    task_type_path: Path,
    attribution_path: Path,
    human_candidates_path: Path,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    performance = performance_rows(task_type_path)
    identities = set(
        (int(repo_id), int(number))
        for repo_id, number in performance[list(IDENTITY_COLUMNS)].itertuples(index=False, name=None)
    )
    attribution = matching_attribution_rows(attribution_path, identities, batch_size)
    if len(attribution) != len(performance):
        raise ValueError("Performance and AIDev attribution inputs have nonmatching identities.")
    added_columns = [column for column in ATTRIBUTION_COLUMNS if column not in IDENTITY_COLUMNS]
    performance = performance.merge(
        attribution[[*IDENTITY_COLUMNS, *added_columns]],
        on=list(IDENTITY_COLUMNS),
        validate="one_to_one",
        how="left",
        sort=False,
    )
    agentic = performance.loc[performance["aidev_attribution_label"].eq("agentic")].copy()

    humans = pd.read_parquet(human_candidates_path)
    require_unique_identities(humans, "Strict human-candidate input")
    required_values = {
        "aidev_task_type": "perf",
        "aidev_attribution_label": "human_candidate",
        "human_filter_label": "human_candidate",
    }
    for column, expected in required_values.items():
        if column not in humans.columns or not humans[column].eq(expected).all():
            raise ValueError(f"Strict human-candidate input requires {column} == {expected!r}.")
    return agentic, humans, len(performance)


def _write_outputs(frames: dict[str, Any], output_dir: Path) -> dict[str, dict[str, object]]:
    outputs = {
        "manifest": (output_dir / "sampling_manifest.parquet", frames["manifest"]),
        "weekly_strata": (output_dir / "weekly_strata.parquet", frames["weekly"]),
        "agentic_sample": (output_dir / "agentic_sample.parquet", frames["agentic_sample"]),
        "human_sample": (output_dir / "human_sample.parquet", frames["human_sample"]),
        "balanced_sample": (output_dir / "balanced_sample.parquet", frames["balanced_sample"]),
        "quality_exclusions": (
            output_dir / "quality_exclusions.parquet",
            frames["quality_exclusions"],
        ),
    }
    metadata: dict[str, dict[str, object]] = {}
    for name, (path, frame) in outputs.items():
        write_parquet(frame, path)
        metadata[name] = {
            "path": str(path.resolve()),
            "rows": len(frame),
            "sha256": sha256_file(path),
        }
    return metadata


def build_weekly_balanced_sample(
    task_type_path: Path,
    attribution_path: Path,
    human_candidates_path: Path,
    output_dir: Path,
    *,
    seed: str = DEFAULT_SEED,
    batch_size: int = 50_000,
    overwrite: bool = False,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for path in (task_type_path, attribution_path, human_candidates_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_paths = (
        output_dir / "sampling_manifest.parquet",
        output_dir / "weekly_strata.parquet",
        output_dir / "agentic_sample.parquet",
        output_dir / "human_sample.parquet",
        output_dir / "balanced_sample.parquet",
        output_dir / "quality_exclusions.parquet",
        output_dir / "summary.json",
    )
    if not overwrite and any(path.exists() for path in output_paths):
        raise FileExistsError("Sampling outputs already exist; pass --overwrite to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)

    agentic, humans, performance_rows_count = _load_populations(
        task_type_path, attribution_path, human_candidates_path, batch_size
    )
    frames = build_balanced_sample_frames(agentic, humans, seed)
    outputs = _write_outputs(frames, output_dir)
    weekly = frames["weekly"]
    summary: dict[str, object] = {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "seed": seed,
        "hash_contract": "SHA256(seed + NUL + repo_id + NUL + number)",
        "stratification": "ISO week of created_at in UTC",
        "sampling": "All agentic PRs; human candidates sampled without replacement to the agentic weekly quota.",
        "eligibility": "Legacy filename, config-only, deleted-repository, and merge-only filters are applied before weekly sampling.",
        "inputs": {
            "task_type": {
                "path": str(task_type_path.resolve()),
                "sha256": sha256_file(task_type_path),
            },
            "aidev_attribution": {
                "path": str(attribution_path.resolve()),
                "sha256": sha256_file(attribution_path),
            },
            "human_candidates": {
                "path": str(human_candidates_path.resolve()),
                "sha256": sha256_file(human_candidates_path),
            },
        },
        "counts": {
            "performance_population": performance_rows_count,
            "agentic_population": len(agentic),
            "human_candidate_population": len(humans),
            "agentic_sample": len(frames["agentic_sample"]),
            "human_sample": len(frames["human_sample"]),
            "balanced_sample": len(frames["balanced_sample"]),
            "agentic_nonempty_weeks": len(weekly),
            "deficient_weeks": 0,
            **frames["population_counts"],
            "quality_exclusions": len(frames["quality_exclusions"]),
        },
        "quality_exclusion_counts": {
            f"{arm}:{reason}": int(count)
            for (arm, reason), count in frames["quality_exclusions"].groupby(
                ["sample_arm", "exclusion_reason"]
            ).size().items()
        },
        "weekly_strata": weekly.to_dict("records"),
        "outputs": outputs,
        "code_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 1:1 AIDev agentic/human performance sample stratified by ISO week."
    )
    parser.add_argument("--task-type", required=True, type=Path)
    parser.add_argument("--attribution", required=True, type=Path)
    parser.add_argument("--human-candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_weekly_balanced_sample(
        args.task_type,
        args.attribution,
        args.human_candidates,
        args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
