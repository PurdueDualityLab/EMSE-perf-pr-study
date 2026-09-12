"""Build the complete three-model, atomic 2-of-3 RQ1 consensus artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


KEY_COLUMNS = ["repo_id", "number"]
MODELS = ("gpt", "gemini", "qwen")
LABEL_COLUMNS = ("high_level_pattern", "sub_pattern")
EXPECTED_REAL_DATA = {
    "rows": 2260,
    "sample_arm_counts": {"agentic": 1130, "human_candidate": 1130},
    "consensus_status_counts": {
        "unanimous": 1238,
        "majority_2_of_3": 845,
        "no_2_of_3_consensus": 177,
    },
    "included_rows": 2083,
    "included_arm_counts": {"agentic": 1049, "human_candidate": 1034},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(catalog: pd.DataFrame) -> set[tuple[str, str]]:
    required = {"High-level Pattern", "Sub pattern"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"Catalog is missing columns: {sorted(missing)}")
    labels = catalog[["High-level Pattern", "Sub pattern"]]
    if labels.isna().any().any():
        raise ValueError("Catalog hierarchy contains missing labels.")
    if labels.duplicated().any():
        raise ValueError("Catalog hierarchy contains duplicate labels.")
    if any(labels[column].astype(str).str.strip().ne(labels[column]).any() for column in labels):
        raise ValueError("Catalog hierarchy contains labels with surrounding whitespace.")
    return set(map(tuple, labels.to_numpy()))


def _validate_sample(sample: pd.DataFrame) -> pd.DataFrame:
    required = {*KEY_COLUMNS, "sample_arm"}
    missing = required - set(sample.columns)
    if missing:
        raise ValueError(f"Official sample is missing columns: {sorted(missing)}")
    if sample.empty or sample.duplicated(KEY_COLUMNS).any():
        raise ValueError("Official sample must have unique PR identities.")
    if not sample["sample_arm"].isin(["agentic", "human_candidate"]).all():
        raise ValueError("Official sample contains an invalid sample arm.")
    return sample


def _validate_labels(
    name: str,
    labels: pd.DataFrame,
    expected: pd.DataFrame,
    valid_labels: set[tuple[str, str]],
) -> None:
    required = {
        *KEY_COLUMNS,
        "sample_arm",
        "classification_status",
        "input_row_sha256",
        "prompt_sha256",
        "prompt_version",
        *LABEL_COLUMNS,
    }
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"{name} labels are missing columns: {sorted(missing)}")
    if labels.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{name} labels contain duplicate PR identities.")
    if len(labels) != len(expected):
        raise ValueError(f"{name} labels do not exactly match the official sample IDs and arms.")
    aligned = expected[KEY_COLUMNS + ["sample_arm"]].merge(
        labels[KEY_COLUMNS + ["sample_arm"]],
        on=KEY_COLUMNS,
        how="outer",
        suffixes=("_sample", "_model"),
        indicator=True,
        validate="one_to_one",
    )
    if not aligned["_merge"].eq("both").all() or not aligned["sample_arm_sample"].eq(
        aligned["sample_arm_model"]
    ).all():
        raise ValueError(f"{name} labels do not exactly match the official sample IDs and arms.")
    if not labels["classification_status"].eq("classified").all():
        counts = labels["classification_status"].value_counts(dropna=False).to_dict()
        raise ValueError(f"{name} contains non-classified rows: {counts}")
    if labels["input_row_sha256"].isna().any():
        raise ValueError(f"{name} contains missing input_row_sha256 values.")
    pairs = set(map(tuple, labels[list(LABEL_COLUMNS)].to_numpy()))
    invalid = pairs - valid_labels
    if invalid:
        raise ValueError(f"{name} contains labels outside the catalog hierarchy: {sorted(invalid)!r}")


def build_consensus(
    gpt: pd.DataFrame,
    gemini: pd.DataFrame,
    qwen: pd.DataFrame,
    sample: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate inputs and vote on indivisible (parent, child) label tuples."""
    sample = _validate_sample(sample)
    valid_labels = load_catalog(catalog)
    frames = {"gpt": gpt, "gemini": gemini, "qwen": qwen}
    for name, frame in frames.items():
        _validate_labels(name.upper(), frame, sample, valid_labels)
    gpt_prompt = gpt[KEY_COLUMNS + ["prompt_version", "prompt_sha256"]].merge(
        gemini[KEY_COLUMNS + ["prompt_version", "prompt_sha256"]],
        on=KEY_COLUMNS,
        suffixes=("_gpt", "_gemini"),
        validate="one_to_one",
    )
    for column in ("prompt_version", "prompt_sha256"):
        if not gpt_prompt[f"{column}_gpt"].eq(gpt_prompt[f"{column}_gemini"]).all():
            raise ValueError(f"GPT and Gemini use different {column} values.")

    base_columns = KEY_COLUMNS + ["sample_arm"]
    for optional in ("repo_full_name", "html_url"):
        if optional in sample.columns:
            base_columns.append(optional)
    result = sample[base_columns].copy()
    hash_columns = []
    for name in MODELS:
        renamed = frames[name][KEY_COLUMNS + [*LABEL_COLUMNS, "input_row_sha256"]].rename(
            columns={column: f"{name}_{column}" for column in (*LABEL_COLUMNS, "input_row_sha256")}
        )
        result = result.merge(renamed, on=KEY_COLUMNS, how="left", validate="one_to_one")
        hash_columns.append(f"{name}_input_row_sha256")
    if not result[hash_columns].nunique(axis=1, dropna=False).eq(1).all():
        raise ValueError("Models use different input_row_sha256 values for at least one PR.")
    result["input_row_sha256"] = result[hash_columns[0]]
    result = result.drop(columns=hash_columns)

    decisions: list[dict[str, Any]] = []
    for row in result.to_dict("records"):
        votes = {
            name: (row[f"{name}_high_level_pattern"], row[f"{name}_sub_pattern"])
            for name in MODELS
        }
        counts = Counter(votes.values())
        winning_label, vote_count = counts.most_common(1)[0]
        if vote_count == 3:
            status = "unanimous"
        elif vote_count == 2:
            status = "majority_2_of_3"
        else:
            status = "no_2_of_3_consensus"
        coalition = "+".join(name for name in MODELS if votes[name] == winning_label) if vote_count >= 2 else ""
        decisions.append(
            {
                "consensus_status": status,
                "consensus_vote_count": vote_count,
                "consensus_coalition": coalition,
                "consensus_high_level_pattern": winning_label[0] if vote_count >= 2 else pd.NA,
                "consensus_sub_pattern": winning_label[1] if vote_count >= 2 else pd.NA,
                "included_in_analysis": vote_count >= 2,
                "inclusion_reason": (
                    "atomic_hierarchical_2_of_3_consensus"
                    if vote_count >= 2
                    else "no_atomic_hierarchical_2_of_3_consensus"
                ),
            }
        )
    result = pd.concat([result, pd.DataFrame(decisions)], axis=1)
    result = result.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)

    status_counts = result["consensus_status"].value_counts().to_dict()
    included = result[result["included_in_analysis"]]
    summary = {
        "rows": len(result),
        "sample_arm_counts": result["sample_arm"].value_counts().sort_index().to_dict(),
        "consensus_status_counts": status_counts,
        "included_rows": len(included),
        "included_arm_counts": included["sample_arm"].value_counts().sort_index().to_dict(),
        "excluded_rows": int((~result["included_in_analysis"]).sum()),
        "voting_unit": ["high_level_pattern", "sub_pattern"],
        "prompt_compatibility": {
            "exact_prompt_equality_claimed": False,
            "reason": "Qwen has a documented prompt version/hash whitespace mismatch; input-row hashes are equal.",
            "versions": {
                name: sorted(map(str, frames[name]["prompt_version"].dropna().unique()))
                for name in MODELS
            },
            "prompt_sha256_counts": {
                name: int(frames[name]["prompt_sha256"].nunique(dropna=False)) for name in MODELS
            },
        },
    }
    return result, summary


def assert_real_data_controls(summary: dict[str, Any]) -> None:
    for field, expected in EXPECTED_REAL_DATA.items():
        if summary[field] != expected:
            raise ValueError(f"Real-data control failed for {field}: expected {expected}, got {summary[field]}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    base = root / "analysis" / "rq1_optimization_patterns"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", type=Path, default=base / "results_gpt" / "optimization_pattern_labels_complete.parquet")
    parser.add_argument("--gemini", type=Path, default=base / "results_gemini" / "optimization_pattern_labels.parquet")
    parser.add_argument("--qwen", type=Path, default=base / "results_qwen" / "optimization_pattern_labels.parquet")
    parser.add_argument("--sample", type=Path, default=root / "data" / "data" / "sample" / "balanced_sample.parquet")
    parser.add_argument("--catalog", type=Path, default=base / "catalog" / "updated_optimization_catalog.csv")
    parser.add_argument("--output-dir", type=Path, default=base / "consensus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {name: getattr(args, name) for name in (*MODELS, "sample", "catalog")}
    consensus, summary = build_consensus(
        pd.read_parquet(args.gpt),
        pd.read_parquet(args.gemini),
        pd.read_parquet(args.qwen),
        pd.read_parquet(args.sample),
        pd.read_csv(args.catalog),
    )
    assert_real_data_controls(summary)
    summary["source_sha256"] = {name: sha256_file(path) for name, path in paths.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    consensus.to_parquet(args.output_dir / "rq1_consensus.parquet", index=False)
    (args.output_dir / "rq1_consensus_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
