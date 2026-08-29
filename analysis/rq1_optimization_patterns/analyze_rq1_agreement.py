"""Measure GPT-Gemini RQ1 agreement and export review candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


KEY_COLUMNS = ["repo_id", "number"]
LABEL_COLUMNS = ["high_level_pattern", "sub_pattern"]


def cohen_kappa(first: pd.Series, second: pd.Series) -> float | None:
    if len(first) != len(second) or first.empty:
        return None
    first = first.fillna("").astype(str)
    second = second.fillna("").astype(str)
    observed = float(first.eq(second).mean())
    first_proportions = first.value_counts(normalize=True)
    second_proportions = second.value_counts(normalize=True)
    expected = sum(
        first_proportions.get(label, 0.0) * second_proportions.get(label, 0.0)
        for label in set(first_proportions.index) | set(second_proportions.index)
    )
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def _metric(first: pd.Series, second: pd.Series) -> dict[str, Any]:
    matches = int(first.fillna("").eq(second.fillna("")).sum())
    return {
        "rows": len(first),
        "matches": matches,
        "agreement_rate": matches / len(first) if len(first) else None,
        "cohen_kappa": cohen_kappa(first, second),
    }


def analyze(
    gpt: pd.DataFrame,
    gemini: pd.DataFrame,
    expected_sample: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        *KEY_COLUMNS,
        *LABEL_COLUMNS,
        "sample_arm",
        "classification_status",
        "study_contract_sha256",
        "provider_config_sha256",
        "input_row_sha256",
        "prompt_sha256",
    }
    for name, frame in (("GPT", gpt), ("Gemini", gemini)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} labels are missing columns: {sorted(missing)}")
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} labels contain duplicate PR identities.")
        arm_counts = frame["sample_arm"].value_counts().to_dict()
        if set(arm_counts) != {"agentic", "human_candidate"} or len(set(arm_counts.values())) != 1:
            raise ValueError(f"{name} labels do not contain the expected balanced sample arms.")
    gpt_ids = set(map(tuple, gpt[KEY_COLUMNS].to_numpy()))
    gemini_ids = set(map(tuple, gemini[KEY_COLUMNS].to_numpy()))
    if gpt_ids != gemini_ids:
        raise ValueError("GPT and Gemini label sets contain different PR identities.")
    if expected_sample is not None:
        expected_ids = set(map(tuple, expected_sample[KEY_COLUMNS].to_numpy()))
        if expected_sample.duplicated(KEY_COLUMNS).any() or gpt_ids != expected_ids:
            raise ValueError("Model labels do not match the expected sample identities.")
    versions = []
    for name, frame in (("GPT", gpt), ("Gemini", gemini)):
        if "prompt_version" not in frame.columns:
            raise ValueError(f"{name} labels do not declare prompt_version.")
        unique = frame["prompt_version"].dropna().unique().tolist()
        if len(unique) != 1:
            raise ValueError(f"{name} labels must declare exactly one prompt_version.")
        versions.append(unique[0])
    if len(set(versions)) > 1:
        raise ValueError(f"Model labels use different prompt versions: {versions}")
    study_hashes = []
    provider_hashes = []
    for name, frame in (("GPT", gpt), ("Gemini", gemini)):
        unique = frame["study_contract_sha256"].dropna().unique().tolist()
        if len(unique) != 1:
            raise ValueError(f"{name} labels must declare one study contract hash.")
        study_hashes.append(unique[0])
        provider_unique = frame["provider_config_sha256"].dropna().unique().tolist()
        if len(provider_unique) != 1:
            raise ValueError(f"{name} labels must declare one provider config hash.")
        provider_hashes.append(provider_unique[0])
    if study_hashes[0] != study_hashes[1]:
        raise ValueError("GPT and Gemini labels use different study contract hashes.")

    coverage = {
        "gpt_status_counts": gpt["classification_status"].value_counts().to_dict(),
        "gemini_status_counts": gemini["classification_status"].value_counts().to_dict(),
    }
    if not gpt["classification_status"].eq("classified").all() or not gemini[
        "classification_status"
    ].eq("classified").all():
        raise ValueError(f"All model errors must be resolved before agreement: {coverage}")

    comparison_columns = ["sample_arm", *LABEL_COLUMNS, "input_row_sha256", "prompt_sha256"]
    merged = gpt[[*KEY_COLUMNS, *comparison_columns]].merge(
        gemini[[*KEY_COLUMNS, *comparison_columns]],
        on=KEY_COLUMNS,
        how="inner",
        suffixes=("_gpt", "_gemini"),
        validate="one_to_one",
    )
    if not merged["sample_arm_gpt"].eq(merged["sample_arm_gemini"]).all():
        raise ValueError("GPT and Gemini sample arms disagree.")
    for column in ("input_row_sha256", "prompt_sha256"):
        if not merged[f"{column}_gpt"].eq(merged[f"{column}_gemini"]).all():
            raise ValueError(f"GPT and Gemini labels use different {column} values.")
    merged["sample_arm"] = merged.pop("sample_arm_gpt")
    merged = merged.drop(columns="sample_arm_gemini")
    merged["high_level_match"] = merged["high_level_pattern_gpt"].fillna("").eq(
        merged["high_level_pattern_gemini"].fillna("")
    )
    merged["sub_pattern_match"] = merged["sub_pattern_gpt"].fillna("").eq(
        merged["sub_pattern_gemini"].fillna("")
    )
    merged["full_match"] = merged["high_level_match"] & merged["sub_pattern_match"]

    def metrics(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": len(frame),
            "high_level_pattern": _metric(
                frame["high_level_pattern_gpt"], frame["high_level_pattern_gemini"]
            ),
            "sub_pattern": _metric(frame["sub_pattern_gpt"], frame["sub_pattern_gemini"]),
            "hierarchical_label": _metric(
                frame["high_level_pattern_gpt"] + " / " + frame["sub_pattern_gpt"],
                frame["high_level_pattern_gemini"] + " / " + frame["sub_pattern_gemini"],
            ),
            "full_matches": int(frame["full_match"].sum()),
            "full_agreement_rate": float(frame["full_match"].mean()) if len(frame) else None,
        }

    summary = {
        "gpt_classified_rows": len(gpt),
        "gemini_classified_rows": len(gemini),
        "overlap_rows": len(merged),
        "prompt_version": versions[0] if versions else None,
        "study_contract_sha256": study_hashes[0],
        "provider_config_sha256": {"gpt": provider_hashes[0], "gemini": provider_hashes[1]},
        "coverage": coverage,
        "overall": metrics(merged),
        "by_sample_arm": {
            arm: metrics(group) for arm, group in merged.groupby("sample_arm", sort=True)
        },
    }
    return merged.sort_values(KEY_COLUMNS).reset_index(drop=True), summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--gemini", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison, summary = analyze(
        pd.read_parquet(args.gpt),
        pd.read_parquet(args.gemini),
        pd.read_parquet(args.sample),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(args.output_dir / "model_comparison.parquet", index=False)
    comparison[comparison["full_match"]].to_parquet(
        args.output_dir / "agreements.parquet", index=False
    )
    comparison[~comparison["full_match"]].to_parquet(
        args.output_dir / "disagreements.parquet", index=False
    )
    adjudication = comparison.loc[
        ~comparison["full_match"],
        [
            *KEY_COLUMNS,
            "sample_arm",
            "high_level_pattern_gpt",
            "sub_pattern_gpt",
            "high_level_pattern_gemini",
            "sub_pattern_gemini",
        ],
    ].copy()
    adjudication["adjudication_status"] = "pending_human_review"
    adjudication["adjudicated_high_level_pattern"] = ""
    adjudication["adjudicated_sub_pattern"] = ""
    adjudication["adjudication_notes"] = ""
    adjudication.to_csv(args.output_dir / "adjudication_template.csv", index=False)
    (args.output_dir / "agreement_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
