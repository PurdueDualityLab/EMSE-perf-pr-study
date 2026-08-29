"""Measure GPT-Gemini RQ2 agreement and export review candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEY_COLUMNS = ["repo_id", "number"]
VALIDATION_TYPES = ("benchmark", "profiling", "static-reasoning", "anecdotal")


def cohen_kappa(first: pd.Series, second: pd.Series) -> float | None:
    if len(first) != len(second) or first.empty:
        return None
    first = first.fillna("").astype(str)
    second = second.fillna("").astype(str)
    observed = float(first.eq(second).mean())
    first_p = first.value_counts(normalize=True)
    second_p = second.value_counts(normalize=True)
    expected = sum(first_p.get(label, 0.0) * second_p.get(label, 0.0)
                   for label in set(first_p.index) | set(second_p.index))
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def _metric(first: pd.Series, second: pd.Series) -> dict[str, Any]:
    matches = int(first.fillna("").eq(second.fillna("")).sum())
    return {"rows": len(first), "matches": matches,
            "agreement_rate": matches / len(first) if len(first) else None,
            "cohen_kappa": cohen_kappa(first, second)}


def _canonical_types(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        labels = set(map(str, value))
    elif value is None or (isinstance(value, float) and pd.isna(value)):
        labels = set()
    else:
        raise ValueError("validation_types must contain lists.")
    unknown = labels - set(VALIDATION_TYPES)
    if unknown:
        raise ValueError(f"Unknown validation types: {sorted(unknown)}")
    return "|".join(label for label in VALIDATION_TYPES if label in labels)


def analyze(gpt: pd.DataFrame, gemini: pd.DataFrame,
            expected_sample: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {*KEY_COLUMNS, "sample_arm", "classification_status", "validation_present",
                "validation_types", "primary_validation_type", "prompt_version",
                "study_contract_sha256", "provider_config_sha256", "input_row_sha256", "prompt_sha256"}
    versions, study_hashes, provider_hashes = [], [], []
    for name, frame in (("GPT", gpt), ("Gemini", gemini)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} labels are missing columns: {sorted(missing)}")
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} labels contain duplicate PR identities.")
        arm_counts = frame["sample_arm"].value_counts().to_dict()
        if set(arm_counts) != {"agentic", "human_candidate"} or len(set(arm_counts.values())) != 1:
            raise ValueError(f"{name} labels do not contain the expected balanced sample arms.")
        for column, target in (("prompt_version", versions), ("study_contract_sha256", study_hashes),
                               ("provider_config_sha256", provider_hashes)):
            unique = frame[column].dropna().unique().tolist()
            if len(unique) != 1:
                raise ValueError(f"{name} labels must declare exactly one {column}.")
            target.append(unique[0])
    gpt_ids = set(map(tuple, gpt[KEY_COLUMNS].to_numpy()))
    gemini_ids = set(map(tuple, gemini[KEY_COLUMNS].to_numpy()))
    if gpt_ids != gemini_ids:
        raise ValueError("GPT and Gemini label sets contain different PR identities.")
    if expected_sample is not None:
        expected_ids = set(map(tuple, expected_sample[KEY_COLUMNS].to_numpy()))
        if expected_sample.duplicated(KEY_COLUMNS).any() or gpt_ids != expected_ids:
            raise ValueError("Model labels do not match the expected sample identities.")
    if versions[0] != versions[1]:
        raise ValueError(f"Model labels use different prompt versions: {versions}")
    if study_hashes[0] != study_hashes[1]:
        raise ValueError("GPT and Gemini labels use different study contract hashes.")
    coverage = {"gpt_status_counts": gpt["classification_status"].value_counts().to_dict(),
                "gemini_status_counts": gemini["classification_status"].value_counts().to_dict()}
    if not gpt["classification_status"].eq("classified").all() or not gemini["classification_status"].eq("classified").all():
        raise ValueError(f"All model errors must be resolved before agreement: {coverage}")

    columns = ["sample_arm", "html_url", "evidence_status", "evidence_complete", "validation_present",
               "validation_types", "primary_validation_type", "evidence_sources", "metrics",
               "evidence_quotes", "validation_description",
               "input_row_sha256", "prompt_sha256"]
    merged = gpt[[*KEY_COLUMNS, *columns]].merge(gemini[[*KEY_COLUMNS, *columns]], on=KEY_COLUMNS,
                                                 suffixes=("_gpt", "_gemini"), validate="one_to_one")
    if not merged["sample_arm_gpt"].eq(merged["sample_arm_gemini"]).all():
        raise ValueError("GPT and Gemini sample arms disagree.")
    for column in ("input_row_sha256", "prompt_sha256"):
        if not merged[f"{column}_gpt"].eq(merged[f"{column}_gemini"]).all():
            raise ValueError(f"GPT and Gemini labels use different {column} values.")
    merged["sample_arm"] = merged.pop("sample_arm_gpt")
    merged = merged.drop(columns="sample_arm_gemini")
    merged["validation_types_gpt_canonical"] = merged["validation_types_gpt"].map(_canonical_types)
    merged["validation_types_gemini_canonical"] = merged["validation_types_gemini"].map(_canonical_types)
    merged["presence_match"] = merged["validation_present_gpt"].eq(merged["validation_present_gemini"])
    merged["primary_type_match"] = merged["primary_validation_type_gpt"].eq(merged["primary_validation_type_gemini"])
    merged["type_set_match"] = merged["validation_types_gpt_canonical"].eq(merged["validation_types_gemini_canonical"])
    merged["full_match"] = (
        merged["presence_match"] & merged["primary_type_match"] & merged["type_set_match"]
    )
    for label in VALIDATION_TYPES:
        merged[f"{label}_gpt"] = merged["validation_types_gpt"].map(lambda values: label in values)
        merged[f"{label}_gemini"] = merged["validation_types_gemini"].map(lambda values: label in values)

    def metrics(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": len(frame),
            "validation_present": _metric(frame["validation_present_gpt"], frame["validation_present_gemini"]),
            "primary_validation_type": _metric(frame["primary_validation_type_gpt"], frame["primary_validation_type_gemini"]),
            "validation_type_set": _metric(frame["validation_types_gpt_canonical"], frame["validation_types_gemini_canonical"]),
            "per_validation_type": {label: _metric(frame[f"{label}_gpt"], frame[f"{label}_gemini"])
                                    for label in VALIDATION_TYPES},
            "full_matches": int(frame["full_match"].sum()),
            "full_agreement_rate": float(frame["full_match"].mean()) if len(frame) else None,
        }
    summary = {
        "gpt_classified_rows": len(gpt), "gemini_classified_rows": len(gemini),
        "overlap_rows": len(merged), "prompt_version": versions[0],
        "study_contract_sha256": study_hashes[0],
        "provider_config_sha256": {"gpt": provider_hashes[0], "gemini": provider_hashes[1]},
        "coverage": coverage, "overall": metrics(merged),
        "by_sample_arm": {arm: metrics(group) for arm, group in merged.groupby("sample_arm", sort=True)},
    }
    return merged.sort_values(KEY_COLUMNS).reset_index(drop=True), summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--gemini", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison, summary = analyze(pd.read_parquet(args.gpt), pd.read_parquet(args.gemini),
                                  pd.read_parquet(args.sample))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(args.output_dir / "model_comparison.parquet", index=False)
    comparison[comparison["full_match"]].to_parquet(args.output_dir / "agreements.parquet", index=False)
    comparison[~comparison["full_match"]].to_parquet(args.output_dir / "disagreements.parquet", index=False)
    adjudication_columns = [
        *KEY_COLUMNS,
        "sample_arm",
        "html_url_gpt",
        "validation_present_gpt",
        "validation_types_gpt_canonical",
        "primary_validation_type_gpt",
        "evidence_quotes_gpt",
        "validation_description_gpt",
        "validation_present_gemini",
        "validation_types_gemini_canonical",
        "primary_validation_type_gemini",
        "evidence_quotes_gemini",
        "validation_description_gemini",
    ]
    adjudication = comparison.loc[~comparison["full_match"], adjudication_columns].copy()
    adjudication["adjudication_status"] = "pending_human_review"
    adjudication["adjudicated_validation_present"] = ""
    adjudication["adjudicated_validation_types"] = ""
    adjudication["adjudicated_primary_validation_type"] = ""
    adjudication["adjudication_notes"] = ""
    adjudication.to_csv(args.output_dir / "adjudication_template.csv", index=False)
    (args.output_dir / "agreement_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
