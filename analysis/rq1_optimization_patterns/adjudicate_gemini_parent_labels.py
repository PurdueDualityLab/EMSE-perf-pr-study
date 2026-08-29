"""Correct Gemini parent labels when the returned sub-pattern has one catalog parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def adjudicate(
    labels_path: Path, catalog_path: Path, provider_dir: Path, audit_path: Path
) -> pd.DataFrame:
    labels = pd.read_parquet(labels_path)
    catalog = pd.read_csv(catalog_path)
    parent_by_sub_pattern = (
        catalog.groupby("Sub pattern")["High-level Pattern"].agg(lambda values: sorted(set(values)))
    )
    error_mask = labels["classification_status"].eq("error")
    error_keys = set(labels.loc[error_mask, "key"])
    returned_labels = {}
    for output_path in sorted(provider_dir.glob("output-*.jsonl")):
        for line in output_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("key") not in error_keys or not item.get("response"):
                continue
            response = item["response"]
            text = "".join(
                str(part.get("text") or "")
                for candidate in response.get("candidates", [])[:1]
                for part in (candidate.get("content") or {}).get("parts", [])
            )
            returned_labels[item["key"]] = json.loads(text)
    audit_rows = []
    for index, row in labels[error_mask].iterrows():
        returned = returned_labels.get(row["key"])
        if not returned:
            raise ValueError(f"No raw model label found for {row['key']}.")
        sub_pattern = returned["sub_pattern"]
        parents = parent_by_sub_pattern.get(sub_pattern, [])
        if not parents:
            canonical_matches = [
                candidate
                for candidate in parent_by_sub_pattern.index
                if candidate.rstrip(")") == sub_pattern.rstrip(")")
            ]
            if len(canonical_matches) == 1:
                sub_pattern = canonical_matches[0]
                parents = parent_by_sub_pattern[sub_pattern]
        if len(parents) != 1:
            raise ValueError(
                f"Cannot uniquely adjudicate {row['key']}: {sub_pattern!r} has parents {parents}."
            )
        corrected_parent = parents[0]
        audit_rows.append(
            {
                "key": row["key"],
                "repo_id": row["repo_id"],
                "number": row["number"],
                "original_high_level_pattern": returned["high_level_pattern"],
                "sub_pattern": sub_pattern,
                "corrected_high_level_pattern": corrected_parent,
                "original_error": row["error"],
                "adjudication_rule": "Preserve the model sub-pattern and use its unique catalog parent.",
                "authorization": "Manual correction authorized by the study owner.",
            }
        )
        labels.at[index, "explanation"] = returned["explanation"]
        labels.at[index, "optimization_comparison"] = returned["optimization_comparison"]
        labels.at[index, "sub_pattern"] = sub_pattern
        labels.at[index, "high_level_pattern"] = corrected_parent
        labels.at[index, "classification_status"] = "classified"
        labels.at[index, "error"] = None

    if not audit_rows:
        raise ValueError("No error rows are available for adjudication.")
    pd.DataFrame(audit_rows).sort_values(["repo_id", "number"]).to_csv(audit_path, index=False)
    labels.to_parquet(labels_path, index=False)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--provider-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    labels = adjudicate(args.labels, args.catalog, args.provider_dir, args.audit)
    print(f"Adjudicated {labels['classification_status'].eq('classified').sum()} total labels")


if __name__ == "__main__":
    main()
