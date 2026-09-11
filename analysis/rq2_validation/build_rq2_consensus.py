"""Build a strict two-stage 2-of-3 consensus for the RQ2 labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEY_COLUMNS = ["repo_id", "number"]
ARMS = ("agentic", "human_candidate")
MODELS = ("gpt", "gemini", "qwen")
VALIDATION_TYPES = ("benchmark", "profiling", "static-reasoning", "anecdotal")
EVIDENCE_SOURCES = ("description", "comments", "reviews", "code_diff", "ci")
REQUIRED_COLUMNS = {
    *KEY_COLUMNS,
    "sample_arm",
    "classification_status",
    "validation_present",
    "validation_types",
    "primary_validation_type",
    "evidence_sources",
    "metrics",
    "evidence_quotes",
    "input_row_sha256",
    "prompt_sha256",
}
OFFICIAL_CONTROLS = {
    "rows": 2258,
    "stage1_positive": 1839,
    "stage1_positive_by_arm": {"agentic": 929, "human_candidate": 910},
    "stage1_negative": 419,
    "stage2_consensus": 1707,
    "stage2_consensus_by_arm": {"agentic": 870, "human_candidate": 837},
    "stage2_unresolved": 132,
    "stage2_unresolved_by_arm": {"agentic": 59, "human_candidate": 73},
    "complete_consensus": 2126,
    "full_unanimous": 1261,
    "exact_majority": 865,
    "unresolved": 132,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_list(value: Any, field: str) -> list[Any]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    raise ValueError(f"{field} must contain lists.")


def canonical_types(value: Any) -> tuple[str, ...]:
    values = _as_list(value, "validation_types")
    if len(values) != len(set(values)):
        raise ValueError("validation_types must not contain duplicates.")
    unknown = set(values) - set(VALIDATION_TYPES)
    if unknown:
        raise ValueError(f"Unknown validation types: {sorted(unknown)}")
    return tuple(label for label in VALIDATION_TYPES if label in values)


def _validate_semantics(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    normalized = frame.copy()
    for index, row in normalized.iterrows():
        prefix = f"{model} row {tuple(row[column] for column in KEY_COLUMNS)}"
        if not isinstance(row["validation_present"], (bool, np.bool_)):
            raise ValueError(f"{prefix}: validation_present must be boolean.")
        types = canonical_types(row["validation_types"])
        primary = row["primary_validation_type"]
        if primary not in {*VALIDATION_TYPES, "none"}:
            raise ValueError(f"{prefix}: unknown primary_validation_type {primary!r}.")
        sources = _as_list(row["evidence_sources"], "evidence_sources")
        if len(sources) != len(set(sources)) or set(sources) - set(EVIDENCE_SOURCES):
            raise ValueError(f"{prefix}: invalid or duplicate evidence_sources.")
        metrics = _as_list(row["metrics"], "metrics")
        quotes = _as_list(row["evidence_quotes"], "evidence_quotes")
        if row["validation_present"]:
            if not types or primary not in types or not sources or not quotes:
                raise ValueError(f"{prefix}: positive label violates RQ2 semantics.")
        elif types or primary != "none" or sources or metrics or quotes:
            raise ValueError(f"{prefix}: absent label violates RQ2 semantics.")
        normalized.at[index, "validation_types"] = list(types)
    return normalized


def _validate_inputs(
    votes: dict[str, pd.DataFrame], expected_sample: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    if tuple(votes) != MODELS:
        raise ValueError(f"Votes must be supplied in model order {MODELS}.")
    sample_required = {*KEY_COLUMNS, "sample_arm"}
    if sample_required - set(expected_sample.columns):
        raise ValueError("Expected sample is missing identity or sample_arm columns.")
    if expected_sample.duplicated(KEY_COLUMNS).any():
        raise ValueError("Expected sample contains duplicate PR identities.")
    if not expected_sample["sample_arm"].isin(ARMS).all():
        raise ValueError("Expected sample contains an unknown sample arm.")
    expected = expected_sample[[*KEY_COLUMNS, "sample_arm"]].sort_values(KEY_COLUMNS).reset_index(drop=True)
    normalized: dict[str, pd.DataFrame] = {}
    for model, original in votes.items():
        missing = REQUIRED_COLUMNS - set(original.columns)
        if missing:
            raise ValueError(f"{model} labels are missing columns: {sorted(missing)}")
        if original.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{model} labels contain duplicate PR identities.")
        actual = original[[*KEY_COLUMNS, "sample_arm"]].sort_values(KEY_COLUMNS).reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError(f"{model} labels do not exactly match expected sample identities and arms.")
        if not original["classification_status"].eq("classified").all():
            counts = original["classification_status"].value_counts().to_dict()
            raise ValueError(f"All {model} rows must be classified: {counts}")
        normalized[model] = _validate_semantics(original, model)

    hashes = None
    for model, frame in normalized.items():
        current = frame[[*KEY_COLUMNS, "input_row_sha256", "prompt_sha256"]].sort_values(KEY_COLUMNS).reset_index(drop=True)
        if current[["input_row_sha256", "prompt_sha256"]].isna().any().any():
            raise ValueError(f"{model} contains missing row or prompt hashes.")
        if hashes is not None and not current.equals(hashes):
            raise ValueError("Models use different input_row_sha256 or prompt_sha256 values.")
        hashes = current
    return normalized


def _coalition(models: list[str]) -> str:
    return "+".join(model for model in MODELS if model in models)


def _summary(frame: pd.DataFrame, source_hashes: dict[str, str] | None) -> dict[str, Any]:
    positive = frame[frame["consensus_validation_present"]]
    included = frame[frame["included_in_stage2_analysis"]]
    unresolved = positive[positive["consensus_status"].eq("unresolved")]
    result = {
        "rows": len(frame),
        "stage1_positive": len(positive),
        "stage1_positive_by_arm": positive["sample_arm"].value_counts().sort_index().to_dict(),
        "stage1_negative": int((~frame["consensus_validation_present"]).sum()),
        "stage2_consensus": len(included),
        "stage2_consensus_by_arm": included["sample_arm"].value_counts().sort_index().to_dict(),
        "stage2_unresolved": len(unresolved),
        "stage2_unresolved_by_arm": unresolved["sample_arm"].value_counts().sort_index().to_dict(),
        "complete_consensus": int((~frame["consensus_status"].eq("unresolved")).sum()),
        "full_unanimous": int(frame["consensus_status"].eq("full_unanimous").sum()),
        "exact_majority": int(frame["consensus_status"].eq("exact_majority").sum()),
        "unresolved": int(frame["consensus_status"].eq("unresolved").sum()),
        "source_sha256": source_hashes or {},
    }
    return result


def build_consensus(
    votes: dict[str, pd.DataFrame],
    expected_sample: pd.DataFrame,
    *,
    source_hashes: dict[str, str] | None = None,
    enforce_official_controls: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate three complete label sets and return deterministic consensus rows."""
    votes = _validate_inputs(votes, expected_sample)
    indexed = {model: frame.set_index(KEY_COLUMNS, drop=False) for model, frame in votes.items()}
    records: list[dict[str, Any]] = []
    sample = expected_sample.sort_values(KEY_COLUMNS, kind="mergesort")
    for _, sample_row in sample.iterrows():
        key = tuple(sample_row[column] for column in KEY_COLUMNS)
        rows = {model: indexed[model].loc[key] for model in MODELS}
        record: dict[str, Any] = {"repo_id": key[0], "number": key[1], "sample_arm": sample_row["sample_arm"]}
        for model, row in rows.items():
            for column in votes[model].columns:
                if column not in KEY_COLUMNS:
                    record[f"{model}_{column}"] = row[column]

        positive_models = [model for model, row in rows.items() if bool(row["validation_present"])]
        present = len(positive_models) >= 2
        winning_presence = positive_models if present else [model for model in MODELS if model not in positive_models]
        record.update({
            "stage1_status": "full_unanimous" if len(winning_presence) == 3 else "exact_majority",
            "stage1_positive_votes": len(positive_models),
            "stage1_negative_votes": 3 - len(positive_models),
            "stage1_coalition": _coalition(winning_presence),
            "consensus_validation_present": present,
        })
        if not present:
            record.update({
                "stage2_status": "not_applicable_absent",
                "stage2_vote_count": 0,
                "stage2_coalition": "",
                "consensus_primary_validation_type": "none",
                "consensus_validation_types": [],
                "consensus_status": record["stage1_status"],
                "included_in_stage2_analysis": False,
                "inclusion_reason": "stage1_consensus_absent",
            })
        else:
            tuples = {
                model: (str(rows[model]["primary_validation_type"]), tuple(rows[model]["validation_types"]))
                for model in positive_models
            }
            counts = Counter(tuples.values())
            winner, count = counts.most_common(1)[0]
            coalition = [model for model in positive_models if tuples[model] == winner]
            resolved = count >= 2
            unanimous = len(positive_models) == 3 and count == 3
            record.update({
                "stage2_status": "full_unanimous" if unanimous else ("exact_majority" if resolved else "unresolved"),
                "stage2_vote_count": count,
                "stage2_coalition": _coalition(coalition) if resolved else "",
                "consensus_primary_validation_type": winner[0] if resolved else None,
                "consensus_validation_types": list(winner[1]) if resolved else [],
                "consensus_status": "full_unanimous" if unanimous else ("exact_majority" if resolved else "unresolved"),
                "included_in_stage2_analysis": resolved,
                "inclusion_reason": "positive_exact_tuple_consensus" if resolved else "positive_without_exact_tuple_majority",
            })
        records.append(record)

    result = pd.DataFrame(records).sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    summary = _summary(result, source_hashes)
    should_enforce = len(result) == OFFICIAL_CONTROLS["rows"] if enforce_official_controls is None else enforce_official_controls
    if should_enforce:
        mismatches = {key: (summary.get(key), expected) for key, expected in OFFICIAL_CONTROLS.items() if summary.get(key) != expected}
        if mismatches:
            raise ValueError(f"Official RQ2 controls failed: {mismatches}")
    return result, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt", type=Path, required=True)
    parser.add_argument("--gemini", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-official-controls", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {model: getattr(args, model) for model in MODELS}
    source_hashes = {model: sha256_file(path) for model, path in paths.items()}
    source_hashes["sample"] = sha256_file(args.sample)
    consensus, summary = build_consensus(
        {model: pd.read_parquet(path) for model, path in paths.items()},
        pd.read_parquet(args.sample),
        source_hashes=source_hashes,
        enforce_official_controls=False if args.skip_official_controls else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    consensus.to_parquet(args.output_dir / "rq2_consensus.parquet", index=False)
    consensus.to_csv(args.output_dir / "rq2_consensus.csv", index=False)
    (args.output_dir / "rq2_consensus_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
