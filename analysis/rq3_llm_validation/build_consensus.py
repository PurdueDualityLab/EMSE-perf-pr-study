"""Build exact 2-of-3 binary consensus for both RQ3 validation tasks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.rq3_llm_validation import experiment

PROVIDERS = ("openai", "gemini", "qwen")
HASH_COLUMNS = ("input_row_sha256", "prompt_sha256", "study_contract_sha256")
REQUIRED = {*experiment.KEYS, "sample_arm", "task", "classification_status", "label", *HASH_COLUMNS}


def _validate(votes: dict[str, pd.DataFrame], task: str) -> dict[str, pd.DataFrame]:
    if tuple(votes) != PROVIDERS:
        raise ValueError(f"Votes must be supplied in provider order {PROVIDERS}.")
    baseline = None
    normalized = {}
    model = experiment.label_model(task)
    for provider, source in votes.items():
        missing = REQUIRED - set(source.columns)
        if missing:
            raise ValueError(f"{provider} labels are missing columns: {sorted(missing)}")
        if source.duplicated(experiment.KEYS).any():
            raise ValueError(f"{provider} contains duplicate PR identities.")
        if not source["classification_status"].eq("classified").all():
            raise ValueError(f"All {provider} rows must be classified; errors or missing rows remain.")
        if not source["task"].eq(task).all():
            raise ValueError(f"{provider} contains labels for the wrong task.")
        for row in source.to_dict("records"):
            fields = set(model.model_fields)
            model.model_validate({key: row[key] for key in fields})
        signature = source[[*experiment.KEYS, "sample_arm", "task", *HASH_COLUMNS]].sort_values(
            experiment.KEYS
        ).reset_index(drop=True)
        if signature[list(HASH_COLUMNS)].isna().any().any():
            raise ValueError(f"{provider} contains missing hashes.")
        if baseline is not None and not signature.equals(baseline):
            raise ValueError("Provider inputs, prompts, study contracts, identities, or arms do not match.")
        baseline = signature
        normalized[provider] = source
    return normalized


def build_consensus(votes: dict[str, pd.DataFrame], task: str,
                    source_hashes: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    votes = _validate(votes, task)
    indexed = {provider: frame.set_index(experiment.KEYS, drop=False) for provider, frame in votes.items()}
    records = []
    for key in indexed["openai"].index.sort_values():
        rows = {provider: indexed[provider].loc[key] for provider in PROVIDERS}
        labels = {provider: str(row["label"]) for provider, row in rows.items()}
        winner, count = Counter(labels.values()).most_common(1)[0]
        if count < 2:
            raise ValueError(f"Binary consensus unexpectedly unresolved for {key}.")
        coalition = [provider for provider in PROVIDERS if labels[provider] == winner]
        record = {"repo_id": key[0], "number": key[1],
                  "sample_arm": rows["openai"]["sample_arm"], "task": task,
                  "consensus_label": winner,
                  "consensus_status": "unanimous" if count == 3 else "majority",
                  "vote_count": count, "coalition": "+".join(coalition)}
        for provider, row in rows.items():
            for column in votes[provider].columns:
                if column not in experiment.KEYS:
                    record[f"{provider}_{column}"] = row[column]
        records.append(record)
    result = pd.DataFrame(records).sort_values(experiment.KEYS).reset_index(drop=True)
    summary = {
        "task": task, "rows": len(result),
        "agreement_counts": result.consensus_status.value_counts().sort_index().to_dict(),
        "arm_counts": result.sample_arm.value_counts().sort_index().to_dict(),
        "class_counts": result.consensus_label.value_counts().sort_index().to_dict(),
        "arm_class_counts": {
            f"{arm}:{label}": int(count)
            for (arm, label), count in result.groupby(["sample_arm", "consensus_label"]).size().items()
        },
        "source_sha256": source_hashes or {},
    }
    return result, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=experiment.TASKS, required=True)
    for provider in PROVIDERS:
        parser.add_argument(f"--{provider}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {provider: getattr(args, provider) for provider in PROVIDERS}
    consensus, summary = build_consensus(
        {provider: pd.read_parquet(path) for provider, path in paths.items()}, args.task,
        {provider: experiment.sha256_file(path) for provider, path in paths.items()},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    consensus.to_parquet(args.output_dir / f"{args.task}_consensus.parquet", index=False)
    experiment.atomic_write_json(args.output_dir / f"{args.task}_consensus_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
