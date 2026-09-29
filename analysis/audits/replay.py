"""Recompute audit summaries from frozen labels and votes, without model calls."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

KEYS = ["repo_id", "number"]
PROVIDERS = ("openai", "gemini", "qwen")
ROOT = Path(__file__).resolve().parents[2]


def regex_summary(frame):
    if frame.duplicated(KEYS + ["occurrence_id"]).any():
        raise ValueError("Duplicate occurrence identity")
    columns = [f"{p}_valid" for p in PROVIDERS]
    if not frame[columns].isin([True, False]).all().all():
        raise ValueError("Occurrence votes must be complete Booleans")
    data = frame.copy()
    data["valid"] = data[columns].sum(axis=1).ge(2)
    if data.groupby(KEYS).sample_arm.nunique().gt(1).any():
        raise ValueError("Inconsistent study arm for a PR")
    prs = data.groupby(KEYS + ["sample_arm"], as_index=False).valid.any()
    counts = {arm: {"true_positive": int(g.valid.sum()), "false_positive": int((~g.valid).sum())}
              for arm, g in prs.groupby("sample_arm")}
    return {"prs": len(prs), "occurrences": len(data), "true_positive": int(prs.valid.sum()),
            "false_positive": int((~prs.valid).sum()), "precision": float(prs.valid.mean()),
            "by_arm": counts, "reference": "three-model consensus, not human ground truth"}


def tradeoff_summary(frame):
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate PR identity")
    allowed = {"tradeoff", "joint_improvement"}
    columns = [f"{p}_label" for p in PROVIDERS]
    if not frame[columns].isin(allowed).all().all():
        raise ValueError("Expected complete binary trade-off votes")
    data = frame.copy()
    data["label"] = [Counter(row).most_common(1)[0][0] for row in data[columns].itertuples(index=False, name=None)]
    return {"prs": len(data), "classes": data.label.value_counts().sort_index().to_dict(),
            "by_arm": {arm: g.label.value_counts().sort_index().to_dict() for arm, g in data.groupby("sample_arm")},
            "reference": "initial binary model classification reported in the submitted manuscript"}


def manual_summary(frame):
    if frame.duplicated(KEYS).any() or not frame.manual_label.isin(["performance", "not_performance", "uncertain"]).all():
        raise ValueError("Invalid manual audit identities or labels")
    data = frame[frame.manual_label.ne("uncertain")]
    predicted = data.aidev_task_type.eq("perf")
    positive = data.manual_label.eq("performance")
    if data.sampling_weight.isna().any() or data.sampling_weight.le(0).any():
        raise ValueError("Sampling weights must be positive")
    if not ((data.sampling_weight * data.inclusion_probability - 1).abs() < 1e-9).all():
        raise ValueError("Sampling weights differ from inverse inclusion probabilities")

    def score(weights):
        masks = {"tp": predicted & positive, "fp": predicted & ~positive,
                 "fn": ~predicted & positive, "tn": ~predicted & ~positive}
        counts = {name: float(weights[mask].sum()) for name, mask in masks.items()}
        tp, fp, fn = (counts[name] for name in ("tp", "fp", "fn"))
        divide = lambda a, b: a / b if b else None
        return {**counts, "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
                "f1": divide(2 * tp, 2 * tp + fp + fn)}

    return {"sample_n": len(frame), "decided_n": len(data),
            "raw_sample": score(pd.Series(1.0, index=data.index)),
            "weighted_population": score(data.sampling_weight),
            "strata": frame.audit_stratum.value_counts().sort_index().to_dict(),
            "note": "Raw estimates describe the audited sample. Weighted estimates target its auditable LLM-classified population; the small negative sample makes recall uncertain."}


def replay(input_dir):
    return {
        "metric_precision": regex_summary(pd.read_csv(input_dir / "regex_occurrence_votes.csv")),
        "tradeoffs": tradeoff_summary(pd.read_csv(input_dir / "tradeoff_votes.csv")),
        "manual_classifier": manual_summary(pd.read_csv(input_dir / "manual_classifier_audit.csv")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "analysis/artifact_inputs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
