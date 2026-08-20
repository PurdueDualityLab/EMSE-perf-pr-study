"""Evaluate the experiment-1 heuristic and PerfMiner cascade against original AIDev labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from schema import atomic_write_text, write_parquet


IDENTITY = ("repo_id", "number")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_unique(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    if frame[list(columns)].isna().any().any() or frame.duplicated(list(columns)).any():
        raise ValueError(f"{label} has null or duplicate identities: {columns}")


def metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, int | float | None]:
    actual = frame["aidev_is_performance"]
    predicted = frame[prediction_column]
    true_positive = int((actual & predicted).sum())
    false_positive = int((~actual & predicted).sum())
    false_negative = int((actual & ~predicted).sum())
    true_negative = int((~actual & ~predicted).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "pr_count": len(frame),
        "positive_ground_truth": int(actual.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_evaluation(
    aidev_prs: pd.DataFrame,
    aidev_task_types: pd.DataFrame,
    experiment_decisions: pd.DataFrame,
    perfminer_prs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    require_unique(aidev_prs, ("id",), "AIDev human pull requests")
    require_unique(aidev_task_types, ("id",), "AIDev human task types")
    if "number" not in perfminer_prs.columns and "pr_number" in perfminer_prs.columns:
        perfminer_prs = perfminer_prs.rename(columns={"pr_number": "number"})
    if "id" not in experiment_decisions.columns:
        raise ValueError("Experiment decisions are missing the GitHub PR ID column.")
    experiment_decisions = experiment_decisions.loc[experiment_decisions["id"].notna()].copy()
    require_unique(experiment_decisions, ("id",), "Experiment decisions with a GitHub PR ID")
    require_unique(perfminer_prs, IDENTITY, "PerfMiner PR predictions")
    required_decisions = {"repo_id", "number", "heuristic_match"}
    required_perfminer = {
        "perfminer_pr_status",
        "perfminer_pr_is_performance",
        "perfminer_pr_evidence_complete",
    }
    if missing := required_decisions - set(experiment_decisions.columns):
        raise ValueError(f"Experiment decisions are missing columns: {sorted(missing)}")
    if missing := required_perfminer - set(perfminer_prs.columns):
        raise ValueError(f"PerfMiner PR predictions are missing columns: {sorted(missing)}")

    labels = aidev_prs[["id"]].merge(
        aidev_task_types[["id", "type"]], on="id", how="inner", validate="one_to_one"
    )
    labels["aidev_task_type"] = labels["type"].fillna("").astype(str).str.strip().str.lower()
    labels = labels.loc[labels["aidev_task_type"].ne("")].drop(columns="type")
    decisions = experiment_decisions[["id", *IDENTITY, "heuristic_match"]].copy()
    evaluation = labels.merge(decisions, on="id", how="left", validate="one_to_one", indicator=True)
    evaluation["in_experiment_decisions"] = evaluation.pop("_merge").eq("both")
    evaluation["aidev_is_performance"] = evaluation["aidev_task_type"].eq("perf")
    evaluation["heuristic_is_performance"] = evaluation["heuristic_match"].fillna(False).astype(bool)
    evaluation = evaluation.merge(
        perfminer_prs[[*IDENTITY, *sorted(required_perfminer)]],
        on=list(IDENTITY),
        how="left",
        validate="many_to_one",
    )
    evaluation["perfminer_is_evaluable"] = evaluation["perfminer_pr_is_performance"].isin([True, False])
    evaluation["perfminer_is_performance"] = evaluation["perfminer_pr_is_performance"].eq(True)
    evaluation["cascade_is_evaluable"] = ~evaluation["heuristic_is_performance"] | evaluation["perfminer_is_evaluable"]
    evaluation["cascade_is_performance"] = (
        evaluation["heuristic_is_performance"] & evaluation["perfminer_is_performance"]
    )

    intersection = evaluation.loc[evaluation["in_experiment_decisions"]].copy()
    heuristic_candidates = intersection.loc[intersection["heuristic_is_performance"]]
    cascade_cohort = intersection.loc[intersection["cascade_is_evaluable"]]
    summary = {
        "aidev_labeled_human_prs": len(labels),
        "aidev_prs_in_experiment_decisions": len(intersection),
        "aidev_prs_missing_from_experiment_decisions": int((~evaluation["in_experiment_decisions"]).sum()),
        "aidev_perf_prs_in_experiment_decisions": int(intersection["aidev_is_performance"].sum()),
        "heuristic": metrics(intersection, "heuristic_is_performance"),
        "perfminer_within_heuristic_candidates": metrics(
            heuristic_candidates.loc[heuristic_candidates["perfminer_is_evaluable"]], "perfminer_is_performance"
        ),
        "perfminer_candidate_coverage": {
            "heuristic_candidate_prs": len(heuristic_candidates),
            "evaluable_perfminer_prs": int(heuristic_candidates["perfminer_is_evaluable"].sum()),
            "not_evaluable_perfminer_prs": int((~heuristic_candidates["perfminer_is_evaluable"]).sum()),
        },
        "cascade": metrics(cascade_cohort, "cascade_is_performance"),
        "cascade_coverage": {
            "full_intersection_prs": len(intersection),
            "evaluable_prs": len(cascade_cohort),
            "excluded_for_missing_perfminer_decision": len(intersection) - len(cascade_cohort),
        },
    }
    return evaluation, summary


def markdown(summary: dict[str, object]) -> str:
    lines = ["# Experiment 1 Evaluation Against AIDev", "", "Original AIDev human task labels are the reference labels.", ""]
    lines.extend(["## Alignment", "", "| Metric | PRs |", "| --- | ---: |"])
    for key in (
        "aidev_labeled_human_prs",
        "aidev_prs_in_experiment_decisions",
        "aidev_prs_missing_from_experiment_decisions",
        "aidev_perf_prs_in_experiment_decisions",
    ):
        lines.append(f"| {key} | {summary[key]:,} |")
    lines.extend(["", "## Metrics", "", "| Method | PRs | TP | FP | FN | TN | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name in ("heuristic", "perfminer_within_heuristic_candidates", "cascade"):
        result = summary[name]
        format_metric = lambda value: "N/A" if value is None else f"{value:.4f}"
        lines.append(
            f"| {name} | {result['pr_count']:,} | {result['true_positive']:,} | {result['false_positive']:,} | "
            f"{result['false_negative']:,} | {result['true_negative']:,} | {format_metric(result['precision'])} | "
            f"{format_metric(result['recall'])} | {format_metric(result['f1'])} |"
        )
    lines.extend(["", "PerfMiner is evaluated conditionally among heuristic candidates with a definitive PR decision. The cascade is `heuristic AND PerfMiner`; rows without a required PerfMiner decision are excluded from its metric cohort and reported in coverage.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate experiment 1 against original AIDev task labels.")
    parser.add_argument("--aidev-human-prs", required=True)
    parser.add_argument("--aidev-human-task-types", required=True)
    parser.add_argument("--experiment-decisions", required=True, type=Path)
    parser.add_argument("--perfminer-prs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    evaluation, summary = build_evaluation(
        pd.read_parquet(args.aidev_human_prs),
        pd.read_parquet(args.aidev_human_task_types),
        pd.read_parquet(args.experiment_decisions),
        pd.read_parquet(args.perfminer_prs),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(evaluation, args.output_dir / "aidev_evaluation.parquet")
    summary["inputs"] = {
        "aidev_human_prs": args.aidev_human_prs,
        "aidev_human_task_types": args.aidev_human_task_types,
        "experiment_decisions": {"path": str(args.experiment_decisions), "sha256": sha256_file(args.experiment_decisions)},
        "perfminer_prs": {"path": str(args.perfminer_prs), "sha256": sha256_file(args.perfminer_prs)},
    }
    atomic_write_text(args.output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_write_text(args.output_dir / "report.md", markdown(summary))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
