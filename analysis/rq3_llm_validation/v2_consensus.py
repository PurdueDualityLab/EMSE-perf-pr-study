"""Aggregate complete v2 occurrence votes and compare the frozen v1 PR labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from analysis.rq3_llm_validation import experiment as v1
from analysis.rq3_llm_validation import v2
from analysis.rq3_llm_validation.run_v2 import checkpoint_path

PROVIDERS = tuple(v2.MODELS)


def consensus_for(request: dict, results: dict) -> tuple[dict, list[dict]]:
    validated = {}
    for provider in PROVIDERS:
        result = results[provider]
        if result["classification_status"] != "classified":
            raise ValueError("Consensus requires three successful provider results.")
        for name in ("custom_id", "prompt_sha256", "study_contract_sha256"):
            if result[name] != request[name]:
                raise ValueError("Provider request identities or hashes do not match.")
        model = v2.RegexResult if request["task"] == "regex_audit" else v2.TradeoffResult
        validated[provider] = v2.validate_result({k: result["label"][k] for k in model.model_fields}, request)
    base = {name: request[name] for name in ("repo_id", "number", "task", "sample_arm", "html_url")}
    occurrences = []
    for index, occurrence_id in enumerate(request["occurrence_ids"]):
        votes = {p: validated[p]["occurrences"][index] for p in PROVIDERS}
        count = sum(v["valid"] for v in votes.values())
        occurrences.append({**base, "occurrence_id": occurrence_id,
                            "dimension": request["occurrence_dimensions"][occurrence_id],
                            "consensus_valid": count >= 2, "valid_votes": count,
                            "agreement": "unanimous" if count in {0, 3} else "majority",
                            **{f"{p}_valid": votes[p]["valid"] for p in PROVIDERS},
                            **{f"{p}_reason": votes[p]["reason"] for p in PROVIDERS}})
    labels = {p: validated[p]["label"] for p in PROVIDERS}
    winner, count = Counter(labels.values()).most_common(1)[0]
    pr_majority = winner if count >= 2 else "indeterminate"
    if request["task"] == "regex_audit":
        label = "true_positive" if any(o["consensus_valid"] for o in occurrences) else "false_positive"
    else:
        label = pr_majority
    row = {**base, "consensus_label": label, "pr_label_majority": pr_majority,
           "consensus_status": "unresolved" if count < 2 else "unanimous" if count == 3 else "majority",
           "occurrence_aggregation_differs": label != pr_majority,
           "valid_occurrences": sum(o["consensus_valid"] for o in occurrences),
           "occurrences": len(occurrences), "activation_count": request["activation_count"],
           **{f"{p}_label": labels[p] for p in PROVIDERS},
           **{f"{p}_rationale": validated[p]["rationale"] for p in PROVIDERS}}
    return row, occurrences


def ready_tasks(output: Path) -> tuple[str, ...]:
    requests = v2.load_requests(output)
    ready = []
    for task in v1.TASKS:
        paths = [checkpoint_path(output, p, r) for r in requests if r["task"] == task for p in PROVIDERS]
        if paths and all(path.exists() and json.loads(path.read_text()).get("classification_status") == "classified"
                         for path in paths):
            ready.append(task)
    return tuple(ready)


def build(output: Path = v2.DEFAULT_OUTPUT, tasks: tuple[str, ...] = v1.TASKS) -> dict:
    if not tasks or not set(tasks) <= set(v1.TASKS):
        raise ValueError("Select one or both RQ3 tasks.")
    requests = [r for r in v2.load_requests(output) if r["task"] in tasks]
    rows, occurrences = [], []
    for request in requests:
        results = {p: json.loads(checkpoint_path(output, p, request).read_text()) for p in PROVIDERS}
        row, detailed = consensus_for(request, results)
        rows.append(row)
        occurrences.extend(detailed)
    frame, details = pd.DataFrame(rows), pd.DataFrame(occurrences)
    directory = output / "consensus"
    directory.mkdir(exist_ok=True)
    summary = {"version": v2.VERSION, "status": "completed" if set(tasks) == set(v1.TASKS) else "partial", "tasks": {}}
    for task in tasks:
        subset = frame[frame.task.eq(task)].copy()
        old_path = ROOT_V1 / "consensus" / task / f"{task}_consensus.parquet"
        old = pd.read_parquet(old_path)[[*v1.KEYS, "consensus_label"]].rename(columns={"consensus_label": "v1_label"})
        subset = subset.merge(old, on=v1.KEYS, validate="one_to_one")
        expected = sum(r["task"] == task for r in requests)
        if len(subset) != expected:
            raise ValueError("v1/v2 comparison changed the frozen population.")
        subset["changed_from_v1"] = subset.consensus_label.ne(subset.v1_label)
        subset.to_parquet(directory / f"{task}_consensus.parquet", index=False)
        subset.to_csv(directory / f"{task}_comparison.csv", index=False)
        details[details.task.eq(task)].to_csv(directory / f"{task}_occurrence_votes.csv", index=False)
        summary["tasks"][task] = {
            "prs": len(subset), "classes": subset.consensus_label.value_counts().to_dict(),
            "changed_from_v1": int(subset.changed_from_v1.sum()),
            "occurrence_vs_pr_majority_differences": int(subset.occurrence_aggregation_differs.sum()),
            "by_arm": {arm: group.consensus_label.value_counts().to_dict()
                       for arm, group in subset.groupby("sample_arm")},
        }
    audit = pd.read_csv(ROOT_V1 / "manual_audit/pr_level_audit.csv")
    reviewed = frame[frame.task.eq("regex_audit")].merge(
        audit[[*v1.KEYS, "independent_verdict", "confidence"]], on=v1.KEYS, validate="one_to_one")
    reviewed["audit_label"] = reviewed.independent_verdict.map({
        "confirmed_false_positive": "false_positive", "overturned_to_true_positive": "true_positive"})
    reviewed["agrees_with_agent_audit"] = reviewed.consensus_label.eq(reviewed.audit_label)
    reviewed.to_csv(directory / "previous_agent_audit_comparison.csv", index=False)
    summary["previous_agent_audit"] = {"prs": len(reviewed),
        "agree": int(reviewed.agrees_with_agent_audit.sum()), "note": "Agent audit, not human ground truth"}
    v1.atomic_write_json(directory / "summary.json", summary)
    lines = ["# RQ3 v2 results", "", "Frozen paired sample: 88 regex-audit PRs (approximately 10%) and all 29 trade-off candidates.", "",
             f"Finalized tasks: {', '.join(tasks)}. Overall status: {summary['status']}.", "",
             "All detected occurrences were exposed through complete matched source records.", ""]
    for task, counts in summary["tasks"].items():
        lines.extend([f"## {task}", "", f"- PRs: {counts['prs']}", f"- Classes: {json.dumps(counts['classes'])}",
                      f"- Changed from v1: {counts['changed_from_v1']}",
                      f"- Occurrence/PR majority differences: {counts['occurrence_vs_pr_majority_differences']}", ""])
    lines.extend(["## Interpretation", "", "Regex consensus uses majority per occurrence, then any valid occurrence at PR level. "
                  "The PR-label majority is also retained for sensitivity analysis. Trade-off labels require two votes; "
                  "no majority yields indeterminate. Unsupported and indeterminate cases remain in the 29-case denominator.", "",
                  "Previous audit judgments are agent assessments, not human ground truth. Review changed labels and disagreements "
                  "before using these results as final study findings.", ""])
    (directory / "summary.md").write_text("\n".join(lines))
    return summary


ROOT_V1 = v1.ROOT / "analysis/rq3_llm_validation/generated"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=v2.DEFAULT_OUTPUT)
    parser.add_argument("--task", choices=v1.TASKS, action="append")
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir, tuple(args.task) if args.task else v1.TASKS), indent=2))
