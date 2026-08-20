"""Materialize the frozen PerfMiner manifest needed to evaluate against original AIDev labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from schema import atomic_write_text, write_parquet


IDENTITY = ("repo_id", "number")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AIDev heuristic candidates from a frozen manifest.")
    parser.add_argument("--aidev-human-prs", required=True)
    parser.add_argument("--aidev-human-task-types", required=True)
    parser.add_argument("--experiment-decisions", required=True, type=Path)
    parser.add_argument("--manifest-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    aidev_prs = pd.read_parquet(args.aidev_human_prs, columns=["id"])
    aidev_types = pd.read_parquet(args.aidev_human_task_types, columns=["id", "type"])
    labels = aidev_prs.merge(aidev_types, on="id", how="inner", validate="one_to_one")
    decisions = pd.read_parquet(args.experiment_decisions)
    decisions = decisions.loc[decisions["id"].notna()].copy()
    candidates = labels.merge(decisions, on="id", how="inner", validate="one_to_one")
    candidates = candidates.loc[candidates["heuristic_match"].fillna(False)].copy()
    if candidates.duplicated(list(IDENTITY)).any():
        raise ValueError("AIDev candidates contain duplicate repository/PR identities.")
    target_keys = set(map(tuple, candidates[list(IDENTITY)].itertuples(index=False, name=None)))
    target_repos = {key[0] for key in target_keys}

    state = json.loads(args.manifest_state.read_text(encoding="utf-8"))
    records = [
        record
        for record in state.get("batches", {}).values()
        if isinstance(record, dict) and record.get("status") == "complete" and record.get("repo_id") in target_repos
    ]
    pr_frames: list[pd.DataFrame] = []
    selected_records: list[dict[str, object]] = []
    for record in records:
        part = record.get("pr_part", {})
        path = Path(str(part.get("path") or ""))
        if not path.is_file():
            raise FileNotFoundError(f"Manifest PR part is missing: {path}")
        frame = pd.read_parquet(path)
        frame = frame.loc[frame[["repo_id", "pr_number"]].apply(tuple, axis=1).isin(target_keys)]
        if not frame.empty:
            pr_frames.append(frame)
            selected_records.append(record)
    pull_requests = pd.concat(pr_frames, ignore_index=True) if pr_frames else pd.DataFrame()
    if pull_requests.duplicated(["repo_id", "pr_number"]).any():
        raise ValueError("Selected manifest contains duplicate PR identities.")
    found_keys = set(map(tuple, pull_requests[["repo_id", "pr_number"]].itertuples(index=False, name=None)))
    commit_frames: list[pd.DataFrame] = []
    for record in selected_records:
        part = record.get("commit_part", {})
        path = Path(str(part.get("path") or ""))
        if not path.is_file():
            raise FileNotFoundError(f"Manifest commit part is missing: {path}")
        frame = pd.read_parquet(path)
        frame = frame.loc[frame[["repo_id", "pr_number"]].apply(tuple, axis=1).isin(found_keys)]
        if not frame.empty:
            commit_frames.append(frame)
    commits = pd.concat(commit_frames, ignore_index=True) if commit_frames else pd.DataFrame()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(candidates, args.output_dir / "aidev_heuristic_candidates.parquet")
    write_parquet(pull_requests, args.output_dir / "pull_requests.parquet")
    write_parquet(commits, args.output_dir / "commits.parquet")
    summary = {
        "aidev_heuristic_candidates": len(candidates),
        "manifest_complete_candidates": len(pull_requests),
        "manifest_unavailable_candidates": len(candidates) - len(pull_requests),
        "commit_rows": len(commits),
    }
    atomic_write_text(args.output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
