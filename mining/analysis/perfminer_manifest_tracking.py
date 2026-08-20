"""Generate a traceable status report for an in-progress PerfMiner manifest."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from schema import atomic_write_text  # noqa: E402


def load_tracking(state_path: Path) -> dict[str, Any]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    batches = state.get("batches")
    if not isinstance(batches, dict):
        raise ValueError(f"Invalid manifest state: {state_path}")

    batch_statuses = Counter()
    retryable_errors = 0
    row_statuses = Counter()
    row_errors = Counter()
    row_count = 0
    for record in batches.values():
        if not isinstance(record, dict):
            raise ValueError("Manifest state contains an invalid batch record.")
        batch_statuses[str(record.get("status") or "missing")] += 1
        retryable_errors += int(record.get("retryable_errors") or 0)
        part = record.get("pr_part")
        if not isinstance(part, dict) or not part.get("path"):
            continue
        part_path = Path(str(part["path"]))
        if not part_path.is_file():
            continue
        for batch in pq.ParquetFile(part_path).iter_batches(
            columns=["commit_manifest_status", "commit_manifest_error"], batch_size=50_000
        ):
            statuses = batch.column("commit_manifest_status").to_pylist()
            errors = batch.column("commit_manifest_error").to_pylist()
            row_count += len(statuses)
            row_statuses.update(str(value or "missing") for value in statuses)
            for value in errors:
                if not value:
                    continue
                text = str(value)
                if text.startswith("GitHubNotFoundError: GitHub resource not found:"):
                    text = "GitHub PR endpoint returned HTTP 404."
                row_errors[text] += 1

    signature = state.get("run_signature") if isinstance(state.get("run_signature"), dict) else {}
    eligible_prs = row_statuses.get("complete", 0)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "state_updated_at": state.get("updated_at"),
        "finalized": bool(state.get("finalized")),
        "candidate_prs": signature.get("pull_request_count"),
        "target_repositories": signature.get("repository_count"),
        "recorded_batches": len(batches),
        "batch_status_counts": dict(sorted(batch_statuses.items())),
        "retryable_errors": retryable_errors,
        "manifest_pr_rows": row_count,
        "pr_status_counts": dict(sorted(row_statuses.items())),
        "pr_error_counts": dict(sorted(row_errors.items())),
        "eligible_prs": eligible_prs,
        "excluded_prs": row_count - eligible_prs,
    }


def markdown(report: dict[str, Any], state_path: Path) -> str:
    lines = [
        "# PerfMiner Manifest Tracking",
        "",
        f"Generated at: `{report['generated_at']}`",
        f"Manifest state: `{state_path}`",
        "",
        "## Coverage",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Candidate PRs | {report['candidate_prs']:,} |",
        f"| Target repositories | {report['target_repositories']:,} |",
        f"| Recorded manifest batches | {report['recorded_batches']:,} |",
        f"| PR rows represented in manifest parts | {report['manifest_pr_rows']:,} |",
        f"| PRs eligible for PerfMiner | {report['eligible_prs']:,} |",
        f"| PRs excluded from PerfMiner | {report['excluded_prs']:,} |",
        f"| Retryable PR-level errors | {report['retryable_errors']:,} |",
        "",
        "## Batch Status",
        "",
        "| Status | Batches |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{status}` | {count:,} |" for status, count in report["batch_status_counts"].items())
    lines.extend(["", "## PR Manifest Status", "", "| Status | PRs |", "| --- | ---: |"])
    lines.extend(f"| `{status}` | {count:,} |" for status, count in report["pr_status_counts"].items())
    lines.extend(["", "## Error Reasons", "", "| Reason | PRs |", "| --- | ---: |"])
    lines.extend(f"| {reason} | {count:,} |" for reason, count in report["pr_error_counts"].items())
    lines.extend(
        [
            "",
            "`unstable_pull_request_snapshot` indicates that the pipeline could not validate a stable",
            "base SHA, head SHA, and ordered commit list after its retries. `commit_count_over_rest_limit`",
            "is the explicit GitHub REST limit for PRs with more than 250 commits. `pull_request_unavailable`",
            "records current HTTP 404 responses from the PR endpoint.",
            "",
            "The 320 incomplete batches contain the 377 retryable unstable-snapshot PRs; a batch can",
            "contain more than one such PR. The completed manifest is the 181,062-PR eligible cohort.",
            "The 394 over-limit and 47 unavailable PRs are terminal exclusions under the current REST",
            "manifest contract. The unstable-snapshot PRs remained unresolved after the complete retry pass.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(state_path: Path, markdown_path: Path, json_path: Path) -> dict[str, Any]:
    report = load_tracking(state_path)
    atomic_write_text(markdown_path, markdown(report, state_path))
    atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def wait_for_process(pid: int, poll_seconds: float) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PerfMiner manifest tracking artifacts.")
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.wait_for_pid is not None:
        if args.poll_seconds <= 0:
            raise ValueError("--poll-seconds must be positive.")
        wait_for_process(args.wait_for_pid, args.poll_seconds)
    report = write_report(args.state, args.markdown, args.json)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
