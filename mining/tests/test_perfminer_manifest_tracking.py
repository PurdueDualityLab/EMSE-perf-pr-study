import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from perfminer_manifest_tracking import load_tracking, write_report  # noqa: E402


def test_tracking_counts_statuses_and_errors(tmp_path):
    part = tmp_path / "part.parquet"
    pq.write_table(
        pa.table(
            {
                "commit_manifest_status": ["complete", "unstable_pull_request_snapshot", "commit_count_over_rest_limit"],
                "commit_manifest_error": ["", "Missing base SHA", "PR exceeds GitHub's 250-commit listing limit."],
            }
        ),
        part,
    )
    state = {
        "updated_at": "2026-08-05T00:00:00+00:00",
        "finalized": False,
        "run_signature": {"pull_request_count": 3, "repository_count": 2},
        "batches": {
            "one": {"status": "complete", "retryable_errors": 0, "pr_part": {"path": str(part)}},
            "two": {"status": "incomplete", "retryable_errors": 2},
        },
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    report = load_tracking(state_path)
    write_report(state_path, tmp_path / "tracking.md", tmp_path / "tracking.json")

    assert report["batch_status_counts"] == {"complete": 1, "incomplete": 1}
    assert report["pr_status_counts"]["complete"] == 1
    assert report["pr_status_counts"]["unstable_pull_request_snapshot"] == 1
    assert report["eligible_prs"] == 1
    assert report["excluded_prs"] == 2
    assert report["retryable_errors"] == 2
    markdown = (tmp_path / "tracking.md").read_text(encoding="utf-8")
    assert "Error Reasons" in markdown
    assert "PRs eligible for PerfMiner" in markdown
