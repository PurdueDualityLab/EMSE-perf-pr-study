from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_rebalanced_dataset
from build_rebalanced_dataset import (  # type: ignore
    chunked,
    github_batch_size,
    initial_resume_state,
    mine_one_repo_for_batch,
    save_repo_checkpoint,
)
from schema import TimeWindow, ensure_output_dirs


def test_github_batch_size_defaults_to_20():
    assert github_batch_size({"github": {}}) == 20


def test_github_batch_size_allows_sequential_mode():
    assert github_batch_size({"github": {"batch_size": 1}}) == 1


def test_chunked_splits_repos_into_expected_batches():
    batches = chunked(list(range(45)), 20)
    assert [len(batch) for batch in batches] == [20, 20, 5]


def test_save_repo_checkpoint_writes_after_each_repo(tmp_path):
    output_dirs = ensure_output_dirs(tmp_path / "outputs")
    raw_path = output_dirs["raw"] / "github_human_prs.parquet"
    state_path = output_dirs["state"] / "state.json"
    state = initial_resume_state({"signature": "test"}, 2)
    report = {"repo_reports": [], "pr_failures": []}
    github_human = pd.DataFrame(
        [{"repo_full_name": "owner/one", "number": 1, "filenames": ["src/a.py"]}]
    )

    completed = {"owner/one"}
    state = save_repo_checkpoint(
        github_human,
        raw_path,
        state_path,
        state,
        completed,
        report,
        "owner/one",
    )
    assert raw_path.exists()
    assert state_path.exists()
    assert state["completed_repo_count"] == 1

    github_human = pd.concat(
        [
            github_human,
            pd.DataFrame(
                [{"repo_full_name": "owner/two", "number": 2, "filenames": ["src/b.py"]}]
            ),
        ],
        ignore_index=True,
    )
    completed.add("owner/two")
    state = save_repo_checkpoint(
        github_human,
        raw_path,
        state_path,
        state,
        completed,
        report,
        "owner/two",
    )

    assert state["completed_repo_count"] == 2
    assert len(pd.read_parquet(raw_path)) == 2


def test_parallel_batch_worker_keeps_success_when_another_repo_fails(monkeypatch):
    def fake_mine(repos, start, end, token_file, per_repo_search_limit, progress, report):
        repo = repos[0]
        name = repo["repo_full_name"]
        if name == "owner/fail":
            report["repo_reports"].append(
                {
                    "repo_full_name": name,
                    "status": "search_failed",
                    "candidate_count": 0,
                    "kept_count": 0,
                    "pr_fetch_failed_count": 0,
                    "error": "search failed",
                }
            )
            return []
        report["repo_reports"].append(
            {
                "repo_full_name": name,
                "status": "completed",
                "candidate_count": 1,
                "kept_count": 1,
                "pr_fetch_failed_count": 0,
                "error": None,
            }
        )
        return [{"repo_full_name": name, "number": 1, "filenames": ["src/a.py"]}]

    monkeypatch.setattr(build_rebalanced_dataset, "mine_human_pull_requests", fake_mine)
    window = TimeWindow(
        start=pd.Timestamp("2025-01-01T00:00:00Z"),
        end=pd.Timestamp("2025-01-02T00:00:00Z"),
    )
    repos = [
        {"repo_full_name": "owner/fail"},
        {"repo_full_name": "owner/pass"},
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                mine_one_repo_for_batch,
                index,
                2,
                repo,
                window,
                None,
                None,
                None,
            )
            for index, repo in enumerate(repos, start=1)
        ]
        results = [future.result() for future in as_completed(futures)]

    statuses = {result["repo_name"]: result["repo_report"]["status"] for result in results}
    assert statuses == {"owner/fail": "search_failed", "owner/pass": "completed"}
