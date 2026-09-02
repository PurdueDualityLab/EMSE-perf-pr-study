from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from collect_sample_evidence import collect_pull_request  # noqa: E402
from github_client import GitHubNotFoundError  # noqa: E402


SAMPLE = {
    "repo_id": 1,
    "number": 2,
    "id": 20,
    "repo_full_name": "org/repo",
    "sample_arm": "agentic",
    "selection_hash": "abc",
    "html_url": "https://github.com/org/repo/pull/2",
    "title": "Improve speed",
    "body": "Faster",
}


class FakeClient:
    def __init__(self, *, missing=False, repo_id=1):
        self.missing = missing
        self.repo_id = repo_id

    def request(self, path, params=None):
        if self.missing:
            raise GitHubNotFoundError(path)
        return {
            "id": 20,
            "node_id": "PR_x",
            "html_url": SAMPLE["html_url"],
            "state": "closed",
            "merged": True,
            "title": SAMPLE["title"],
            "body": SAMPLE["body"],
            "base": {"repo": {"id": self.repo_id}, "sha": "base"},
            "head": {"sha": "head"},
            "user": {"login": "author", "type": "User"},
            "changed_files": 1,
        }

    def request_paginated_list(self, path, params=None, item_key=None):
        if path.endswith("/files"):
            return [{"filename": "a.py", "patch": "+faster", "additions": 1, "deletions": 0}]
        if path.endswith("/commits"):
            return [{"sha": "head", "commit": {"message": "perf"}}]
        if path.endswith("/check-runs"):
            return [{"id": 4, "name": "benchmark", "app": {}, "output": {}}]
        if path.endswith("/actions/runs"):
            return [{"id": 3, "head_sha": "head", "name": "ci"}]
        return []


def test_collect_pull_request_normalizes_code_and_ci_evidence():
    result = collect_pull_request(FakeClient(), SAMPLE, "2025-01-02T00:00:00+00:00")

    assert result["pull_requests"][0]["identity_verified"] is True
    assert result["pull_request_files"][0]["filename"] == "a.py"
    assert result["commits"][0]["sha"] == "head"
    assert result["workflow_runs"][0]["name"] == "ci"
    assert result["check_runs"][0]["name"] == "benchmark"
    assert result["collection_status"][0]["status"] == "complete"


def test_collect_pull_request_records_not_found_as_observability_failure():
    result = collect_pull_request(FakeClient(missing=True), SAMPLE, "2025-01-02T00:00:00+00:00")

    assert result["pull_requests"][0]["fetch_status"] == "not_found"
    assert result["collection_status"][0]["status"] == "not_found"
    assert result["collection_status"][0]["pull_request_files_status"] == "not_requested"
    assert result["pull_request_files"] == []


def test_collect_pull_request_rejects_repository_identity_mismatch():
    result = collect_pull_request(FakeClient(repo_id=99), SAMPLE, "2025-01-02T00:00:00+00:00")

    assert result["pull_requests"][0]["fetch_status"] == "identity_mismatch"
    assert result["collection_status"][0]["status"] == "identity_mismatch"
    assert result["pull_request_files"] == []
