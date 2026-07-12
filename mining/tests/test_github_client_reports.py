from datetime import date
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import github_client
from github_client import (
    GitHubClient,
    GitHubClientError,
    GitHubTokenPool,
    github_tokens,
    github_tokens_from_file,
    mine_human_pull_requests,
    monthly_date_ranges,
)


class FakeClient:
    def search_pull_requests(self, repo_full_name, start, end, per_repo_limit=None):
        return [{"number": 1}, {"number": 2}]

    def fetch_pull_request_record(
        self,
        repo_full_name,
        number,
        base_record=None,
        include_commits=False,
    ):
        if number == 2:
            raise GitHubClientError("fetch failed")
        record = dict(base_record or {})
        record.update(
            {
                "repo_full_name": repo_full_name,
                "number": number,
                "html_url": f"https://github.com/{repo_full_name}/pull/{number}",
                "filenames": ["src/a.py"],
                "deleted_repo": False,
            }
        )
        return record


class FakeResponse:
    def __init__(self, status_code, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text)

    def json(self):
        return self._payload


class RotatingSession:
    def __init__(self):
        self.authorizations = []
        self.responses = [
            FakeResponse(
                403,
                text="API rate limit exceeded",
                headers={
                    "x-ratelimit-remaining": "0",
                    "x-ratelimit-reset": str(int(time.time()) + 3600),
                },
            ),
            FakeResponse(200, payload={"ok": True}, headers={"x-ratelimit-remaining": "4999"}),
        ]

    def get(self, url, headers=None, params=None, timeout=None):
        self.authorizations.append(headers["Authorization"])
        return self.responses.pop(0)


class SharedRecordingSession:
    def __init__(self, authorizations, lock):
        self.authorizations = authorizations
        self.lock = lock

    def get(self, url, headers=None, params=None, timeout=None):
        with self.lock:
            self.authorizations.append(headers["Authorization"])
        return FakeResponse(200, payload={"ok": True}, headers={"x-ratelimit-remaining": "4999"})


class TokenOneLimitedSession:
    def __init__(self):
        self.attempted_authorizations = []
        self.success_authorizations = []
        self.lock = threading.Lock()

    def get(self, url, headers=None, params=None, timeout=None):
        authorization = headers["Authorization"]
        with self.lock:
            self.attempted_authorizations.append(authorization)
        if authorization == "Bearer token-one":
            return FakeResponse(
                403,
                text="API rate limit exceeded",
                headers={
                    "x-ratelimit-remaining": "0",
                    "x-ratelimit-reset": str(int(time.time()) + 3600),
                },
            )
        with self.lock:
            self.success_authorizations.append(authorization)
        return FakeResponse(200, payload={"ok": True}, headers={"x-ratelimit-remaining": "4999"})


class AllTokensLimitedThenAvailableSession:
    def __init__(self):
        self.calls = 0

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls += 1
        if self.calls <= 2:
            return FakeResponse(
                403,
                text="API rate limit exceeded",
                headers={
                    "x-ratelimit-remaining": "0",
                    "x-ratelimit-reset": str(int(time.time())),
                },
            )
        return FakeResponse(200, payload={"ok": True}, headers={"x-ratelimit-remaining": "4999"})


class SearchRecordingClient(GitHubClient):
    def __init__(self):
        super().__init__(tokens=["token-one"], sleep_seconds=0)
        self.ranges = []

    def search_pull_requests_for_range(self, repo_full_name, start, end):
        self.ranges.append((repo_full_name, start, end))
        return [
            {
                "number": len(self.ranges),
                "html_url": f"https://github.com/{repo_full_name}/pull/{len(self.ranges)}",
            }
        ]


def test_github_tokens_from_file_reads_one_token_per_line(tmp_path):
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("token-one\n\ntoken-two\n# comment\ntoken-three\n", encoding="utf-8")

    assert github_tokens_from_file(token_file) == ["token-one", "token-two", "token-three"]


def test_github_tokens_from_file_dedupes_while_preserving_order(tmp_path):
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("token-one\ntoken-two\ntoken-one\n", encoding="utf-8")

    assert github_tokens_from_file(token_file) == ["token-one", "token-two"]


def test_github_tokens_from_file_requires_existing_file(tmp_path):
    missing = tmp_path / "missing.txt"

    with pytest.raises(GitHubClientError, match="GitHub token file not found"):
        github_tokens_from_file(missing)


def test_github_tokens_requires_token_file(tmp_path):
    with pytest.raises(GitHubClientError, match="github.token_file is required"):
        github_tokens(None)


def test_github_tokens_uses_file(tmp_path):
    token_file = tmp_path / "tokens.txt"
    token_file.write_text("file-token\n", encoding="utf-8")

    assert github_tokens(str(token_file)) == ["file-token"]


def test_monthly_date_ranges_split_partial_months():
    assert monthly_date_ranges(date(2024, 12, 24), date(2025, 2, 2)) == [
        (date(2024, 12, 24), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 1, 31)),
        (date(2025, 2, 1), date(2025, 2, 2)),
    ]


def test_search_pull_requests_queries_monthly_ranges():
    client = SearchRecordingClient()

    results = client.search_pull_requests("owner/repo", date(2024, 12, 24), date(2025, 2, 2))

    assert [item[1:] for item in client.ranges] == [
        (date(2024, 12, 24), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 1, 31)),
        (date(2025, 2, 1), date(2025, 2, 2)),
    ]
    assert [item["number"] for item in results] == [1, 2, 3]


def test_github_client_rotates_to_next_numbered_token_on_rate_limit():
    session = RotatingSession()
    client = GitHubClient(tokens=["token-one", "token-two"], session=session, sleep_seconds=0)

    assert client.request("/rate-limited-once") == {"ok": True}
    assert session.authorizations == ["Bearer token-one", "Bearer token-two"]
    assert client.token == "token-two"


def test_github_token_pool_distributes_concurrent_requests():
    authorizations = []
    lock = threading.Lock()
    pool = GitHubTokenPool(tokens=["token-one", "token-two", "token-three"])

    def request_once():
        client = GitHubClient(
            token_pool=pool,
            session=SharedRecordingSession(authorizations, lock),
            sleep_seconds=0,
        )
        return client.request("/ok")

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(lambda _: request_once(), range(6)))

    assert results == [{"ok": True}] * 6
    assert set(authorizations) == {
        "Bearer token-one",
        "Bearer token-two",
        "Bearer token-three",
    }


def test_concurrent_requests_retry_after_shared_token_hits_rate_limit():
    pool = GitHubTokenPool(tokens=["token-one", "token-two", "token-three"])
    session = TokenOneLimitedSession()

    def request_once():
        client = GitHubClient(token_pool=pool, session=session, sleep_seconds=0)
        return client.request("/ok")

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(lambda _: request_once(), range(40)))

    assert results == [{"ok": True}] * 40
    assert "Bearer token-one" in session.attempted_authorizations
    assert "Bearer token-one" not in session.success_authorizations
    assert set(session.success_authorizations) <= {"Bearer token-two", "Bearer token-three"}
    assert 0 in pool.rate_limited_until


def test_github_client_waits_until_limited_tokens_become_available():
    session = AllTokensLimitedThenAvailableSession()
    client = GitHubClient(tokens=["token-one"], session=session, sleep_seconds=0)

    assert client.request("/eventually-ok") == {"ok": True}
    assert session.calls == 3


def test_github_token_pool_waits_when_all_tokens_are_limited():
    pool = GitHubTokenPool(tokens=["token-one"])
    pool.mark_rate_limited(
        0,
        FakeResponse(
            403,
            text="API rate limit exceeded",
            headers={
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": str(int(time.time())),
            },
        ),
    )

    assert pool.acquire_waiting("/ok") == (0, "token-one")


def test_mine_human_pull_requests_records_failures(monkeypatch):
    fake_client = FakeClient()
    monkeypatch.setattr(
        github_client.GitHubClient,
        "from_token_file",
        classmethod(lambda cls, token_file=None: fake_client),
    )

    report: dict = {"repo_reports": [], "pr_failures": []}
    mined = mine_human_pull_requests(
        [{"repo_id": 7, "repo_full_name": "example/repo"}],
        date(2025, 1, 1),
        date(2025, 1, 2),
        report=report,
    )

    assert len(mined) == 1
    assert report["completed_repos"] == 1
    assert report["pr_fetch_failed"] == 1
    assert report["kept_rows"] == 1
    assert len(report["repo_reports"]) == 1
    repo_report = report["repo_reports"][0]
    assert repo_report["status"] == "completed"
    assert repo_report["candidate_count"] == 2
    assert repo_report["kept_count"] == 1
    assert repo_report["pr_fetch_failed_count"] == 1
    assert len(report["pr_failures"]) == 1
    assert report["pr_failures"][0]["number"] == 2
