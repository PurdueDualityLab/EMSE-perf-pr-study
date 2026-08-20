from datetime import date
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
import threading

import pytest
import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import github_client
from github_client import (
    GitHubClient,
    GitHubAuthenticationError,
    GitHubClientError,
    GitHubNotFoundError,
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


class SubdividingSearchClient(GitHubClient):
    def __init__(self):
        super().__init__(tokens=["token-one"], sleep_seconds=0)
        self.queries = []
        self.full_queries = []

    def request(self, path, params=None):
        self.full_queries.append(params["q"])
        date_range = params["q"].split("created:", 1)[1]
        self.queries.append(date_range)
        if date_range == "2025-01-01..2025-01-04":
            return {"total_count": 1200, "incomplete_results": False, "items": []}
        if date_range == "2025-01-01..2025-01-02":
            return {"total_count": 1100, "incomplete_results": False, "items": []}
        number = {
            "2025-01-01..2025-01-01": 1,
            "2025-01-02..2025-01-02": 2,
            "2025-01-03..2025-01-04": 3,
        }[date_range]
        return {
            "total_count": 1,
            "incomplete_results": False,
            "items": [{"number": number}],
        }


class SequenceSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.authorizations = []
        self.calls = 0

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls += 1
        self.authorizations.append(headers["Authorization"])
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


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


def test_search_pull_requests_recursively_subdivides_ranges_over_cap():
    client = SubdividingSearchClient()

    results = client.search_pull_requests_for_range(
        "owner/repo", date(2025, 1, 1), date(2025, 1, 4)
    )

    assert [item["number"] for item in results] == [1, 2, 3]
    assert client.queries == [
        "2025-01-01..2025-01-04",
        "2025-01-01..2025-01-02",
        "2025-01-01..2025-01-01",
        "2025-01-02..2025-01-02",
        "2025-01-03..2025-01-04",
    ]


def test_search_qualifiers_are_preserved_when_ranges_are_subdivided():
    client = SubdividingSearchClient()

    client.search_pull_requests_for_range(
        "owner/repo",
        date(2025, 1, 1),
        date(2025, 1, 4),
        search_qualifiers="head:codex/",
    )

    assert len(client.full_queries) == 5
    assert all("repo:owner/repo is:pr head:codex/ created:" in query for query in client.full_queries)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"incomplete_results": False, "items": []},
        {"total_count": "invalid", "incomplete_results": False, "items": []},
        {"total_count": True, "incomplete_results": False, "items": []},
        {"total_count": -1, "incomplete_results": False, "items": []},
        {"total_count": 0, "incomplete_results": "false", "items": []},
        {"total_count": 0, "incomplete_results": False, "items": {}},
        {"total_count": 0, "incomplete_results": False, "items": [{"number": 1}]},
    ],
)
def test_strict_search_rejects_malformed_payloads(monkeypatch, payload):
    client = GitHubClient(tokens=["token-one"], sleep_seconds=0)
    monkeypatch.setattr(client, "request", lambda path, params=None: payload)

    with pytest.raises(GitHubClientError):
        client.search_pull_requests_for_range(
            "owner/repo",
            date(2025, 1, 1),
            date(2025, 1, 2),
            search_qualifiers="head:codex/",
            strict=True,
        )


def test_strict_search_rejects_duplicate_results_across_pages(monkeypatch):
    client = GitHubClient(tokens=["token-one"], sleep_seconds=0)
    first_items = [{"number": number} for number in range(1, 101)]

    def fake_request(path, params=None):
        items = first_items if params["page"] == 1 else [{"number": 100}]
        return {"total_count": 101, "incomplete_results": False, "items": items}

    monkeypatch.setattr(client, "request", fake_request)

    with pytest.raises(GitHubClientError, match="duplicate PR number"):
        client.search_pull_requests_for_range(
            "owner/repo",
            date(2025, 1, 1),
            date(2025, 1, 2),
            search_qualifiers="head:codex/",
            strict=True,
        )


def test_strict_search_rejects_later_page_overdelivery(monkeypatch):
    client = GitHubClient(tokens=["token-one"], sleep_seconds=0)
    first_items = [{"number": number} for number in range(1, 101)]

    def fake_request(path, params=None):
        items = first_items if params["page"] == 1 else [{"number": 101}, {"number": 102}]
        return {"total_count": 101, "incomplete_results": False, "items": items}

    monkeypatch.setattr(client, "request", fake_request)

    with pytest.raises(GitHubClientError, match="more items than total_count"):
        client.search_pull_requests_for_range(
            "owner/repo",
            date(2025, 1, 1),
            date(2025, 1, 2),
            search_qualifiers="head:codex/",
            strict=True,
        )


def test_search_pull_requests_subdivides_incomplete_ranges(monkeypatch):
    client = GitHubClient(tokens=["token-one"], sleep_seconds=0)
    client.max_retries = 0

    def fake_request(path, params=None):
        date_range = params["q"].split("created:", 1)[1]
        if date_range == "2025-01-01..2025-01-02":
            return {"total_count": 2, "incomplete_results": True, "items": []}
        number = 1 if date_range.startswith("2025-01-01") else 2
        return {
            "total_count": 1,
            "incomplete_results": False,
            "items": [{"number": number}],
        }

    monkeypatch.setattr(client, "request", fake_request)

    results = client.search_pull_requests_for_range(
        "owner/repo", date(2025, 1, 1), date(2025, 1, 2)
    )

    assert [item["number"] for item in results] == [1, 2]


def test_search_pull_requests_fails_when_one_day_exceeds_cap(monkeypatch):
    client = GitHubClient(tokens=["token-one"], sleep_seconds=0)
    monkeypatch.setattr(
        client,
        "request",
        lambda path, params=None: {
            "total_count": 1001,
            "incomplete_results": False,
            "items": [],
        },
    )

    with pytest.raises(GitHubClientError, match="one-day range cannot be subdivided"):
        client.search_pull_requests_for_range(
            "owner/repo", date(2025, 1, 1), date(2025, 1, 1)
        )


def test_one_day_search_can_return_a_bounded_prefix_over_cap(monkeypatch):
    client = GitHubClient(tokens=["token-one"], sleep_seconds=0)
    items = [{"number": number} for number in range(1, 101)]
    monkeypatch.setattr(
        client,
        "request",
        lambda path, params=None: {
            "total_count": 1200,
            "incomplete_results": False,
            "items": items,
        },
    )

    results = client.search_pull_requests(
        "owner/repo",
        date(2025, 1, 1),
        date(2025, 1, 1),
        per_repo_limit=10,
    )

    assert [item["number"] for item in results] == list(range(1, 11))


def test_search_retries_transient_incomplete_results(monkeypatch):
    client = GitHubClient(
        tokens=["token-one"],
        sleep_seconds=0,
        retry_backoff_seconds=0,
    )
    calls = 0

    def fake_request(path, params=None):
        nonlocal calls
        calls += 1
        return {
            "total_count": 1,
            "incomplete_results": calls == 1,
            "items": [{"number": 1}],
        }

    monkeypatch.setattr(client, "request", fake_request)

    results = client.search_pull_requests_for_range(
        "owner/repo", date(2025, 1, 1), date(2025, 1, 1)
    )

    assert results == [{"number": 1}]
    assert calls == 2


def test_github_client_rotates_to_next_numbered_token_on_rate_limit():
    session = RotatingSession()
    client = GitHubClient(tokens=["token-one", "token-two"], session=session, sleep_seconds=0)

    assert client.request("/rate-limited-once") == {"ok": True}
    assert session.authorizations == ["Bearer token-one", "Bearer token-two"]
    assert client.token == "token-two"


def test_github_client_rotates_past_invalid_token():
    session = SequenceSession(
        [
            FakeResponse(401, text="Bad credentials"),
            FakeResponse(200, payload={"ok": True}),
        ]
    )
    client = GitHubClient(tokens=["bad-token", "good-token"], session=session, sleep_seconds=0)

    assert client.request("/ok") == {"ok": True}
    assert session.authorizations == ["Bearer bad-token", "Bearer good-token"]
    assert client.token_pool.invalid_token_indexes == {0}


def test_github_client_fails_when_all_tokens_are_invalid():
    session = SequenceSession(
        [
            FakeResponse(401, text="Bad credentials"),
            FakeResponse(401, text="Bad credentials"),
        ]
    )
    client = GitHubClient(tokens=["bad-one", "bad-two"], session=session, sleep_seconds=0)

    with pytest.raises(GitHubAuthenticationError, match="All 2 configured"):
        client.request("/never-succeeds")


def test_github_client_retries_timeout_and_server_error():
    session = SequenceSession(
        [
            requests.Timeout("timed out"),
            FakeResponse(503, text="unavailable"),
            FakeResponse(200, payload={"ok": True}),
        ]
    )
    client = GitHubClient(
        tokens=["token-one"],
        session=session,
        sleep_seconds=0,
        retry_backoff_seconds=0,
        max_retries=2,
    )

    assert client.request("/eventually-ok") == {"ok": True}
    assert session.calls == 3


def test_token_pool_uses_retry_after_for_secondary_rate_limit(monkeypatch):
    monkeypatch.setattr(github_client.time, "time", lambda: 100.0)
    pool = GitHubTokenPool(tokens=["token-one"])

    pool.mark_rate_limited(
        0,
        FakeResponse(403, text="secondary rate limit", headers={"Retry-After": "7"}),
    )

    assert pool.rate_limited_until[0] == 107.0


def test_token_pool_applies_minimum_delay_to_stale_reset(monkeypatch):
    monkeypatch.setattr(github_client.time, "time", lambda: 100.0)
    pool = GitHubTokenPool(tokens=["token-one"])

    pool.mark_rate_limited(
        0,
        FakeResponse(
            403,
            text="API rate limit exceeded",
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "100"},
        ),
    )

    assert pool.rate_limited_until[0] == 100.5


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
    assert report["completed_repos"] == 0
    assert report["pr_fetch_failed"] == 1
    assert report["kept_rows"] == 1
    assert len(report["repo_reports"]) == 1
    repo_report = report["repo_reports"][0]
    assert repo_report["status"] == "partial"
    assert repo_report["candidate_count"] == 2
    assert repo_report["kept_count"] == 1
    assert repo_report["pr_fetch_failed_count"] == 1
    assert len(report["pr_failures"]) == 1
    assert report["pr_failures"][0]["number"] == 2


def test_failed_repository_search_is_not_completed(monkeypatch):
    class SearchFailingClient:
        def search_pull_requests(self, repo_full_name, start, end, per_repo_limit=None):
            raise GitHubClientError("search failed")

    monkeypatch.setattr(
        github_client.GitHubClient,
        "from_token_file",
        classmethod(lambda cls, token_file=None: SearchFailingClient()),
    )
    report: dict = {"repo_reports": [], "pr_failures": []}

    mined = mine_human_pull_requests(
        [{"repo_id": 7, "repo_full_name": "example/repo"}],
        date(2025, 1, 1),
        date(2025, 1, 2),
        report=report,
    )

    assert mined == []
    assert report["completed_repos"] == 0
    assert report["repo_reports"][0]["status"] == "search_failed"


def test_repository_search_does_not_swallow_authentication_exhaustion(monkeypatch):
    class AuthenticationFailingClient:
        def search_pull_requests(self, repo_full_name, start, end, per_repo_limit=None):
            raise GitHubAuthenticationError("all tokens invalid")

    monkeypatch.setattr(
        github_client.GitHubClient,
        "from_token_file",
        classmethod(lambda cls, token_file=None: AuthenticationFailingClient()),
    )

    with pytest.raises(GitHubAuthenticationError, match="all tokens invalid"):
        mine_human_pull_requests(
            [{"repo_id": 7, "repo_full_name": "example/repo"}],
            date(2025, 1, 1),
            date(2025, 1, 2),
        )


def test_fetch_record_preserves_commit_metadata_without_commit_fetch(monkeypatch):
    client = GitHubClient(tokens=["token-one"], sleep_seconds=0)
    monkeypatch.setattr(
        client,
        "fetch_pull_request",
        lambda repo, number: {"id": 1, "user": {}, "html_url": "https://github.com/a/b/pull/1"},
    )
    monkeypatch.setattr(client, "fetch_pull_files", lambda repo, number: [])
    monkeypatch.setattr(
        client,
        "fetch_pull_commits",
        lambda repo, number: pytest.fail("commits should not be fetched"),
    )
    existing_commits = [{"sha": "abc"}]

    result = client.fetch_pull_request_record(
        "a/b",
        1,
        base_record={"commit_messages": ["existing"], "commits": existing_commits},
        include_commits=False,
    )

    assert result["commit_messages"] == ["existing"]
    assert result["commits"] == existing_commits


def test_files_endpoint_404_is_not_classified_as_deleted_repo(monkeypatch):
    client = GitHubClient(tokens=["token-one"], sleep_seconds=0)
    monkeypatch.setattr(client, "fetch_pull_request", lambda repo, number: {"id": 1, "user": {}})
    monkeypatch.setattr(
        client,
        "fetch_pull_files",
        lambda repo, number: (_ for _ in ()).throw(GitHubNotFoundError("files not found")),
    )

    with pytest.raises(GitHubNotFoundError, match="files not found"):
        client.fetch_pull_request_record("a/b", 1)


@pytest.mark.parametrize("value", [None, "", "[]", float("nan"), pd.NA, [], [None]])
def test_has_filenames_rejects_missing_values(value):
    assert not github_client._has_filenames({"filenames": value})


@pytest.mark.parametrize("value", [["src/a.py"], "src/a.py"])
def test_has_filenames_accepts_real_values(value):
    assert github_client._has_filenames({"filenames": value})


@pytest.mark.parametrize("value", [None, pd.NA, float("nan"), "", "1.5", 0, -1])
def test_parse_pr_number_rejects_invalid_values(value):
    assert github_client._parse_pr_number(value) is None


@pytest.mark.parametrize("value", [1, 1.0, "1"])
def test_parse_pr_number_accepts_positive_integers(value):
    assert github_client._parse_pr_number(value) == 1
