from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import requests


class GitHubClientError(RuntimeError):
    pass


class GitHubRateLimitError(GitHubClientError):
    pass


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "?:??"
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_repo_full_name(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    patterns = (
        r"github\.com/(?:repos/)?([^/\s]+/[^/\s]+)",
        r"api\.github\.com/repos/([^/\s]+/[^/\s]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).removesuffix(".git")
    if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", text):
        return text
    return None


def parse_token_value(value: str | None) -> list[str]:
    if not value:
        return []
    token = value.strip()
    return [token] if token else []


def github_tokens_from_file(token_file: str | Path) -> list[str]:
    path = Path(token_file)
    if not path.is_file():
        raise GitHubClientError(
            f"GitHub token file not found: {path}. Set github.token_file to an existing file with at least one token per line."
        )
    tokens: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.extend(parse_token_value(line))

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        deduped.append(token)
        seen.add(token)
    return deduped


def github_tokens(token_file: str | None) -> list[str]:
    if not token_file:
        raise GitHubClientError("github.token_file is required for real GitHub mining.")
    return github_tokens_from_file(token_file)


@dataclass
class GitHubTokenPool:
    tokens: list[str]
    next_index: int = 0
    rate_limited_until: dict[int, int] = field(default_factory=dict, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.tokens:
            raise GitHubClientError("At least one GitHub token is required.")

    def label(self, index: int) -> str:
        return f"{index + 1}/{len(self.tokens)}"

    def _is_available_unlocked(self, index: int, now: int) -> bool:
        reset_at = self.rate_limited_until.get(index)
        return reset_at is None or reset_at <= now

    def acquire(self) -> tuple[int, str]:
        with self.lock:
            now = int(time.time())
            for offset in range(len(self.tokens)):
                index = (self.next_index + offset) % len(self.tokens)
                if self._is_available_unlocked(index, now):
                    self.next_index = (index + 1) % len(self.tokens)
                    return index, self.tokens[index]
        raise self.all_rate_limited_error("GitHub API")

    def mark_rate_limited(self, index: int, response: requests.Response) -> None:
        reset_at = response.headers.get("x-ratelimit-reset")
        reset_value = int(reset_at) if reset_at and reset_at.isdigit() else int(time.time()) + 60
        with self.lock:
            self.rate_limited_until[index] = reset_value

    def next_available_label(self) -> str | None:
        with self.lock:
            now = int(time.time())
            for offset in range(len(self.tokens)):
                index = (self.next_index + offset) % len(self.tokens)
                if self._is_available_unlocked(index, now):
                    return self.label(index)
        return None

    def all_rate_limited_error(self, url: str) -> GitHubRateLimitError:
        with self.lock:
            now = int(time.time())
            reset_values = [
                reset_at
                for reset_at in self.rate_limited_until.values()
                if reset_at > now
            ]
        reset_note = ""
        if reset_values:
            wait_seconds = max(min(reset_values) - int(time.time()), 0)
            reset_note = f" Earliest token reset in about {format_seconds(wait_seconds)}."
        return GitHubRateLimitError(
            f"GitHub rate limit hit for all {len(self.tokens)} configured token(s) while requesting {url}.{reset_note}"
        )


_TOKEN_POOLS: dict[tuple[str, tuple[str, ...]], GitHubTokenPool] = {}
_TOKEN_POOLS_LOCK = threading.Lock()


def github_token_pool(token_file: str | None) -> GitHubTokenPool:
    tokens = github_tokens(token_file=token_file)
    if not tokens:
        raise GitHubClientError(
            f"No GitHub tokens were loaded from {token_file}. "
            "Set github.token_file to a file with at least one token per line."
        )
    key = (str(token_file), tuple(tokens))
    with _TOKEN_POOLS_LOCK:
        pool = _TOKEN_POOLS.get(key)
        if pool is None:
            pool = GitHubTokenPool(tokens=tokens)
            _TOKEN_POOLS[key] = pool
        return pool


@dataclass
class GitHubClient:
    tokens: list[str] | None = None
    api_base: str = "https://api.github.com"
    sleep_seconds: float = 0.05
    session: requests.Session = field(default_factory=requests.Session, repr=False)
    token_pool: GitHubTokenPool | None = None
    last_token_index: int = 0

    @classmethod
    def from_token_file(cls, token_file: str | None) -> "GitHubClient":
        return cls(token_pool=github_token_pool(token_file=token_file))

    def __post_init__(self) -> None:
        if self.token_pool is None:
            self.token_pool = GitHubTokenPool(tokens=list(self.tokens or []))
        self.tokens = self.token_pool.tokens

    @property
    def token(self) -> str:
        return self.token_pool.tokens[self.last_token_index]

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    @staticmethod
    def _is_rate_limited(response: requests.Response) -> bool:
        remaining = response.headers.get("x-ratelimit-remaining")
        if response.status_code in {403, 429} and remaining == "0":
            return True
        text = response.text.lower()
        return response.status_code in {403, 429} and (
            "rate limit" in text
            or "secondary rate limit" in text
            or "api rate limit exceeded" in text
        )

    def request(self, path: str, params: dict | None = None) -> dict | list:
        url = path if path.startswith("https://") else f"{self.api_base}{path}"
        for _ in range(len(self.token_pool.tokens)):
            token_index, token = self.token_pool.acquire()
            self.last_token_index = token_index
            try:
                response = self.session.get(url, headers=self._headers(token), params=params, timeout=20)
            except requests.RequestException as exc:
                raise GitHubClientError(f"GitHub request failed for {url}: {exc}") from exc
            if self._is_rate_limited(response):
                self.token_pool.mark_rate_limited(token_index, response)
                next_label = self.token_pool.next_available_label()
                if next_label is None:
                    raise self.token_pool.all_rate_limited_error(url)
                print(
                    f"[github-token] rate limit on token {self.token_pool.label(token_index)}; switching to token {next_label}",
                    flush=True,
                )
                continue
            if response.status_code == 404:
                raise GitHubClientError(f"GitHub resource not found: {url}")
            try:
                response.raise_for_status()
            except requests.RequestException as exc:
                raise GitHubClientError(f"GitHub request failed for {url}: {exc}") from exc
            time.sleep(self.sleep_seconds)
            return response.json()
        raise self.token_pool.all_rate_limited_error(url)

    def search_pull_requests(
        self,
        repo_full_name: str,
        start: date,
        end: date,
        per_repo_limit: int | None = None,
    ) -> list[dict]:
        query = f"repo:{repo_full_name} is:pr created:{start.isoformat()}..{end.isoformat()}"
        collected: list[dict] = []
        page = 1
        while True:
            payload = self.request(
                "/search/issues",
                {
                    "q": query,
                    "sort": "created",
                    "order": "asc",
                    "per_page": 100,
                    "page": page,
                },
            )
            items = payload.get("items", []) if isinstance(payload, dict) else []
            if not items:
                break
            collected.extend(items)
            if per_repo_limit and len(collected) >= per_repo_limit:
                return collected[:per_repo_limit]
            if len(items) < 100:
                break
            page += 1
        return collected

    def fetch_pull_request(self, repo_full_name: str, number: int) -> dict:
        return self.request(f"/repos/{repo_full_name}/pulls/{number}")

    def fetch_pull_files(self, repo_full_name: str, number: int) -> list[dict]:
        files: list[dict] = []
        page = 1
        while True:
            payload = self.request(
                f"/repos/{repo_full_name}/pulls/{number}/files",
                {"per_page": 100, "page": page},
            )
            if not isinstance(payload, list) or not payload:
                break
            files.extend(payload)
            if len(payload) < 100:
                break
            page += 1
        return files

    def fetch_pull_commits(self, repo_full_name: str, number: int) -> list[dict]:
        commits: list[dict] = []
        page = 1
        while True:
            payload = self.request(
                f"/repos/{repo_full_name}/pulls/{number}/commits",
                {"per_page": 100, "page": page},
            )
            if not isinstance(payload, list) or not payload:
                break
            commits.extend(payload)
            if len(payload) < 100:
                break
            page += 1
        return commits

    def fetch_pull_request_record(
        self,
        repo_full_name: str,
        number: int,
        base_record: dict | None = None,
        include_commits: bool = False,
    ) -> dict:
        record = dict(base_record or {})
        record["repo_full_name"] = repo_full_name
        record["number"] = number
        try:
            pr = self.fetch_pull_request(repo_full_name, number)
            files = self.fetch_pull_files(repo_full_name, number)
            commits = self.fetch_pull_commits(repo_full_name, number) if include_commits else []
        except GitHubClientError as exc:
            if "resource not found" in str(exc).lower():
                record["deleted_repo"] = True
                record.setdefault("filenames", [])
                record.setdefault("files", [])
                record.setdefault("commit_messages", [])
                record.setdefault("commits", [])
                return record
            raise

        record.update(
            {
                "id": pr.get("id", record.get("id")),
                "title": pr.get("title", record.get("title")),
                "body": pr.get("body", record.get("body")),
                "user": (pr.get("user") or {}).get("login", record.get("user")),
                "user_type": (pr.get("user") or {}).get("type", record.get("user_type")),
                "state": pr.get("state", record.get("state")),
                "created_at": pr.get("created_at", record.get("created_at")),
                "closed_at": pr.get("closed_at", record.get("closed_at")),
                "merged_at": pr.get("merged_at", record.get("merged_at")),
                "html_url": pr.get("html_url", record.get("html_url")),
                "additions": pr.get("additions", record.get("additions")),
                "deletions": pr.get("deletions", record.get("deletions")),
                "changed_files": pr.get("changed_files", record.get("changed_files")),
                "filenames": [file.get("filename") for file in files],
                "files": files,
                "commit_messages": [
                    (commit.get("commit") or {}).get("message") for commit in commits
                ],
                "commits": commits,
                "deleted_repo": False,
            }
        )
        return record


def enrich_pull_request_records(
    rows: Iterable[dict],
    token_file: str | None = None,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 10,
    report: dict[str, Any] | None = None,
) -> list[dict]:
    client = GitHubClient.from_token_file(token_file=token_file)
    rows_list = list(rows)
    total = len(rows_list)
    if report is not None:
        report.setdefault("total_rows", total)
        report.setdefault("already_has_filenames", 0)
        report.setdefault("missing_repo_or_number", 0)
        report.setdefault("fetched", 0)
        report.setdefault("deleted_repo", 0)
        report.setdefault("failed", 0)
        report.setdefault("failures", [])
    enriched: list[dict] = []
    for index, row in enumerate(rows_list, start=1):
        record = dict(row)
        if _has_filenames(record):
            enriched.append(record)
            if report is not None:
                report["already_has_filenames"] = int(report.get("already_has_filenames", 0)) + 1
            if progress and (index == 1 or index == total or index % progress_every == 0):
                progress(f"[enrich {index}/{total}] already has filenames")
            continue
        repo_full_name = (
            parse_repo_full_name(record.get("repo_full_name"))
            or parse_repo_full_name(record.get("repo_url"))
            or parse_repo_full_name(record.get("html_url"))
            or parse_repo_full_name(record.get("url"))
        )
        number = record.get("number")
        if not repo_full_name or number is None:
            if progress:
                progress(f"[enrich {index}/{total}] missing repo/number, keeping row as-is")
            if report is not None:
                report["missing_repo_or_number"] = int(report.get("missing_repo_or_number", 0)) + 1
                report["failures"].append(
                    {
                        "status": "missing_repo_or_number",
                        "repo_full_name": repo_full_name,
                        "number": int(number) if number is not None else None,
                        "html_url": record.get("html_url"),
                        "title": record.get("title"),
                    }
                )
            enriched.append(record)
            continue
        try:
            enriched_record = client.fetch_pull_request_record(
                repo_full_name,
                int(number),
                base_record=record,
            )
            enriched.append(enriched_record)
            if progress and (index == 1 or index == total or index % progress_every == 0):
                progress(f"[enrich {index}/{total}] fetched {repo_full_name}#{number}")
            if report is not None:
                if enriched_record.get("deleted_repo"):
                    report["deleted_repo"] = int(report.get("deleted_repo", 0)) + 1
                    report["failures"].append(
                        {
                            "status": "deleted_repo",
                            "repo_full_name": repo_full_name,
                            "number": int(number),
                            "html_url": enriched_record.get("html_url"),
                            "title": enriched_record.get("title"),
                        }
                    )
                else:
                    report["fetched"] = int(report.get("fetched", 0)) + 1
        except GitHubRateLimitError:
            raise
        except GitHubClientError:
            if progress:
                progress(f"[enrich {index}/{total}] failed to fetch {repo_full_name}#{number}, keeping row as-is")
            if report is not None:
                report["failed"] = int(report.get("failed", 0)) + 1
                report["failures"].append(
                    {
                        "status": "fetch_failed",
                        "repo_full_name": repo_full_name,
                        "number": int(number),
                        "html_url": record.get("html_url"),
                        "title": record.get("title"),
                    }
                )
            enriched.append(record)
    return enriched


def _has_filenames(row: dict) -> bool:
    filenames = row.get("filenames")
    if filenames is None:
        return False
    if isinstance(filenames, str):
        return bool(filenames.strip())
    try:
        return len(filenames) > 0
    except TypeError:
        return bool(filenames)


def mine_human_pull_requests(
    repos: Iterable[dict],
    start: date,
    end: date,
    token_file: str | None = None,
    per_repo_search_limit: int | None = None,
    progress: Callable[[str], None] | None = None,
    report: dict[str, Any] | None = None,
) -> list[dict]:
    client = GitHubClient.from_token_file(token_file=token_file)
    repos_list = list(repos)
    total_repos = len(repos_list)
    if report is not None:
        report.setdefault("total_repos", total_repos)
        report.setdefault("completed_repos", 0)
        report.setdefault("missing_repo_name", 0)
        report.setdefault("search_failed", 0)
        report.setdefault("search_empty", 0)
        report.setdefault("search_candidates", 0)
        report.setdefault("kept_rows", 0)
        report.setdefault("pr_fetch_failed", 0)
        report.setdefault("repo_reports", [])
        report.setdefault("pr_failures", [])
    mined: list[dict] = []
    started_at = time.perf_counter()
    for repo_index, repo in enumerate(repos_list, start=1):
        repo_id = repo.get("repo_id") or repo.get("id")
        full_name = repo.get("repo_full_name") or repo.get("full_name") or repo.get("name_with_owner")
        repo_report = {
            "repo_index": repo_index,
            "repo_total": total_repos,
            "repo_id": repo_id,
            "repo_full_name": full_name,
            "status": "pending",
            "search_limit": per_repo_search_limit,
            "candidate_count": 0,
            "kept_count": 0,
            "pr_fetch_failed_count": 0,
            "error": None,
        }
        if not full_name:
            if progress:
                progress(f"[github {repo_index}/{total_repos}] missing repo name, skipping")
            repo_report["status"] = "missing_repo_name"
            if report is not None:
                report["missing_repo_name"] = int(report.get("missing_repo_name", 0)) + 1
                report["repo_reports"].append(repo_report)
            continue
        if progress:
            elapsed = time.perf_counter() - started_at
            avg = elapsed / max(repo_index - 1, 1)
            eta = avg * max(total_repos - repo_index + 1, 0)
            progress(
                f"[github {repo_index}/{total_repos}] {full_name} | mined={len(mined)} | eta={format_seconds(eta)}"
            )
        try:
            search_items = client.search_pull_requests(
                str(full_name), start, end, per_repo_limit=per_repo_search_limit
            )
        except GitHubRateLimitError:
            raise
        except GitHubClientError as exc:
            if progress:
                progress(f"[github {repo_index}/{total_repos}] {full_name} search failed: {exc}")
            repo_report["status"] = "search_failed"
            repo_report["error"] = str(exc)
            if report is not None:
                report["search_failed"] = int(report.get("search_failed", 0)) + 1
                report["repo_reports"].append(repo_report)
            continue
        if progress:
            progress(f"[github {repo_index}/{total_repos}] {full_name} search returned {len(search_items)} candidates")
        repo_report["candidate_count"] = len(search_items)
        if report is not None:
            report["search_candidates"] = int(report.get("search_candidates", 0)) + len(search_items)
            if not search_items:
                report["search_empty"] = int(report.get("search_empty", 0)) + 1
        repo_mined = 0
        repo_pr_failed = 0
        for item in search_items:
            number = int(item["number"])
            try:
                record = client.fetch_pull_request_record(
                    str(full_name),
                    number,
                    base_record={
                        "repo_id": repo_id,
                        "repo_full_name": full_name,
                        "number": number,
                    },
                    include_commits=False,
                )
            except GitHubRateLimitError:
                raise
            except GitHubClientError as exc:
                if progress:
                    progress(f"  [skip {full_name}#{number}] {exc}")
                repo_pr_failed += 1
                if report is not None:
                    report["pr_fetch_failed"] = int(report.get("pr_fetch_failed", 0)) + 1
                    report["pr_failures"].append(
                        {
                            "repo_full_name": str(full_name),
                            "number": int(number),
                            "error": str(exc),
                        }
                    )
                continue
            mined.append(record)
            repo_mined += 1
        if progress:
            progress(
                f"[github {repo_index}/{total_repos}] {full_name} done | kept={repo_mined} | cumulative={len(mined)}"
            )
        repo_report["status"] = "completed"
        repo_report["kept_count"] = repo_mined
        repo_report["pr_fetch_failed_count"] = repo_pr_failed
        if report is not None:
            report["completed_repos"] = int(report.get("completed_repos", 0)) + 1
            report["kept_rows"] = int(report.get("kept_rows", 0)) + repo_mined
            report["repo_reports"].append(repo_report)
    return mined
