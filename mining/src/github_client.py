from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from collections.abc import Callable, Iterable
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests


class GitHubClientError(RuntimeError):
    pass


class GitHubRateLimitError(GitHubClientError):
    pass


class GitHubAuthenticationError(GitHubClientError):
    pass


class GitHubNotFoundError(GitHubClientError):
    pass


SEARCH_RESULT_CAP = 1000


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


def monthly_date_ranges(start: date, end: date) -> list[tuple[date, date]]:
    if end < start:
        return []
    ranges: list[tuple[date, date]] = []
    current = start
    while current <= end:
        if current.month == 12:
            next_month = date(current.year + 1, 1, 1)
        else:
            next_month = date(current.year, current.month + 1, 1)
        chunk_end = min(end, next_month - timedelta(days=1))
        ranges.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return ranges


def response_header(response: requests.Response, name: str) -> str | None:
    for key, value in response.headers.items():
        if key.lower() == name.lower():
            return str(value)
    return None


def retry_after_seconds(response: requests.Response) -> float | None:
    raw_value = response_header(response, "retry-after")
    if not raw_value:
        return None
    try:
        return max(float(raw_value), 0.0)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw_value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0)


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
    rate_limited_until: dict[int, float] = field(default_factory=dict, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    invalid_token_indexes: set[int] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if not self.tokens:
            raise GitHubClientError("At least one GitHub token is required.")

    def label(self, index: int) -> str:
        return f"{index + 1}/{len(self.tokens)}"

    def _is_available_unlocked(self, index: int, now: float) -> bool:
        if index in self.invalid_token_indexes:
            return False
        reset_at = self.rate_limited_until.get(index)
        return reset_at is None or reset_at <= now

    def acquire(self) -> tuple[int, str]:
        with self.lock:
            now = time.time()
            for offset in range(len(self.tokens)):
                index = (self.next_index + offset) % len(self.tokens)
                if self._is_available_unlocked(index, now):
                    self.next_index = (index + 1) % len(self.tokens)
                    return index, self.tokens[index]
            all_invalid = len(self.invalid_token_indexes) == len(self.tokens)
        if all_invalid:
            raise GitHubAuthenticationError(
                f"All {len(self.tokens)} configured GitHub token(s) were rejected with HTTP 401."
            )
        raise self.all_rate_limited_error("GitHub API")

    def seconds_until_next_available(self) -> float | None:
        with self.lock:
            now = time.time()
            reset_values = [
                reset_at
                for index, reset_at in self.rate_limited_until.items()
                if index not in self.invalid_token_indexes and reset_at > now
            ]
        if not reset_values:
            return None
        return max(min(reset_values) - time.time(), 0.0)

    def acquire_waiting(self, url: str, progress: Callable[[str], None] | None = None) -> tuple[int, str]:
        while True:
            try:
                return self.acquire()
            except GitHubRateLimitError:
                wait_seconds = self.seconds_until_next_available()
                if wait_seconds is None:
                    raise self.all_rate_limited_error(url)
                if progress is not None:
                    progress(
                        f"[github-token] all {len(self.tokens)} token(s) are rate limited; "
                        f"waiting about {format_seconds(wait_seconds)} before retrying"
                    )
                time.sleep(wait_seconds)

    def mark_rate_limited(
        self,
        index: int,
        response: requests.Response,
        minimum_delay: float = 0.5,
    ) -> None:
        now = time.time()
        retry_after = retry_after_seconds(response)
        reset_at = response_header(response, "x-ratelimit-reset")
        if retry_after is not None:
            reset_value = now + retry_after
        else:
            try:
                reset_value = float(reset_at) if reset_at is not None else now + 60
            except ValueError:
                reset_value = now + 60
        reset_value = max(reset_value, now + max(minimum_delay, 0.0))
        with self.lock:
            self.rate_limited_until[index] = reset_value

    def mark_invalid(self, index: int) -> None:
        with self.lock:
            self.invalid_token_indexes.add(index)
            self.rate_limited_until.pop(index, None)

    def next_available_label(self) -> str | None:
        with self.lock:
            now = time.time()
            for offset in range(len(self.tokens)):
                index = (self.next_index + offset) % len(self.tokens)
                if self._is_available_unlocked(index, now):
                    return self.label(index)
        return None

    def all_rate_limited_error(self, url: str) -> GitHubRateLimitError:
        with self.lock:
            now = time.time()
            reset_values = [
                reset_at
                for index, reset_at in self.rate_limited_until.items()
                if index not in self.invalid_token_indexes and reset_at > now
            ]
        reset_note = ""
        if reset_values:
            wait_seconds = max(min(reset_values) - time.time(), 0.0)
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
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5

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
        remaining = response_header(response, "x-ratelimit-remaining")
        if response.status_code == 429:
            return True
        if response.status_code in {403, 429} and remaining == "0":
            return True
        text = str(response.text or "").lower()
        return response.status_code in {403, 429} and (
            "rate limit" in text
            or "secondary rate limit" in text
            or "api rate limit exceeded" in text
        )

    def _transient_retry_delay(
        self,
        retry_number: int,
        response: requests.Response | None = None,
    ) -> float:
        if response is not None:
            retry_after = retry_after_seconds(response)
            if retry_after is not None:
                return retry_after
        return self.retry_backoff_seconds * (2 ** retry_number)

    def request(self, path: str, params: dict | None = None) -> dict | list:
        url = path if path.startswith("https://") else f"{self.api_base}{path}"
        transient_retries = 0
        while True:
            token_index, token = self.token_pool.acquire_waiting(url, progress=print)
            self.last_token_index = token_index
            try:
                response = self.session.get(url, headers=self._headers(token), params=params, timeout=20)
            except (requests.Timeout, requests.ConnectionError) as exc:
                if transient_retries >= self.max_retries:
                    raise GitHubClientError(
                        f"GitHub request failed for {url} after {transient_retries + 1} attempts: {exc}"
                    ) from exc
                delay = self._transient_retry_delay(transient_retries)
                transient_retries += 1
                time.sleep(delay)
                continue
            except requests.RequestException as exc:
                raise GitHubClientError(f"GitHub request failed for {url}: {exc}") from exc
            if response.status_code == 401:
                self.token_pool.mark_invalid(token_index)
                next_label = self.token_pool.next_available_label()
                if next_label is not None:
                    print(
                        f"[github-token] token {self.token_pool.label(token_index)} was rejected with HTTP 401; "
                        f"switching to token {next_label}",
                        flush=True,
                    )
                continue
            if self._is_rate_limited(response):
                self.token_pool.mark_rate_limited(
                    token_index,
                    response,
                    minimum_delay=max(self.retry_backoff_seconds, 0.1),
                )
                next_label = self.token_pool.next_available_label()
                if next_label is None:
                    wait_seconds = self.token_pool.seconds_until_next_available()
                    if wait_seconds is None:
                        raise self.token_pool.all_rate_limited_error(url)
                    print(
                        f"[github-token] rate limit on token {self.token_pool.label(token_index)}; "
                        f"all {len(self.token_pool.tokens)} token(s) limited; "
                        f"waiting about {format_seconds(wait_seconds)} before retrying",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                    continue
                print(
                    f"[github-token] rate limit on token {self.token_pool.label(token_index)}; switching to token {next_label}",
                    flush=True,
                )
                continue
            if 500 <= response.status_code < 600:
                if transient_retries >= self.max_retries:
                    raise GitHubClientError(
                        f"GitHub request failed for {url} with HTTP {response.status_code} "
                        f"after {transient_retries + 1} attempts."
                    )
                delay = self._transient_retry_delay(transient_retries, response=response)
                transient_retries += 1
                time.sleep(delay)
                continue
            if response.status_code == 404:
                raise GitHubNotFoundError(f"GitHub resource not found: {url}")
            try:
                response.raise_for_status()
            except Exception as exc:
                raise GitHubClientError(f"GitHub request failed for {url}: {exc}") from exc
            time.sleep(self.sleep_seconds)
            return response.json()

    def search_pull_requests(
        self,
        repo_full_name: str,
        start: date,
        end: date,
        per_repo_limit: int | None = None,
    ) -> list[dict]:
        if per_repo_limit is not None and per_repo_limit <= 0:
            raise ValueError("per_repo_limit must be a positive integer when set.")
        collected: list[dict] = []
        seen_urls: set[str] = set()
        seen_numbers: set[int] = set()
        for chunk_start, chunk_end in monthly_date_ranges(start, end):
            remaining = (
                per_repo_limit - len(collected)
                if per_repo_limit is not None
                else None
            )
            if remaining is not None and remaining <= 0:
                break
            if remaining is None:
                range_items = self.search_pull_requests_for_range(
                    repo_full_name, chunk_start, chunk_end
                )
            else:
                range_items = self.search_pull_requests_for_range(
                    repo_full_name, chunk_start, chunk_end, result_limit=remaining
                )
            for item in range_items:
                url = str(item.get("html_url") or item.get("url") or "")
                number = item.get("number")
                if url and url in seen_urls:
                    continue
                if not url and isinstance(number, int) and number in seen_numbers:
                    continue
                if url:
                    seen_urls.add(url)
                if isinstance(number, int):
                    seen_numbers.add(number)
                collected.append(item)
                if per_repo_limit and len(collected) >= per_repo_limit:
                    return collected[:per_repo_limit]
        return collected

    def search_pull_requests_for_range(
        self,
        repo_full_name: str,
        start: date,
        end: date,
        result_limit: int | None = None,
        *,
        search_qualifiers: str | None = None,
        strict: bool = False,
    ) -> list[dict]:
        if end < start:
            return []
        query_parts = [f"repo:{repo_full_name}", "is:pr"]
        normalized_qualifiers = " ".join(str(search_qualifiers or "").split())
        if normalized_qualifiers:
            query_parts.append(normalized_qualifiers)
        query_parts.append(f"created:{start.isoformat()}..{end.isoformat()}")
        query = " ".join(query_parts)
        params = {
            "q": query,
            "sort": "created",
            "order": "asc",
            "per_page": 100,
            "page": 1,
        }
        def request_page(page: int) -> dict | list:
            payload: dict | list = {}
            for retry_number in range(self.max_retries + 1):
                payload = self.request("/search/issues", {**params, "page": page})
                if not isinstance(payload, dict) or not bool(
                    payload.get("incomplete_results", False)
                ):
                    return payload
                if retry_number < self.max_retries:
                    time.sleep(self._transient_retry_delay(retry_number))
            return payload

        payload = request_page(1)
        if not isinstance(payload, dict):
            if strict:
                raise GitHubClientError(
                    f"GitHub search for {repo_full_name} on {start.isoformat()}..{end.isoformat()} "
                    "returned a non-object response."
                )
            return []

        raw_total = payload.get("total_count")
        if strict:
            if isinstance(raw_total, bool) or not isinstance(raw_total, int) or raw_total < 0:
                raise GitHubClientError(
                    f"GitHub search for {repo_full_name} on {start.isoformat()}..{end.isoformat()} "
                    f"returned invalid total_count={raw_total!r}."
                )
            incomplete_value = payload.get("incomplete_results")
            if not isinstance(incomplete_value, bool):
                raise GitHubClientError(
                    f"GitHub search for {repo_full_name} on {start.isoformat()}..{end.isoformat()} "
                    "returned invalid incomplete_results."
                )
            total_count = raw_total
        else:
            try:
                total_count = int(raw_total) if raw_total is not None else None
            except (TypeError, ValueError):
                total_count = None
        incomplete = bool(payload.get("incomplete_results", False))
        exceeds_cap = total_count is not None and total_count > SEARCH_RESULT_CAP

        def subdivide_or_fail(reason: str) -> list[dict]:
            if start == end:
                raise GitHubClientError(reason + "; the one-day range cannot be subdivided.")
            midpoint = start + timedelta(days=(end - start).days // 2)
            left = self.search_pull_requests_for_range(
                repo_full_name,
                start,
                midpoint,
                result_limit=result_limit,
                search_qualifiers=normalized_qualifiers,
                strict=strict,
            )
            if result_limit is not None and len(left) >= result_limit:
                return left[:result_limit]
            right_limit = None if result_limit is None else result_limit - len(left)
            right = self.search_pull_requests_for_range(
                repo_full_name,
                midpoint + timedelta(days=1),
                end,
                result_limit=right_limit,
                search_qualifiers=normalized_qualifiers,
                strict=strict,
            )
            return left + right

        bounded_one_day = (
            start == end
            and result_limit is not None
            and result_limit <= SEARCH_RESULT_CAP
        )
        if (exceeds_cap and not bounded_one_day) or incomplete:
            if exceeds_cap:
                reason = (
                    f"GitHub search for {repo_full_name} on {start.isoformat()} returned "
                    f"{total_count} results, exceeding the {SEARCH_RESULT_CAP}-result cap"
                )
            else:
                reason = (
                    f"GitHub search for {repo_full_name} on {start.isoformat()}..{end.isoformat()} "
                    "returned incomplete_results"
                )
            return subdivide_or_fail(reason)

        target_count = total_count
        if result_limit is not None:
            target_count = (
                result_limit
                if target_count is None
                else min(target_count, result_limit)
            )
        raw_items = payload.get("items")
        if strict and not isinstance(raw_items, list):
            raise GitHubClientError(
                f"GitHub search for {repo_full_name} on {start.isoformat()}..{end.isoformat()} "
                "returned invalid items."
            )
        collected = list(raw_items or [])
        seen_numbers: set[int] = set()
        if strict:
            for item in collected:
                number = item.get("number") if isinstance(item, dict) else None
                if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
                    raise GitHubClientError(
                        f"GitHub search for {repo_full_name} returned an item without a valid PR number."
                    )
                if number in seen_numbers:
                    raise GitHubClientError(
                        f"GitHub search for {repo_full_name} returned duplicate PR number {number}."
                    )
                seen_numbers.add(number)
            if total_count is not None and len(collected) > total_count:
                raise GitHubClientError(
                    f"GitHub search for {repo_full_name} returned more items than total_count."
                )
        if target_count is not None:
            collected = collected[:target_count]
            if strict:
                seen_numbers = {item["number"] for item in collected}
        page = 2
        while True:
            if target_count is not None and len(collected) >= target_count:
                break
            if len(collected) < (page - 1) * 100:
                break
            page_payload = request_page(page)
            if isinstance(page_payload, dict) and bool(page_payload.get("incomplete_results", False)):
                return subdivide_or_fail(
                    f"GitHub search for {repo_full_name} on {start.isoformat()}..{end.isoformat()} "
                    "returned incomplete_results"
                )
            if strict and not isinstance(page_payload, dict):
                raise GitHubClientError(
                    f"GitHub search page {page} for {repo_full_name} returned a non-object response."
                )
            raw_page_items = page_payload.get("items", []) if isinstance(page_payload, dict) else []
            if strict:
                page_incomplete = page_payload.get("incomplete_results")
                page_total = page_payload.get("total_count")
                if not isinstance(page_incomplete, bool):
                    raise GitHubClientError(
                        f"GitHub search page {page} for {repo_full_name} returned invalid incomplete_results."
                    )
                if (
                    isinstance(page_total, bool)
                    or not isinstance(page_total, int)
                    or page_total != total_count
                ):
                    raise GitHubClientError(
                        f"GitHub search page {page} for {repo_full_name} returned inconsistent total_count."
                    )
            if strict and not isinstance(raw_page_items, list):
                raise GitHubClientError(
                    f"GitHub search page {page} for {repo_full_name} returned invalid items."
                )
            items = raw_page_items or []
            if not items:
                break
            if strict:
                page_numbers: set[int] = set()
                for item in items:
                    number = item.get("number") if isinstance(item, dict) else None
                    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
                        raise GitHubClientError(
                            f"GitHub search page {page} for {repo_full_name} returned an item "
                            "without a valid PR number."
                        )
                    if number in page_numbers or number in seen_numbers:
                        raise GitHubClientError(
                            f"GitHub search for {repo_full_name} returned duplicate PR number {number}."
                        )
                    page_numbers.add(number)
                seen_numbers.update(page_numbers)
            collected.extend(items)
            if strict and total_count is not None and len(collected) > total_count:
                raise GitHubClientError(
                    f"GitHub search for {repo_full_name} returned more items than total_count."
                )
            if target_count is not None:
                collected = collected[:target_count]
            if len(items) < 100:
                break
            page += 1
        if target_count is not None and len(collected) < target_count:
            raise GitHubClientError(
                f"GitHub search for {repo_full_name} on {start.isoformat()}..{end.isoformat()} "
                f"returned {len(collected)} of {target_count} required results."
            )
        return collected

    def fetch_pull_request(self, repo_full_name: str, number: int) -> dict:
        return self.request(f"/repos/{repo_full_name}/pulls/{number}")

    def fetch_repository(self, repo_full_name: str) -> dict:
        return self.request(f"/repos/{repo_full_name}")

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
        except GitHubNotFoundError:
            record["deleted_repo"] = True
            record.setdefault("filenames", [])
            record.setdefault("files", [])
            record.setdefault("commit_messages", [])
            record.setdefault("commits", [])
            return record

        files = self.fetch_pull_files(repo_full_name, number)
        commits = self.fetch_pull_commits(repo_full_name, number) if include_commits else None

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
                "deleted_repo": False,
            }
        )
        if commits is None:
            record.setdefault("commit_messages", [])
            record.setdefault("commits", [])
        else:
            record["commit_messages"] = [
                (commit.get("commit") or {}).get("message") for commit in commits
            ]
            record["commits"] = commits
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
        number = _parse_pr_number(record.get("number"))
        if not repo_full_name or number is None:
            if progress:
                progress(f"[enrich {index}/{total}] missing repo/number, keeping row as-is")
            if report is not None:
                report["missing_repo_or_number"] = int(report.get("missing_repo_or_number", 0)) + 1
                report["failures"].append(
                    {
                        "status": "missing_repo_or_number",
                        "repo_full_name": repo_full_name,
                        "number": number,
                        "html_url": record.get("html_url"),
                        "title": record.get("title"),
                    }
                )
            enriched.append(record)
            continue
        try:
            enriched_record = client.fetch_pull_request_record(
                repo_full_name,
                number,
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
        except (GitHubRateLimitError, GitHubAuthenticationError):
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
        return filenames.strip().lower() not in {"", "[]", "nan", "none", "<na>"}
    if isinstance(filenames, float) and math.isnan(filenames):
        return False
    if filenames.__class__.__name__ == "NAType":
        return False
    try:
        return any(str(value).strip().lower() not in {"", "nan", "none", "<na>"} for value in filenames)
    except TypeError:
        return bool(filenames)


def _parse_pr_number(value: object) -> int | None:
    if value is None or value.__class__.__name__ == "NAType":
        return None
    try:
        number = int(value)
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric_value) or numeric_value != number or number <= 0:
        return None
    return number


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
        except (GitHubRateLimitError, GitHubAuthenticationError):
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
            except (GitHubRateLimitError, GitHubAuthenticationError):
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
        repo_report["status"] = "partial" if repo_pr_failed else "completed"
        repo_report["kept_count"] = repo_mined
        repo_report["pr_fetch_failed_count"] = repo_pr_failed
        if report is not None:
            if not repo_pr_failed:
                report["completed_repos"] = int(report.get("completed_repos", 0)) + 1
            report["kept_rows"] = int(report.get("kept_rows", 0)) + repo_mined
            report["repo_reports"].append(repo_report)
    return mined
