from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from github_client import (
    GitHubAuthenticationError,
    GitHubClient,
    GitHubClientError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    parse_repo_full_name,
)
from schema import atomic_write_text


METHOD_NAME = "aidev_published_searches_v1"
METHOD_VERSION = 1
AIDEV_PAPER_URL = "https://arxiv.org/abs/2507.15003"
REPOSITORY_RESOLUTION_PREDECESSOR_SHA256 = (
    "6eef661b732ab2e3a4534d0b786f2317c59d1557bad9fad10ab76aadd2a094f2"
)


@dataclass(frozen=True)
class AIDevSearchRule:
    rule_id: str
    agent: str
    search_qualifiers: str
    start_date: date
    required_login: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "rule_id": self.rule_id,
            "agent": self.agent,
            "search_qualifiers": self.search_qualifiers,
            "start_date": self.start_date.isoformat(),
            "required_login": self.required_login,
        }


AIDEV_SEARCH_RULES = (
    AIDevSearchRule("openai_codex_head", "openai_codex", "head:codex/", date(2025, 5, 16)),
    AIDevSearchRule(
        "devin_author",
        "devin",
        "author:devin-ai-integration[bot]",
        date(2024, 12, 24),
    ),
    AIDevSearchRule(
        "github_copilot_head",
        "github_copilot",
        "head:copilot/",
        date(2025, 1, 1),
        required_login="copilot",
    ),
    AIDevSearchRule("cursor_head", "cursor", "head:cursor/", date(2025, 1, 1)),
    AIDevSearchRule(
        "claude_code_coauthor",
        "claude_code",
        '"Co-Authored-By: Claude"',
        date(2025, 2, 24),
    ),
)
AIDEV_START_DATE = min(rule.start_date for rule in AIDEV_SEARCH_RULES)

ATTRIBUTION_FIELDS = (
    pa.field("aidev_attribution_label", pa.string()),
    pa.field("aidev_attribution_agent", pa.string()),
    pa.field("aidev_attribution_rule", pa.string()),
    pa.field("aidev_attribution_evidence", pa.string()),
    pa.field("aidev_attribution_status", pa.string()),
    pa.field("aidev_attribution_method", pa.string()),
    pa.field("aidev_attribution_head_ref", pa.string()),
    pa.field("aidev_attribution_head_label", pa.string()),
    pa.field("aidev_attribution_author_login", pa.string()),
    pa.field("aidev_attribution_author_type", pa.string()),
    pa.field("aidev_attribution_detail_status", pa.string()),
)
ATTRIBUTION_COLUMNS = frozenset(field.name for field in ATTRIBUTION_FIELDS)
IDENTITY_COLUMNS = (
    "html_url",
    "aidev_source_html_url",
    "repo_full_name",
    "full_name",
    "name_with_owner",
    "repo_url",
    "repository_url",
    "repo_id",
    "repository_id",
    "number",
    "pr_number",
    "aidev_source_pr_number",
    "created_at",
)

WEB_PR_URL = re.compile(
    r"^(?:https?://)?github\.com/([^/?#]+)/([^/?#]+)/pull/(\d+)(?:[/?#].*)?$",
    re.IGNORECASE,
)
API_PR_URL = re.compile(
    r"^https?://api\.github\.com/repos/([^/?#]+)/([^/?#]+)/pulls/(\d+)(?:[/?#].*)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PullRequestIdentity:
    repo_full_name: str
    number: int

    @property
    def key(self) -> str:
        return f"{self.repo_full_name.casefold()}#{self.number}"


@dataclass
class RepositoryTarget:
    repo_full_name: str
    identities: dict[str, PullRequestIdentity]
    repository_ids: set[int] = field(default_factory=set)
    row_count: int = 0

    @property
    def by_number(self) -> dict[int, str]:
        return {identity.number: key for key, identity in self.identities.items()}


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def iso_date(value: str) -> date:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"expected an ISO date, got {value!r}") from error
    if pd.isna(parsed):
        raise argparse.ArgumentTypeError(f"expected an ISO date, got {value!r}")
    return parsed.date()


def state_path(output: Path) -> Path:
    return output.with_suffix(".state.json")


def summary_path(output: Path) -> Path:
    return output.with_suffix(".summary.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def parse_positive_number(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not parsed.is_integer() or parsed <= 0:
        return None
    return int(parsed)


def parse_pr_url(value: object) -> PullRequestIdentity | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "<na>"}:
        return None
    for pattern in (WEB_PR_URL, API_PR_URL):
        match = pattern.match(text)
        if match:
            return PullRequestIdentity(f"{match.group(1)}/{match.group(2)}", int(match.group(3)))
    return None


def row_identity(row: Mapping[str, object]) -> PullRequestIdentity | None:
    url_identity = parse_pr_url(row.get("html_url"))
    if url_identity is None:
        url_identity = parse_pr_url(row.get("aidev_source_html_url"))

    explicit_numbers = {
        number
        for column in ("number", "pr_number", "aidev_source_pr_number")
        if (number := parse_positive_number(row.get(column))) is not None
    }
    if len(explicit_numbers) > 1:
        return None
    explicit_number = next(iter(explicit_numbers), None)

    if url_identity is not None:
        if explicit_number is not None and explicit_number != url_identity.number:
            return None
        return url_identity

    repo_full_name = None
    for column in ("repo_full_name", "full_name", "name_with_owner", "repo_url", "repository_url"):
        repo_full_name = parse_repo_full_name(row.get(column))
        if repo_full_name:
            break
    if repo_full_name and explicit_number is not None:
        return PullRequestIdentity(repo_full_name, explicit_number)
    return None


def row_repository_id(row: Mapping[str, object]) -> int | None:
    repository_ids = {
        repository_id
        for column in ("repo_id", "repository_id")
        if (repository_id := parse_positive_number(row.get(column))) is not None
    }
    return next(iter(repository_ids)) if len(repository_ids) == 1 else None


def created_date(value: object) -> date | None:
    try:
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.date()


def input_projection(schema: pa.Schema) -> list[str]:
    columns = [column for column in IDENTITY_COLUMNS if column in schema.names]
    if "created_at" not in columns:
        raise ValueError("Input must contain created_at.")
    if not any(
        column in columns
        for column in (
            "html_url",
            "aidev_source_html_url",
            "repo_full_name",
            "full_name",
            "name_with_owner",
            "repo_url",
            "repository_url",
        )
    ):
        raise ValueError("Input must contain a PR URL or repository identity column.")
    return columns


def output_schema(input_schema: pa.Schema) -> pa.Schema:
    collisions = sorted(ATTRIBUTION_COLUMNS.intersection(input_schema.names))
    if collisions:
        raise ValueError("Input already contains AIDev attribution columns: " + ", ".join(collisions))
    return pa.schema(list(input_schema) + list(ATTRIBUTION_FIELDS), metadata=input_schema.metadata)


def collect_targets(
    input_path: Path,
    end_date: date,
    *,
    batch_size: int = 100_000,
) -> tuple[dict[str, RepositoryTarget], dict[str, int]]:
    source = pq.ParquetFile(input_path)
    columns = input_projection(source.schema_arrow)
    targets: dict[str, RepositoryTarget] = {}
    counts: Counter[str] = Counter()
    for batch in source.iter_batches(batch_size=batch_size, columns=columns):
        for row in pa.Table.from_batches([batch]).to_pylist():
            counts["input_rows"] += 1
            row_date = created_date(row.get("created_at"))
            if row_date is None:
                counts["invalid_created_at"] += 1
                continue
            if row_date < AIDEV_START_DATE or row_date > end_date:
                counts["outside_scope"] += 1
                continue
            identity = row_identity(row)
            if identity is None:
                counts["invalid_identity"] += 1
                continue
            repository_id = row_repository_id(row)
            if repository_id is None:
                counts["unverifiable_repository_rows"] += 1
                continue
            repo_key = identity.repo_full_name.casefold()
            target = targets.setdefault(repo_key, RepositoryTarget(identity.repo_full_name, {}))
            target.identities[identity.key] = identity
            target.row_count += 1
            target.repository_ids.add(repository_id)
            counts["eligible_rows"] += 1
    unverifiable = [key for key, target in targets.items() if len(target.repository_ids) != 1]
    for key in unverifiable:
        target = targets.pop(key)
        counts["unverifiable_repository_rows"] += target.row_count
        counts["eligible_rows"] -= target.row_count
    counts["eligible_repositories"] = len(targets)
    counts["eligible_unique_prs"] = sum(len(target.identities) for target in targets.values())
    return targets, {key: int(value) for key, value in counts.items()}


def item_login(item: Mapping[str, object]) -> str | None:
    user = item.get("user")
    if not isinstance(user, Mapping):
        return None
    login = user.get("login")
    if not isinstance(login, str):
        return None
    normalized = login.strip()
    return normalized or None


def error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"[:1000]


def search_repository(
    target: RepositoryTarget,
    end_date: date,
    client: GitHubClient,
    previous: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    previous = previous or {}
    expected_repository_ids = set(target.repository_ids)
    if len(expected_repository_ids) != 1:
        repository_resolution = {
            "status": "failed",
            "error": "Input repository identity is ambiguous.",
            "expected_repository_ids": sorted(expected_repository_ids),
        }
        return {
            "repo_full_name": target.repo_full_name,
            "query_repo_full_name": "",
            "repository_resolution": repository_resolution,
            "status": "incomplete",
            "searches": {},
            "details": {},
            "matched_input_count": 0,
        }
    expected_repository_id = next(iter(expected_repository_ids))
    try:
        repository_payload = client.request(f"/repositories/{expected_repository_id}")
    except GitHubNotFoundError as error:
        repository_resolution = {
            "status": "unavailable",
            "error": error_text(error),
            "source_repo_full_name": target.repo_full_name,
            "resolved_repo_full_name": "",
            "resolved_repository_id": expected_repository_id,
            "identity_verified": True,
            "renamed": False,
        }
        return {
            "repo_full_name": target.repo_full_name,
            "query_repo_full_name": "",
            "repository_resolution": repository_resolution,
            "status": "complete",
            "searches": {},
            "details": {},
            "matched_input_count": 0,
        }
    except (GitHubAuthenticationError, GitHubRateLimitError):
        raise
    except GitHubClientError as error:
        repository_resolution = {"status": "failed", "error": error_text(error)}
        return {
            "repo_full_name": target.repo_full_name,
            "query_repo_full_name": "",
            "repository_resolution": repository_resolution,
            "status": "incomplete",
            "searches": {},
            "details": {},
            "matched_input_count": 0,
        }
    query_repo_full_name = (
        parse_repo_full_name(repository_payload.get("full_name"))
        if isinstance(repository_payload, Mapping)
        else None
    )
    resolved_repository_id = (
        parse_positive_number(repository_payload.get("id"))
        if isinstance(repository_payload, Mapping)
        else None
    )
    if (
        not query_repo_full_name
        or resolved_repository_id is None
        or resolved_repository_id != expected_repository_id
    ):
        repository_resolution = {
            "status": "failed",
            "error": "GitHub repository identity could not be verified.",
            "expected_repository_ids": sorted(expected_repository_ids),
            "resolved_repository_id": resolved_repository_id,
        }
        return {
            "repo_full_name": target.repo_full_name,
            "query_repo_full_name": query_repo_full_name or "",
            "repository_resolution": repository_resolution,
            "status": "incomplete",
            "searches": {},
            "details": {},
            "matched_input_count": 0,
        }
    repository_resolution = {
        "status": "complete",
        "source_repo_full_name": target.repo_full_name,
        "resolved_repo_full_name": query_repo_full_name,
        "resolved_repository_id": resolved_repository_id,
        "identity_verified": bool(expected_repository_ids),
        "renamed": query_repo_full_name.casefold() != target.repo_full_name.casefold(),
    }
    previous_searches = previous.get("searches")
    searches = dict(previous_searches) if isinstance(previous_searches, Mapping) else {}
    target_by_number = target.by_number

    for rule in AIDEV_SEARCH_RULES:
        prior = searches.get(rule.rule_id)
        if isinstance(prior, Mapping) and prior.get("status") in {"complete", "skipped"}:
            continue
        if rule.start_date > end_date:
            searches[rule.rule_id] = {
                "status": "skipped",
                "reason": "rule_starts_after_requested_end_date",
                "matched_identities": [],
            }
            continue
        try:
            items = client.search_pull_requests_for_range(
                query_repo_full_name,
                rule.start_date,
                end_date,
                search_qualifiers=rule.search_qualifiers,
                strict=True,
            )
        except (GitHubAuthenticationError, GitHubRateLimitError):
            raise
        except GitHubClientError as error:
            searches[rule.rule_id] = {
                "status": "failed",
                "error": error_text(error),
                "matched_identities": [],
            }
            continue

        verified_items = []
        matched_identities: set[str] = set()
        invalid_result_numbers = 0
        missing_required_logins = 0
        for item in items:
            if not isinstance(item, Mapping):
                invalid_result_numbers += 1
                continue
            if rule.required_login:
                login = item_login(item)
                if login is None:
                    missing_required_logins += 1
                    continue
                if login.casefold() != rule.required_login.casefold():
                    continue
            verified_items.append(item)
            number = parse_positive_number(item.get("number"))
            if number is None:
                invalid_result_numbers += 1
                continue
            identity_key = target_by_number.get(number)
            if identity_key:
                matched_identities.add(identity_key)
        status = "failed" if invalid_result_numbers or missing_required_logins else "complete"
        searches[rule.rule_id] = {
            "status": status,
            "search_qualifiers": rule.search_qualifiers,
            "start_date": rule.start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "raw_result_count": len(items),
            "verified_result_count": len(verified_items),
            "invalid_result_number_count": invalid_result_numbers,
            "missing_required_login_count": missing_required_logins,
            "matched_input_count": len(matched_identities),
            "matched_identities": sorted(matched_identities),
        }
        if invalid_result_numbers or missing_required_logins:
            searches[rule.rule_id]["error"] = "GitHub search returned unverifiable results."

    matched_keys = sorted(
        {
            str(identity_key)
            for result in searches.values()
            if isinstance(result, Mapping) and result.get("status") == "complete"
            for identity_key in result.get("matched_identities", [])
        }
    )
    previous_details = previous.get("details")
    details = dict(previous_details) if isinstance(previous_details, Mapping) else {}
    for identity_key in matched_keys:
        prior = details.get(identity_key)
        if isinstance(prior, Mapping) and prior.get("status") == "complete":
            continue
        identity = target.identities[identity_key]
        try:
            pull_request = client.fetch_pull_request(query_repo_full_name, identity.number)
        except (GitHubAuthenticationError, GitHubRateLimitError):
            raise
        except GitHubClientError as error:
            details[identity_key] = {"status": "failed", "error": error_text(error)}
            continue
        head = pull_request.get("head") if isinstance(pull_request.get("head"), Mapping) else {}
        user = pull_request.get("user") if isinstance(pull_request.get("user"), Mapping) else {}
        details[identity_key] = {
            "status": "complete",
            "head_ref": str(head.get("ref") or ""),
            "head_label": str(head.get("label") or ""),
            "author_login": str(user.get("login") or ""),
            "author_type": str(user.get("type") or ""),
        }

    searches_complete = all(
        isinstance(searches.get(rule.rule_id), Mapping)
        and searches[rule.rule_id].get("status") in {"complete", "skipped"}
        for rule in AIDEV_SEARCH_RULES
    )
    details_complete = all(
        isinstance(details.get(identity_key), Mapping)
        and details[identity_key].get("status") == "complete"
        for identity_key in matched_keys
    )
    return {
        "repo_full_name": target.repo_full_name,
        "query_repo_full_name": query_repo_full_name,
        "repository_resolution": repository_resolution,
        "status": "complete" if searches_complete and details_complete else "incomplete",
        "searches": searches,
        "details": details,
        "matched_input_count": len(matched_keys),
    }


def repository_worker(
    target: RepositoryTarget,
    end_date: date,
    token_file: Path,
    previous: Mapping[str, object] | None,
) -> dict[str, Any]:
    client = GitHubClient.from_token_file(str(token_file))
    return search_repository(target, end_date, client, previous)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_run_signature(
    input_path: Path,
    source: pq.ParquetFile,
    input_sha256: str,
    end_date: date,
) -> dict[str, Any]:
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "input_path": str(input_path.resolve()),
        "input_sha256": input_sha256,
        "input_row_count": int(source.metadata.num_rows),
        "input_schema": str(source.schema_arrow),
        "input_schema_sha256": schema_sha256(source.schema_arrow),
        "end_date": end_date.isoformat(),
        "rules": [rule.to_dict() for rule in AIDEV_SEARCH_RULES],
        "code_sha256": {
            Path(__file__).name: sha256_file(Path(__file__)),
            "github_client.py": sha256_file(Path(__file__).with_name("github_client.py")),
        },
    }


def can_adopt_repository_resolution_upgrade(
    existing: Mapping[str, object],
    current: Mapping[str, object],
) -> bool:
    existing_code = existing.get("code_sha256")
    current_code = current.get("code_sha256")
    if not isinstance(existing_code, Mapping) or not isinstance(current_code, Mapping):
        return False
    if (
        existing_code.get(Path(__file__).name)
        != REPOSITORY_RESOLUTION_PREDECESSOR_SHA256
        or existing_code.get("github_client.py") != current_code.get("github_client.py")
    ):
        return False
    existing_without_code = dict(existing)
    current_without_code = dict(current)
    existing_without_code.pop("code_sha256", None)
    current_without_code.pop("code_sha256", None)
    return existing_without_code == current_without_code


def run_searches(
    targets: Mapping[str, RepositoryTarget],
    end_date: date,
    token_file: Path,
    workers: int,
    state: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_every: int,
) -> None:
    repositories = state.setdefault("repositories", {})
    pending = [
        (key, target)
        for key, target in targets.items()
        if not isinstance(repositories.get(key), Mapping)
        or repositories[key].get("status") != "complete"
    ]
    if not pending:
        return
    fatal_error: Exception | None = None
    dirty_repositories = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as executor:
        futures = {
            executor.submit(
                repository_worker,
                target,
                end_date,
                token_file,
                repositories.get(key),
            ): (key, target)
            for key, target in pending
        }
        completed = len(targets) - len(pending)
        for future in as_completed(futures):
            key, target = futures[future]
            try:
                result = future.result()
            except Exception as error:
                if fatal_error is None:
                    fatal_error = error
                    for queued in futures:
                        if queued is not future:
                            queued.cancel()
                continue
            repositories[key] = result
            completed += 1
            dirty_repositories += 1
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            if dirty_repositories >= checkpoint_every:
                write_json(checkpoint_path, state)
                dirty_repositories = 0
            print(
                f"[aidev {completed}/{len(targets)}] {target.repo_full_name} "
                f"status={repositories[key]['status']} matches={repositories[key]['matched_input_count']}",
                flush=True,
            )
    if dirty_repositories:
        write_json(checkpoint_path, state)
    if fatal_error is not None:
        raise fatal_error


def attribution_for_row(
    row: Mapping[str, object],
    repositories: Mapping[str, object],
    end_date: date,
) -> dict[str, str]:
    result = {field.name: "" for field in ATTRIBUTION_FIELDS}
    result["aidev_attribution_method"] = METHOD_NAME
    result["aidev_attribution_detail_status"] = "not_requested"

    row_date = created_date(row.get("created_at"))
    if row_date is None:
        result.update(
            aidev_attribution_label="unresolved",
            aidev_attribution_status="invalid_created_at",
        )
        return result
    if row_date < AIDEV_START_DATE or row_date > end_date:
        result.update(
            aidev_attribution_label="outside_scope",
            aidev_attribution_status="outside_scope",
        )
        return result

    identity = row_identity(row)
    if identity is None:
        result.update(
            aidev_attribution_label="unresolved",
            aidev_attribution_status="invalid_identity",
        )
        return result
    repository_id = row_repository_id(row)
    if repository_id is None:
        result.update(
            aidev_attribution_label="unresolved",
            aidev_attribution_status="invalid_repository_identity",
        )
        return result
    repository = repositories.get(identity.repo_full_name.casefold())
    if not isinstance(repository, Mapping):
        result.update(
            aidev_attribution_label="unresolved",
            aidev_attribution_status="search_not_run",
        )
        return result
    resolution = repository.get("repository_resolution")
    resolved_repository_id = (
        parse_positive_number(resolution.get("resolved_repository_id"))
        if isinstance(resolution, Mapping)
        else None
    )
    if resolved_repository_id != repository_id:
        result.update(
            aidev_attribution_label="unresolved",
            aidev_attribution_status="repository_identity_mismatch",
        )
        return result

    searches = repository.get("searches")
    searches = searches if isinstance(searches, Mapping) else {}
    matched_rules = []
    incomplete_rules = []
    for rule in AIDEV_SEARCH_RULES:
        if row_date < rule.start_date:
            continue
        search = searches.get(rule.rule_id)
        if not isinstance(search, Mapping) or search.get("status") != "complete":
            incomplete_rules.append(rule)
            continue
        if identity.key in search.get("matched_identities", []):
            matched_rules.append(rule)

    if matched_rules:
        agents = sorted({rule.agent for rule in matched_rules})
        detail_map = repository.get("details")
        detail_map = detail_map if isinstance(detail_map, Mapping) else {}
        detail = detail_map.get(identity.key)
        detail = detail if isinstance(detail, Mapping) else {"status": "missing"}
        status_parts = ["matched_multiple_agents" if len(agents) > 1 else "matched"]
        if incomplete_rules:
            status_parts.append("search_incomplete")
        if detail.get("status") != "complete":
            status_parts.append("detail_incomplete")
        result.update(
            aidev_attribution_label="agentic",
            aidev_attribution_agent="|".join(agents),
            aidev_attribution_rule="|".join(rule.rule_id for rule in matched_rules),
            aidev_attribution_evidence="|".join(rule.search_qualifiers for rule in matched_rules),
            aidev_attribution_status="_and_".join(status_parts),
            aidev_attribution_head_ref=str(detail.get("head_ref") or ""),
            aidev_attribution_head_label=str(detail.get("head_label") or ""),
            aidev_attribution_author_login=str(detail.get("author_login") or ""),
            aidev_attribution_author_type=str(detail.get("author_type") or ""),
            aidev_attribution_detail_status=str(detail.get("status") or "missing"),
        )
        return result

    if incomplete_rules:
        result.update(
            aidev_attribution_label="unresolved",
            aidev_attribution_rule="|".join(rule.rule_id for rule in incomplete_rules),
            aidev_attribution_status="search_incomplete",
        )
        return result

    result.update(
        aidev_attribution_label="human_candidate",
        aidev_attribution_status="complete_no_match",
    )
    return result


def write_enriched_parquet(
    input_path: Path,
    output_path: Path,
    repositories: Mapping[str, object],
    end_date: date,
    *,
    batch_size: int,
    expected_input_sha256: str | None = None,
) -> tuple[int, dict[str, int]]:
    source = pq.ParquetFile(input_path)
    enriched_schema = output_schema(source.schema_arrow)
    projection = input_projection(source.schema_arrow)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    row_count = 0
    labels: Counter[str] = Counter()
    try:
        with pq.ParquetWriter(temporary_path, enriched_schema, compression="snappy") as writer:
            for batch in source.iter_batches(batch_size=batch_size):
                table = pa.Table.from_batches([batch])
                rows = table.select(projection).to_pylist()
                attributions = [attribution_for_row(row, repositories, end_date) for row in rows]
                for attribution in attributions:
                    labels[attribution["aidev_attribution_label"]] += 1
                arrays = list(batch.columns)
                arrays.extend(
                    pa.array([attribution[field.name] for attribution in attributions], type=field.type)
                    for field in ATTRIBUTION_FIELDS
                )
                writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=enriched_schema))
                row_count += batch.num_rows
        finalized = pq.ParquetFile(temporary_path)
        if finalized.metadata.num_rows != source.metadata.num_rows:
            raise ValueError(
                f"Enriched output has {finalized.metadata.num_rows} rows; "
                f"expected {source.metadata.num_rows}."
            )
        if not finalized.schema_arrow.equals(enriched_schema):
            raise ValueError("Enriched output does not have the expected schema.")
        if expected_input_sha256 and sha256_file(input_path) != expected_input_sha256:
            raise ValueError("Input changed while AIDev enrichment was running.")
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return row_count, {key: int(value) for key, value in labels.items()}


def build_summary(
    input_path: Path,
    output_path: Path,
    end_date: date,
    target_counts: Mapping[str, int],
    state: Mapping[str, object],
    label_counts: Mapping[str, int],
    input_sha256: str,
) -> dict[str, Any]:
    repositories = state.get("repositories")
    repositories = repositories if isinstance(repositories, Mapping) else {}
    repository_statuses = Counter(
        str(value.get("status") or "missing")
        for value in repositories.values()
        if isinstance(value, Mapping)
    )
    rule_summary = {}
    for rule in AIDEV_SEARCH_RULES:
        outcomes = [
            repository.get("searches", {}).get(rule.rule_id, {})
            for repository in repositories.values()
            if isinstance(repository, Mapping)
        ]
        rule_summary[rule.rule_id] = {
            "agent": rule.agent,
            "search_qualifiers": rule.search_qualifiers,
            "start_date": rule.start_date.isoformat(),
            "complete_repositories": sum(outcome.get("status") == "complete" for outcome in outcomes),
            "failed_repositories": sum(outcome.get("status") == "failed" for outcome in outcomes),
            "matched_input_prs": sum(int(outcome.get("matched_input_count", 0)) for outcome in outcomes),
        }
    complete = bool(repositories) and all(
        isinstance(repository, Mapping) and repository.get("status") == "complete"
        for repository in repositories.values()
    )
    if not repositories and int(target_counts.get("eligible_repositories", 0)) == 0:
        complete = True
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "paper": AIDEV_PAPER_URL,
        "scope": {
            "start_date": AIDEV_START_DATE.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "rules": [rule.to_dict() for rule in AIDEV_SEARCH_RULES],
        "input": {
            "path": str(input_path.resolve()),
            "sha256": input_sha256,
            **{key: int(value) for key, value in target_counts.items()},
        },
        "output": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "labels": {key: int(value) for key, value in sorted(label_counts.items())},
        },
        "repositories": {
            "total": len(repositories),
            "statuses": {key: int(value) for key, value in sorted(repository_statuses.items())},
            "renamed": sum(
                bool(repository.get("repository_resolution", {}).get("renamed"))
                for repository in repositories.values()
                if isinstance(repository, Mapping)
            ),
        },
        "searches": rule_summary,
        "complete": complete,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich a PR Parquet using the five published AIDev agent searches."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--end-date", required=True, type=iso_date)
    parser.add_argument("--token-file", type=Path, default=Path("mining/github_tokens.txt"))
    parser.add_argument("--workers", type=positive_integer, default=5)
    parser.add_argument("--row-batch-size", type=positive_integer, default=100_000)
    parser.add_argument("--checkpoint-every", type=positive_integer, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--adopt-repository-resolution-upgrade", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")
    if args.end_date < AIDEV_START_DATE:
        parser.error(f"--end-date must be on or after {AIDEV_START_DATE.isoformat()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = state_path(args.output)
    report_path = summary_path(args.output)
    existing_artifacts = [path for path in (args.output, checkpoint_path, report_path) if path.exists()]
    if existing_artifacts and not args.resume:
        raise FileExistsError(
            "AIDev enrichment artifacts already exist; use --resume or choose another output: "
            + ", ".join(str(path) for path in existing_artifacts)
        )
    if args.resume and not checkpoint_path.exists() and (args.output.exists() or report_path.exists()):
        raise ValueError("Cannot resume AIDev enrichment artifacts without checkpoint state.")

    source = pq.ParquetFile(args.input)
    output_schema(source.schema_arrow)
    input_sha256 = sha256_file(args.input)
    signature = build_run_signature(args.input, source, input_sha256, args.end_date)
    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if state.get("run_signature") != signature:
            existing_signature = state.get("run_signature")
            can_adopt = (
                args.resume
                and args.adopt_repository_resolution_upgrade
                and not state.get("finalized", False)
                and isinstance(existing_signature, Mapping)
                and can_adopt_repository_resolution_upgrade(existing_signature, signature)
            )
            if not can_adopt:
                raise ValueError("Existing AIDev enrichment state does not match this run.")
            state.setdefault("migrations", []).append(
                {
                    "name": "repository_id_resolution_upgrade",
                    "adopted_at": datetime.now(timezone.utc).isoformat(),
                    "predecessor_code_sha256": REPOSITORY_RESOLUTION_PREDECESSOR_SHA256,
                    "replacement_code_sha256": signature["code_sha256"][Path(__file__).name],
                }
            )
            state["run_signature"] = signature
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(checkpoint_path, state)
        recorded_output_hash = state.get("output_sha256")
        if args.output.exists() and recorded_output_hash and sha256_file(args.output) != recorded_output_hash:
            raise ValueError("Existing enriched output does not match its checkpoint SHA-256.")
    else:
        state = {
            "version": METHOD_VERSION,
            "run_signature": signature,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "repositories": {},
            "finalized": False,
        }
        write_json(checkpoint_path, state)

    targets, target_counts = collect_targets(
        args.input,
        args.end_date,
        batch_size=args.row_batch_size,
    )
    if targets:
        run_searches(
            targets,
            args.end_date,
            args.token_file,
            args.workers,
            state,
            checkpoint_path,
            args.checkpoint_every,
        )

    row_count, label_counts = write_enriched_parquet(
        args.input,
        args.output,
        state.get("repositories", {}),
        args.end_date,
        batch_size=args.row_batch_size,
        expected_input_sha256=input_sha256,
    )
    if row_count != source.metadata.num_rows:
        raise ValueError(f"Enriched {row_count} rows; expected {source.metadata.num_rows}.")
    summary = build_summary(
        args.input,
        args.output,
        args.end_date,
        target_counts,
        state,
        label_counts,
        input_sha256,
    )
    write_json(report_path, summary)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["finalized"] = bool(summary["complete"])
    state["output_sha256"] = summary["output"]["sha256"]
    state["summary_path"] = str(report_path.resolve())
    write_json(checkpoint_path, state)

    print(
        f"Wrote {row_count} rows to {args.output}; "
        f"complete={summary['complete']} labels={summary['output']['labels']}",
        flush=True,
    )
    if not summary["complete"]:
        raise RuntimeError("AIDev enrichment has incomplete repositories; rerun with --resume.")


if __name__ == "__main__":
    main()
