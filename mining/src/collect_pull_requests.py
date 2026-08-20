"""Collect the frozen GitHub pull-request population used by the study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections.abc import Callable
from typing import Any

import pandas as pd
import yaml

from author_filter import filter_by_author_with_details
from classify_task_type import attach_task_type
from github_client import GitHubRateLimitError, enrich_pull_request_records, mine_human_pull_requests, parse_repo_full_name
from load_aidev import load_aidev_tables
from resume import (
    checkpoint_dataframe,
    completed_repo_set,
    initial_resume_state,
    load_checkpoint_dataframe,
    load_resume_state,
    reset_output_tree,
    resume_state_path,
    save_checkpoint_dataframe,
    save_resume_state,
    state_matches_signature, update_resume_state,
)
from quality_filters import apply_quality_filters_with_details
from schema import (
    REPO_ID_COLUMNS,
    REPO_NAME_COLUMNS,
    STAR_COLUMNS,
    TimeWindow,
    atomic_write_text,
    ensure_datetime,
    ensure_output_dirs,
    first_existing_column,
    require_column,
    write_parquet,
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def log(message: str) -> None:
    print(message, flush=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def repo_star_column(repos: pd.DataFrame) -> str:
    return require_column(repos, STAR_COLUMNS, "repository star filtering")


def filter_repositories(repos: pd.DataFrame, star_floor: int) -> pd.DataFrame:
    stars = repo_star_column(repos)
    return repos.loc[pd.to_numeric(repos[stars], errors="coerce").fillna(0) >= star_floor].copy()


def ensure_repo_full_names(repos: pd.DataFrame, pr_table: pd.DataFrame) -> pd.DataFrame:
    result = repos.copy()
    existing = first_existing_column(result, REPO_NAME_COLUMNS)
    if existing:
        result["repo_full_name"] = result[existing]
        return result

    for url_col in ("repo_url", "html_url", "url"):
        if url_col in result.columns:
            result["repo_full_name"] = result[url_col].map(parse_repo_full_name)
            if result["repo_full_name"].notna().any():
                return result

    repo_id_col = first_existing_column(result, REPO_ID_COLUMNS)
    pr_repo_id_col = first_existing_column(pr_table, REPO_ID_COLUMNS)
    pr_url_col = first_existing_column(pr_table, ("repo_url", "repository_url"))
    if repo_id_col and pr_repo_id_col and pr_url_col:
        mapping = (
            pr_table[[pr_repo_id_col, pr_url_col]]
            .dropna()
            .drop_duplicates(pr_repo_id_col)
            .assign(repo_full_name=lambda df: df[pr_url_col].map(parse_repo_full_name))
            .dropna(subset=["repo_full_name"])
            .set_index(pr_repo_id_col)["repo_full_name"]
            .to_dict()
        )
        result["repo_full_name"] = result[repo_id_col].map(mapping)
    return result


def derive_time_window(agent_prs: pd.DataFrame) -> TimeWindow:
    created_col = require_column(agent_prs, ("created_at", "createdAt"), "time window derivation")
    created = ensure_datetime(agent_prs[created_col]).dropna()
    if created.empty:
        raise ValueError("Cannot derive time window because created_at is empty.")
    return TimeWindow(start=created.min(), end=created.max())


def configured_time_window(config: dict[str, Any], fallback: TimeWindow) -> TimeWindow:
    criteria = config.get("criteria", {})
    start = criteria.get("start_date") or criteria.get("window_start")
    end = criteria.get("end_date") or criteria.get("window_end")
    if start is None and end is None:
        return fallback
    return TimeWindow(
        start=ensure_datetime(pd.Series([start])).iloc[0] if start is not None else fallback.start,
        end=ensure_datetime(pd.Series([end])).iloc[0] if end is not None else fallback.end,
    )


def filter_prs_to_repos_and_window(
    prs: pd.DataFrame,
    repos: pd.DataFrame,
    window: TimeWindow,
) -> pd.DataFrame:
    if prs.empty:
        return prs.copy()
    pr_repo_col = first_existing_column(prs, REPO_ID_COLUMNS)
    repo_repo_col = first_existing_column(repos, REPO_ID_COLUMNS)
    result = prs.copy()
    if pr_repo_col and repo_repo_col:
        result = result.loc[result[pr_repo_col].isin(set(repos[repo_repo_col]))].copy()

    created_col = first_existing_column(result, ("created_at", "createdAt"))
    if created_col:
        created = ensure_datetime(result[created_col])
        result = result.loc[(created >= window.start) & (created <= window.end)].copy()
    return result


def filter_by_author(df: pd.DataFrame, arm: str) -> pd.DataFrame:
    filtered, _ = filter_by_author_with_details(df, arm)
    return filtered


def filter_by_author_with_removed(df: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return filter_by_author_with_details(df, arm)


def filter_perf(df: pd.DataFrame, target_type: str) -> pd.DataFrame:
    if "task_type" not in df.columns:
        return df.iloc[0:0].copy()
    return df.loc[df["task_type"].astype(str).str.lower() == target_type.lower()].copy()


def dedupe_prs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    for column in ("html_url", "url"):
        if column in df.columns:
            return df.drop_duplicates(subset=[column], keep="first").copy()
    if {"repo_id", "number"}.issubset(df.columns):
        return df.drop_duplicates(subset=["repo_id", "number"], keep="first").copy()
    return df.drop_duplicates(keep="first").copy()


def canonical_pull_request_url(value: object) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    patterns = (
        r"https?://(?:www\.)?github\.com/([^/?#]+)/([^/?#]+)/pull/(\d+)",
        r"https?://api\.github\.com/repos/([^/?#]+)/([^/?#]+)/pulls/(\d+)",
    )
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            owner, repo, number = match.groups()
            return f"https://github.com/{owner.lower()}/{repo.lower()}/pull/{number}"
    return text.split("?", 1)[0].split("#", 1)[0].rstrip("/").lower()


def anti_join_agentic_urls(human_prs: pd.DataFrame, agentic_prs: pd.DataFrame) -> pd.DataFrame:
    if human_prs.empty or agentic_prs.empty:
        return human_prs.copy()
    url_columns = ("html_url", "url", "pull_request_url", "pr_url")
    known_urls: set[str] = set()
    for column in url_columns:
        if column in agentic_prs.columns:
            known_urls.update(
                url
                for url in agentic_prs[column].map(canonical_pull_request_url).tolist()
                if url is not None
            )
    if not known_urls:
        return human_prs.copy()

    matched = pd.Series(False, index=human_prs.index)
    for column in url_columns:
        if column in human_prs.columns:
            matched |= human_prs[column].map(canonical_pull_request_url).isin(known_urls)
    return human_prs.loc[~matched].copy()


def repo_records_for_github(repos: pd.DataFrame, limit_repos: int | None) -> list[dict]:
    repo_name_col = first_existing_column(repos, REPO_NAME_COLUMNS)
    if repo_name_col is None:
        return []
    selected = repos.head(limit_repos) if limit_repos else repos
    records = []
    for _, row in selected.iterrows():
        record = row.to_dict()
        record["repo_full_name"] = record.get(repo_name_col)
        records.append(record)
    return records


def repository_identity_hash(records: list[dict[str, Any]]) -> str:
    identities: list[dict[str, str | None]] = []
    for record in records:
        raw_repo_id = None
        for column in ("repo_id", "repository_id", "id"):
            candidate = record.get(column)
            if candidate is None:
                continue
            try:
                if bool(pd.isna(candidate)):
                    continue
            except (TypeError, ValueError):
                pass
            raw_repo_id = candidate
            break
        repo_id = None
        if raw_repo_id is not None:
            repo_id = str(json_safe(raw_repo_id))
        raw_name = record.get("repo_full_name")
        repo_name = parse_repo_full_name(raw_name)
        identities.append(
            {
                "repo_id": repo_id,
                "repo_full_name": repo_name or (str(raw_name).strip() if raw_name is not None else None),
            }
        )
    payload = json.dumps(identities, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def github_batch_size(config: dict[str, Any]) -> int:
    raw_value = config.get("github", {}).get("batch_size", 20)
    if raw_value is None:
        return 20
    batch_size = int(raw_value)
    if batch_size < 1:
        raise ValueError("github.batch_size must be at least 1.")
    return batch_size


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def build_pipeline_signature(
    config: dict[str, Any],
    window: TimeWindow,
    repo_count: int,
    limit_repos: int | None,
    repository_identity_hash: str | None = None,
) -> dict[str, Any]:
    source_cfg = config.get("source", {})
    criteria_cfg = config.get("criteria", {})
    github_cfg = config.get("github", {})
    search_limit = github_cfg.get("per_repo_search_limit")
    return {
        "source_mode": source_cfg.get("mode"),
        "aidev_dataset": source_cfg.get("aidev_dataset"),
        "aidev_revision": source_cfg.get("aidev_revision") or source_cfg.get("revision"),
        "data_root": source_cfg.get("data_root"),
        "star_floor": int(criteria_cfg.get("star_floor", 100)),
        "task_type": str(criteria_cfg.get("task_type", "perf")),
        "window": window.to_dict(),
        "repo_count": int(repo_count),
        "limit_repos": limit_repos,
        "github_enabled": bool(github_cfg.get("enabled", False)),
        "token_file": github_cfg.get("token_file"),
        "per_repo_search_limit": int(search_limit) if search_limit is not None else None,
        "repository_identity_hash": repository_identity_hash,
    }


def mine_one_repo_for_batch(
    repo_position: int,
    repo_total: int,
    repo: dict[str, Any],
    window: TimeWindow,
    token_file: str | None,
    per_repo_search_limit: int | None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    repo_name = str(repo.get("repo_full_name"))
    aggregate_report: dict[str, Any] = {"repo_reports": [], "pr_failures": []}

    def scoped_progress(message: str) -> None:
        if progress is None:
            return
        progress(message.replace("[github 1/1]", f"[github {repo_position}/{repo_total}]"))

    mined_rows = mine_human_pull_requests(
        [repo],
        window.start.date(),
        window.end.date(),
        token_file=token_file,
        per_repo_search_limit=per_repo_search_limit,
        progress=scoped_progress,
        report=aggregate_report,
    )
    repo_reports = aggregate_report.get("repo_reports", [])
    repo_report = repo_reports[0] if repo_reports else {
        "status": "completed",
        "candidate_count": 0,
        "kept_count": len(mined_rows),
        "pr_fetch_failed_count": 0,
        "error": None,
    }
    repo_report.update(
        {
            "repo_index": repo_position,
            "repo_total": repo_total,
            "repo_id": json_safe(repo.get("repo_id") or repo.get("id")),
            "repo_full_name": repo_name,
            "search_limit": per_repo_search_limit,
        }
    )
    return {
        "repo_name": repo_name,
        "repo_position": repo_position,
        "mined_rows": mined_rows,
        "repo_report": repo_report,
        "pr_failures": aggregate_report.get("pr_failures", []),
    }


def save_repo_checkpoint(
    github_human: pd.DataFrame,
    raw_path: Path,
    state_path: Path,
    resume_state: dict[str, Any],
    completed_repos: set[str],
    repo_mining_report: dict[str, Any],
    repo_name: str,
) -> dict[str, Any]:
    updated_state = update_resume_state(
        resume_state,
        completed_repos,
        len(github_human),
        last_completed_repo=repo_name if repo_name in completed_repos else None,
    )
    updated_state["repo_mining_reports"] = repo_mining_report["repo_reports"]
    updated_state["pr_failures"] = repo_mining_report["pr_failures"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(
        dir=raw_path.parent,
        prefix=f".{raw_path.name}.",
        suffix=".pending.parquet",
    )
    os.close(descriptor)
    staged_path = Path(staged_name)
    journal_path = state_path.with_name("checkpoint_pending.json")
    raw_replaced = False
    try:
        checkpoint_dataframe(github_human).to_parquet(staged_path, index=False)
        updated_state["raw_checkpoint_sha256"] = sha256_file(staged_path)
        save_resume_state(
            journal_path,
            {
                "version": 1,
                "raw_path": str(raw_path.resolve()),
                "raw_sha256": updated_state["raw_checkpoint_sha256"],
                "staged_path": str(staged_path.resolve()),
                "state": updated_state,
            },
        )
        os.replace(staged_path, raw_path)
        raw_replaced = True
        save_resume_state(state_path, updated_state)
        journal_path.unlink(missing_ok=True)
    except Exception:
        if not raw_replaced:
            staged_path.unlink(missing_ok=True)
            journal_path.unlink(missing_ok=True)
        raise
    return updated_state


def merge_repo_mining_rows(
    existing: pd.DataFrame,
    mined_rows: list[dict[str, Any]],
    repo_name: str,
    status: str,
) -> pd.DataFrame:
    if existing.empty:
        return pd.DataFrame(mined_rows) if mined_rows else existing.copy()
    if "repo_full_name" not in existing.columns:
        return dedupe_prs(
            pd.concat([existing, pd.DataFrame(mined_rows)], ignore_index=True, sort=False)
        )

    existing_repo_names = existing["repo_full_name"].map(parse_repo_full_name)
    if mined_rows and "number" in existing.columns:
        fetched_numbers = {
            str(row["number"])
            for row in mined_rows
            if row.get("number") is not None
        }
        same_fetched_pr = (existing_repo_names == repo_name) & existing["number"].map(str).isin(
            fetched_numbers
        )
        base = existing.loc[~same_fetched_pr].copy()
    else:
        base = existing.copy()

    if not mined_rows:
        return base.reset_index(drop=True)
    return pd.concat([base, pd.DataFrame(mined_rows)], ignore_index=True, sort=False)


def enrich_dataframe_with_github(
    df: pd.DataFrame,
    token_file: str | None,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 10,
    report: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    enriched = enrich_pull_request_records(
        df.to_dict(orient="records"),
        token_file=token_file,
        progress=progress,
        progress_every=progress_every,
        report=report,
    )
    if not enriched:
        return df.iloc[0:0].copy()
    return pd.DataFrame(enriched)


def build_summary(
    window: TimeWindow,
    repo_count: int,
    agent_before_quality_count: int,
    agent_after_quality_count: int,
    agent_perf_count: int,
    human_before_quality_count: int,
    human_after_quality_count: int,
    task_type_sources: dict[str, int],
    removed_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "time_window": window.to_dict(),
        "repositories_at_or_above_star_floor": repo_count,
        "agent_perf_prs_before_quality_filters": agent_before_quality_count,
        "agent_perf_prs_after_quality_filters": agent_after_quality_count,
        "agent_perf_prs_matched_criteria": agent_perf_count,
        "human_perf_prs_before_quality_filters": human_before_quality_count,
        "human_perf_prs_after_quality_filters": human_after_quality_count,
        "human_task_type_sources": task_type_sources,
        "quality_filter_removed_counts": removed_counts,
        "methodological_note": "same inclusion criteria, no imposed target count",
    }


def describe_removed_rows(df: pd.DataFrame, reason: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if df.empty:
        return records
    for _, row in df.iterrows():
        record: dict[str, Any] = {"reason": reason}
        for column in (
            "repo_full_name",
            "repo_url",
            "html_url",
            "url",
            "number",
            "id",
            "title",
            "task_type",
            "task_type_source",
            "user",
            "user_type",
        ):
            if column in row and row[column] is not None and not pd.isna(row[column]):
                value = row[column]
                if isinstance(value, (list, tuple)):
                    record[column] = [str(json_safe(item)) for item in value]
                else:
                    record[column] = str(value) if column == "title" else json_safe(value)
        if "filenames" in row:
            filenames = row["filenames"]
            if isinstance(filenames, str):
                filenames = [part.strip() for part in filenames.split(",") if part.strip()]
            elif isinstance(filenames, tuple):
                filenames = list(filenames)
            if isinstance(filenames, list) and filenames:
                record["filenames"] = [str(json_safe(item)) for item in filenames]
        records.append(record)
    return records


def build_detailed_report(
    summary: dict[str, Any],
    source_counts: dict[str, int],
    stage_counts: dict[str, int],
    author_filter_removed_counts: dict[str, int],
    author_filter_removed_records: dict[str, list[dict[str, Any]]],
    repo_mining_report: dict[str, Any],
    enrichment_reports: dict[str, dict[str, Any]],
    quality_filter_removed_records: dict[str, dict[str, list[dict[str, Any]]]],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    repo_reports = repo_mining_report.get("repo_reports", [])
    pr_failures = repo_mining_report.get("pr_failures", [])
    repo_failures = [
        report
        for report in repo_reports
        if report.get("status") != "completed"
    ]

    detailed = {
        "methodological_note": summary["methodological_note"],
        "time_window": summary["time_window"],
        "repositories_at_or_above_star_floor": summary["repositories_at_or_above_star_floor"],
        "source_counts": source_counts,
        "stage_counts": stage_counts,
        "summary_counts": {
            "agent_perf_prs": summary["agent_perf_prs_matched_criteria"],
            "human_perf_prs": stage_counts["human_perf_after_quality"],
            "total_perf_prs": stage_counts["agent_perf_after_quality"] + stage_counts["human_perf_after_quality"],
            "agent_perf_prs_before_quality_filters": summary["agent_perf_prs_before_quality_filters"],
            "agent_perf_prs_after_quality_filters": summary["agent_perf_prs_after_quality_filters"],
            "human_perf_prs_before_quality_filters": summary["human_perf_prs_before_quality_filters"],
            "human_perf_prs_after_quality_filters": summary["human_perf_prs_after_quality_filters"],
            "raw_github_human_prs": stage_counts["raw_github_human_prs"],
            "repo_mining_failures": len(repo_failures),
            "pr_fetch_failures": len(pr_failures),
        },
        "author_filter": {
            "removed_counts": author_filter_removed_counts,
            "removed_records": author_filter_removed_records,
        },
        "repo_mining": {
            "attempted": len(repo_reports),
            "completed": sum(1 for report in repo_reports if report.get("status") == "completed"),
            "failed": len(repo_failures),
            "reports": repo_reports,
        },
        "pr_fetch_failures": pr_failures,
        "enrichment_reports": enrichment_reports,
        "quality_filter_removed_records": quality_filter_removed_records,
        "files": output_paths,
    }
    return detailed


def write_detailed_report_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Rebalancing Detailed Report",
        "",
        f"- Time window: `{report['time_window']['start']}` to `{report['time_window']['end']}`",
        f"- Repositories >= star floor: `{report['repositories_at_or_above_star_floor']}`",
        "",
        "## Totals",
    ]
    counts = report.get("summary_counts", {})
    lines.extend(
        [
            f"- AI perf PRs: `{counts.get('agent_perf_prs', 0)}`",
            f"- Human perf PRs: `{counts.get('human_perf_prs', 0)}`",
            f"- Total perf PRs: `{counts.get('total_perf_prs', 0)}`",
            f"- Raw GitHub human PRs: `{counts.get('raw_github_human_prs', 0)}`",
            f"- Repo mining failures: `{counts.get('repo_mining_failures', 0)}`",
            f"- PR fetch failures: `{counts.get('pr_fetch_failures', 0)}`",
        ]
    )
    lines.append("")
    lines.append("## Repository Mining")
    repo_mining = report.get("repo_mining", {})
    lines.append(f"- Attempted repos: `{repo_mining.get('attempted', 0)}`")
    lines.append(f"- Completed repos: `{repo_mining.get('completed', 0)}`")
    lines.append(f"- Failed repos: `{repo_mining.get('failed', 0)}`")
    failed_repos = [item for item in repo_mining.get("reports", []) if item.get("status") != "completed"]
    if failed_repos:
        lines.append("")
        lines.append("Failed repos")
        for item in failed_repos:
            repo_name = item.get("repo_full_name") or "missing-repo-name"
            status = item.get("status")
            error = item.get("error")
            suffix = f" - {error}" if error else ""
            lines.append(f"- `{repo_name}`: `{status}`{suffix}")
    lines.append("")
    lines.append("## Author Filter")
    author_filter = report.get("author_filter", {})
    removed_counts = author_filter.get("removed_counts", {})
    lines.append(f"- Agent-arm removals: `{removed_counts.get('agent', 0)}`")
    lines.append(f"- Human-arm removals: `{removed_counts.get('human', 0)}`")
    lines.append("")
    lines.append("## Quality Filters")
    for arm, filters in report.get("quality_filter_removed_records", {}).items():
        lines.append(f"### {arm}")
        for reason, records in filters.items():
            lines.append(f"- `{reason}`: `{len(records)}`")
    lines.append("")
    lines.append("Full row-level details are stored in the JSON report.")
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Rebalancing Summary",
        "",
        f"- Time window: `{summary['time_window']['start']}` to `{summary['time_window']['end']}`",
        f"- Repositories >= star floor: `{summary['repositories_at_or_above_star_floor']}`",
        f"- AI perf PRs: `{summary['agent_perf_prs_matched_criteria']}`",
        f"- Human perf PRs: `{summary['human_perf_prs_before_quality_filters']}` before quality / `{summary['human_perf_prs_after_quality_filters']}` after quality",
        f"- Repo mining failures: `{summary.get('repo_mining_failures', 0)}`",
        f"- PR fetch failures: `{summary.get('pr_fetch_failures', 0)}`",
        "",
        "Source counts:",
    ]
    for source, count in summary.get("source_counts", {}).items():
        lines.append(f"- `{source}`: `{count}`")
    lines.extend(
        [
            "",
            "Author filter removals:",
        ]
    )
    for arm, count in summary.get("author_filter_removed_counts", {}).items():
        lines.append(f"- `{arm}`: `{count}`")
    lines.extend(
        [
            "",
            "Quality filter removals:",
        ]
    )
    for reason, count in summary["quality_filter_removed_counts"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(
        [
            "",
            "Agent quality filter removals:",
        ]
    )
    for reason, count in summary.get("agent_quality_filter_removed_counts", {}).items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(
        [
            "",
            "Human quality filter removals:",
        ]
    )
    for reason, count in summary.get("human_quality_filter_removed_counts", {}).items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(
        [
            "",
            "Human task type sources:",
        ]
    )
    for source, count in summary["human_task_type_sources"].items():
        lines.append(f"- `{source}`: `{count}`")
    lines.extend(
        [
            "",
            f"Detailed report: `{summary.get('detailed_report_json', 'n/a')}`",
            "",
            f"Methodological note: {summary['methodological_note']}.",
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")


def recover_pending_repo_checkpoint(raw_path: Path, state_path: Path) -> None:
    journal_path = state_path.with_name("checkpoint_pending.json")
    if not journal_path.is_file():
        return
    journal = load_resume_state(journal_path)
    expected_raw_path = journal.get("raw_path")
    expected_hash = journal.get("raw_sha256")
    pending_state = journal.get("state")
    if (
        expected_raw_path != str(raw_path.resolve())
        or not isinstance(expected_hash, str)
        or not isinstance(pending_state, dict)
    ):
        raise ValueError(f"Invalid pending checkpoint journal: {journal_path}")

    if raw_path.is_file() and sha256_file(raw_path) == expected_hash:
        pending_state["raw_checkpoint_sha256"] = expected_hash
        save_resume_state(state_path, pending_state)
        log("[resume] recovered state for an atomically written raw checkpoint")
    else:
        log("[resume] discarded an unfinished checkpoint transaction before raw replacement")

    staged_value = journal.get("staged_path")
    if isinstance(staged_value, str):
        staged_path = Path(staged_value)
        if staged_path.parent == raw_path.resolve().parent:
            staged_path.unlink(missing_ok=True)
    journal_path.unlink(missing_ok=True)


def cleanup_orphaned_staged_checkpoints(raw_path: Path) -> None:
    pattern = f".{raw_path.name}.*.pending.parquet"
    for staged_path in raw_path.parent.glob(pattern):
        staged_path.unlink(missing_ok=True)
        log(f"[resume] removed orphaned staged checkpoint {staged_path.name}")


def prepare_resume_checkpoint(
    output_dir: Path,
    output_dirs: dict[str, Path],
    resume_cfg: dict[str, Any],
    signature: dict[str, Any],
    repo_count: int,
) -> tuple[pd.DataFrame, dict[str, Any], set[str], Path]:
    state_path = resume_state_path(output_dir)
    raw_path = output_dirs["raw"] / "github_human_prs.parquet"
    reset_requested = bool(resume_cfg.get("reset", False))

    if reset_requested:
        log("[resume] reset requested; clearing previous outputs")
        reset_output_tree(output_dir)
        output_dirs.update(ensure_output_dirs(output_dir))

    recover_pending_repo_checkpoint(raw_path, state_path)
    cleanup_orphaned_staged_checkpoints(raw_path)
    raw_checkpoint_exists = raw_path.exists()
    state = load_resume_state(state_path)
    raw_df = load_checkpoint_dataframe(raw_path)

    if not raw_df.empty:
        log(f"[resume] raw checkpoint found: {len(raw_df)} rows")

    if state and not state_matches_signature(state, signature):
        raise ValueError(
            "Existing resume state does not match the current configuration. "
            "Set resume.reset: true to start from scratch."
        )

    recorded_raw_hash = state.get("raw_checkpoint_sha256") if state else None
    if raw_checkpoint_exists and state:
        actual_raw_hash = sha256_file(raw_path)
        if recorded_raw_hash and actual_raw_hash != recorded_raw_hash:
            raise ValueError(
                f"Raw checkpoint integrity mismatch for {raw_path}; "
                "restore the matching checkpoint or reset the run."
            )
        if not recorded_raw_hash:
            legacy_completed = completed_repo_set(state)
            if legacy_completed and not bool(
                resume_cfg.get("trust_legacy_checkpoint", False)
            ):
                raise ValueError(
                    "Existing resume state predates raw-checkpoint hashes and marks "
                    f"{len(legacy_completed)} repositories complete. Verify the raw parquet, "
                    "then set resume.trust_legacy_checkpoint: true once to adopt it."
                )
            state["version"] = 2
            state["raw_checkpoint_sha256"] = actual_raw_hash
            save_resume_state(state_path, state)
            log("[resume] adopted legacy raw checkpoint with a SHA-256 integrity hash")

    if state:
        state.setdefault("repo_mining_reports", [])
        state.setdefault("pr_failures", [])

    if not state and raw_checkpoint_exists:
        raise ValueError(
            f"Raw checkpoint exists without its resume state: {raw_path}. "
            "Restore the matching state file or move the orphan checkpoint before starting a new run."
        )

    if not state:
        state = initial_resume_state(signature, repo_count)
        completed: set[str] = set()
        log("[resume] no checkpoint found; starting fresh")
        save_resume_state(state_path, state)
    else:
        completed = completed_repo_set(state)
        state = update_resume_state(state, completed, len(raw_df))
        save_resume_state(state_path, state)

    if not raw_checkpoint_exists and completed:
        log("[resume] state exists but raw checkpoint is missing; starting from scratch")
        state = initial_resume_state(signature, repo_count)
        completed = set()
        save_resume_state(state_path, state)

    if completed and raw_df.empty:
        log(f"[resume] loaded checkpoint: {len(completed)} completed repos, {max(repo_count - len(completed), 0)} remaining")
    elif completed and not raw_df.empty:
        log(f"[resume] loaded checkpoint: {len(completed)} completed repos, {max(repo_count - len(completed), 0)} remaining")

    return raw_df, state, completed, state_path


def run_pipeline(config: dict[str, Any], limit_repos: int | None = None) -> dict[str, Any]:
    criteria = config.get("criteria", {})
    star_floor = int(criteria.get("star_floor", 100))
    target_type = str(criteria.get("task_type", "perf"))
    output_dir = Path(config.get("outputs", {}).get("dir", "mining/outputs"))
    output_dirs = ensure_output_dirs(output_dir)
    github_cfg = config.get("github", {})
    if not github_cfg.get("enabled", False):
        raise ValueError("Real mining requires github.enabled: true.")
    token_file = github_cfg.get("token_file")
    batch_size = github_batch_size(config)
    resume_cfg = config.get("resume", {})
    resume_enabled = bool(resume_cfg.get("enabled", True))

    log("[1/6] Loading AIDev tables")
    tables = load_aidev_tables(config)
    log(
        f"  loaded pull_request={len(tables['pull_request'])}, repository={len(tables['repository'])}, "
        f"human_pull_request={len(tables['human_pull_request'])}"
    )

    log(f"[2/6] Filtering repositories with stars >= {star_floor}")
    repos = ensure_repo_full_names(filter_repositories(tables["repository"], star_floor), tables["pull_request"])
    log(f"  repositories kept: {len(repos)}")
    window = configured_time_window(config, derive_time_window(tables["pull_request"]))
    agent_prs = filter_prs_to_repos_and_window(tables["pull_request"], repos, window)
    records = repo_records_for_github(repos, limit_repos)
    log(f"  time window: {window.start.isoformat()} -> {window.end.isoformat()}")
    signature = build_pipeline_signature(
        config,
        window,
        len(repos),
        limit_repos,
        repository_identity_hash=repository_identity_hash(records),
    )

    log("[3/6] Preparing AI arm")
    agent_typed = attach_task_type(agent_prs, tables.get("pr_task_type"))
    agent_authored, agent_author_removed = filter_by_author_with_removed(agent_typed, "agent")
    agent_perf_candidates = filter_perf(agent_authored, target_type)
    log(f"  typed AI PRs: {len(agent_typed)}")
    log(f"  agent-authored AI PRs: {len(agent_authored)}")
    log(f"  agent-arm removals: {len(agent_author_removed)}")
    log(f"  AI perf PRs before GitHub enrichment: {len(agent_perf_candidates)}")
    agent_enrichment_report: dict[str, Any] = {"arm": "agent"}
    agent_perf = enrich_dataframe_with_github(
        agent_perf_candidates,
        token_file,
        progress=log,
        progress_every=25,
        report=agent_enrichment_report,
    )
    agent_perf_before_quality = len(agent_perf)
    agent_perf, agent_quality_removed_counts, agent_quality_removed_records = apply_quality_filters_with_details(agent_perf)
    agent_removed = agent_quality_removed_counts
    log(f"  AI perf PRs after quality filters: {len(agent_perf)}")
    log(f"  AI quality removals: {agent_removed}")

    raw_path = output_dirs["raw"] / "github_human_prs.parquet"
    state_path = resume_state_path(output_dir)
    if resume_enabled:
        log("[4/6] Preparing resume checkpoint")
        github_human, resume_state, completed_repos, state_path = prepare_resume_checkpoint(
            output_dir=output_dir,
            output_dirs=output_dirs,
            resume_cfg=resume_cfg,
            signature=signature,
            repo_count=len(records),
        )
    else:
        log("[4/6] Mining human PRs from GitHub")
        github_human = pd.DataFrame()
        resume_state = initial_resume_state(signature, len(records))
        completed_repos = set()

    repo_mining_report: dict[str, Any] = {
        "repo_reports": list(resume_state.get("repo_mining_reports", [])),
        "pr_failures": list(resume_state.get("pr_failures", [])),
    }

    log(f"  repos queued for GitHub mining: {len(records)}")
    if resume_enabled:
        log(f"  repos already completed: {len(completed_repos)}")
        log(f"  repos remaining: {max(len(records) - len(completed_repos), 0)}")

    pending_items = [
        (repo_position, record)
        for repo_position, record in enumerate(records, start=1)
        if record["repo_full_name"] not in completed_repos
    ]
    batches = chunked(pending_items, batch_size)
    if not pending_items:
        log("  nothing left to mine; rebuilding outputs from checkpoint")
    else:
        log(f"  GitHub batch size: {batch_size}")
        log(f"  GitHub batches queued: {len(batches)}")

    for batch_index, batch in enumerate(batches, start=1):
        if not batch:
            continue
        batch_start = batch[0][0]
        batch_end = batch[-1][0]
        workers = min(batch_size, len(batch))
        log(f"[batch {batch_index}/{len(batches)}] starting repos {batch_start}-{batch_end}/{len(records)} | workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    mine_one_repo_for_batch,
                    repo_position,
                    len(records),
                    repo,
                    window,
                    token_file,
                    github_cfg.get("per_repo_search_limit"),
                    log,
                ): (repo_position, repo)
                for repo_position, repo in batch
            }
            for future in as_completed(future_map):
                repo_position, repo = future_map[future]
                repo_name = str(repo.get("repo_full_name"))
                try:
                    result = future.result()
                except GitHubRateLimitError:
                    log(f"[github {repo_position}/{len(records)}] {repo_name} stopped because all configured tokens are rate limited")
                    raise

                mined_rows = result["mined_rows"]
                repo_report = result["repo_report"]
                result_repo_name = result["repo_name"]
                status = str(repo_report.get("status"))
                github_human = merge_repo_mining_rows(
                    github_human,
                    mined_rows,
                    result_repo_name,
                    status,
                )
                github_human = checkpoint_dataframe(github_human)
                repo_mining_report["repo_reports"] = [
                    item
                    for item in repo_mining_report["repo_reports"]
                    if str(item.get("repo_full_name")) != result_repo_name
                ]
                repo_mining_report["pr_failures"] = [
                    item
                    for item in repo_mining_report["pr_failures"]
                    if str(item.get("repo_full_name")) != result_repo_name
                ]
                repo_mining_report["repo_reports"].append(repo_report)
                repo_mining_report["pr_failures"].extend(result.get("pr_failures", []))
                if status == "completed":
                    completed_repos.add(result_repo_name)
                else:
                    completed_repos.discard(result_repo_name)
                log(
                    f"[github {repo_report.get('repo_index', repo_position)}/{len(records)}] "
                    f"{result_repo_name} done | status={status} | kept={repo_report.get('kept_count', 0)}"
                )
                if resume_enabled:
                    resume_state = save_repo_checkpoint(
                        github_human,
                        raw_path,
                        state_path,
                        resume_state,
                        completed_repos,
                        repo_mining_report,
                        result_repo_name,
                    )
                    log(
                        f"[resume] checkpoint saved after {result_repo_name}: "
                        f"completed={len(completed_repos)}/{len(records)} raw_rows={len(github_human)}"
                    )

    if not resume_enabled or not raw_path.exists():
        save_checkpoint_dataframe(github_human, raw_path)
    log(f"  raw GitHub human PRs available: {len(github_human)}")

    log("[5/6] Joining and typing human PRs")
    existing_human = tables.get("human_pull_request", pd.DataFrame())
    human_candidates = dedupe_prs(pd.concat([existing_human, github_human], ignore_index=True, sort=False))
    before_agentic_url_anti_join = len(human_candidates)
    human_candidates = anti_join_agentic_urls(human_candidates, tables["pull_request"])
    human_agentic_url_removed = before_agentic_url_anti_join - len(human_candidates)
    log(f"  known AIDev agentic URLs removed from human arm: {human_agentic_url_removed}")
    human_candidates = filter_prs_to_repos_and_window(human_candidates, repos, window)
    human_candidates_before_author = len(human_candidates)
    human_candidates, human_author_removed = filter_by_author_with_removed(human_candidates, "human")
    log(f"  human candidates after repo/window/author filters: {len(human_candidates)}")
    log(f"  human-arm removals: {len(human_author_removed)}")
    human_typed = attach_task_type(human_candidates, tables.get("human_pr_task_type"))
    log(f"  human typed rows: {len(human_typed)}")
    write_parquet(human_typed, output_dirs["intermediate"] / "human_prs_task_typed.parquet")

    human_perf_before_quality = filter_perf(human_typed, target_type)
    log(f"  human perf PRs before GitHub enrichment: {len(human_perf_before_quality)}")
    human_enrichment_report: dict[str, Any] = {"arm": "human"}
    human_perf_before_quality = enrich_dataframe_with_github(
        human_perf_before_quality,
        token_file,
        progress=log,
        progress_every=25,
        report=human_enrichment_report,
    )
    human_perf_before_quality_count = len(human_perf_before_quality)
    log(f"  human perf PRs before quality filters: {human_perf_before_quality_count}")
    human_perf, human_quality_removed_counts, human_quality_removed_records = apply_quality_filters_with_details(
        human_perf_before_quality
    )
    human_removed = human_quality_removed_counts
    log(f"  human perf PRs after quality filters: {len(human_perf)}")
    log(f"  human quality removals: {human_removed}")
    write_parquet(human_perf, output_dirs["intermediate"] / "human_prs_filtered.parquet")

    log("[6/6] Writing final outputs")
    write_parquet(agent_perf, output_dirs["final"] / "agent_perf_prs_matched_criteria.parquet")
    write_parquet(human_perf, output_dirs["final"] / "human_perf_prs_matched_criteria.parquet")

    agent_author_removed_records = describe_removed_rows(agent_author_removed, "non_agent_authored")
    human_author_removed_records = describe_removed_rows(human_author_removed, "agent_authored")
    source_counts = {
        "pull_request": int(len(tables.get("pull_request", pd.DataFrame()))),
        "repository": int(len(tables.get("repository", pd.DataFrame()))),
        "human_pull_request": int(len(tables.get("human_pull_request", pd.DataFrame()))),
        "pr_task_type": int(len(tables.get("pr_task_type", pd.DataFrame()))),
        "human_pr_task_type": int(len(tables.get("human_pr_task_type", pd.DataFrame()))),
    }
    stage_counts = {
        "repositories_after_star_floor": int(len(repos)),
        "agent_typed": int(len(agent_typed)),
        "agent_author_kept": int(len(agent_authored)),
        "agent_author_removed": int(len(agent_author_removed)),
        "agent_perf_before_enrichment": int(len(agent_perf_candidates)),
        "agent_perf_after_enrichment": int(agent_perf_before_quality),
        "agent_perf_after_quality": int(len(agent_perf)),
        "human_agentic_url_removed": int(human_agentic_url_removed),
        "human_candidates_before_author": int(human_candidates_before_author),
        "human_author_kept": int(len(human_candidates)),
        "human_author_removed": int(len(human_author_removed)),
        "human_typed": int(len(human_typed)),
        "human_perf_before_enrichment": int(len(filter_perf(human_typed, target_type))),
        "human_perf_after_enrichment": int(human_perf_before_quality_count),
        "human_perf_after_quality": int(len(human_perf)),
        "raw_github_human_prs": int(len(github_human)),
    }
    author_filter_removed_counts = {
        "agent": int(len(agent_author_removed)),
        "human": int(len(human_author_removed)),
    }
    quality_filter_removed_counts = {
        reason: int(agent_quality_removed_counts.get(reason, 0) + human_quality_removed_counts.get(reason, 0))
        for reason in sorted(set(agent_quality_removed_counts) | set(human_quality_removed_counts))
    }
    combined_quality_removed_records = {
        "agent": agent_quality_removed_records,
        "human": human_quality_removed_records,
    }
    enrichment_reports = {
        "agent": agent_enrichment_report,
        "human": human_enrichment_report,
    }

    task_type_sources = (
        human_typed.get("task_type_source", pd.Series(dtype=str)).fillna("missing").value_counts().to_dict()
    )
    summary = build_summary(
        window=window,
        repo_count=len(repos),
        agent_before_quality_count=agent_perf_before_quality,
        agent_after_quality_count=len(agent_perf),
        agent_perf_count=len(agent_perf),
        human_before_quality_count=human_perf_before_quality_count,
        human_after_quality_count=len(human_perf),
        task_type_sources={str(key): int(value) for key, value in task_type_sources.items()},
        removed_counts={str(key): int(value) for key, value in quality_filter_removed_counts.items()},
    )
    detailed_report = build_detailed_report(
        summary=summary,
        source_counts=source_counts,
        stage_counts=stage_counts,
        author_filter_removed_counts=author_filter_removed_counts,
        author_filter_removed_records={
            "agent": agent_author_removed_records,
            "human": human_author_removed_records,
        },
        repo_mining_report=repo_mining_report,
        enrichment_reports=enrichment_reports,
        quality_filter_removed_records=combined_quality_removed_records,
        output_paths={
            "agent_perf_prs": str(output_dirs["final"] / "agent_perf_prs_matched_criteria.parquet"),
            "human_perf_prs": str(output_dirs["final"] / "human_perf_prs_matched_criteria.parquet"),
            "human_task_typed": str(output_dirs["intermediate"] / "human_prs_task_typed.parquet"),
            "human_filtered": str(output_dirs["intermediate"] / "human_prs_filtered.parquet"),
            "raw_github_human_prs": str(raw_path),
            "summary_json": str(output_dirs["final"] / "rebalancing_summary.json"),
            "summary_md": str(output_dirs["final"] / "rebalancing_summary.md"),
            "detailed_report_json": str(output_dirs["final"] / "rebalancing_detailed_report.json"),
            "detailed_report_md": str(output_dirs["final"] / "rebalancing_detailed_report.md"),
        },
    )
    summary.update(
        {
            "source_counts": source_counts,
            "stage_counts": stage_counts,
            "author_filter_removed_counts": author_filter_removed_counts,
            "agent_quality_filter_removed_counts": {str(k): int(v) for k, v in agent_quality_removed_counts.items()},
            "human_quality_filter_removed_counts": {str(k): int(v) for k, v in human_quality_removed_counts.items()},
            "repo_mining_failures": int(detailed_report["repo_mining"]["failed"]),
            "pr_fetch_failures": int(len(detailed_report["pr_fetch_failures"])),
            "detailed_report_json": str(output_dirs["final"] / "rebalancing_detailed_report.json"),
            "detailed_report_md": str(output_dirs["final"] / "rebalancing_detailed_report.md"),
        }
    )
    summary_json = output_dirs["final"] / "rebalancing_summary.json"
    atomic_write_text(summary_json, json.dumps(summary, indent=2) + "\n")
    write_summary_markdown(summary, output_dirs["final"] / "rebalancing_summary.md")
    detailed_json = output_dirs["final"] / "rebalancing_detailed_report.json"
    atomic_write_text(detailed_json, json.dumps(detailed_report, indent=2) + "\n")
    write_detailed_report_markdown(detailed_report, output_dirs["final"] / "rebalancing_detailed_report.md")
    log(f"Done. Summary written to {summary_json}")
    log(f"Done. Detailed report written to {detailed_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine and filter pull requests from GitHub.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit-repos", type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    load_env_file(config_path.parent / ".env")
    load_env_file(config_path.parent / ".env.local")
    summary = run_pipeline(load_config(config_path), limit_repos=args.limit_repos)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
