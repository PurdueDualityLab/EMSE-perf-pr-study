from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
from typing import Any, Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
from pydriller import Git

from github_client import parse_repo_full_name
from parquet_parts import (
    finalize_parts,
    part_path,
    schema_sha256,
    sha256_file,
    validate_part,
    write_json,
    write_part,
)
from perfminer_reproduction import (
    PERFMINER_MAX_DIFF_CHARS,
    PERFMINER_MAX_SOURCE_BEFORE_CHARS,
    PERFMINER_REVISION,
    PERFMINER_SUPPORTED_EXTENSIONS,
    commit_manifest_sha256,
    inspect_commit,
)


METHOD_NAME = "perfminer_operational_evidence_v1"
METHOD_VERSION = 1

EVIDENCE_SCHEMA = pa.schema(
    [
        pa.field("repo_id", pa.int64()),
        pa.field("source_repo_full_name", pa.string()),
        pa.field("resolved_repo_full_name", pa.string()),
        pa.field("pr_number", pa.int64()),
        pa.field("pr_html_url", pa.string()),
        pa.field("pr_base_sha", pa.string()),
        pa.field("pr_head_sha", pa.string()),
        pa.field("commit_manifest_sha256", pa.string()),
        pa.field("pr_manifest_complete", pa.bool_()),
        pa.field("commit_index", pa.int64()),
        pa.field("commit_sha", pa.string()),
        pa.field("commit_message_api", pa.large_string()),
        pa.field("parent_count_api", pa.int64()),
        pa.field("commit_html_url", pa.string()),
        pa.field("api_message_matches_git", pa.bool_()),
        pa.field("api_parent_count_matches_git", pa.bool_()),
        pa.field("commit_message_raw", pa.large_string()),
        pa.field("commit_message", pa.large_string()),
        pa.field("commit_message_word_count", pa.int64()),
        pa.field("commit_date", pa.string()),
        pa.field("parent_count", pa.int64()),
        pa.field("changed_file_count", pa.int64()),
        pa.field("filename", pa.string()),
        pa.field("change_type", pa.string()),
        pa.field("is_binary", pa.bool_()),
        pa.field("changed_method_count", pa.int64()),
        pa.field("changed_method_name", pa.string()),
        pa.field("changed_method_start_line", pa.int64()),
        pa.field("changed_method_end_line", pa.int64()),
        pa.field("code_diff", pa.large_string()),
        pa.field("diff_chars", pa.int64()),
        pa.field("source_before_chars", pa.int64()),
        pa.field("source_after_chars", pa.int64()),
        pa.field("source_pair_md5", pa.string()),
        pa.field("perfminer_evidence_status", pa.string()),
        pa.field("perfminer_exclusion_reason", pa.string()),
        pa.field("perfminer_evidence_error", pa.string()),
    ]
)


@dataclass(frozen=True)
class CommitTarget:
    repo_id: int
    source_repo_full_name: str
    resolved_repo_full_name: str
    pr_number: int
    pr_html_url: str
    pr_base_sha: str
    pr_head_sha: str
    commit_manifest_sha256: str
    pr_manifest_complete: bool
    commit_index: int
    commit_sha: str
    commit_message_api: str
    parent_count_api: int
    commit_html_url: str


@dataclass
class RepositoryTarget:
    repo_id: int
    resolved_repo_full_name: str
    commits: list[CommitTarget] = field(default_factory=list)
    checkpoint_index: int = 0

    @property
    def pull_request_numbers(self) -> list[int]:
        return sorted({commit.pr_number for commit in self.commits})

    @property
    def state_key(self) -> str:
        return f"{self.repo_id}:{self.checkpoint_index}"


class CommitInspectionTimeout(TimeoutError):
    pass


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def collect_targets(
    commits_path: Path,
    pull_requests_path: Path,
) -> dict[int, RepositoryTarget]:
    pr_required = {
        "repo_id",
        "resolved_repo_full_name",
        "pr_number",
        "pr_base_sha",
        "pr_head_sha",
        "commit_manifest_sha256",
        "expected_commit_count",
        "listed_commit_count",
        "commit_manifest_status",
    }
    pr_source = pq.ParquetFile(pull_requests_path)
    pr_missing = sorted(pr_required - set(pr_source.schema_arrow.names))
    if pr_missing:
        raise ValueError("PR manifest is missing columns: " + ", ".join(pr_missing))
    pr_snapshots: dict[tuple[int, int], dict[str, object]] = {}
    for batch in pr_source.iter_batches(batch_size=50_000, columns=sorted(pr_required)):
        for row in pa.Table.from_batches([batch]).to_pylist():
            key = (int(row["repo_id"]), int(row["pr_number"]))
            if key in pr_snapshots:
                raise ValueError(f"PR manifest contains duplicate row: {key}")
            pr_snapshots[key] = row

    required = {
        "repo_id",
        "source_repo_full_name",
        "resolved_repo_full_name",
        "pr_number",
        "pr_html_url",
        "pr_base_sha",
        "pr_head_sha",
        "commit_manifest_sha256",
        "pr_manifest_complete",
        "commit_index",
        "commit_sha",
        "commit_message_api",
        "parent_count_api",
        "commit_html_url",
    }
    source = pq.ParquetFile(commits_path)
    missing = sorted(required - set(source.schema_arrow.names))
    if missing:
        raise ValueError("Commit manifest is missing columns: " + ", ".join(missing))

    targets: dict[int, RepositoryTarget] = {}
    seen_rows: set[tuple[int, int, str]] = set()
    commit_pairs: dict[tuple[int, int], list[tuple[int, str]]] = {}
    for batch in source.iter_batches(batch_size=50_000, columns=sorted(required)):
        for row in pa.Table.from_batches([batch]).to_pylist():
            repo_id = int(row["repo_id"])
            resolved_name = parse_repo_full_name(row["resolved_repo_full_name"])
            sha = str(row["commit_sha"] or "").strip()
            if repo_id <= 0 or not resolved_name or not sha:
                raise ValueError("Commit manifest contains an invalid repository or commit identity.")
            target = targets.setdefault(repo_id, RepositoryTarget(repo_id, resolved_name))
            if target.resolved_repo_full_name.casefold() != resolved_name.casefold():
                raise ValueError(f"Repository {repo_id} has conflicting resolved names.")
            key = (repo_id, int(row["pr_number"]), sha)
            if key in seen_rows:
                raise ValueError(f"Commit manifest contains duplicate PR/commit row: {key}")
            seen_rows.add(key)
            pr_key = key[:2]
            snapshot = pr_snapshots.get(pr_key)
            if snapshot is None:
                raise ValueError(f"Commit manifest references PR absent from PR manifest: {pr_key}")
            if str(snapshot["commit_manifest_status"] or "") != "complete":
                raise ValueError(f"Incomplete PR manifest unexpectedly contains commit rows: {pr_key}")
            for column in (
                "resolved_repo_full_name",
                "pr_base_sha",
                "pr_head_sha",
                "commit_manifest_sha256",
            ):
                if str(row[column] or "") != str(snapshot[column] or ""):
                    raise ValueError(f"Commit and PR manifests disagree on {column}: {pr_key}")
            if not bool(row["pr_manifest_complete"]):
                raise ValueError(f"Commit row is not bound to a complete PR manifest: {key}")
            commit_pairs.setdefault(pr_key, []).append((int(row["commit_index"]), sha))
            target.commits.append(
                CommitTarget(
                    repo_id=repo_id,
                    source_repo_full_name=str(row["source_repo_full_name"] or ""),
                    resolved_repo_full_name=resolved_name,
                    pr_number=int(row["pr_number"]),
                    pr_html_url=str(row["pr_html_url"] or ""),
                    pr_base_sha=str(row["pr_base_sha"] or ""),
                    pr_head_sha=str(row["pr_head_sha"] or ""),
                    commit_manifest_sha256=str(row["commit_manifest_sha256"] or ""),
                    pr_manifest_complete=bool(row["pr_manifest_complete"]),
                    commit_index=int(row["commit_index"]),
                    commit_sha=sha,
                    commit_message_api=str(row["commit_message_api"] or ""),
                    parent_count_api=int(row["parent_count_api"]),
                    commit_html_url=str(row["commit_html_url"] or ""),
                )
            )
    for key, snapshot in pr_snapshots.items():
        pairs = sorted(commit_pairs.get(key, []))
        if str(snapshot["commit_manifest_status"] or "") != "complete":
            if pairs:
                raise ValueError(f"Incomplete PR manifest unexpectedly has commit rows: {key}")
            continue
        listed_count = int(snapshot["listed_commit_count"] or 0)
        expected_count = int(snapshot["expected_commit_count"] or 0)
        if len(pairs) != listed_count or listed_count != expected_count:
            raise ValueError(f"Commit and PR manifest counts disagree: {key}")
        if [index for index, _ in pairs] != list(range(1, listed_count + 1)):
            raise ValueError(f"Commit manifest indices are not contiguous: {key}")
        digest = commit_manifest_sha256(pairs)
        if digest != str(snapshot["commit_manifest_sha256"] or ""):
            raise ValueError(f"Commit manifest digest does not match PR snapshot: {key}")
        if not pairs or pairs[-1][1] != str(snapshot["pr_head_sha"] or ""):
            raise ValueError(f"Commit manifest does not end at the frozen PR head: {key}")
    return targets


def checkpoint_targets(
    targets: Mapping[int, RepositoryTarget],
    commits_per_part: int,
) -> list[RepositoryTarget]:
    batches: list[RepositoryTarget] = []
    for target in sorted(targets.values(), key=lambda item: item.repo_id):
        by_pull_request: dict[int, list[CommitTarget]] = {}
        for commit in sorted(target.commits, key=lambda item: (item.pr_number, item.commit_index)):
            by_pull_request.setdefault(commit.pr_number, []).append(commit)
        pending: list[CommitTarget] = []
        checkpoint_index = 0
        for pull_request_commits in by_pull_request.values():
            if pending and len(pending) + len(pull_request_commits) > commits_per_part:
                checkpoint_index += 1
                batches.append(
                    RepositoryTarget(
                        repo_id=target.repo_id,
                        resolved_repo_full_name=target.resolved_repo_full_name,
                        commits=pending,
                        checkpoint_index=checkpoint_index,
                    )
                )
                pending = []
            pending.extend(pull_request_commits)
        if pending:
            checkpoint_index += 1
            batches.append(
                RepositoryTarget(
                    repo_id=target.repo_id,
                    resolved_repo_full_name=target.resolved_repo_full_name,
                    commits=pending,
                    checkpoint_index=checkpoint_index,
                )
            )
    return batches


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int,
    check: bool = True,
    input_text: str | None = None,
    environment_overrides: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment.update(environment_overrides or {})
    result = subprocess.run(
        command,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=environment,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()[-1000:]
        raise RuntimeError(f"Command failed ({' '.join(command[:3])}): {detail}")
    return result


def ensure_repository(
    target: RepositoryTarget,
    cache_root: Path,
    *,
    git_timeout: int,
) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    repository = cache_root / f"repo-{target.repo_id}"
    if repository.exists():
        if not (repository / ".git").is_dir():
            raise ValueError(f"Repository cache path is not a Git worktree: {repository}")
    else:
        temporary = repository.with_name(f".{repository.name}.tmp")
        if (temporary / ".git").is_dir():
            os.replace(temporary, repository)
        else:
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            run_command(
                ["git", "init", "--quiet"],
                cwd=temporary,
                timeout=git_timeout,
            )
            run_command(
                [
                    "git",
                    "remote",
                    "add",
                    "origin",
                    f"https://github.com/{target.resolved_repo_full_name}.git",
                ],
                cwd=temporary,
                timeout=git_timeout,
            )
            os.replace(temporary, repository)

    remote_url = f"https://github.com/{target.resolved_repo_full_name}.git"
    existing_remote = run_command(
        ["git", "remote", "get-url", "origin"],
        cwd=repository,
        timeout=git_timeout,
        check=False,
    )
    remote_action = "set-url" if existing_remote.returncode == 0 else "add"
    run_command(
        ["git", "remote", remote_action, "origin", remote_url],
        cwd=repository,
        timeout=git_timeout,
    )
    for key, value in (
        ("extensions.partialClone", "origin"),
        ("remote.origin.promisor", "true"),
        ("remote.origin.partialclonefilter", "blob:none"),
    ):
        run_command(
            ["git", "config", key, value],
            cwd=repository,
            timeout=git_timeout,
        )
    return repository


def chunks(values: list[int], size: int) -> Iterator[list[int]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def fetch_pull_refs(
    repository: Path,
    target: RepositoryTarget,
    *,
    git_timeout: int,
) -> set[int]:
    failed: set[int] = set()
    snapshots: dict[int, tuple[str, int]] = {}
    for commit in target.commits:
        previous = snapshots.get(commit.pr_number)
        candidate = (commit.pr_head_sha, commit.commit_index + 1)
        if previous is not None and previous[0] != candidate[0]:
            raise ValueError(f"PR {commit.pr_number} has conflicting frozen head SHAs.")
        snapshots[commit.pr_number] = (
            candidate[0], max(previous[1] if previous else 0, candidate[1])
        )
    for batch in chunks(sorted(snapshots), 50):
        refspecs = [
            f"+{snapshots[number][0]}:refs/perfminer/pull/{number}" for number in batch
        ]
        depth = max(snapshots[number][1] for number in batch)
        result = run_command(
            [
                "git",
                "fetch",
                "--quiet",
                "--no-tags",
                "--filter=blob:none",
                f"--depth={depth}",
                "origin",
                *refspecs,
            ],
            cwd=repository,
            timeout=git_timeout,
            check=False,
        )
        if result.returncode == 0:
            continue
        detail = (result.stderr or result.stdout or "").casefold()
        if not is_missing_remote_object_error(detail):
            raise RuntimeError(f"Git fetch failed: {detail[-1000:]}")
        for number, refspec in zip(batch, refspecs, strict=True):
            retry = run_command(
                [
                    "git",
                    "fetch",
                    "--quiet",
                    "--no-tags",
                    "--filter=blob:none",
                    f"--depth={snapshots[number][1]}",
                    "origin",
                    refspec,
                ],
                cwd=repository,
                timeout=git_timeout,
                check=False,
            )
            if retry.returncode != 0:
                retry_detail = (retry.stderr or retry.stdout or "").casefold()
                if is_missing_remote_object_error(retry_detail):
                    failed.add(number)
                else:
                    raise RuntimeError(f"Git fetch failed: {retry_detail[-1000:]}")
    return failed


def is_missing_remote_object_error(detail: str) -> bool:
    return any(
        marker in detail
        for marker in (
            "couldn't find remote ref",
            "not our ref",
            "unadvertised object",
        )
    )


def fetch_missing_commits(
    repository: Path,
    shas: list[str],
    *,
    git_timeout: int,
) -> set[str]:
    unavailable: set[str] = set()
    for sha in shas:
        result = run_command(
            [
                "git",
                "fetch",
                "--quiet",
                "--no-tags",
                "--filter=blob:none",
                "--depth=2",
                "origin",
                sha,
            ],
            cwd=repository,
            timeout=git_timeout,
            check=False,
        )
        if result.returncode == 0:
            continue
        detail = (result.stderr or result.stdout or "").casefold()
        if is_missing_remote_object_error(detail):
            unavailable.add(sha)
        else:
            raise RuntimeError(f"Git fetch failed: {detail[-1000:]}")
    return unavailable


def available_commits(repository: Path, shas: list[str], *, git_timeout: int) -> set[str]:
    unique = list(dict.fromkeys(shas))
    if not unique:
        return set()
    expressions = "".join(f"{sha}^{{commit}}\n" for sha in unique)
    result = run_command(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
        cwd=repository,
        timeout=git_timeout,
        input_text=expressions,
        environment_overrides={"GIT_NO_LAZY_FETCH": "1"},
    )
    available: set[str] = set()
    for sha, line in zip(unique, result.stdout.splitlines(), strict=True):
        fields = line.split()
        if len(fields) == 2 and fields[1] == "commit":
            available.add(sha)
    return available


@contextmanager
def inspection_timeout(seconds: int):
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(_signum, _frame):
        raise CommitInspectionTimeout(f"commit inspection exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def unavailable_evidence(sha: str, reason: str, error: str = "") -> dict[str, object]:
    return {
        "commit_sha": sha,
        "commit_message_raw": "",
        "commit_message": "",
        "commit_message_word_count": None,
        "commit_date": "",
        "parent_count": None,
        "changed_file_count": None,
        "filename": "",
        "change_type": "",
        "is_binary": None,
        "changed_method_count": None,
        "changed_method_name": "",
        "changed_method_start_line": None,
        "changed_method_end_line": None,
        "code_diff": "",
        "diff_chars": None,
        "source_before_chars": None,
        "source_after_chars": None,
        "source_pair_md5": "",
        "perfminer_evidence_status": "unavailable",
        "perfminer_exclusion_reason": reason,
        "perfminer_evidence_error": error[:1000],
    }


def identity_row(target: CommitTarget) -> dict[str, object]:
    return {
        "repo_id": target.repo_id,
        "source_repo_full_name": target.source_repo_full_name,
        "resolved_repo_full_name": target.resolved_repo_full_name,
        "pr_number": target.pr_number,
        "pr_html_url": target.pr_html_url,
        "pr_base_sha": target.pr_base_sha,
        "pr_head_sha": target.pr_head_sha,
        "commit_manifest_sha256": target.commit_manifest_sha256,
        "pr_manifest_complete": target.pr_manifest_complete,
        "commit_index": target.commit_index,
        "commit_sha": target.commit_sha,
        "commit_message_api": target.commit_message_api,
        "parent_count_api": target.parent_count_api,
        "commit_html_url": target.commit_html_url,
    }


def extract_repository(
    target: RepositoryTarget,
    cache_root: Path,
    *,
    git_timeout: int,
    commit_timeout: int,
    commit_attempts: int,
) -> dict[str, Any]:
    try:
        repository = ensure_repository(target, cache_root, git_timeout=git_timeout)
    except Exception as error:
        return {
            "status": "incomplete",
            "error": f"{type(error).__name__}: {error}"[:1000],
            "rows": [],
        }

    try:
        failed_refs = fetch_pull_refs(repository, target, git_timeout=git_timeout)
        shas = [commit.commit_sha for commit in target.commits]
        available = available_commits(repository, shas, git_timeout=git_timeout)
        missing_shas = [sha for sha in dict.fromkeys(shas) if sha not in available]
        if missing_shas:
            fetch_missing_commits(repository, missing_shas, git_timeout=git_timeout)
            available = available_commits(repository, shas, git_timeout=git_timeout)
    except Exception as error:
        return {
            "status": "incomplete",
            "error": f"{type(error).__name__}: {error}"[:1000],
            "rows": [],
        }
    git_repository = Git(str(repository))
    evidence_by_sha: dict[str, dict[str, object]] = {}
    for sha in dict.fromkeys(shas):
        if sha not in available:
            evidence_by_sha[sha] = unavailable_evidence(sha, "commit_object_unavailable")
            continue
        for attempt in range(1, commit_attempts + 1):
            try:
                with inspection_timeout(commit_timeout):
                    evidence = inspect_commit(git_repository.get_commit(sha))
                evidence["perfminer_evidence_error"] = ""
                evidence_by_sha[sha] = evidence
                break
            except CommitInspectionTimeout as error:
                if attempt == commit_attempts:
                    evidence_by_sha[sha] = unavailable_evidence(
                        sha, "commit_timeout", str(error)
                    )
            except Exception as error:
                if attempt == commit_attempts:
                    evidence_by_sha[sha] = unavailable_evidence(
                        sha,
                        "commit_extraction_error",
                        f"{type(error).__name__}: {error}",
                    )

    rows: list[dict[str, object]] = []
    for commit_target in target.commits:
        evidence = dict(evidence_by_sha[commit_target.commit_sha])
        raw_message = str(evidence.get("commit_message_raw") or "")
        parent_count = int(evidence.get("parent_count") or 0)
        evidence_available = evidence["perfminer_evidence_status"] != "unavailable"
        message_matches = (
            raw_message.strip() == commit_target.commit_message_api.strip()
            if evidence_available
            else False
        )
        parent_count_matches = (
            parent_count == commit_target.parent_count_api if evidence_available else False
        )
        if evidence_available and (not message_matches or not parent_count_matches):
            evidence = unavailable_evidence(
                commit_target.commit_sha,
                "api_git_identity_mismatch",
                "Commit message or parent count differs between GitHub API and Git.",
            )
        row = {
            **identity_row(commit_target),
            "api_message_matches_git": message_matches,
            "api_parent_count_matches_git": parent_count_matches,
            **evidence,
        }
        rows.append(row)
    return {
        "status": "complete",
        "error": "",
        "failed_pull_ref_count": len(failed_refs),
        "available_commit_count": len(available),
        "rows": rows,
    }


def build_run_signature(
    args: argparse.Namespace,
    commits: pq.ParquetFile,
    pull_requests: pq.ParquetFile,
    targets: Mapping[int, RepositoryTarget],
) -> dict[str, object]:
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "perfminer_revision": PERFMINER_REVISION,
        "commits_path": str(args.commits.resolve()),
        "commits_sha256": sha256_file(args.commits),
        "commits_rows": int(commits.metadata.num_rows),
        "commits_schema_sha256": schema_sha256(commits.schema_arrow),
        "pull_requests_path": str(args.pull_requests.resolve()),
        "pull_requests_sha256": sha256_file(args.pull_requests),
        "pull_requests_rows": int(pull_requests.metadata.num_rows),
        "pull_requests_schema_sha256": schema_sha256(pull_requests.schema_arrow),
        "repository_count": len(targets),
        "commits_per_part": args.commits_per_part,
        "git_timeout_seconds": args.git_timeout,
        "commit_timeout_seconds": args.commit_timeout,
        "commit_attempts": args.commit_attempts,
        "filters": {
            "supported_extensions": list(PERFMINER_SUPPORTED_EXTENSIONS),
            "max_diff_chars": PERFMINER_MAX_DIFF_CHARS,
            "max_source_before_chars": PERFMINER_MAX_SOURCE_BEFORE_CHARS,
            "one_file": True,
            "one_changed_method": True,
        },
        "runtime": {
            "pydriller": importlib.metadata.version("pydriller"),
            "lizard": importlib.metadata.version("lizard"),
            "gitpython": importlib.metadata.version("gitpython"),
            "git": run_command(["git", "--version"], timeout=30).stdout.strip(),
        },
        "code_sha256": {
            Path(__file__).name: sha256_file(Path(__file__)),
            "perfminer_reproduction.py": sha256_file(
                Path(__file__).with_name("perfminer_reproduction.py")
            ),
            "parquet_parts.py": sha256_file(Path(__file__).with_name("parquet_parts.py")),
        },
    }


def validate_recorded_part(metadata: Mapping[str, object]) -> Path:
    path = Path(str(metadata.get("path") or ""))
    validate_part(
        path,
        EVIDENCE_SCHEMA,
        expected_rows=int(metadata.get("row_count", -1)),
        expected_sha256=str(metadata.get("sha256") or ""),
    )
    return path


def summarize(output: Path) -> dict[str, object]:
    table = pq.read_table(
        output,
        columns=["perfminer_evidence_status", "perfminer_exclusion_reason"],
    )
    rows = table.to_pylist()
    statuses = Counter(row["perfminer_evidence_status"] for row in rows)
    reasons = Counter(
        row["perfminer_exclusion_reason"]
        for row in rows
        if row["perfminer_exclusion_reason"]
    )
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "row_count": len(rows),
        "evidence_status_counts": dict(sorted(statuses.items())),
        "exclusion_reason_counts": dict(sorted(reasons.items())),
        "output": {"path": str(output.resolve()), "sha256": sha256_file(output)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract commit-level PerfMiner message/diff evidence from frozen PR commits."
    )
    parser.add_argument("--pull-requests", required=True, type=Path)
    parser.add_argument("--commits", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--repo-cache",
        type=Path,
        default=Path("mining/query_cache/perfminer_repositories"),
    )
    parser.add_argument("--git-timeout", type=positive_integer, default=1800)
    parser.add_argument("--commit-timeout", type=positive_integer, default=10)
    parser.add_argument("--commit-attempts", type=positive_integer, default=3)
    parser.add_argument("--commits-per-part", type=positive_integer, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    commit_source = pq.ParquetFile(args.commits)
    pr_source = pq.ParquetFile(args.pull_requests)
    targets_by_id = collect_targets(args.commits, args.pull_requests)
    target_batches = checkpoint_targets(targets_by_id, args.commits_per_part)
    targets = list(enumerate(target_batches, start=1))
    signature = build_run_signature(args, commit_source, pr_source, targets_by_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output_dir / "parts" / "commit_evidence"
    state_file = args.output_dir / "commit_evidence.state.json"
    summary_file = args.output_dir / "commit_evidence.summary.json"
    output = args.output_dir / "commit_evidence.parquet"
    has_artifacts = any(path.exists() for path in (state_file, summary_file, output)) or parts_dir.exists()
    if has_artifacts and not args.resume:
        raise FileExistsError(f"Commit-evidence artifacts already exist in {args.output_dir}; use --resume.")
    if args.resume and not state_file.exists() and has_artifacts:
        raise ValueError("Cannot resume commit-evidence artifacts without matching state.")

    if state_file.exists():
        state: dict[str, Any] = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("run_signature") != signature:
            raise ValueError("Existing commit-evidence run signature does not match the request.")
    else:
        state = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "run_signature": signature,
            "batches": {},
            "finalized": False,
        }
        write_json(state_file, state)

    batches = state.setdefault("batches", {})
    for index, target in targets:
        key = target.state_key
        previous = batches.get(key)
        if isinstance(previous, Mapping) and previous.get("status") == "complete":
            validate_recorded_part(previous.get("part", {}))
            continue
        result = extract_repository(
            target,
            args.repo_cache,
            git_timeout=args.git_timeout,
            commit_timeout=args.commit_timeout,
            commit_attempts=args.commit_attempts,
        )
        rows = result.pop("rows")
        if result["status"] == "complete":
            part = write_part(part_path(parts_dir, index), rows, EVIDENCE_SCHEMA)
            result["part"] = part
        batches[key] = {
            **result,
            "part_index": index,
            "repo_full_name": target.resolved_repo_full_name,
            "repo_id": target.repo_id,
            "checkpoint_index": target.checkpoint_index,
            "commit_manifest_rows": len(target.commits),
        }
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(state_file, state)
        print(
            f"[commit-evidence {index}/{len(targets)}] {target.resolved_repo_full_name} "
            f"status={result['status']} rows={len(rows)}",
            flush=True,
        )

    incomplete = [
        key
        for key, record in batches.items()
        if not isinstance(record, Mapping) or record.get("status") != "complete"
    ]
    if incomplete:
        raise RuntimeError(
            f"Commit evidence has {len(incomplete)} incomplete repositories; rerun with --resume."
        )

    parts = []
    for index, target in targets:
        record = batches[target.state_key]
        if int(record.get("part_index", -1)) != index:
            raise ValueError(f"Repository {target.repo_id} has an unexpected part index.")
        parts.append(validate_recorded_part(record["part"]))
    state["output"] = finalize_parts(parts, output, EVIDENCE_SCHEMA)
    summary = summarize(output)
    write_json(summary_file, summary)
    state["summary"] = {"path": str(summary_file.resolve()), "sha256": sha256_file(summary_file)}
    state["finalized"] = True
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(state_file, state)
    print(f"finalized_commit_evidence={summary['row_count']}", flush=True)


if __name__ == "__main__":
    main()
