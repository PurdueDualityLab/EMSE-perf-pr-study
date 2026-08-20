from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping

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
from parquet_parts import (
    finalize_parts,
    part_path,
    schema_sha256,
    sha256_file,
    validate_part,
    write_json,
    write_part,
)
from perfminer_reproduction import commit_manifest_sha256


METHOD_NAME = "pr_commit_manifest_v1"
METHOD_VERSION = 1

PR_SCHEMA = pa.schema(
    [
        pa.field("repo_id", pa.int64()),
        pa.field("source_repo_full_name", pa.string()),
        pa.field("resolved_repo_full_name", pa.string()),
        pa.field("pr_number", pa.int64()),
        pa.field("pr_html_url", pa.string()),
        pa.field("pr_base_sha", pa.string()),
        pa.field("pr_head_sha", pa.string()),
        pa.field("commit_manifest_sha256", pa.string()),
        pa.field("expected_commit_count", pa.int64()),
        pa.field("listed_commit_count", pa.int64()),
        pa.field("commit_manifest_status", pa.string()),
        pa.field("commit_manifest_error", pa.string()),
    ]
)

COMMIT_SCHEMA = pa.schema(
    [
        pa.field("repo_id", pa.int64()),
        pa.field("source_repo_full_name", pa.string()),
        pa.field("resolved_repo_full_name", pa.string()),
        pa.field("pr_number", pa.int64()),
        pa.field("pr_html_url", pa.string()),
        pa.field("pr_base_sha", pa.string()),
        pa.field("pr_head_sha", pa.string()),
        pa.field("commit_manifest_sha256", pa.string()),
        pa.field("commit_index", pa.int64()),
        pa.field("commit_sha", pa.string()),
        pa.field("commit_message_api", pa.string()),
        pa.field("parent_count_api", pa.int64()),
        pa.field("commit_html_url", pa.string()),
        pa.field("pr_manifest_complete", pa.bool_()),
    ]
)


@dataclass(frozen=True)
class PullRequestTarget:
    number: int
    html_url: str


@dataclass
class RepositoryTarget:
    repo_id: int
    source_names: set[str] = field(default_factory=set)
    pull_requests: dict[int, PullRequestTarget] = field(default_factory=dict)
    checkpoint_index: int = 0

    @property
    def source_name(self) -> str:
        return sorted(self.source_names, key=str.casefold)[0]

    @property
    def state_key(self) -> str:
        return f"{self.repo_id}:{self.checkpoint_index}"


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def parse_positive_integer(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not parsed.is_integer() or parsed <= 0:
        return None
    return int(parsed)


def first_value(row: Mapping[str, object], columns: tuple[str, ...]) -> object:
    for column in columns:
        value = row.get(column)
        if value is not None:
            return value
    return None


def collect_targets(input_path: Path, limit: int | None = None) -> dict[int, RepositoryTarget]:
    source = pq.ParquetFile(input_path)
    required_candidates = {
        "repo_id": ("repo_id", "repository_id"),
        "repo_name": ("repo_full_name", "full_name", "name_with_owner"),
        "number": ("number", "pr_number", "aidev_source_pr_number"),
        "html_url": ("html_url", "aidev_source_html_url"),
    }
    selected = sorted(
        {
            column
            for candidates in required_candidates.values()
            for column in candidates
            if column in source.schema_arrow.names
        }
    )
    for label, candidates in required_candidates.items():
        if not any(column in selected for column in candidates):
            raise ValueError(f"Input is missing a {label} column.")

    targets: dict[int, RepositoryTarget] = {}
    seen_repo_names: dict[str, int] = {}
    remaining = limit
    for batch in source.iter_batches(batch_size=50_000, columns=selected):
        rows = pa.Table.from_batches([batch]).to_pylist()
        if remaining is not None:
            rows = rows[:remaining]
            remaining -= len(rows)
        for row in rows:
            repo_id = parse_positive_integer(first_value(row, required_candidates["repo_id"]))
            repo_name = parse_repo_full_name(first_value(row, required_candidates["repo_name"]))
            number = parse_positive_integer(first_value(row, required_candidates["number"]))
            html_url_value = first_value(row, required_candidates["html_url"])
            html_url = str(html_url_value or "").strip()
            if repo_id is None or repo_name is None or number is None:
                raise ValueError("Every input row must have a valid repository ID, name, and PR number.")
            name_key = repo_name.casefold()
            previous_repo_id = seen_repo_names.setdefault(name_key, repo_id)
            if previous_repo_id != repo_id:
                raise ValueError(f"Repository name {repo_name} maps to multiple repository IDs.")
            target = targets.setdefault(repo_id, RepositoryTarget(repo_id))
            target.source_names.add(repo_name)
            normalized_url = html_url or f"https://github.com/{repo_name}/pull/{number}"
            previous = target.pull_requests.get(number)
            candidate = PullRequestTarget(number, normalized_url)
            if previous is not None and previous != candidate:
                raise ValueError(f"Conflicting input rows for repository {repo_id} PR {number}.")
            target.pull_requests[number] = candidate
        if remaining == 0:
            break
    return targets


def checkpoint_targets(
    targets: Mapping[int, RepositoryTarget],
    pull_requests_per_part: int,
) -> list[RepositoryTarget]:
    batches: list[RepositoryTarget] = []
    for target in sorted(targets.values(), key=lambda item: item.repo_id):
        pull_requests = sorted(target.pull_requests.values(), key=lambda item: item.number)
        for start in range(0, len(pull_requests), pull_requests_per_part):
            selected = pull_requests[start : start + pull_requests_per_part]
            batches.append(
                RepositoryTarget(
                    repo_id=target.repo_id,
                    source_names=set(target.source_names),
                    pull_requests={item.number: item for item in selected},
                    checkpoint_index=start // pull_requests_per_part + 1,
                )
            )
    return batches


def error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"[:1000]


def unresolved_pr_row(
    target: RepositoryTarget,
    pull_request: PullRequestTarget,
    status: str,
    error: str,
    *,
    resolved_name: str = "",
) -> dict[str, object]:
    return {
        "repo_id": target.repo_id,
        "source_repo_full_name": target.source_name,
        "resolved_repo_full_name": resolved_name,
        "pr_number": pull_request.number,
        "pr_html_url": pull_request.html_url,
        "pr_base_sha": "",
        "pr_head_sha": "",
        "commit_manifest_sha256": "",
        "expected_commit_count": None,
        "listed_commit_count": 0,
        "commit_manifest_status": status,
        "commit_manifest_error": error,
    }


def resolve_repository(
    target: RepositoryTarget,
    client: GitHubClient,
) -> tuple[str | None, str | None, bool]:
    try:
        payload = client.request(f"/repositories/{target.repo_id}")
    except GitHubNotFoundError as error:
        return None, error_text(error), False
    except (GitHubAuthenticationError, GitHubRateLimitError):
        raise
    except GitHubClientError as error:
        return None, error_text(error), True
    if not isinstance(payload, Mapping):
        return None, "GitHub repository payload is not an object.", True
    resolved_id = parse_positive_integer(payload.get("id"))
    resolved_name = parse_repo_full_name(payload.get("full_name"))
    if resolved_id != target.repo_id or resolved_name is None:
        return None, "GitHub repository identity does not match the input repository ID.", False
    return resolved_name, None, False


def fetch_pull_request_manifest(
    target: RepositoryTarget,
    pull_request: PullRequestTarget,
    resolved_name: str,
    client: GitHubClient,
) -> tuple[dict[str, object], list[dict[str, object]], bool]:
    def request_pull() -> Mapping[str, object]:
        payload = client.fetch_pull_request(resolved_name, pull_request.number)
        if not isinstance(payload, Mapping):
            raise GitHubClientError("GitHub PR payload is not an object.")
        return payload

    def snapshot(payload: Mapping[str, object]) -> dict[str, object]:
        base = payload.get("base") if isinstance(payload.get("base"), Mapping) else {}
        base_repo = base.get("repo") if isinstance(base.get("repo"), Mapping) else {}
        head = payload.get("head") if isinstance(payload.get("head"), Mapping) else {}
        return {
            "base_repo_id": parse_positive_integer(base_repo.get("id")),
            "base_sha": str(base.get("sha") or "").strip(),
            "head_sha": str(head.get("sha") or "").strip(),
            "expected_count": parse_positive_integer(payload.get("commits")),
            "html_url": str(payload.get("html_url") or pull_request.html_url),
        }

    last_snapshot: dict[str, object] = {}
    last_error = "PR head or commit list changed while the snapshot was being read."
    for attempt in range(3):
        try:
            before_payload = request_pull()
        except GitHubNotFoundError as error:
            return (
                unresolved_pr_row(
                    target,
                    pull_request,
                    "pull_request_unavailable",
                    error_text(error),
                    resolved_name=resolved_name,
                ),
                [],
                False,
            )
        except (GitHubAuthenticationError, GitHubRateLimitError):
            raise
        except GitHubClientError as error:
            last_error = error_text(error)
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
                continue
            return (
                unresolved_pr_row(
                    target,
                    pull_request,
                    "api_error",
                    last_error,
                    resolved_name=resolved_name,
                ),
                [],
                True,
            )

        before = snapshot(before_payload)
        last_snapshot = before
        if before["base_repo_id"] != target.repo_id:
            return (
                unresolved_pr_row(
                    target,
                    pull_request,
                    "repository_identity_mismatch",
                    "PR base repository ID does not match the input repository ID.",
                    resolved_name=resolved_name,
                ),
                [],
                False,
            )
        expected_count = before["expected_count"]
        if expected_count is not None and int(expected_count) > 250:
            return (
                {
                    **unresolved_pr_row(
                        target,
                        pull_request,
                        "commit_count_over_rest_limit",
                        "PR exceeds GitHub's 250-commit listing limit.",
                        resolved_name=resolved_name,
                    ),
                    "pr_base_sha": before["base_sha"],
                    "pr_head_sha": before["head_sha"],
                    "expected_commit_count": expected_count,
                    "pr_html_url": before["html_url"],
                },
                [],
                False,
            )
        if not before["base_sha"] or not before["head_sha"] or expected_count is None:
            last_error = "PR payload has no valid base SHA, head SHA, or commit count."
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
                continue
            break

        try:
            commits = client.fetch_pull_commits(resolved_name, pull_request.number)
            after_payload = request_pull()
        except GitHubNotFoundError as error:
            return (
                {
                    **unresolved_pr_row(
                        target,
                        pull_request,
                        "commit_list_unavailable",
                        error_text(error),
                        resolved_name=resolved_name,
                    ),
                    "pr_base_sha": before["base_sha"],
                    "pr_head_sha": before["head_sha"],
                    "expected_commit_count": expected_count,
                    "pr_html_url": before["html_url"],
                },
                [],
                False,
            )
        except (GitHubAuthenticationError, GitHubRateLimitError):
            raise
        except GitHubClientError as error:
            last_error = error_text(error)
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
                continue
            break

        after = snapshot(after_payload)
        if before != after:
            last_error = "PR snapshot changed between commit-list reads."
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
                continue
            break

        errors: list[str] = []
        commit_rows: list[dict[str, object]] = []
        seen_shas: set[str] = set()
        for index, item in enumerate(commits, start=1):
            if not isinstance(item, Mapping):
                errors.append(f"commit {index} is not an object")
                continue
            sha = str(item.get("sha") or "").strip()
            commit_payload = item.get("commit") if isinstance(item.get("commit"), Mapping) else {}
            message = commit_payload.get("message")
            parents = item.get("parents")
            if (
                not sha
                or sha in seen_shas
                or not isinstance(message, str)
                or not isinstance(parents, list)
            ):
                errors.append(f"commit {index} has invalid identity, message, or parents")
                continue
            seen_shas.add(sha)
            commit_rows.append(
                {
                    "repo_id": target.repo_id,
                    "source_repo_full_name": target.source_name,
                    "resolved_repo_full_name": resolved_name,
                    "pr_number": pull_request.number,
                    "pr_html_url": before["html_url"],
                    "pr_base_sha": before["base_sha"],
                    "pr_head_sha": before["head_sha"],
                    "commit_manifest_sha256": "",
                    "commit_index": index,
                    "commit_sha": sha,
                    "commit_message_api": message,
                    "parent_count_api": len(parents),
                    "commit_html_url": str(item.get("html_url") or ""),
                    "pr_manifest_complete": True,
                }
            )

        if len(commits) != expected_count:
            errors.append(f"listed {len(commits)} of {expected_count} expected commits")
        if len(commit_rows) != len(commits):
            errors.append("one or more listed commits could not be validated")
        if commit_rows and commit_rows[-1]["commit_sha"] != before["head_sha"]:
            errors.append("last listed commit does not match the PR head SHA")
        if not commit_rows:
            errors.append("PR commit list is empty")
        if errors:
            last_error = "; ".join(errors)
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
                continue
            break

        manifest_sha256 = commit_manifest_sha256(
            (int(row["commit_index"]), str(row["commit_sha"])) for row in commit_rows
        )
        for row in commit_rows:
            row["commit_manifest_sha256"] = manifest_sha256
        return (
            {
                "repo_id": target.repo_id,
                "source_repo_full_name": target.source_name,
                "resolved_repo_full_name": resolved_name,
                "pr_number": pull_request.number,
                "pr_html_url": before["html_url"],
                "pr_base_sha": before["base_sha"],
                "pr_head_sha": before["head_sha"],
                "commit_manifest_sha256": manifest_sha256,
                "expected_commit_count": expected_count,
                "listed_commit_count": len(commit_rows),
                "commit_manifest_status": "complete",
                "commit_manifest_error": "",
            },
            commit_rows,
            False,
        )

    return (
        {
            **unresolved_pr_row(
                target,
                pull_request,
                "unstable_pull_request_snapshot",
                last_error,
                resolved_name=resolved_name,
            ),
            "pr_base_sha": str(last_snapshot.get("base_sha") or ""),
            "pr_head_sha": str(last_snapshot.get("head_sha") or ""),
            "expected_commit_count": last_snapshot.get("expected_count"),
            "pr_html_url": str(last_snapshot.get("html_url") or pull_request.html_url),
        },
        [],
        True,
    )


def fetch_repository_manifest(target: RepositoryTarget, token_file: Path) -> dict[str, Any]:
    client = GitHubClient.from_token_file(str(token_file))
    try:
        resolved_name, resolution_error, retryable = resolve_repository(target, client)
    except (GitHubAuthenticationError, GitHubRateLimitError):
        raise
    if resolved_name is None:
        status = "api_error" if retryable else "repository_unavailable"
        pr_rows = [
            unresolved_pr_row(target, pull_request, status, resolution_error or "")
            for pull_request in target.pull_requests.values()
        ]
        return {
            "status": "incomplete" if retryable else "complete",
            "resolved_repo_full_name": "",
            "pr_rows": pr_rows,
            "commit_rows": [],
            "retryable_errors": int(retryable),
        }

    pr_rows: list[dict[str, object]] = []
    commit_rows: list[dict[str, object]] = []
    retryable_errors = 0
    for pull_request in sorted(target.pull_requests.values(), key=lambda item: item.number):
        try:
            pr_row, rows, retryable = fetch_pull_request_manifest(
                target, pull_request, resolved_name, client
            )
        except (GitHubAuthenticationError, GitHubRateLimitError):
            raise
        pr_rows.append(pr_row)
        commit_rows.extend(rows)
        retryable_errors += int(retryable)
    return {
        "status": "incomplete" if retryable_errors else "complete",
        "resolved_repo_full_name": resolved_name,
        "pr_rows": pr_rows,
        "commit_rows": commit_rows,
        "retryable_errors": retryable_errors,
    }


def build_run_signature(
    args: argparse.Namespace,
    source: pq.ParquetFile,
    targets: Mapping[int, RepositoryTarget],
) -> dict[str, object]:
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "input_row_count": int(source.metadata.num_rows),
        "input_schema_sha256": schema_sha256(source.schema_arrow),
        "limit": args.limit,
        "pull_requests_per_part": args.pull_requests_per_part,
        "repository_count": len(targets),
        "pull_request_count": sum(len(target.pull_requests) for target in targets.values()),
        "code_sha256": {
            Path(__file__).name: sha256_file(Path(__file__)),
            "github_client.py": sha256_file(Path(__file__).with_name("github_client.py")),
            "parquet_parts.py": sha256_file(Path(__file__).with_name("parquet_parts.py")),
            "perfminer_reproduction.py": sha256_file(
                Path(__file__).with_name("perfminer_reproduction.py")
            ),
        },
    }


def validate_recorded_part(metadata: Mapping[str, object], schema: pa.Schema) -> Path:
    path = Path(str(metadata.get("path") or ""))
    validate_part(
        path,
        schema,
        expected_rows=int(metadata.get("row_count", -1)),
        expected_sha256=str(metadata.get("sha256") or ""),
    )
    return path


def run_workers(
    targets: list[tuple[int, RepositoryTarget]],
    args: argparse.Namespace,
    state: dict[str, Any],
    pr_parts_dir: Path,
    commit_parts_dir: Path,
    state_file: Path,
) -> None:
    batches = state.setdefault("batches", {})
    pending = [
        (index, target)
        for index, target in targets
        if not isinstance(batches.get(target.state_key), Mapping)
        or batches[target.state_key].get("status") != "complete"
    ]
    if not pending:
        return
    dirty = 0
    completed = len(targets) - len(pending)
    fatal_error: Exception | None = None
    with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
        futures = {
            executor.submit(fetch_repository_manifest, target, args.token_file): (index, target)
            for index, target in pending
        }
        for future in as_completed(futures):
            index, target = futures[future]
            try:
                result = future.result()
            except Exception as error:
                if fatal_error is None:
                    fatal_error = error
                    for queued in futures:
                        if queued is not future:
                            queued.cancel()
                continue
            pr_part = write_part(part_path(pr_parts_dir, index), result.pop("pr_rows"), PR_SCHEMA)
            commit_part = write_part(
                part_path(commit_parts_dir, index), result.pop("commit_rows"), COMMIT_SCHEMA
            )
            batches[target.state_key] = {
                **result,
                "part_index": index,
                "source_repo_full_name": target.source_name,
                "repo_id": target.repo_id,
                "checkpoint_index": target.checkpoint_index,
                "pull_request_count": len(target.pull_requests),
                "pr_part": pr_part,
                "commit_part": commit_part,
            }
            completed += 1
            dirty += 1
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            if dirty >= args.checkpoint_every:
                write_json(state_file, state)
                dirty = 0
            print(
                f"[commit-manifest {completed}/{len(targets)}] {target.source_name} "
                f"status={result['status']} prs={len(target.pull_requests)} "
                f"commits={commit_part['row_count']}",
                flush=True,
            )
    if dirty:
        write_json(state_file, state)
    if fatal_error is not None:
        raise fatal_error


def summarize(pr_output: Path, commit_output: Path) -> dict[str, object]:
    statuses = Counter(
        row["commit_manifest_status"]
        for row in pq.read_table(pr_output, columns=["commit_manifest_status"]).to_pylist()
    )
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "pull_request_rows": int(pq.ParquetFile(pr_output).metadata.num_rows),
        "commit_rows": int(pq.ParquetFile(commit_output).metadata.num_rows),
        "pull_request_status_counts": dict(sorted(statuses.items())),
        "outputs": {
            "pull_requests": {
                "path": str(pr_output.resolve()),
                "sha256": sha256_file(pr_output),
            },
            "commits": {
                "path": str(commit_output.resolve()),
                "sha256": sha256_file(commit_output),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze each input PR head and enumerate its commits for PerfMiner reproduction."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--workers", type=positive_integer, default=5)
    parser.add_argument("--checkpoint-every", type=positive_integer, default=1)
    parser.add_argument("--pull-requests-per-part", type=positive_integer, default=50)
    parser.add_argument("--limit", type=positive_integer)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = pq.ParquetFile(args.input)
    targets_by_id = collect_targets(args.input, args.limit)
    target_batches = checkpoint_targets(targets_by_id, args.pull_requests_per_part)
    targets = list(enumerate(target_batches, start=1))
    run_signature = build_run_signature(args, source, targets_by_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.output_dir / "commit_manifest.state.json"
    summary_file = args.output_dir / "commit_manifest.summary.json"
    pr_output = args.output_dir / "pull_requests.parquet"
    commit_output = args.output_dir / "commits.parquet"
    pr_parts_dir = args.output_dir / "parts" / "pull_requests"
    commit_parts_dir = args.output_dir / "parts" / "commits"
    artifact_paths = (state_file, summary_file, pr_output, commit_output)
    has_parts = pr_parts_dir.exists() or commit_parts_dir.exists()

    if not args.resume and (any(path.exists() for path in artifact_paths) or has_parts):
        raise FileExistsError(f"Commit-manifest artifacts already exist in {args.output_dir}; use --resume.")
    if args.resume and not state_file.exists() and (any(path.exists() for path in artifact_paths[1:]) or has_parts):
        raise ValueError("Cannot resume commit-manifest artifacts without matching state.")

    state: dict[str, Any]
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("run_signature") != run_signature:
            raise ValueError("Existing commit-manifest run signature does not match the request.")
    else:
        state = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "run_signature": run_signature,
            "batches": {},
            "finalized": False,
        }
        write_json(state_file, state)

    for _batch_key, record in state.get("batches", {}).items():
        if not isinstance(record, Mapping) or record.get("status") != "complete":
            continue
        validate_recorded_part(record.get("pr_part", {}), PR_SCHEMA)
        validate_recorded_part(record.get("commit_part", {}), COMMIT_SCHEMA)

    run_workers(targets, args, state, pr_parts_dir, commit_parts_dir, state_file)
    incomplete = sorted(
        batch_key
        for batch_key, record in state.get("batches", {}).items()
        if not isinstance(record, Mapping) or record.get("status") != "complete"
    )
    missing = sorted(target.state_key for _, target in targets if target.state_key not in state["batches"])
    if incomplete or missing:
        raise RuntimeError(
            f"Commit manifest has {len(incomplete) + len(missing)} incomplete repositories; rerun with --resume."
        )

    pr_parts = []
    commit_parts = []
    for index, target in targets:
        record = state["batches"][target.state_key]
        if int(record.get("part_index", -1)) != index:
            raise ValueError(f"Repository {target.repo_id} has an unexpected part index.")
        pr_parts.append(validate_recorded_part(record["pr_part"], PR_SCHEMA))
        commit_parts.append(validate_recorded_part(record["commit_part"], COMMIT_SCHEMA))
    state["outputs"] = {
        "pull_requests": finalize_parts(pr_parts, pr_output, PR_SCHEMA),
        "commits": finalize_parts(commit_parts, commit_output, COMMIT_SCHEMA),
    }
    summary = summarize(pr_output, commit_output)
    write_json(summary_file, summary)
    state["summary"] = {"path": str(summary_file.resolve()), "sha256": sha256_file(summary_file)}
    state["finalized"] = True
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(state_file, state)
    print(
        f"finalized_prs={summary['pull_request_rows']} finalized_commits={summary['commit_rows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
