"""Collect GitHub code and discussion evidence for the final balanced sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from github_client import GitHubClient, GitHubClientError, GitHubNotFoundError
from schema import atomic_write_text, write_parquet


COMMON_COLUMNS = (
    "repo_id",
    "number",
    "repo_full_name",
    "sample_arm",
    "selection_hash",
    "snapshot_at",
)
TABLES = (
    "pull_requests",
    "pull_request_files",
    "commits",
    "issue_comments",
    "review_comments",
    "reviews",
    "workflow_runs",
    "check_runs",
    "collection_status",
)
TABLE_COLUMNS = {
    "pull_requests": (*COMMON_COLUMNS, "pr_id", "pr_node_id", "html_url", "state", "draft", "merged", "title", "body", "author_login", "author_type", "created_at", "updated_at", "closed_at", "merged_at", "base_sha", "head_sha", "commits_count", "comments_count", "review_comments_count", "additions", "deletions", "changed_files", "fetch_status", "identity_verified"),
    "pull_request_files": (*COMMON_COLUMNS, "file_index", "filename", "previous_filename", "status", "additions", "deletions", "changes", "patch", "patch_available", "blob_url"),
    "commits": (*COMMON_COLUMNS, "commit_index", "sha", "html_url", "message", "author_login", "author_date", "committer_date", "verified"),
    "issue_comments": (*COMMON_COLUMNS, "comment_id", "html_url", "author_login", "author_type", "author_association", "body", "created_at", "updated_at"),
    "review_comments": (*COMMON_COLUMNS, "comment_id", "review_id", "html_url", "author_login", "author_type", "author_association", "body", "created_at", "updated_at", "path", "diff_hunk", "commit_id"),
    "reviews": (*COMMON_COLUMNS, "review_id", "html_url", "author_login", "author_type", "author_association", "state", "body", "commit_id", "submitted_at"),
    "workflow_runs": (*COMMON_COLUMNS, "workflow_run_id", "workflow_id", "name", "display_title", "event", "status", "conclusion", "head_sha", "created_at", "updated_at", "html_url"),
    "check_runs": (*COMMON_COLUMNS, "checked_sha", "check_run_id", "name", "app_id", "app_slug", "status", "conclusion", "started_at", "completed_at", "details_url", "output_title", "output_summary"),
    "collection_status": (*COMMON_COLUMNS, "status", "pull_request_files_status", "commits_status", "issue_comments_status", "review_comments_status", "reviews_status", "workflow_runs_status", "check_runs_status"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sample(path: Path, expected_rows: int, expected_sha256: str) -> pd.DataFrame:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Balanced-sample SHA-256 does not match the configured official artifact.")
    frame = pd.read_parquet(path)
    required = {*COMMON_COLUMNS[:5], "selected", "id", "title", "body", "html_url"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Balanced sample is missing columns: " + ", ".join(missing))
    if len(frame) != expected_rows:
        raise ValueError(f"Balanced sample has {len(frame)} rows; expected {expected_rows}.")
    if frame.duplicated(["repo_id", "number"]).any():
        raise ValueError("Balanced sample contains duplicate immutable identities.")
    if not frame["selected"].eq(True).all():
        raise ValueError("Balanced sample contains unselected rows.")
    arm_counts = frame["sample_arm"].value_counts().to_dict()
    if set(arm_counts) != {"agentic", "human_candidate"} or len(set(arm_counts.values())) != 1:
        raise ValueError("Balanced sample is not the expected 1:1 final sample.")
    return frame


def _common(sample: dict[str, Any], snapshot_at: str) -> dict[str, Any]:
    return {
        "repo_id": int(sample["repo_id"]),
        "number": int(sample["number"]),
        "repo_full_name": str(sample["repo_full_name"]),
        "sample_arm": str(sample["sample_arm"]),
        "selection_hash": str(sample["selection_hash"]),
        "snapshot_at": snapshot_at,
    }


def _user(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    user = payload.get("user") or {}
    return user.get("login"), user.get("type")


def collect_pull_request(
    client: GitHubClient, sample: dict[str, Any], snapshot_at: str
) -> dict[str, list[dict[str, Any]]]:
    common = _common(sample, snapshot_at)
    repo = common["repo_full_name"]
    number = common["number"]
    base = f"/repos/{repo}"
    result = {name: [] for name in TABLES}
    endpoint_status: dict[str, str] = {}
    try:
        pr = client.request(f"{base}/pulls/{number}")
        if not isinstance(pr, dict):
            raise GitHubClientError("Pull-request response is not an object.")
    except GitHubNotFoundError:
        result["pull_requests"].append(
            {**common, "pr_id": int(sample["id"]), "html_url": sample["html_url"], "title": sample["title"], "body": sample["body"], "head_sha": None, "fetch_status": "not_found", "identity_verified": False}
        )
        result["collection_status"].append(
            {**common, "status": "not_found", "pull_request_files_status": "not_requested", "commits_status": "not_requested", "issue_comments_status": "not_requested", "review_comments_status": "not_requested", "reviews_status": "not_requested", "workflow_runs_status": "not_requested", "check_runs_status": "not_requested"}
        )
        return result

    repo_id = ((pr.get("base") or {}).get("repo") or {}).get("id")
    identity_verified = repo_id == common["repo_id"]
    head_sha = (pr.get("head") or {}).get("sha")
    author_login, author_type = _user(pr)
    result["pull_requests"].append(
        {
            **common,
            "pr_id": pr.get("id"),
            "pr_node_id": pr.get("node_id"),
            "html_url": pr.get("html_url") or sample["html_url"],
            "state": pr.get("state"),
            "draft": pr.get("draft"),
            "merged": pr.get("merged"),
            "title": pr.get("title") or sample["title"],
            "body": pr.get("body") if pr.get("body") is not None else sample["body"],
            "author_login": author_login,
            "author_type": author_type,
            "created_at": pr.get("created_at"),
            "updated_at": pr.get("updated_at"),
            "closed_at": pr.get("closed_at"),
            "merged_at": pr.get("merged_at"),
            "base_sha": (pr.get("base") or {}).get("sha"),
            "head_sha": head_sha,
            "commits_count": pr.get("commits"),
            "comments_count": pr.get("comments"),
            "review_comments_count": pr.get("review_comments"),
            "additions": pr.get("additions"),
            "deletions": pr.get("deletions"),
            "changed_files": pr.get("changed_files"),
            "fetch_status": "complete" if identity_verified else "identity_mismatch",
            "identity_verified": identity_verified,
        }
    )
    if not identity_verified:
        result["collection_status"].append({**common, "status": "identity_mismatch"})
        return result

    endpoints = (
        ("pull_request_files", f"{base}/pulls/{number}/files", None),
        ("commits", f"{base}/pulls/{number}/commits", None),
        ("issue_comments", f"{base}/issues/{number}/comments", None),
        ("review_comments", f"{base}/pulls/{number}/comments", None),
        ("reviews", f"{base}/pulls/{number}/reviews", None),
    )
    payloads: dict[str, list[dict[str, Any]]] = {}
    for table, path, item_key in endpoints:
        try:
            payloads[table] = client.request_paginated_list(path, item_key=item_key)
            endpoint_status[table] = "complete"
        except (GitHubClientError, GitHubNotFoundError):
            payloads[table] = []
            endpoint_status[table] = "failed"

    for index, item in enumerate(payloads["pull_request_files"]):
        result["pull_request_files"].append(
            {**common, "file_index": index, "filename": item.get("filename"), "previous_filename": item.get("previous_filename"), "status": item.get("status"), "additions": item.get("additions"), "deletions": item.get("deletions"), "changes": item.get("changes"), "patch": item.get("patch"), "patch_available": bool(item.get("patch")), "blob_url": item.get("blob_url")}
        )
    for index, item in enumerate(payloads["commits"]):
        commit = item.get("commit") or {}
        author = item.get("author") or {}
        result["commits"].append(
            {**common, "commit_index": index, "sha": item.get("sha"), "html_url": item.get("html_url"), "message": commit.get("message"), "author_login": author.get("login"), "author_date": (commit.get("author") or {}).get("date"), "committer_date": (commit.get("committer") or {}).get("date"), "verified": (commit.get("verification") or {}).get("verified")}
        )
    for table in ("issue_comments", "review_comments"):
        for item in payloads[table]:
            login, user_type = _user(item)
            row = {**common, "comment_id": item.get("id"), "html_url": item.get("html_url"), "author_login": login, "author_type": user_type, "author_association": item.get("author_association"), "body": item.get("body"), "created_at": item.get("created_at"), "updated_at": item.get("updated_at")}
            if table == "review_comments":
                row.update({"review_id": item.get("pull_request_review_id"), "path": item.get("path"), "diff_hunk": item.get("diff_hunk"), "commit_id": item.get("commit_id")})
            result[table].append(row)
    for item in payloads["reviews"]:
        login, user_type = _user(item)
        result["reviews"].append(
            {**common, "review_id": item.get("id"), "html_url": item.get("html_url"), "author_login": login, "author_type": user_type, "author_association": item.get("author_association"), "state": item.get("state"), "body": item.get("body"), "commit_id": item.get("commit_id"), "submitted_at": item.get("submitted_at")}
        )

    for table, path, key in (
        ("workflow_runs", f"{base}/actions/runs", "workflow_runs"),
        ("check_runs", f"{base}/commits/{head_sha}/check-runs", "check_runs"),
    ):
        try:
            params = {"event": "pull_request", "head_sha": head_sha} if table == "workflow_runs" else None
            payloads[table] = client.request_paginated_list(path, params=params, item_key=key)
            endpoint_status[table] = "complete"
        except (GitHubClientError, GitHubNotFoundError):
            payloads[table] = []
            endpoint_status[table] = "failed"
    for item in payloads["workflow_runs"]:
        if item.get("head_sha") != head_sha:
            continue
        result["workflow_runs"].append(
            {**common, "workflow_run_id": item.get("id"), "workflow_id": item.get("workflow_id"), "name": item.get("name"), "display_title": item.get("display_title"), "event": item.get("event"), "status": item.get("status"), "conclusion": item.get("conclusion"), "head_sha": item.get("head_sha"), "created_at": item.get("created_at"), "updated_at": item.get("updated_at"), "html_url": item.get("html_url")}
        )
    for item in payloads["check_runs"]:
        app = item.get("app") or {}
        output = item.get("output") or {}
        result["check_runs"].append(
            {**common, "checked_sha": head_sha, "check_run_id": item.get("id"), "name": item.get("name"), "app_id": app.get("id"), "app_slug": app.get("slug"), "status": item.get("status"), "conclusion": item.get("conclusion"), "started_at": item.get("started_at"), "completed_at": item.get("completed_at"), "details_url": item.get("details_url"), "output_title": output.get("title"), "output_summary": output.get("summary")}
        )
    status = "complete" if all(value == "complete" for value in endpoint_status.values()) else "partial"
    result["collection_status"].append(
        {**common, "status": status, **{f"{name}_status": endpoint_status.get(name, "not_requested") for name in ("pull_request_files", "commits", "issue_comments", "review_comments", "reviews", "workflow_runs", "check_runs")}}
    )
    return result


def _fragment_path(output_dir: Path, repo_id: int, number: int) -> Path:
    return output_dir / "fragments" / f"{repo_id}_{number}.json"


def collect_sample_evidence(config: dict[str, Any], *, resume: bool = False, limit: int | None = None) -> dict[str, Any]:
    input_config = config["input"]
    output_dir = Path(config["output"]["dir"])
    sample = validate_sample(Path(input_config["balanced_sample"]), int(input_config["expected_rows"]), str(input_config["expected_sha256"]))
    state_path = output_dir / "state.json"
    if state_path.exists():
        if not resume:
            raise FileExistsError("Evidence state exists; pass --resume.")
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"version": 1, "input_sha256": input_config["expected_sha256"], "snapshot_at": datetime.now(timezone.utc).isoformat(), "completed": []}
    if state["input_sha256"] != input_config["expected_sha256"]:
        raise ValueError("Resume state has an incompatible input signature.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fragments").mkdir(exist_ok=True)
    completed = set(state["completed"])
    client = GitHubClient.from_token_file(config["github"]["token_file"])
    pending = sample.loc[~sample.apply(lambda row: f"{int(row.repo_id)}:{int(row.number)}" in completed, axis=1)]
    if limit is not None:
        pending = pending.head(limit)
    for position, row in enumerate(pending.to_dict("records"), start=1):
        key = f"{int(row['repo_id'])}:{int(row['number'])}"
        result = collect_pull_request(client, row, state["snapshot_at"])
        fragment = _fragment_path(output_dir, int(row["repo_id"]), int(row["number"]))
        atomic_write_text(fragment, json.dumps(result, ensure_ascii=True) + "\n")
        if result["collection_status"][0]["status"] in {"complete", "not_found", "identity_mismatch"}:
            completed.add(key)
        state["completed"] = sorted(completed)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_text(state_path, json.dumps(state, indent=2) + "\n")
        print(f"[{position}/{len(pending)}] {key} {result['collection_status'][0]['status']}", flush=True)
    finalize_evidence(output_dir)
    return state


def finalize_evidence(output_dir: Path) -> None:
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLES}
    for path in sorted((output_dir / "fragments").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for name in TABLES:
            tables[name].extend(payload.get(name, []))
    final_dir = output_dir / "final"
    for name, rows in tables.items():
        frame = pd.DataFrame(rows).reindex(columns=TABLE_COLUMNS[name])
        write_parquet(frame, final_dir / f"{name}.parquet")
    status = pd.DataFrame(tables["collection_status"])
    summary = {"snapshot_at": status["snapshot_at"].iloc[0] if not status.empty else None, "sample_rows": len(status), "status_counts": status["status"].value_counts().to_dict(), "table_rows": {name: len(rows) for name, rows in tables.items()}}
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    collect_sample_evidence(config, resume=args.resume, limit=args.limit)


if __name__ == "__main__":
    main()
