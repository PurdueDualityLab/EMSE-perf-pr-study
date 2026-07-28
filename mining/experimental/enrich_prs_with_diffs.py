from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_rebalanced_dataset import load_config
from github_client import GitHubClient, github_tokens
from schema import write_parquet


def as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        return as_list(value.tolist())
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass
    return [str(value)]


def working_tokens(tokens: list[str]) -> list[str]:
    valid = []
    for index, token in enumerate(tokens, start=1):
        client = GitHubClient(tokens=[token], sleep_seconds=0)
        try:
            client.request("/rate_limit")
        except Exception as exc:
            print(f"[token-check] ignoring token {index}/{len(tokens)}: {exc}", flush=True)
            continue
        valid.append(token)
    print(f"[token-check] using {len(valid)}/{len(tokens)} working token(s)", flush=True)
    if not valid:
        raise ValueError("No working GitHub tokens available.")
    return valid


def existing_output(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def key(row: pd.Series) -> str:
    html_url_value = row.get("html_url")
    html_url = "" if html_url_value is None or bool(pd.isna(html_url_value)) else str(html_url_value)
    if html_url:
        return html_url
    return f"{row.get('repo_full_name')}#{row.get('number')}"


def write_rows(rows: list[dict], path: Path) -> None:
    result = pd.DataFrame(rows)
    if "html_url" in result.columns:
        result = result.drop_duplicates(subset=["html_url"], keep="last")
    write_parquet(result, path)


def fetch_files(client: GitHubClient, repo: str, number: int, max_files: int, max_patch_chars: int) -> dict[str, Any]:
    files = client.fetch_pull_files(repo, number)[:max_files]
    filenames = []
    statuses = []
    patches = []
    total_patch_chars = 0
    for item in files:
        filename = str(item.get("filename") or "")
        status = str(item.get("status") or "")
        patch = str(item.get("patch") or "")
        remaining = max(max_patch_chars - total_patch_chars, 0)
        if remaining <= 0:
            patch = ""
        elif len(patch) > remaining:
            patch = patch[:remaining]
        total_patch_chars += len(patch)
        filenames.append(filename)
        statuses.append(status)
        patches.append(patch)
    return {
        "diff_filenames": filenames,
        "diff_statuses": statuses,
        "diff_patches": patches,
        "diff_patch_chars": total_patch_chars,
    }


def fetch_commits(client: GitHubClient, repo: str, number: int) -> list[str]:
    commits = client.fetch_pull_commits(repo, number)
    messages = []
    for item in commits:
        message = item.get("commit", {}).get("message")
        if message:
            messages.append(str(message))
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich a PR parquet with GitHub diff patches and commit messages.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("mining/config.local.yaml"))
    parser.add_argument("--max-files-per-pr", type=int, default=50)
    parser.add_argument("--max-patch-chars", type=int, default=50000)
    parser.add_argument("--batch-write-size", type=int, default=100)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    client = GitHubClient(tokens=working_tokens(github_tokens(config.get("github", {}).get("token_file"))), sleep_seconds=0.2)
    source = pd.read_parquet(args.input)
    if args.limit is not None:
        source = source.head(args.limit).copy()

    existing = existing_output(args.output)
    done = set(existing.loc[existing.get("diff_fetch_status", pd.Series(dtype=str)).eq("ok")].apply(key, axis=1)) if not existing.empty else set()
    rows = existing.to_dict("records") if not existing.empty else []
    pending_batch = 0

    for index, row in source.iterrows():
        row_key = key(row)
        if row_key in done:
            continue
        record = row.to_dict()
        repo = str(row.get("repo_full_name") or "")
        number = row.get("number")
        try:
            number_int = int(number)
            file_data = fetch_files(client, repo, number_int, args.max_files_per_pr, args.max_patch_chars)
            commit_messages = fetch_commits(client, repo, number_int)
            record.update(file_data)
            record["commit_messages_enriched"] = commit_messages or as_list(row.get("commit_messages"))
            record["diff_fetch_status"] = "ok"
            record["diff_fetch_error"] = ""
        except Exception as exc:
            record["diff_filenames"] = []
            record["diff_statuses"] = []
            record["diff_patches"] = []
            record["diff_patch_chars"] = 0
            record["commit_messages_enriched"] = as_list(row.get("commit_messages"))
            record["diff_fetch_status"] = "failed"
            record["diff_fetch_error"] = str(exc)
        rows.append(record)
        pending_batch += 1
        if pending_batch >= args.batch_write_size:
            write_rows(rows, args.output)
            pending_batch = 0
            print(f"processed={index + 1} rows_written={len(rows)}", flush=True)

    write_rows(rows, args.output)


if __name__ == "__main__":
    main()
