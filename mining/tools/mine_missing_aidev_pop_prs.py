from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_rebalanced_dataset import load_config
from github_client import GitHubClient, github_tokens, parse_repo_full_name
from load_aidev import load_aidev_tables
from schema import atomic_write_text, ensure_datetime, write_parquet


AIDEV_START = pd.Timestamp("2024-12-24T00:23:09+00:00")
AIDEV_END = pd.Timestamp("2025-07-30T19:36:13+00:00")


def log(message: str) -> None:
    print(message, flush=True)


def aidev_pop_urls_by_repo(config: dict[str, Any]) -> pd.DataFrame:
    tables = load_aidev_tables(config)
    parts = []
    for arm, table_name in (("human", "human_pull_request"), ("agentic", "pull_request")):
        df = tables[table_name].copy()
        created = ensure_datetime(df["created_at"])
        df = df.loc[(created >= AIDEV_START) & (created <= AIDEV_END)].copy()
        df["aidev_arm"] = arm
        parts.append(df)
    combined = pd.concat(parts, ignore_index=True, sort=False).dropna(subset=["html_url"])
    combined["repo_full_name"] = combined["html_url"].astype(str).str.extract(r"https://github.com/([^/]+/[^/]+)/pull/")[0]
    combined["number_from_url"] = combined["html_url"].astype(str).str.extract(r"/pull/(\d+)")[0]
    combined["number_from_url"] = pd.to_numeric(combined["number_from_url"], errors="coerce").astype("Int64")
    return combined.drop_duplicates("html_url")


def existing_raw_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path, columns=["html_url"])
    return set(df["html_url"].dropna().astype(str))


def missing_prs(config: dict[str, Any], raw_path: Path, limit_prs: int | None) -> pd.DataFrame:
    aidev = aidev_pop_urls_by_repo(config)
    raw_urls = existing_raw_urls(raw_path)
    missing = aidev.loc[~aidev["html_url"].astype(str).isin(raw_urls)].copy()
    missing = missing.dropna(subset=["repo_full_name", "number_from_url"])
    missing = missing.sort_values(["repo_full_name", "number_from_url"])
    if limit_prs is not None:
        missing = missing.head(limit_prs)
    repo_counts = missing["repo_full_name"].dropna().value_counts()
    log(f"AIDev-Pop missing URLs: {len(missing)} across {len(repo_counts)} repos")
    return missing


def fetch_pr_direct(client: GitHubClient, repo_full_name: str, number: int) -> dict | None:
    try:
        pr = client.request(f"/repos/{repo_full_name}/pulls/{number}")
    except Exception as exc:
        log(f"  fetch_failed {repo_full_name}#{number}: {exc}")
        return None
    if not isinstance(pr, dict):
        return None
    parsed_repo = parse_repo_full_name(str(pr.get("url") or pr.get("html_url") or ""))
    pr["repo_full_name"] = parsed_repo or repo_full_name
    return pr


def append_parquet(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
        if "html_url" in combined.columns:
            combined = combined.drop_duplicates("html_url", keep="first")
    else:
        combined = new_df
    write_parquet(combined, path)


def save_state(path: Path, state: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(state, indent=2) + "\n")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed_repos": [], "reports": []}
    return json.loads(path.read_text(encoding="utf-8"))


def working_tokens(tokens: list[str]) -> list[str]:
    valid = []
    for index, token in enumerate(tokens, start=1):
        client = GitHubClient(tokens=[token], sleep_seconds=0)
        try:
            client.request("/rate_limit")
        except Exception as exc:
            log(f"[token-check] ignoring token {index}/{len(tokens)}: {exc}")
            continue
        valid.append(token)
    log(f"[token-check] using {len(valid)}/{len(tokens)} working token(s)")
    if not valid:
        raise ValueError("No working GitHub tokens available.")
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-mine AIDev-Pop PRs missing from the raw GitHub checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("mining/config.local.yaml"))
    parser.add_argument("--raw", type=Path, default=Path("mining/outputs_2026_06_01/raw/github_human_prs.parquet"))
    parser.add_argument("--output", type=Path, default=Path("mining/query_cache/missing_aidev_pop_prs.parquet"))
    parser.add_argument("--state", type=Path, default=Path("mining/query_cache/missing_aidev_pop_prs_state.json"))
    parser.add_argument("--limit-prs", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    missing = missing_prs(config, args.raw, args.limit_prs)
    tokens = working_tokens(github_tokens(config.get("github", {}).get("token_file")))
    client = GitHubClient(tokens=tokens, sleep_seconds=0.2)
    state = load_state(args.state)
    completed = set(state.get("completed_urls", []))

    for index, row in enumerate(missing.itertuples(index=False), start=1):
        html_url = str(row.html_url)
        if html_url in completed:
            continue
        repo = str(row.repo_full_name)
        number = int(row.number_from_url)
        log(f"[{index}/{len(missing)}] {repo}#{number}")
        pr = fetch_pr_direct(client, repo, number)
        if pr is None:
            state.setdefault("failed_urls", []).append(html_url)
            state.setdefault("reports", []).append({"html_url": html_url, "repo": repo, "number": number, "status": "failed"})
            save_state(args.state, state)
            continue
        append_parquet(args.output, [pr])
        completed.add(html_url)
        state.setdefault("reports", []).append({"html_url": html_url, "repo": repo, "number": number, "status": "completed"})
        state["completed_urls"] = sorted(completed)
        save_state(args.state, state)
        log("  rows=1")


if __name__ == "__main__":
    main()
