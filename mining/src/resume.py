from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from github_client import parse_repo_full_name
from schema import atomic_write_text, write_parquet


RESUME_STATE_FILENAME = "rebalancing_state.json"
CHECKPOINT_DROP_COLUMNS = ("files", "commits")


def resume_state_path(output_dir: Path) -> Path:
    return output_dir / "state" / RESUME_STATE_FILENAME


def reset_output_tree(output_dir: Path) -> None:
    shutil.rmtree(output_dir, ignore_errors=True)


def load_resume_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_resume_state(path: Path, state: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(state, indent=2) + "\n")


def load_checkpoint_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        parquet_file = pq.ParquetFile(path)
        columns = [
            name
            for name in parquet_file.schema_arrow.names
            if name not in CHECKPOINT_DROP_COLUMNS
        ]
        return pd.read_parquet(path, columns=columns)


def checkpoint_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[column for column in CHECKPOINT_DROP_COLUMNS if column in df.columns])


def save_checkpoint_dataframe(df: pd.DataFrame, path: Path) -> None:
    write_parquet(checkpoint_dataframe(df), path)


def completed_repo_names_from_checkpoint(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    completed: set[str] = set()
    for column in ("repo_full_name", "full_name", "name_with_owner", "repo_url", "html_url", "url"):
        if column not in df.columns:
            continue
        for value in df[column].dropna().tolist():
            repo_name = parse_repo_full_name(value)
            if repo_name:
                completed.add(repo_name)
    return completed


def initial_resume_state(signature: dict[str, Any], repo_count: int) -> dict[str, Any]:
    return {
        "version": 2,
        "signature": signature,
        "repo_count": int(repo_count),
        "completed_repo_names": [],
        "completed_repo_count": 0,
        "raw_rows_count": 0,
        "raw_checkpoint_sha256": None,
        "last_completed_repo": None,
        "repo_mining_reports": [],
        "pr_failures": [],
        "updated_at": None,
    }


def update_resume_state(
    state: dict[str, Any],
    completed_repo_names: set[str],
    raw_rows_count: int,
    last_completed_repo: str | None = None,
) -> dict[str, Any]:
    updated = dict(state)
    completed_sorted = sorted(completed_repo_names)
    updated["completed_repo_names"] = completed_sorted
    updated["completed_repo_count"] = len(completed_sorted)
    updated["raw_rows_count"] = int(raw_rows_count)
    if last_completed_repo is not None:
        updated["last_completed_repo"] = last_completed_repo
    updated["updated_at"] = datetime.now(timezone.utc).isoformat()
    return updated


def completed_repo_set(state: dict[str, Any]) -> set[str]:
    return set(state.get("completed_repo_names", []))


def state_matches_signature(state: dict[str, Any], signature: dict[str, Any]) -> bool:
    if not state:
        return False
    return state.get("signature") == signature
