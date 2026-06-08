from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


AGENT_NAMES = (
    "openai codex",
    "codex",
    "devin",
    "github copilot",
    "copilot",
    "cursor",
    "claude code",
    "claude",
)

AGENT_LOGIN_PATTERNS = (
    "codex",
    "openai",
    "devin",
    "copilot",
    "github-actions",
    "claude",
    "cursor",
)

CONFIG_FILE_SUFFIXES = (
    ".yml",
    ".yaml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".json",
    ".lock",
)

CONFIG_FILE_PREFIXES = (
    ".github/",
    ".circleci/",
    ".travis",
    ".pre-commit",
    "config/",
    "configs/",
)

PR_ID_COLUMNS = ("pr_id", "pull_request_id", "id")
REPO_ID_COLUMNS = ("repo_id", "repository_id")
REPO_NAME_COLUMNS = ("repo_full_name", "full_name", "name_with_owner")
STAR_COLUMNS = ("stargazers_count", "stars", "star_count", "watchers_count")
TASK_TYPE_COLUMNS = ("type", "task_type", "label")


@dataclass(frozen=True)
class TimeWindow:
    start: pd.Timestamp
    end: pd.Timestamp

    def to_dict(self) -> dict[str, str]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def require_column(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    column = first_existing_column(df, candidates)
    if column is None:
        expected = ", ".join(candidates)
        raise ValueError(f"{label} requires one of these columns: {expected}")
    return column


def ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def normalize_type(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    return text or None


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "raw": output_dir / "raw",
        "intermediate": output_dir / "intermediate",
        "final": output_dir / "final",
        "state": output_dir / "state",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
