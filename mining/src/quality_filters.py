from __future__ import annotations

import ast
from collections import Counter
from typing import Any
from typing import Iterable

import pandas as pd

from schema import CONFIG_FILE_PREFIXES, CONFIG_FILE_SUFFIXES


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None or pd.isna(value):
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (SyntaxError, ValueError):
            pass
        return [part.strip() for part in value.split(",") if part.strip()]
    return [value]


def _json_safe(value: Any) -> Any:
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


def filenames_for_row(row: pd.Series) -> list[str]:
    for column in ("filenames", "files", "changed_files_list"):
        if column in row and row[column] is not None:
            values = _as_list(row[column])
            filenames = []
            for item in values:
                if isinstance(item, dict):
                    name = item.get("filename") or item.get("path") or item.get("name")
                else:
                    name = item
                if name is not None:
                    filenames.append(str(name))
            return filenames
    if "filename" in row and row["filename"] is not None:
        return [str(row["filename"])]
    return []


def is_config_file(filename: str) -> bool:
    path = filename.strip().lower()
    if not path:
        return False
    if path in {"dockerfile", "makefile"}:
        return True
    if any(path.startswith(prefix) for prefix in CONFIG_FILE_PREFIXES):
        return True
    return path.endswith(CONFIG_FILE_SUFFIXES)


def is_config_only_row(row: pd.Series) -> bool:
    filenames = filenames_for_row(row)
    return bool(filenames) and all(is_config_file(name) for name in filenames)


def is_empty_filename_row(row: pd.Series) -> bool:
    filenames = filenames_for_row(row)
    return not filenames or any(not str(name).strip() for name in filenames)


def is_deleted_repo_row(row: pd.Series) -> bool:
    for column in ("deleted_repo", "repo_deleted", "repository_deleted"):
        if column in row:
            value = row[column]
            if isinstance(value, bool):
                return value
            if str(value).strip().lower() in {"true", "1", "yes"}:
                return True
    return False


def is_merge_only_row(row: pd.Series) -> bool:
    for column in ("is_merge_commit", "merge_only", "is_merge_only"):
        if column in row:
            value = row[column]
            if isinstance(value, bool):
                return value
            if str(value).strip().lower() in {"true", "1", "yes"}:
                return True

    messages: Iterable[object] = []
    for column in ("commit_messages", "commits"):
        if column in row:
            messages = _as_list(row[column])
            break
    message_text = " ".join(
        item.get("message", "") if isinstance(item, dict) else str(item)
        for item in messages
    ).strip().lower()
    title = str(row.get("title") or "").strip().lower()
    return bool(message_text or title) and (
        message_text.startswith("merge ")
        or title.startswith("merge ")
        or "merge branch" in message_text
    )


def quality_filter_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)
    flags["empty_filename"] = df.apply(is_empty_filename_row, axis=1)
    flags["config_only"] = df.apply(is_config_only_row, axis=1)
    flags["deleted_repo"] = df.apply(is_deleted_repo_row, axis=1)
    flags["merge_only"] = df.apply(is_merge_only_row, axis=1)
    return flags


def quality_filter_removed_record(row: pd.Series, reason: str) -> dict[str, Any]:
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
    ):
        if column in row and row[column] is not None and not pd.isna(row[column]):
            value = row[column]
            if isinstance(value, (list, tuple)):
                record[column] = [str(_json_safe(item)) for item in value]
            else:
                record[column] = str(value) if column == "title" else _json_safe(value)
    filenames = filenames_for_row(row)
    if filenames:
        record["filenames"] = filenames
    return record


def apply_quality_filters_with_details(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, list[dict[str, Any]]]]:
    removed_records: dict[str, list[dict[str, Any]]] = {
        "empty_filename": [],
        "config_only": [],
        "deleted_repo": [],
        "merge_only": [],
    }
    if df.empty:
        return df.copy(), {key: 0 for key in removed_records}, removed_records

    flags = quality_filter_flags(df)
    removed_counts = Counter()
    keep = pd.Series(True, index=df.index)
    for reason in ("empty_filename", "config_only", "deleted_repo", "merge_only"):
        reason_mask = flags[reason] & keep
        removed_counts[reason] = int(reason_mask.sum())
        if reason_mask.any():
            for _, row in df.loc[reason_mask].iterrows():
                removed_records[reason].append(quality_filter_removed_record(row, reason))
        keep &= ~flags[reason]

    return df.loc[keep].copy(), dict(removed_counts), removed_records


def apply_quality_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if df.empty:
        return df.copy(), {
            "empty_filename": 0,
            "config_only": 0,
            "deleted_repo": 0,
            "merge_only": 0,
        }

    filtered, removed_counts, _ = apply_quality_filters_with_details(df)
    return filtered, removed_counts
