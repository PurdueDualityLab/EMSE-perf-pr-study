from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from classify_task_type import compatible_task_type
from schema import atomic_write_text, write_parquet


IDENTITY_COLUMNS = ("repo_id", "number")
TITLE_PERF_PREFIX = re.compile(r"^\s*perf(?:\([^)]+\))?:", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_unique_identities(frame: pd.DataFrame, label: str) -> None:
    missing = [column for column in IDENTITY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing identity columns: {', '.join(missing)}")
    if frame[list(IDENTITY_COLUMNS)].isna().any().any():
        raise ValueError(f"{label} contains null immutable identities.")
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{label} contains duplicate immutable identities.")


def join_raw_and_attribution(raw: pd.DataFrame, attribution: pd.DataFrame) -> pd.DataFrame:
    require_unique_identities(raw, "Raw input")
    require_unique_identities(attribution, "Attribution input")
    raw_keys = set(map(tuple, raw[list(IDENTITY_COLUMNS)].itertuples(index=False, name=None)))
    attribution_keys = set(
        map(tuple, attribution[list(IDENTITY_COLUMNS)].itertuples(index=False, name=None))
    )
    if raw_keys != attribution_keys:
        raise ValueError("Raw and attribution inputs have nonmatching immutable identities.")

    added_columns = [
        column for column in attribution.columns if column not in raw.columns and column not in IDENTITY_COLUMNS
    ]
    return raw.merge(
        attribution[[*IDENTITY_COLUMNS, *added_columns]],
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    )


def parse_timestamp(value: str | None, argument: str) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{argument} must be an ISO timestamp with a timezone.")
    return timestamp.tz_convert("UTC")


def add_decisions(
    frame: pd.DataFrame,
    heuristic: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("--start-date must be earlier than or equal to --end-date.")
    result = frame.copy()
    if start_date is not None or end_date is not None:
        if "created_at" not in result.columns:
            raise ValueError("Date filtering requires a created_at column.")
        timestamps = pd.to_datetime(result["created_at"], utc=True, errors="coerce")
        result["date_eligible"] = timestamps.notna()
        result["date_reason"] = "invalid_created_at"
        if start_date is not None:
            before_start = timestamps < start_date
            result.loc[before_start, "date_eligible"] = False
            result.loc[before_start, "date_reason"] = "before_start_date"
        if end_date is not None:
            after_end = timestamps > end_date
            result.loc[after_end, "date_eligible"] = False
            result.loc[after_end, "date_reason"] = "after_end_date"
        in_window = timestamps.notna()
        if start_date is not None:
            in_window &= timestamps >= start_date
        if end_date is not None:
            in_window &= timestamps <= end_date
        result.loc[in_window, "date_reason"] = "within_window"
    else:
        result["date_eligible"] = True
        result["date_reason"] = "no_date_window"

    if heuristic == "project":
        classifications = result.apply(compatible_task_type, axis=1)
        result["selection_task_type"] = classifications.map(lambda value: value[0])
        result["selection_reason"] = classifications.map(lambda value: value[2])
        result["heuristic_match"] = result["selection_task_type"].eq("perf")
    else:
        titles = result.get("title", pd.Series(pd.NA, index=result.index))
        result["heuristic_match"] = titles.fillna("").astype(str).str.match(TITLE_PERF_PREFIX)
        result["selection_task_type"] = pd.NA
        result["selection_reason"] = result["heuristic_match"].map(
            {True: "title_perf_prefix", False: "title_not_perf_prefix"}
        )

    result["selection_heuristic"] = heuristic
    result["selected"] = result["date_eligible"] & result["heuristic_match"]
    return result


def select_candidates(
    raw_path: Path,
    attribution_path: Path,
    output_dir: Path,
    heuristic: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    raw = pd.read_parquet(raw_path)
    attribution = pd.read_parquet(attribution_path)
    decisions = add_decisions(join_raw_and_attribution(raw, attribution), heuristic, start_date, end_date)
    candidates = decisions.loc[decisions["selected"]].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(decisions, output_dir / "decisions.parquet")
    write_parquet(candidates, output_dir / "candidates.parquet")
    summary = {
        "raw_rows": len(raw),
        "attribution_rows": len(attribution),
        "decision_rows": len(decisions),
        "candidate_rows": len(candidates),
        "heuristic": heuristic,
        "start_date": start_date.isoformat() if start_date is not None else None,
        "end_date": end_date.isoformat() if end_date is not None else None,
        "raw_sha256": sha256_file(raw_path),
        "attribution_sha256": sha256_file(attribution_path),
        "note": "title_perf_prefix is provisional and not a verified AIDev reproduction.",
    }
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--attribution", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--heuristic", required=True, choices=("project", "title_perf_prefix"))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = select_candidates(
        args.raw,
        args.attribution,
        args.output_dir,
        args.heuristic,
        parse_timestamp(args.start_date, "--start-date"),
        parse_timestamp(args.end_date, "--end-date"),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
