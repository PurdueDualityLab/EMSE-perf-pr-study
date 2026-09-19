"""Recover the GitHub creation timestamp of every RQ2 PR identity.

The balanced sample stores `created_at` and the ISO-week strata, but that
parquet lives in the private dataset submodule. This script rebuilds the same
immutable field from GitHub so the temporal analysis can run from the compact
labels published in this repository. It batches GraphQL aliases through the
authenticated `gh` CLI and falls back to REST for identities GraphQL cannot
resolve, such as renamed repositories.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

KEY_COLUMNS = ["repo_id", "number"]
OUTPUT_COLUMNS = [*KEY_COLUMNS, "repo_full_name", "created_at", "source"]


def run_gh(arguments: list[str]) -> dict:
    process = subprocess.run(
        ["gh", *arguments], capture_output=True, text=True, check=False
    )
    if process.returncode != 0 and not process.stdout.strip():
        raise RuntimeError(f"gh {' '.join(arguments)} failed: {process.stderr.strip()}")
    return json.loads(process.stdout)


def graphql_batch(rows: pd.DataFrame) -> dict[int, str]:
    parts = []
    for position, row in enumerate(rows.itertuples(index=False)):
        owner, _, name = row.repo_full_name.partition("/")
        parts.append(
            f'  p{position}: repository(owner:{json.dumps(owner)}, name:{json.dumps(name)}) '
            f'{{ pullRequest(number:{int(row.number)}) {{ createdAt }} }}'
        )
    payload = run_gh(["api", "graphql", "-f", "query={\n" + "\n".join(parts) + "\n}"])
    data = payload.get("data") or {}
    resolved: dict[int, str] = {}
    for position in range(len(rows)):
        node = data.get(f"p{position}") or {}
        pull_request = node.get("pullRequest") or {}
        created_at = pull_request.get("createdAt")
        if created_at:
            resolved[position] = created_at
    return resolved


def rest_created_at(repo_full_name: str, number: int) -> str:
    payload = run_gh(["api", f"repos/{repo_full_name}/pulls/{number}"])
    created_at = payload.get("created_at")
    if not created_at:
        raise RuntimeError(f"{repo_full_name}#{number} has no created_at")
    return created_at


def fetch(frame: pd.DataFrame, *, batch_size: int, known: pd.DataFrame) -> pd.DataFrame:
    resolved = known.set_index(KEY_COLUMNS) if len(known) else None
    records: list[dict[str, object]] = []
    pending = []
    for row in frame.itertuples(index=False):
        key = (row.repo_id, row.number)
        if resolved is not None and key in resolved.index:
            cached = resolved.loc[key]
            records.append({
                "repo_id": row.repo_id,
                "number": row.number,
                "repo_full_name": row.repo_full_name,
                "created_at": cached["created_at"],
                "source": cached["source"],
            })
            continue
        pending.append(row)
    print(f"cached {len(records)}, fetching {len(pending)}", flush=True)
    for start in range(0, len(pending), batch_size):
        chunk = pd.DataFrame(pending[start:start + batch_size])
        found = graphql_batch(chunk)
        for position, row in enumerate(chunk.itertuples(index=False)):
            created_at = found.get(position)
            source = "graphql"
            if created_at is None:
                created_at = rest_created_at(row.repo_full_name, int(row.number))
                source = "rest"
            records.append({
                "repo_id": row.repo_id,
                "number": row.number,
                "repo_full_name": row.repo_full_name,
                "created_at": created_at,
                "source": source,
            })
        print(f"fetched {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)
    result = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    return result.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=root / "analysis/classification_labels/rq2_labels.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "pr_created_at.csv")
    parser.add_argument("--batch-size", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels, usecols=[*KEY_COLUMNS, "repo_full_name"])
    if labels.duplicated(KEY_COLUMNS).any():
        raise ValueError("Labels contain duplicate PR identities.")
    known = (
        pd.read_csv(args.output)
        if args.output.exists()
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    result = fetch(labels, batch_size=args.batch_size, known=known)
    if len(result) != len(labels) or result["created_at"].isna().any():
        raise ValueError("Creation timestamps are incomplete.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(json.dumps({"rows": len(result), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
