"""Validate and publish the approved study artifacts to a private HF dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_DIR = ROOT / "mining" / "release_bundle"
ABSOLUTE_PATH_PATTERN = re.compile(r"(?:^|[\s\"'])(?:/home/|/Users/|[A-Za-z]:\\)")


@dataclass(frozen=True)
class Artifact:
    local_path: str
    remote_path: str
    stage: str
    rows: int
    sha256: str


ARTIFACTS = (
    Artifact("mining/outputs_2026_06_01/raw/github_human_prs.parquet", "data/raw/github_pull_requests.parquet", "mining", 1_603_213, "a3d737ce2cc32e15b6e14b91578217873fcd9739bebc664fdc04f31209256bef"),
    Artifact("mining/aidev_attribution_all_available_dates/aidev_all_available_dates.parquet", "data/agentic/aidev_attribution.parquet", "agentic_attribution", 1_603_213, "305238366a898fc95a963a96f86d026166d3d4f68a968711b7aee3fe16d8b6a0"),
    Artifact("mining/experiments/aidev_luna_full_v1/task_type_decisions_with_retry.parquet", "data/performance/task_type_decisions.parquet", "performance_classification", 1_603_213, "16a9a281029321541c7370c3445a69c896c05dd7fce2a0dbcb4c7ce70b2afbf4"),
    Artifact("mining/aidev_human_candidates_v1/decisions.parquet", "data/human/decisions.parquet", "human_filtering", 28_127, "72922a804a87063867927f12dc39e8eabfb3a1230905153aa9ff31ba13b5105b"),
    Artifact("mining/aidev_human_candidates_v1/human_candidates.parquet", "data/human/human_candidates.parquet", "human_filtering", 24_680, "b08525d2c6e634c6ed667000bf24d22cc956b85ca5a844a6acbbeb24ff79637f"),
    Artifact("mining/aidev_human_candidates_v1/excluded_from_human_candidates.parquet", "data/human/excluded.parquet", "human_filtering", 3_447, "5c6813a81251e963bac43ec03f5a5249d24d4ab5125b2734544d399a1b9a6e19"),
    Artifact("mining/aidev_weekly_balanced_sample_v1/agentic_sample.parquet", "data/sample/agentic_sample.parquet", "weekly_sampling", 1_356, "d2e18627c868a67491c7580d50cb57d275ec21813c41d4e68d80701849eb1cb1"),
    Artifact("mining/aidev_weekly_balanced_sample_v1/human_sample.parquet", "data/sample/human_sample.parquet", "weekly_sampling", 1_356, "1891c8c679ee82df50fe910e2379b4ff55a6f310e7a2662524ccfc71e44da6e5"),
    Artifact("mining/aidev_weekly_balanced_sample_v1/balanced_sample.parquet", "data/sample/balanced_sample.parquet", "weekly_sampling", 2_712, "8f72f6d4d92a6817abbbcdc06db61ec45e0674560eaca93c6071e1f85166d078"),
    Artifact("mining/aidev_weekly_balanced_sample_v1/sampling_manifest.parquet", "data/sample/sampling_manifest.parquet", "weekly_sampling", 26_036, "4ef5ff011a65162ae7717a064eb8dcf34aab6c14d69f1eb7fb46e8a20d3096c2"),
    Artifact("mining/aidev_weekly_balanced_sample_v1/weekly_strata.parquet", "data/sample/weekly_strata.parquet", "weekly_sampling", 67, "d3a7db0c0fd1f1abab1e1d4239958c94074605723ff1609712e729ba7477ca62"),
)

SUMMARIES = {
    "mining/aidev_attribution_all_available_dates/summary.json": "metadata/agentic_summary.json",
    "mining/experiments/aidev_luna_full_v1/summary_with_retry.json": "metadata/performance_summary.json",
    "mining/aidev_human_candidates_v1/summary.json": "metadata/human_filter_summary.json",
    "mining/aidev_weekly_balanced_sample_v1/summary.json": "metadata/sampling_summary.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_value(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: portable_value(item, path_map) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_value(item, path_map) for item in value]
    if not isinstance(value, str):
        return value
    normalized = str(Path(value).resolve()) if Path(value).is_absolute() else value
    if normalized in path_map:
        return path_map[normalized]
    if Path(value).is_absolute():
        return Path(value).name
    return value


def assert_portable(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if ABSOLUTE_PATH_PATTERN.search(text):
        raise ValueError(f"Generated metadata contains an absolute user path: {path}")
    lowered = text.lower()
    for marker in ("github_token", "openai_api_key", "hf_token"):
        if marker in lowered:
            raise ValueError(f"Generated metadata contains a secret marker: {marker}")


def dataset_card() -> str:
    return """---
license: other
task_categories:
- text-classification
tags:
- software-engineering
- pull-requests
- performance
- coding-agents
---

# Performance Pull Request Study

This private dataset contains the approved artifacts for a study comparing
agentic and human-candidate performance pull requests.

## Cohorts

- Raw pull requests: 1,603,213
- Performance pull requests: 29,483
- Agentic performance pull requests: 1,356
- Strict human candidates: 24,680
- Final balanced sample: 1,356 agentic and 1,356 human candidates
- Weekly strata: 67 ISO weeks in UTC

"Human candidate" means no selected observable coding-agent signal was found;
it does not establish human authorship. The performance classifier retains six
explicit API errors, which are not included among performance-labeled rows.

## Layout

The `data/` directory follows the five pipeline stages. `metadata/` contains
sanitized summaries, a manifest, and checksums. Operational checkpoints, API
payloads, retries, secrets, and local paths are intentionally excluded.

## Terms

The repository's processing code is MIT licensed. This dataset contains public
GitHub metadata and derived labels; no blanket license is asserted over source
repository content. Users are responsible for complying with GitHub and source
repository terms.
"""


def build_bundle(bundle_dir: Path) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    path_map = {
        str((ROOT / artifact.local_path).resolve()): artifact.remote_path
        for artifact in ARTIFACTS
    }
    manifest_entries = []
    uploads: list[tuple[Path, str]] = []
    for artifact in ARTIFACTS:
        path = ROOT / artifact.local_path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_rows = pq.ParquetFile(path).metadata.num_rows
        if actual_rows != artifact.rows:
            raise ValueError(f"Row-count mismatch for {path}: {actual_rows} != {artifact.rows}")
        actual_hash = sha256_file(path)
        if actual_hash != artifact.sha256:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual_hash} != {artifact.sha256}")
        manifest_entries.append(
            {
                "path": artifact.remote_path,
                "stage": artifact.stage,
                "rows": actual_rows,
                "bytes": path.stat().st_size,
                "sha256": actual_hash,
                "columns": pq.ParquetFile(path).schema.names,
            }
        )
        uploads.append((path, artifact.remote_path))

    for source_name, remote_path in SUMMARIES.items():
        source = ROOT / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        output = bundle_dir / remote_path
        output.parent.mkdir(parents=True, exist_ok=True)
        summary = portable_value(json.loads(source.read_text(encoding="utf-8")), path_map)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        assert_portable(output)
        uploads.append((output, remote_path))

    card_path = bundle_dir / "README.md"
    card_path.write_text(dataset_card(), encoding="utf-8")
    uploads.append((card_path, "README.md"))

    manifest_path = bundle_dir / "metadata" / "publication_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "artifact_count": len(manifest_entries),
                "artifacts": manifest_entries,
                "sampling_seed": "emse-primary-human-sample-v1",
                "sampling_hash_contract": "SHA256(seed + NUL + repo_id + NUL + number)",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert_portable(manifest_path)
    uploads.append((manifest_path, "metadata/publication_manifest.json"))

    checksums_path = bundle_dir / "metadata" / "checksums.sha256"
    checksums_path.write_text(
        "".join(f"{sha256_file(path)}  {remote}\n" for path, remote in sorted(uploads, key=lambda item: item[1])),
        encoding="utf-8",
    )
    uploads.append((checksums_path, "metadata/checksums.sha256"))
    return manifest_entries, uploads


def upload(repo_id: str, uploads: list[tuple[Path, str]]) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    if not info.private:
        raise ValueError(f"Refusing to upload study records to public dataset {repo_id}.")
    operations = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(path))
        for path, remote in uploads
    ]
    result = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Publish validated study artifacts",
    )
    return result.commit_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.environ.get("HF_DATASET_REPO"))
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if args.upload and args.dry_run:
        parser.error("Choose either --dry-run or --upload.")
    if args.upload and not args.repo_id:
        parser.error("--repo-id or HF_DATASET_REPO is required with --upload.")
    return args


def main() -> None:
    args = parse_args()
    manifest, uploads = build_bundle(args.bundle_dir)
    total_bytes = sum(entry["bytes"] for entry in manifest)
    result: dict[str, Any] = {
        "validated_artifacts": len(manifest),
        "upload_files": len(uploads),
        "data_bytes": total_bytes,
        "bundle_dir": str(args.bundle_dir),
    }
    if args.upload:
        result["commit_url"] = upload(args.repo_id, uploads)
        result["repo_id"] = args.repo_id
    else:
        result["mode"] = "dry-run"
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
