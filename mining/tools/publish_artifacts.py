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

import pandas as pd
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
    format: str = "parquet"


ARTIFACTS = (
    Artifact("mining/outputs_2026_06_01/raw/github_human_prs.parquet", "data/raw/github_pull_requests.parquet", "mining", 1_603_213, "a3d737ce2cc32e15b6e14b91578217873fcd9739bebc664fdc04f31209256bef"),
    Artifact("mining/aidev_attribution_all_available_dates/aidev_all_available_dates.parquet", "data/agentic/aidev_attribution.parquet", "agentic_attribution", 1_603_213, "305238366a898fc95a963a96f86d026166d3d4f68a968711b7aee3fe16d8b6a0"),
    Artifact("mining/experiments/aidev_luna_full_v1/task_type_decisions_with_retry.parquet", "data/performance/task_type_decisions.parquet", "performance_classification", 1_603_213, "16a9a281029321541c7370c3445a69c896c05dd7fce2a0dbcb4c7ce70b2afbf4"),
    Artifact("mining/aidev_human_candidates_v1/decisions.parquet", "data/human/decisions.parquet", "human_filtering", 28_127, "72922a804a87063867927f12dc39e8eabfb3a1230905153aa9ff31ba13b5105b"),
    Artifact("mining/aidev_human_candidates_v1/human_candidates.parquet", "data/human/human_candidates.parquet", "human_filtering", 24_680, "b08525d2c6e634c6ed667000bf24d22cc956b85ca5a844a6acbbeb24ff79637f"),
    Artifact("mining/aidev_human_candidates_v1/excluded_from_human_candidates.parquet", "data/human/excluded.parquet", "human_filtering", 3_447, "5c6813a81251e963bac43ec03f5a5249d24d4ab5125b2734544d399a1b9a6e19"),
    Artifact("mining/aidev_weekly_balanced_sample_v1/agentic_sample.parquet", "data/sample/agentic_sample.parquet", "weekly_sampling", 1_130, "66643a58480668ef62295e1d3cf3c162cb89a79a81d6741fac7bec2ddbb1820e"),
    Artifact("mining/aidev_weekly_balanced_sample_v1/human_sample.parquet", "data/sample/human_sample.parquet", "weekly_sampling", 1_130, "66970968eba6666b752f1306fe204d215f424e8aab086d71bfc78084e06ea835"),
    Artifact("mining/aidev_weekly_balanced_sample_v1/balanced_sample.parquet", "data/sample/balanced_sample.parquet", "weekly_sampling", 2_260, "10f46fae9c880325f6574ca308f1857a4798c2ac1548bd0511502cb8687b357f"),
    Artifact("mining/aidev_weekly_balanced_sample_v1/sampling_manifest.parquet", "data/sample/sampling_manifest.parquet", "weekly_sampling", 25_365, "ad11ad502cc689608278993225bb59e1568e51377d710fed3da072f649b3a1b5"),
    Artifact("mining/aidev_weekly_balanced_sample_v1/weekly_strata.parquet", "data/sample/weekly_strata.parquet", "weekly_sampling", 65, "670eabaaf28672a10966de3112067000ebfc70f6e0aa0016249191da5e878027"),
    Artifact("mining/aidev_weekly_balanced_sample_v1/quality_exclusions.parquet", "data/sample/quality_exclusions.parquet", "weekly_sampling", 671, "09a67ecf51e4cb4945b5f00ab65939708cdf912fed437600a0c82cf05c6b055b"),
    Artifact("mining/curated_labels_v1/pull_request_labels.parquet", "data/curated/pull_request_labels.parquet", "curated_labels", 1_603_213, "f058825dfbc1c56f621a03ccef9480d06dcedab13e138e7e4cd97fbd1a60117a"),
    Artifact("analysis/rq1_optimization_patterns/catalog/original_optimization_catalog.csv", "data/catalog/original_optimization_catalog.csv", "rq1_catalog", 43, "a951a9acf91ebe9a775009e95d187a3ea59f5a4ae4b2875d5ebc9788c1bc6d1e", "csv"),
    Artifact("analysis/rq1_optimization_patterns/catalog/updated_optimization_catalog.csv", "data/catalog/updated_optimization_catalog.csv", "rq1_catalog", 58, "8a4b2e59a83a3994ee228515ba47ce2712c43112df3070640ea0b3ef7fcf803c", "csv"),
    Artifact("analysis/rq1_optimization_patterns/results_gpt/optimization_pattern_labels_complete.parquet", "data/rq1/optimization_pattern_labels_gpt.parquet", "rq1_model_labels", 2_260, "6a34304ec4de221cd28108265b6d5787c5d684d2f444240253ff8cc898ce632b"),
    Artifact("analysis/rq1_optimization_patterns/results_gemini/optimization_pattern_labels.parquet", "data/rq1/optimization_pattern_labels_gemini.parquet", "rq1_model_labels", 2_260, "fcb5e01a0da93f7718e6f2fa1dd4417c8e467b8c5b1c9da7df70205d4ec4e96a"),
    Artifact("analysis/rq1_optimization_patterns/results_qwen/optimization_pattern_labels.parquet", "data/rq1/optimization_pattern_labels_qwen.parquet", "rq1_model_labels", 2_260, "69df410be22b738fbaef95385e04f8df755564e6a12578a85870ca9e833f719f"),
    Artifact("analysis/rq2_validation/results_gpt/validation_labels_complete.parquet", "data/rq2/performance_validation_labels_gpt.parquet", "rq2_model_labels", 2_258, "88fe2f918cc4d436a459f0c13ea5718a6ef37f896884088d552d8a876801f2a6"),
    Artifact("analysis/rq2_validation/results_gemini/validation_labels.parquet", "data/rq2/performance_validation_labels_gemini.parquet", "rq2_model_labels", 2_258, "0a8e964ac6d9623e916e3478a9f1c688ec4c8667af5c74fbc79fe97bc7e675aa"),
    Artifact("analysis/rq2_validation/results_qwen/validation_labels.parquet", "data/rq2/performance_validation_labels_qwen.parquet", "rq2_model_labels", 2_258, "7e455ad4028bcecdfd42ec3a74443195fdf3667b20c9b5f995415a06211eeb33"),
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
pretty_name: "How Do Coding Agents Optimize Software and Report Performance Validation?"
size_categories:
- 1M<n<10M
task_categories:
- text-classification
tags:
- software-engineering
- pull-requests
- performance
- coding-agents
configs:
- config_name: balanced-sample
  data_files:
  - split: sample
    path: data/sample/balanced_sample.parquet
  default: true
- config_name: agentic-sample
  data_files:
  - split: sample
    path: data/sample/agentic_sample.parquet
- config_name: human-sample
  data_files:
  - split: sample
    path: data/sample/human_sample.parquet
- config_name: human-filter
  data_files:
  - split: decisions
    path: data/human/decisions.parquet
  - split: candidates
    path: data/human/human_candidates.parquet
  - split: excluded
    path: data/human/excluded.parquet
- config_name: sampling-frame
  data_files:
  - split: frame
    path: data/sample/sampling_manifest.parquet
- config_name: weekly-strata
  data_files:
  - split: strata
    path: data/sample/weekly_strata.parquet
- config_name: performance-classification
  data_files:
  - split: full
    path: data/performance/task_type_decisions.parquet
- config_name: agentic-attribution
  data_files:
  - split: full
    path: data/agentic/aidev_attribution.parquet
- config_name: curated-labels
  data_files:
  - split: full
    path: data/curated/pull_request_labels.parquet
- config_name: raw
  data_files:
  - split: full
    path: data/raw/github_pull_requests.parquet
- config_name: optimization-catalog-original
  data_files:
  - split: catalog
    path: data/catalog/original_optimization_catalog.csv
- config_name: optimization-catalog-updated
  data_files:
  - split: catalog
    path: data/catalog/updated_optimization_catalog.csv
- config_name: rq1-model-labels
  data_files:
  - split: gpt
    path: data/rq1/optimization_pattern_labels_gpt.parquet
  - split: gemini
    path: data/rq1/optimization_pattern_labels_gemini.parquet
  - split: qwen
    path: data/rq1/optimization_pattern_labels_qwen.parquet
- config_name: rq2-model-labels
  data_files:
  - split: gpt
    path: data/rq2/performance_validation_labels_gpt.parquet
  - split: gemini
    path: data/rq2/performance_validation_labels_gemini.parquet
  - split: qwen
    path: data/rq2/performance_validation_labels_qwen.parquet
---

# How Do Coding Agents Optimize Software and Report Performance Validation?

## A Large-Scale Empirical Study of Open-Source Pull Requests

This access-controlled dataset contains the approved artifacts for a study
comparing agentic and human-authored performance pull requests. The stored
`human_candidate` label denotes the manuscript's human-authored group.
Offline statistics and figure reproduction, without full-data downloads, are
documented in the source repository's `REPRODUCING.md`.

## Cohorts

- Raw pull requests: 1,603,213
- Performance pull requests: 29,483
- Eligible agentic performance pull requests after legacy quality filters: 1,130
- Strict human candidates: 24,680
- Final balanced sample: 1,130 agentic and 1,130 human candidates
- Weekly strata: 65 ISO weeks in UTC

"Human candidate" means no selected observable coding-agent signal was found;
it does not establish human authorship. The performance classifier retains six
explicit API errors, which are not included among performance-labeled rows.

## Layout

The `data/` directory follows the five pipeline stages. `metadata/` contains
sanitized summaries, a manifest, and checksums. Operational checkpoints, API
payloads, retries, secrets, and local paths are intentionally excluded.

The `balanced-sample` subset is the default. Other subsets expose the full
mining, attribution, classification, filtering, and sampling artifacts without
combining tables that have different schemas.

The optimization catalog subsets expose the original and executed taxonomies
as separate tables. The executed snapshot contains 58 patterns; the source
repository documents the discrepancy with the submitted paper's count of 59.
Historical `rq1` paths answer journal RQ2, `rq2` paths answer journal RQ3,
and the historical `rq3` extraction bundle answers journal RQ4.

The `rq1-model-labels` and `rq2-model-labels` subsets expose the complete,
validated outputs from GPT-5.6-sol, Gemini 3.1 Pro Preview, and Qwen3.8 27B as
separate splits. These are model-specific labels for agreement analysis, not a
human-adjudicated ground truth. Provider responses, prompts, and operational
retry checkpoints are intentionally excluded.

The `curated-labels` subset provides one compact row per pull request with
agentic attribution, task type, the derived performance indicator, strict
human-filter decisions, and sampling labels. Human-filter and sampling fields
are null for pull requests outside the populations evaluated by those stages.

```python
from datasets import load_dataset

sample = load_dataset(
    "rcalvome/EMSE-perf-pr-study",
    "balanced-sample",
    token=True,
)
```

Downloads require an authenticated account with the necessary dataset access
and acceptance of any applicable access conditions. Availability of the Dataset
Viewer depends on the Hub's current access and account settings.

## Terms

The repository's processing code is MIT licensed. This dataset contains public
GitHub metadata and derived labels; no blanket license is asserted over source
repository content. Users are responsible for complying with GitHub and source
repository terms.
"""


def inspect_table(path: Path, format: str) -> tuple[int, list[str]]:
    if format == "parquet":
        parquet = pq.ParquetFile(path)
        return parquet.metadata.num_rows, parquet.schema.names
    if format == "csv":
        frame = pd.read_csv(path)
        return len(frame), list(frame.columns)
    raise ValueError(f"Unsupported artifact format: {format}")


def validate_sources(dataset_dir: Path | None = None) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    """Inspect the allowlisted source tables or a downloaded dataset without writes."""
    manifest_entries = []
    uploads: list[tuple[Path, str]] = []
    for artifact in ARTIFACTS:
        root = (dataset_dir or ROOT).resolve()
        relative = artifact.remote_path if dataset_dir is not None else artifact.local_path
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Artifact escapes the selected root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_rows, columns = inspect_table(path, artifact.format)
        if actual_rows != artifact.rows:
            raise ValueError(f"Row-count mismatch for {path}: {actual_rows} != {artifact.rows}")
        actual_hash = sha256_file(path)
        if actual_hash != artifact.sha256:
            raise ValueError(f"SHA-256 mismatch for {path}: {actual_hash} != {artifact.sha256}")
        manifest_entries.append(
            {
                "path": artifact.remote_path,
                "stage": artifact.stage,
                "format": artifact.format,
                "rows": actual_rows,
                "bytes": path.stat().st_size,
                "sha256": actual_hash,
                "columns": columns,
            }
        )
        uploads.append((path, artifact.remote_path))
    return manifest_entries, uploads


def inventory_text() -> str:
    lines = [
        '# Official Core Artifact Inventory', '',
        'Generated from the allowlist in `mining/tools/publish_artifacts.py`.',
        'Paths below are relative to the Hugging Face dataset repository root.',
        'This is the core mining/catalog/model-label bundle. The historical RQ4',
        'extraction bundle has separate metadata; offline paper inputs and audits',
        'are listed in the root `artifact_manifest.json`.', '',
        '| Stage | Artifact | Rows | SHA-256 |', '| --- | --- | ---: | --- |',
    ]
    lines.extend(f'| {a.stage} | `{a.remote_path}` | {a.rows:,} | `{a.sha256}` |' for a in ARTIFACTS)
    lines += ['',
        'The executed optimization catalog has **58 patterns**. The submitted paper',
        'reports 59; the documented consolidation and manuscript history are in',
        '`analysis/rq1_optimization_patterns/catalog/README.md`.', '',
        'Historical `rq1` and `rq2` dataset paths refer to journal RQ2 and RQ3.',
        '`human_candidate` is the stored label for the manuscript\'s human-authored',
        'group; it does not establish independently verified human authorship.', '',
        'Full-data access requires authentication and the dataset\'s access conditions.',
        'Publication tooling retains its private-only upload guard.', '',
        'Validate installed dataset files without writes:', '',
        '```bash', 'python mining/tools/publish_artifacts.py --validate-only --dataset-dir data', '```', '',
        '`--dry-run` validates original local sources and **builds** sanitized metadata',
        'under `--bundle-dir`; it is not a read-only validation command.', '',
        'Maintainers can check or regenerate this inventory and `mining/artifacts.sha256`:', '',
        '```bash', 'python mining/tools/publish_artifacts.py --check-inventory',
        'python mining/tools/publish_artifacts.py --write-inventory', '```', '',
    ]
    return '\n'.join(lines)


def checksum_text() -> str:
    return ''.join(f'{a.sha256}  {a.local_path}\n' for a in ARTIFACTS)


def check_inventory() -> None:
    for path, expected in [(ROOT / 'mining/ARTIFACTS.md', inventory_text()),
                           (ROOT / 'mining/artifacts.sha256', checksum_text())]:
        if path.read_text() != expected:
            raise ValueError(f'Stale artifact inventory: {path}; use --write-inventory after reviewing the allowlist.')


def build_bundle(bundle_dir: Path) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    manifest_entries, uploads = validate_sources()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    path_map = {
        str((ROOT / artifact.local_path).resolve()): artifact.remote_path
        for artifact in ARTIFACTS
    }

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
    parser.add_argument("--validate-only", action="store_true", help="Read-only validation; do not construct or upload a bundle.")
    parser.add_argument("--dataset-dir", type=Path, help="Validate the downloaded dataset layout instead of original local source paths.")
    parser.add_argument("--check-inventory", action="store_true")
    parser.add_argument("--write-inventory", action="store_true")
    args = parser.parse_args()
    if args.upload and args.dry_run:
        parser.error("Choose either --dry-run or --upload.")
    if args.upload and not args.repo_id:
        parser.error("--repo-id or HF_DATASET_REPO is required with --upload.")
    if sum([args.dry_run, args.upload, args.validate_only, args.check_inventory, args.write_inventory]) > 1:
        parser.error("Choose one validation, inventory, build, or upload mode.")
    if args.dataset_dir is not None and not args.validate_only:
        parser.error("--dataset-dir requires --validate-only.")
    return args


def main() -> None:
    args = parse_args()
    if args.check_inventory or args.write_inventory:
        if args.write_inventory:
            (ROOT / 'mining/ARTIFACTS.md').write_text(inventory_text())
            (ROOT / 'mining/artifacts.sha256').write_text(checksum_text())
        check_inventory()
        print(json.dumps({'inventory_artifacts': len(ARTIFACTS), 'mode': 'inventory'}))
        return
    if args.validate_only:
        entries, _ = validate_sources(args.dataset_dir)
        print(json.dumps({'validated_artifacts': len(entries), 'mode': 'validate-only'}))
        return
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
