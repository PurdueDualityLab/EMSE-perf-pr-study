"""Build compact RQ1-RQ3 label exports and their provenance manifest."""

from __future__ import annotations

import hashlib
import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.rq3_pattern_and_validation.refresh_type_layer import rebuild_types

OUTPUT_DIR = ROOT / "reproduction/labels"
KEYS = ["repo_id", "number"]
ARMS = ("agentic", "human_candidate")
DIMENSIONS = [f"D{index}" for index in range(10)]

SOURCES = {
    "rq1_labels.csv": ROOT / "analysis/rq1_optimization_patterns/consensus/rq1_consensus.parquet",
    "rq2_labels.csv": ROOT / "analysis/rq2_validation/consensus/rq2_consensus.parquet",
    "rq2_exclusions.csv": ROOT / "analysis/rq2_validation/sample/evidence_exclusions.parquet",
    "rq3_labels.csv": ROOT / "data/data/rq3/current/data/rq3_pr_level.csv",
}

BASE_COLUMNS = [*KEYS, "sample_arm"]
RQ1_COLUMNS = BASE_COLUMNS + ["repo_full_name", "html_url"] + [
    f"{provider}_{label}"
    for provider in ("gpt", "gemini", "qwen")
    for label in ("high_level_pattern", "sub_pattern")
] + [
    "consensus_status", "consensus_vote_count", "consensus_coalition",
    "consensus_high_level_pattern", "consensus_sub_pattern",
    "included_in_analysis", "inclusion_reason",
]
RQ2_COLUMNS = BASE_COLUMNS + ["gpt_repo_full_name", "gpt_html_url"] + [
    f"{provider}_{label}"
    for provider in ("gpt", "gemini", "qwen")
    for label in (
        "model", "classification_status", "validation_present",
        "primary_validation_type", "validation_types",
    )
] + [
    "stage1_status", "stage1_positive_votes", "stage1_negative_votes",
    "stage1_coalition", "consensus_validation_present", "stage2_status",
    "stage2_vote_count", "stage2_coalition",
    "consensus_primary_validation_type", "consensus_status",
    "included_in_stage2_analysis", "inclusion_reason", "multilabel_status",
    "multilabel_vote_count", "multilabel_coalition",
    "consensus_validation_types", "included_in_multilabel_analysis",
    "multilabel_inclusion_reason",
]
RQ2_EXCLUSION_COLUMNS = BASE_COLUMNS + [
    "exclusion_reason", "paired_with_repo_id", "paired_with_number",
]
RQ3_COLUMNS = BASE_COLUMNS + [
    "html_url", "pattern", "sub_pattern", "rq1_label_source",
    "validation_present", "validation_type", "in_metric_layer", "in_type_layer",
    *DIMENSIONS, "n_dims", "n_dims_specific", "dims",
    *(f"{dimension}_nodiff" for dimension in DIMENSIONS),
    "n_dims_nodiff", "n_dims_description_only",
]

COLUMNS = {
    "rq1_labels.csv": RQ1_COLUMNS,
    "rq2_labels.csv": RQ2_COLUMNS,
    "rq2_exclusions.csv": RQ2_EXCLUSION_COLUMNS,
    "rq3_labels.csv": RQ3_COLUMNS,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize_type_set(value: object) -> str:
    if isinstance(value, str):
        value = json.loads(value) if value else None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return ""
    return json.dumps(list(value))


def load_source(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def export_frame(name: str, source: Path) -> pd.DataFrame:
    frame = load_source(source)[COLUMNS[name]].copy()
    frame = frame.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    if frame.duplicated(KEYS).any() or not frame["sample_arm"].isin(ARMS).all():
        raise ValueError(f"{name} contains invalid identities or sample arms.")
    frame = frame.rename(columns={
        "gpt_repo_full_name": "repo_full_name",
        "gpt_html_url": "html_url",
    })
    for column in frame.columns:
        if column.endswith("_validation_types"):
            frame[column] = frame[column].map(serialize_type_set)
    return frame


def validate_exports(frames: dict[str, pd.DataFrame]) -> None:
    rq1 = frames["rq1_labels.csv"]
    rq2 = frames["rq2_labels.csv"]
    exclusions = frames["rq2_exclusions.csv"]
    rq3 = frames["rq3_labels.csv"]
    if (len(rq1), len(rq2), len(exclusions), len(rq3)) != (2260, 2258, 2, 2081):
        raise ValueError("Unexpected compact export row counts.")
    controls = {
        "rq1 resolved": (int(rq1["included_in_analysis"].sum()), 2083),
        "rq2 positive": (int(rq2["consensus_validation_present"].sum()), 1839),
        "rq2 primary resolved": (int(rq2["included_in_stage2_analysis"].sum()), 1819),
        "rq2 multilabel resolved": (int(rq2["included_in_multilabel_analysis"].sum()), 1707),
        "rq3 metric layer": (int(rq3["in_metric_layer"].sum()), 1699),
        "rq3 primary type layer": (int(rq3["in_type_layer"].sum()), 1684),
    }
    failures = {name: values for name, values in controls.items() if values[0] != values[1]}
    if failures:
        raise ValueError(f"Compact export controls failed: {failures}")
    identities = lambda frame: set(frame[KEYS].itertuples(index=False, name=None))
    if identities(rq1) != identities(rq2) | identities(exclusions):
        raise ValueError("RQ2 identities and exclusions do not reconstruct the RQ1 sample.")
    if identities(rq2) & identities(exclusions):
        raise ValueError("RQ2 labels overlap evidence exclusions.")
    if int((rq2["consensus_validation_present"] & ~rq2["included_in_stage2_analysis"]).sum()) != 20:
        raise ValueError("Unexpected unresolved primary-type count.")
    if int((rq2["consensus_validation_present"] & ~rq2["included_in_multilabel_analysis"]).sum()) != 132:
        raise ValueError("Unexpected unresolved multi-label count.")
    unresolved_rq3 = rq3["validation_type"].eq("unresolved") & rq3["in_metric_layer"] & ~rq3["in_type_layer"]
    if int(unresolved_rq3.sum()) != 15:
        raise ValueError("Unexpected unresolved RQ3 primary-type count.")


def build_manifest(frames: dict[str, pd.DataFrame], sources=None, output_dir=OUTPUT_DIR) -> dict[str, object]:
    sources = sources or SOURCES
    manifest: dict[str, object] = {
        "models": {
            "gpt": "gpt-5.6-sol",
            "gemini": "gemini-3.1-pro-preview",
            "qwen": "qwen3.8:27b",
        },
        "files": {},
    }
    files = manifest["files"]
    assert isinstance(files, dict)
    for name, frame in frames.items():
        source = sources[name].resolve()
        entry: dict[str, object] = {
            "rows": len(frame),
            "columns": frame.columns.tolist(),
            "arms": frame["sample_arm"].value_counts().to_dict(),
            "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else source.name,
            "source_sha256": sha256_file(source),
            "sha256": sha256_file(output_dir / name),
        }
        if name == "rq3_labels.csv":
            entry["label_method"] = "Deterministic D0-D9 quantitative-claim extraction, not LLM classification"
            entry["type_refresh_consensus_sha256"] = sha256_file(sources["rq2_labels.csv"])
            entry["notes"] = "Metric flags and validation presence come from the frozen extraction; primary types are refreshed against the supplied RQ3 consensus."
        files[name] = entry
    manifest["summary"] = {
        "rq1_resolved": 2083,
        "rq1_unresolved": 177,
        "rq2_positive": 1839,
        "rq2_negative": 419,
        "rq2_resolved_positive_primary_types": 1819,
        "rq2_unresolved_positive_primary_types": 20,
        "rq2_multilabel_consensus": 1707,
        "rq2_multilabel_unresolved": 132,
        "rq3_analytic": 2081,
        "rq3_metric_layer": 1699,
        "rq3_resolved_primary_type_layer": 1684,
        "rq3_unresolved_positive_primary_types": 15,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rq1-consensus', type=Path, default=SOURCES['rq1_labels.csv'])
    parser.add_argument('--rq2-consensus', type=Path, default=SOURCES['rq2_labels.csv'])
    parser.add_argument('--rq2-exclusions', type=Path, default=SOURCES['rq2_exclusions.csv'])
    parser.add_argument('--metric-source', type=Path, default=SOURCES['rq3_labels.csv'])
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    sources = dict(zip(SOURCES, [args.rq1_consensus, args.rq2_consensus, args.rq2_exclusions, args.metric_source]))
    frames = {name: export_frame(name, source) for name, source in sources.items()}
    refreshed, _, _ = rebuild_types(frames['rq3_labels.csv'], frames['rq2_labels.csv'])
    frames['rq3_labels.csv'] = refreshed[RQ3_COLUMNS]
    validate_exports(frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(args.output_dir / name, index=False)
    manifest = build_manifest(frames, sources, args.output_dir)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": {name: len(frame) for name, frame in frames.items()}, "validated": True}, sort_keys=True))


if __name__ == "__main__":
    main()
