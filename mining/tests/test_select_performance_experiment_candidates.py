import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from select_performance_experiment_candidates import (  # noqa: E402
    add_decisions,
    join_raw_and_attribution,
    parse_timestamp,
    select_candidates,
)


def frames():
    raw = pd.DataFrame(
        [
            {"repo_id": 1, "number": 1, "title": "Improve cache", "body": "performance improvement", "created_at": "2025-01-01T00:00:00Z"},
            {"repo_id": 1, "number": 2, "title": "PERF(cache): reduce misses", "body": "", "created_at": "2025-01-02T00:00:00Z"},
            {"repo_id": 1, "number": 3, "title": "docs: benchmark notes", "body": "", "created_at": "bad-date"},
        ]
    )
    attribution = pd.DataFrame(
        [{"repo_id": 1, "number": number, "aidev_attribution_label": "human_candidate"} for number in (1, 2, 3)]
    )
    return raw, attribution


def test_project_heuristic_uses_compatible_classifier():
    raw, attribution = frames()
    decisions = add_decisions(join_raw_and_attribution(raw, attribution), "project")

    assert decisions["heuristic_match"].tolist() == [True, True, False]
    assert decisions["selection_reason"].tolist() == ["performance_keyword_heuristic", "conventional_commit_prefix", "conventional_commit_prefix"]
    assert decisions["aidev_attribution_label"].tolist() == ["human_candidate"] * 3


def test_title_perf_prefix_only_matches_conventional_perf_prefix():
    raw, attribution = frames()
    decisions = add_decisions(join_raw_and_attribution(raw, attribution), "title_perf_prefix")

    assert decisions["heuristic_match"].tolist() == [False, True, False]
    assert decisions["selection_reason"].tolist() == ["title_not_perf_prefix", "title_perf_prefix", "title_not_perf_prefix"]


def test_join_rejects_duplicate_and_nonmatching_identities():
    raw, attribution = frames()
    with pytest.raises(ValueError, match="duplicate immutable identities"):
        join_raw_and_attribution(pd.concat([raw, raw.iloc[[0]]], ignore_index=True), attribution)
    with pytest.raises(ValueError, match="nonmatching immutable identities"):
        join_raw_and_attribution(raw, attribution.iloc[:2])


def test_date_window_records_reasons_and_excludes_outside_rows():
    raw, attribution = frames()
    decisions = add_decisions(
        join_raw_and_attribution(raw, attribution),
        "title_perf_prefix",
        parse_timestamp("2025-01-02T00:00:00Z", "--start-date"),
        parse_timestamp("2025-01-02T00:00:00Z", "--end-date"),
    )

    assert decisions["date_reason"].tolist() == ["before_start_date", "within_window", "invalid_created_at"]
    assert decisions["selected"].tolist() == [False, True, False]


def test_outputs_include_all_decisions_counts_hashes_and_candidates(tmp_path):
    raw, attribution = frames()
    raw_path = tmp_path / "raw.parquet"
    attribution_path = tmp_path / "attribution.parquet"
    output_dir = tmp_path / "output"
    raw.to_parquet(raw_path, index=False)
    attribution.to_parquet(attribution_path, index=False)

    summary = select_candidates(raw_path, attribution_path, output_dir, "title_perf_prefix")
    decisions = pd.read_parquet(output_dir / "decisions.parquet")
    candidates = pd.read_parquet(output_dir / "candidates.parquet")
    persisted = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert len(decisions) == 3
    assert len(candidates) == 1
    assert candidates["number"].tolist() == [2]
    assert summary["decision_rows"] == 3
    assert summary["candidate_rows"] == 1
    assert len(summary["raw_sha256"]) == 64
    assert persisted == summary
    assert "provisional" in summary["note"]
