from pathlib import Path

import pandas as pd
import pytest

from analysis.rq2_validation.temporal.analyze_rq2_temporal import (
    add_window_half,
    check_controls,
    period_table,
    prepare,
    trend_model,
)


ROOT = Path(__file__).resolve().parents[2]
TEMPORAL = ROOT / "analysis/rq2_validation/temporal"


@pytest.fixture(scope="module")
def temporal_frame():
    labels = pd.read_csv(ROOT / "analysis/classification_labels/rq2_labels.csv")
    dates = pd.read_csv(TEMPORAL / "pr_created_at.csv")
    return add_window_half(prepare(labels, dates))


def test_temporal_sample_reproduces_weekly_balanced_design(temporal_frame):
    controls = check_controls(temporal_frame)

    assert controls["rows"] == 2258
    assert controls["strata"] == 65
    assert controls["weekly_balance_verified"] is True
    assert controls["window_start"] == "2025-01-02T13:44:32+00:00"
    assert controls["window_end"] == "2026-06-01T16:55:18+00:00"


def test_temporal_split_uses_median_observed_iso_week(temporal_frame):
    halves = period_table(temporal_frame, temporal_frame["validation_present"], "window_half")

    assert halves["period"].tolist() == ["2025-W01..2025-W42", "2025-W43..2026-W23"]
    assert halves[["agentic_total", "human_candidate_total"]].to_numpy().tolist() == [
        [347, 347],
        [782, 782],
    ]
    assert halves["agentic_rate"].tolist() == pytest.approx([241 / 347, 688 / 782])
    assert halves["human_candidate_rate"].tolist() == pytest.approx([257 / 347, 653 / 782])


def test_temporal_models_reproduce_clustered_trends(temporal_frame):
    presence = trend_model(temporal_frame, temporal_frame["validation_present"], "validation_present")
    resolved = temporal_frame[temporal_frame["included_in_stage2_analysis"].astype(bool)]
    benchmark = trend_model(
        resolved,
        resolved["consensus_primary_validation_type"].eq("benchmark"),
        "benchmark_primary_type",
    )

    assert presence["terms"]["time_human_candidate"]["cluster_robust"]["odds_ratio"] == pytest.approx(3.063, rel=1e-3)
    assert presence["agentic_time_slope"]["odds_ratio"] == pytest.approx(6.475, rel=1e-3)
    assert presence["terms"]["author_type_by_time"]["cluster_robust"]["p_value"] == pytest.approx(0.0928, abs=1e-4)
    assert benchmark["terms"]["author_type_by_time"]["cluster_robust"]["odds_ratio"] == pytest.approx(3.693, rel=1e-3)
    assert benchmark["terms"]["author_type_by_time"]["cluster_robust"]["p_value"] == pytest.approx(0.0187, abs=1e-4)
