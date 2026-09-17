from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "rq2_validation"))

from analyze_rq2_comparison import (  # noqa: E402
    analyze,
    binary_comparison,
    cluster_bootstrap_cramers_v,
    holm_adjust,
)


def test_binary_statistics_handle_zero_cells():
    frame = pd.DataFrame({"sample_arm": ["agentic"] * 4 + ["human_candidate"] * 4})
    result = binary_comparison(frame, pd.Series([True] * 4 + [False] * 4), "event")
    assert result["risk_difference"] == 1
    assert result["risk_ratio"] is None
    assert result["odds_ratio"] is None
    assert all(value > 0 for value in result["risk_ratio_ci95"])


def test_binary_statistics_handle_all_zero_events():
    frame = pd.DataFrame({"sample_arm": ["agentic"] * 2 + ["human_candidate"] * 2})
    result = binary_comparison(frame, pd.Series(False, index=frame.index), "event")
    assert result["pearson_p"] == 1
    assert result["phi_cramers_v"] == 0


def test_binary_statistics_preserve_historical_yates_correction():
    frame = pd.DataFrame({"sample_arm": ["agentic"] * 10 + ["human_candidate"] * 10})
    result = binary_comparison(
        frame,
        pd.Series([True] * 7 + [False] * 3 + [True] * 3 + [False] * 7),
        "event",
    )

    assert result["pearson_chi2"] == pytest.approx(1.8)
    assert result["pearson_p"] == pytest.approx(0.1797124949)


def test_holm_is_monotone_in_sorted_p_values():
    adjusted = holm_adjust([0.04, 0.001, 0.03, 0.2])
    ordered = sorted(zip([0.04, 0.001, 0.03, 0.2], adjusted))
    assert [value for _, value in ordered] == sorted(value for _, value in ordered)
    assert adjusted[1] == pytest.approx(0.004)


def _consensus_frame():
    rows = []
    types = ["benchmark", "profiling", "static-reasoning", "anecdotal"]
    for arm_index, arm in enumerate(("agentic", "human_candidate")):
        for index in range(16):
            primary = types[(index + arm_index) % 4]
            unresolved = index == 0
            rows.append({
                "repo_id": index // 2 + arm_index * 20,
                "number": arm_index * 100 + index,
                "sample_arm": arm,
                "consensus_validation_present": index < 14,
                "consensus_status": "unresolved" if unresolved else "exact_majority",
                "consensus_primary_validation_type": None if unresolved else (primary if index < 14 else "none"),
                "consensus_validation_types": [] if unresolved or index >= 14 else [primary],
                "included_in_stage2_analysis": index < 14 and not unresolved,
                "multilabel_status": "unresolved" if unresolved else ("exact_majority" if index < 14 else "not_applicable_absent"),
                "included_in_multilabel_analysis": index < 14 and not unresolved,
            })
    return pd.DataFrame(rows)


def test_cluster_bootstrap_is_deterministic():
    frame = _consensus_frame()
    stage2 = frame[frame["included_in_stage2_analysis"]]
    first = cluster_bootstrap_cramers_v(stage2, iterations=40, seed=7)
    second = cluster_bootstrap_cramers_v(stage2, iterations=40, seed=7)
    assert first == second


def test_analysis_has_omnibus_binary_holm_and_unresolved_results():
    result = analyze(_consensus_frame(), bootstrap_iterations=20, bootstrap_seed=3)
    assert result["stage1_rows"] == 32
    assert result["stage2_positive_consensus_rows"] == 26
    assert result["multilabel_positive_consensus_rows"] == 26
    assert len(result["stage2_primary"]["expected_cells"]) == 2
    assert len(result["stage2_per_primary"]) == 4
    assert len(result["stage2_per_validation_type"]) == 4
    assert all("holm_adjusted_p" in row for row in result["stage2_per_primary"])
    assert result["stage2_unresolved"]["agentic_events"] == 1
