from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis" / "rq1_optimization_patterns"))

from analyze_rq1_comparison import (  # noqa: E402
    analyze_comparison,
    category_effects,
    cluster_bootstrap_cramers_v,
    cramers_v,
    holm_adjust,
)


def test_canonical_statistics_and_zero_cell_correction():
    table = np.array([[20, 30], [10, 40]])

    assert cramers_v(table) == pytest.approx(0.2182178902)
    effects = category_effects(table)
    assert effects["risk_difference"] == pytest.approx(0.2)
    assert effects["risk_ratio"] == pytest.approx(2.0)
    assert effects["odds_ratio"] == pytest.approx(8 / 3)
    assert effects["continuity_corrected"] is False

    zero_effects = category_effects(np.array([[0, 10], [5, 5]]))
    assert zero_effects["continuity_corrected"] is True
    assert all(np.isfinite(zero_effects[key]) for key in ("risk_ratio", "odds_ratio", "risk_ratio_ci_low", "odds_ratio_ci_high"))


def test_holm_adjustment_is_order_preserving_and_monotone():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def _consensus_frame():
    rows = []
    patterns = {
        "agentic": ["A", "A", "A", "B", "B", "C"],
        "human_candidate": ["A", "B", "B", "B", "C", "C"],
    }
    repo = 0
    for arm, labels in patterns.items():
        for index, label in enumerate(labels):
            repo += 1
            rows.append(
                {
                    "repo_id": (repo + 1) // 2,
                    "number": repo,
                    "sample_arm": arm,
                    "consensus_high_level_pattern": label,
                    "consensus_sub_pattern": f"{label}{index % 2}",
                    "included_in_analysis": True,
                }
            )
    rows.append(
        {
            "repo_id": 99,
            "number": 99,
            "sample_arm": "agentic",
            "consensus_high_level_pattern": None,
            "consensus_sub_pattern": None,
            "included_in_analysis": False,
        }
    )
    return pd.DataFrame(rows)


def test_cluster_bootstrap_is_deterministic_with_fast_iterations():
    frame = _consensus_frame().query("included_in_analysis")

    first = cluster_bootstrap_cramers_v(frame, iterations=40, seed=7)
    second = cluster_bootstrap_cramers_v(frame, iterations=40, seed=7)

    assert first == second
    assert 0 <= first["ci_low"] <= first["ci_high"] <= 1


def test_analysis_emits_primary_and_secondary_results():
    distribution, expected, tests, summary = analyze_comparison(
        _consensus_frame(), bootstrap_iterations=30, permutation_iterations=30, seed=3
    )

    assert distribution["count"].sum() == 12
    assert expected.shape == (2, 4)
    assert set(tests["high_level_pattern"]) == {"A", "B", "C"}
    assert tests["holm_adjusted_p_value"].between(0, 1).all()
    assert summary["included_rows"] == 12
    assert summary["subpattern_richness"]["role"] == "secondary_exploratory_analysis"
