from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_perfminer_against_aidev import build_evaluation


def test_evaluation_uses_aidev_labels_and_reports_cascade_coverage():
    aidev_prs = pd.DataFrame({"id": [1, 2, 3, 4]})
    labels = pd.DataFrame({"id": [1, 2, 3, 4], "type": ["perf", "fix", "perf", "fix"]})
    decisions = pd.DataFrame(
        {
            "id": [1, 2, 3, None],
            "repo_id": [10, 10, 11, 12],
            "number": [1, 2, 1, 1],
            "heuristic_match": [True, True, False, False],
        }
    )
    perfminer = pd.DataFrame(
        {
            "repo_id": [10, 10],
            "pr_number": [1, 2],
            "perfminer_pr_status": ["performance_positive", "unresolved"],
            "perfminer_pr_is_performance": [True, None],
            "perfminer_pr_evidence_complete": [True, False],
        }
    )

    evaluation, summary = build_evaluation(aidev_prs, labels, decisions, perfminer)

    assert len(evaluation) == 4
    assert summary["aidev_prs_in_experiment_decisions"] == 3
    assert summary["aidev_prs_missing_from_experiment_decisions"] == 1
    assert summary["heuristic"] == {
        "pr_count": 3,
        "positive_ground_truth": 2,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 0,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert summary["perfminer_within_heuristic_candidates"]["pr_count"] == 1
    assert summary["cascade"]["pr_count"] == 2
    assert summary["cascade"]["true_positive"] == 1
    assert summary["cascade_coverage"]["excluded_for_missing_perfminer_decision"] == 1
