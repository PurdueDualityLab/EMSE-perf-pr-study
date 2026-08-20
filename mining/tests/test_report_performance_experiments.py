from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from report_performance_experiments import build_comparison, report  # noqa: E402


def decisions(heuristic):
    return pd.DataFrame(
        [
            {"repo_id": 1, "number": 1, "html_url": "https://example/1", "selected": heuristic == "project", "heuristic_match": heuristic == "project", "selection_reason": heuristic, "date_eligible": True, "date_reason": "within_window", "aidev_attribution_label": "human_candidate"},
            {"repo_id": 1, "number": 2, "html_url": "https://example/2", "selected": True, "heuristic_match": True, "selection_reason": heuristic, "date_eligible": True, "date_reason": "within_window", "aidev_attribution_label": "agentic"},
        ]
    )


def test_build_comparison_keeps_all_prs_and_optional_results():
    perfminer = pd.DataFrame([{"repo_id": 1, "pr_number": 2, "perfminer_pr_status": "performance_positive", "perfminer_pr_is_performance": True}])
    llm = pd.DataFrame([{"repo_id": 1, "number": 2, "llm_label": "performance", "llm_status": "classified"}])

    result = build_comparison(decisions("project"), decisions("prefix"), perfminer, llm)

    assert len(result) == 2
    assert result.loc[result["number"] == 2, "perfminer_pr_status"].item() == "performance_positive"
    assert result.loc[result["number"] == 2, "llm_label"].item() == "performance"


def test_build_comparison_rejects_mismatched_decisions():
    with pytest.raises(ValueError, match="nonmatching identities"):
        build_comparison(decisions("project"), decisions("prefix").iloc[:1])


def test_report_writes_parquet_and_summary(tmp_path):
    project = tmp_path / "project.parquet"
    prefix = tmp_path / "prefix.parquet"
    decisions("project").to_parquet(project, index=False)
    decisions("prefix").to_parquet(prefix, index=False)

    summary = report(project, prefix, tmp_path / "out")

    assert summary["rows"] == 2
    assert summary["project_selected"] == 2
    assert (tmp_path / "out" / "comparison.parquet").is_file()
