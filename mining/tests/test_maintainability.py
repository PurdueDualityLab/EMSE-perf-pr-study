from pathlib import Path
import sys

import pytest

pytest.importorskip('lizard')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'analysis/maintainability'))
from run_maintainability import metrics, paired_change, aggregate


def test_branch_increases_cyclomatic_complexity():
    before = metrics('a.py', 'def f(x):\n    return x\n')
    after = metrics('a.py', 'def f(x):\n    if x:\n        return 1\n    return 0\n')
    assert before['avg_ccn'] == 1
    assert after['avg_ccn'] == 2
    assert paired_change(before['avg_ccn'], after['avg_ccn']) == (1, 100)


def test_zero_and_missing_baselines_are_not_zero_changes():
    assert paired_change(0, 2) == (2, None)
    assert paired_change(None, 2) == (None, None)
    assert metrics('a.py', 'x = 1\n')['avg_ccn'] is None


def test_partial_file_pairs_excluded_from_complete_pr_analysis():
    common = dict(repo_id=1, number=2, sample_arm='agentic')
    values = dict(nloc=10, avg_ccn=2, function_count=1)
    rows = [{**common, 'status': 'complete', 'before': values, 'after': values},
            {**common, 'status': 'failed'}]
    result = aggregate(rows)
    assert not result.complete_pr.any()
    assert result.metric_pairs.eq(1).all()
