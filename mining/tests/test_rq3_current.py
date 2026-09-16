from pathlib import Path
import sys
import os
import subprocess

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis/rq3_pattern_and_validation"))
from run_current import join_consensus, assemble_corpus
from extract_metrics import extract_dimensions


def inputs():
    sample = pd.DataFrame({"repo_id": [1, 2, 3], "number": [7, 7, 7],
                           "sample_arm": ["agentic", "human_candidate", "agentic"]})
    rq1 = sample.assign(included_in_analysis=True, consensus_high_level_pattern="Algorithm-Level Optimizations",
                        consensus_sub_pattern="Example", consensus_status="majority")
    rq2 = sample.assign(consensus_validation_present=[True, True, False],
                        included_in_stage2_analysis=[True, False, False],
                        consensus_primary_validation_type=["benchmark", None, None])
    status = sample.assign(status="complete")
    return sample, rq1, rq2, status


def test_unresolved_type_does_not_become_negative_presence():
    result = join_consensus(*inputs())
    assert result.validation_present.tolist() == [True, True, False]
    assert result.in_metric_layer.tolist() == [True, True, False]
    assert result.in_type_layer.tolist() == [True, False, False]
    assert result.validation_type.tolist() == ["benchmark", "unresolved", "none"]


def test_metric_tests_include_unresolved_types_but_type_tests_exclude_them(monkeypatch):
    import rq3_statistics as stats

    result = join_consensus(*inputs())
    result["author_type"] = result.sample_arm
    result["pattern"] = result.consensus_high_level_pattern
    result["category"] = result.pattern
    result["any_dim"] = True
    result["n_dims"] = 1
    result["n_dims_specific"] = 1
    for dim in stats.DIMS:
        result[dim] = dim == "D1"
    saved = {}
    monkeypatch.setattr(stats, "AUTHORS", ["agentic", "human_candidate"])
    monkeypatch.setattr(stats, "TESTS", [])
    monkeypatch.setattr(stats, "LINES", [])
    monkeypatch.setattr(stats, "save", lambda frame, name: saved.update({name: frame}))

    val = stats.section_B(result)
    assert len(val) == 2
    assert saved["T2_1_metric_profile_human_candidate"].loc["All", "n validated"] == 1
    assert saved["T2_4_metric_profile_by_validation_type"]["n"].sum() == 1
    tests = {test["test_label"]: test for test in stats.TESTS}
    assert tests["Author × any dimension reported"]["n"] == 2
    assert tests["Benchmark evidence × any dimension reported"]["n"] == 1


def test_duplicate_and_incomplete_evidence_rejected():
    sample, rq1, rq2, status = inputs()
    with pytest.raises(ValueError, match="Duplicate"):
        join_consensus(sample, pd.concat([rq1, rq1]), rq2, status)
    status.loc[0, "status"] = "partial"
    with pytest.raises(ValueError, match="incomplete evidence"):
        join_consensus(sample, rq1, rq2, status)


def test_arm_mismatch_rejected():
    sample, rq1, rq2, status = inputs()
    rq2.loc[0, "sample_arm"] = "human_candidate"
    with pytest.raises(ValueError, match="arm mismatch"):
        join_consensus(sample, rq1, rq2, status)


def test_corpus_joins_use_repository_identity_and_preserve_long_text():
    sample, *_ = inputs()
    keys = sample[["repo_id", "number"]]
    text = "padding " * 3000 + "latency reduced from 120 ms to 80 ms"
    tables = {
        "pull_requests": keys.assign(title="PR", body="Description"),
        "issue_comments": keys.assign(body=[text, "unrelated", ""], author_type="User"),
        "review_comments": keys.assign(body=""),
        "commits": keys.assign(message=""),
        "workflow_runs": keys.assign(name=""),
        "pull_request_files": keys.assign(patch=""),
    }
    corpus, _ = assemble_corpus(sample, tables)
    assert corpus[(1, 7)]["issue_comments"] == text
    assert "D1" in extract_dimensions(corpus[(1, 7)]["issue_comments"])
    assert not extract_dimensions(corpus[(2, 7)]["issue_comments"])


def test_quantitative_ties_are_hash_seed_independent():
    source = Path(__file__).resolve().parents[2] / "analysis/rq3_pattern_and_validation"
    script = (
        "import json; from extract_metrics import extract_dimensions; "
        "print(json.dumps(extract_dimensions('latency 20% faster memory 2x throughput 100 req/s'), sort_keys=True))"
    )
    results = [subprocess.check_output([sys.executable, "-c", script], cwd=source,
        env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True) for seed in (1, 2, 42)]
    assert len(set(results)) == 1
