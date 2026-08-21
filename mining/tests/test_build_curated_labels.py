from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_curated_labels import build_curated_labels_frame  # noqa: E402


def _inputs():
    attribution = pd.DataFrame(
        {
            "repo_id": [1, 1],
            "repo_full_name": ["org/repo", "org/repo"],
            "number": [1, 2],
            "html_url": ["https://example/1", "https://example/2"],
            "created_at": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True),
            "aidev_attribution_label": ["agentic", "human_candidate"],
            "aidev_attribution_agent": ["agent", ""],
            "aidev_attribution_rule": ["rule", ""],
            "aidev_attribution_evidence": ["evidence", ""],
            "aidev_attribution_status": ["matched", "complete_no_match"],
            "aidev_attribution_method": ["method", "method"],
        }
    )
    task_types = pd.DataFrame(
        {
            "repo_id": [1, 1],
            "number": [1, 2],
            "aidev_task_type": ["perf", "fix"],
            "aidev_task_type_reason": ["reason", "reason"],
            "aidev_task_type_confidence": [9, 8],
            "aidev_task_type_method": ["llm", "llm"],
            "aidev_task_type_status": ["classified", "classified"],
            "aidev_task_type_model": ["model", "model"],
            "aidev_task_type_classifier": ["classifier", "classifier"],
        }
    )
    human = pd.DataFrame(
        {
            "repo_id": [1],
            "number": [2],
            "human_filter_label": ["human_candidate"],
            "human_filter_reasons": [""],
            "human_filter_primary_reason": ["no_observable_agent_signal"],
            "human_filter_method": ["filter"],
        }
    )
    sampling = pd.DataFrame(
        {
            "repo_id": [1],
            "number": [1],
            "sample_arm": ["agentic"],
            "sampling_iso_year": [2025],
            "sampling_iso_week": [1],
            "sampling_stratum": ["2025-W01"],
            "stratum_population": [1],
            "stratum_quota": [1],
            "inclusion_probability": [1.0],
            "selected": [True],
            "sampling_seed": ["seed"],
            "sampling_method": ["sampling"],
        }
    )
    return attribution, task_types, human, sampling


def test_build_curated_labels_frame_combines_optional_stage_labels():
    result = build_curated_labels_frame(*_inputs())

    assert result["is_performance"].tolist() == [True, False]
    assert result.loc[0, "sample_arm"] == "agentic"
    assert bool(result.loc[0, "selected"]) is True
    assert pd.isna(result.loc[0, "human_filter_label"])
    assert result.loc[1, "human_filter_label"] == "human_candidate"
    assert pd.isna(result.loc[1, "sample_arm"])


def test_build_curated_labels_frame_rejects_nonmatching_full_populations():
    attribution, task_types, human, sampling = _inputs()
    task_types.loc[1, "number"] = 3

    with pytest.raises(ValueError, match="nonmatching identities"):
        build_curated_labels_frame(attribution, task_types, human, sampling)


def test_build_curated_labels_frame_rejects_duplicate_optional_stage_identity():
    attribution, task_types, human, sampling = _inputs()
    sampling = pd.concat([sampling, sampling], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate identities"):
        build_curated_labels_frame(attribution, task_types, human, sampling)
