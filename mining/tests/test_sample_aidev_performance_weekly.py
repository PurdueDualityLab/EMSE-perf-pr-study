from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sample_aidev_performance_weekly import (
    DEFAULT_SEED,
    add_weekly_strata,
    build_balanced_sample_frames,
    selection_hash,
)


def row(repo_id, number, created_at):
    return {
        "repo_id": repo_id,
        "number": number,
        "html_url": f"https://github.com/owner/repo/pull/{number}",
        "created_at": created_at,
    }


def test_selection_hash_is_deterministic_and_seeded():
    first = selection_hash(DEFAULT_SEED, 123, 45)

    assert first == selection_hash(DEFAULT_SEED, 123, 45)
    assert first != selection_hash("another-seed", 123, 45)
    assert first != selection_hash(DEFAULT_SEED, 123, 46)
    assert len(first) == 64


def test_weekly_strata_use_iso_year_in_utc():
    frame = pd.DataFrame(
        [
            row(1, 1, "2024-12-29T23:59:59Z"),
            row(1, 2, "2024-12-30T00:00:00Z"),
            row(1, 3, "2025-01-05T23:59:59Z"),
            row(1, 4, "2025-01-06T00:00:00Z"),
        ]
    )

    result = add_weekly_strata(frame)

    assert result["sampling_stratum"].tolist() == [
        "2024-W52",
        "2025-W01",
        "2025-W01",
        "2025-W02",
    ]


def test_weekly_sampling_uses_all_agents_and_equal_human_quotas():
    agentic = pd.DataFrame(
        [
            row(1, 1, "2025-01-01T00:00:00Z"),
            row(1, 2, "2025-01-02T00:00:00Z"),
            row(2, 3, "2025-01-08T00:00:00Z"),
        ]
    )
    humans = pd.DataFrame(
        [
            row(10, 1, "2025-01-01T00:00:00Z"),
            row(10, 2, "2025-01-02T00:00:00Z"),
            row(10, 3, "2025-01-03T00:00:00Z"),
            row(20, 4, "2025-01-08T00:00:00Z"),
            row(20, 5, "2025-01-09T00:00:00Z"),
            row(30, 6, "2025-02-01T00:00:00Z"),
        ]
    )

    result = build_balanced_sample_frames(agentic, humans)
    selected = result["manifest"].loc[result["manifest"]["selected"]]
    weekly = selected.groupby(["sampling_stratum", "sample_arm"]).size().unstack(fill_value=0)

    assert len(result["agentic_sample"]) == 3
    assert len(result["human_sample"]) == 3
    assert len(result["balanced_sample"]) == 6
    assert weekly["agentic"].equals(weekly["human_candidate"])
    assert "2025-W05" not in set(result["human_sample"]["sampling_stratum"])
    assert result["human_sample"][["repo_id", "number"]].duplicated().sum() == 0


def test_same_seed_reproduces_human_sample_and_different_seed_changes_it():
    agentic = pd.DataFrame([row(1, 1, "2025-01-01T00:00:00Z")])
    humans = pd.DataFrame(
        [row(2, number, "2025-01-01T00:00:00Z") for number in range(1, 21)]
    )

    first = build_balanced_sample_frames(agentic, humans, "seed-a")["human_sample"]
    repeated = build_balanced_sample_frames(agentic, humans, "seed-a")["human_sample"]
    different = build_balanced_sample_frames(agentic, humans, "seed-b")["human_sample"]

    assert first[["repo_id", "number"]].equals(repeated[["repo_id", "number"]])
    assert not first[["repo_id", "number"]].equals(different[["repo_id", "number"]])


def test_weekly_sampling_rejects_insufficient_human_candidates():
    agentic = pd.DataFrame(
        [
            row(1, 1, "2025-01-01T00:00:00Z"),
            row(1, 2, "2025-01-02T00:00:00Z"),
        ]
    )
    humans = pd.DataFrame([row(2, 1, "2025-01-01T00:00:00Z")])

    with pytest.raises(ValueError, match="Insufficient human candidates"):
        build_balanced_sample_frames(agentic, humans)


def test_weekly_sampling_rejects_overlapping_populations():
    population = pd.DataFrame([row(1, 1, "2025-01-01T00:00:00Z")])

    with pytest.raises(ValueError, match="overlap"):
        build_balanced_sample_frames(population, population)
