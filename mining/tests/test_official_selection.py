import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_official_selection import (
    add_heuristic_columns,
    add_quality_decisions,
    assign_author_arms,
    load_or_create_manifest,
    output_paths,
    parquet_artifact,
    filter_dates,
    filter_heuristic,
    inclusive_date_mask,
    run_quality_filter,
    validate_pr_urls,
    verify_parquet_artifact,
    write_checksum_lock,
)


def test_date_mask_keeps_both_window_boundaries():
    values = pd.Series(
        [
            "2024-12-24T00:23:09Z",
            "2025-07-30T19:36:13Z",
            "2024-12-24T00:23:08Z",
            "2025-07-30T19:36:14Z",
            "not-a-date",
        ]
    )
    mask = inclusive_date_mask(
        values,
        pd.Timestamp("2024-12-24T00:23:09+00:00"),
        pd.Timestamp("2025-07-30T19:36:13+00:00"),
    )
    assert mask.tolist() == [True, True, False, False, False]


def test_heuristic_preserves_conventional_prefix_precedence():
    frame = pd.DataFrame(
        [
            {"title": "perf: reduce allocations", "body": ""},
            {"title": "fix: improve cache performance", "body": ""},
            {"title": "Improve parser", "body": "Lower latency under load."},
            {"title": "docs: update readme", "body": ""},
        ]
    )
    result = add_heuristic_columns(frame)
    assert result["heuristic_match"].tolist() == [True, False, True, False]
    assert result["heuristic_reason"].tolist() == [
        "conventional_commit_prefix",
        "conventional_commit_prefix",
        "performance_keyword_heuristic",
        "conventional_commit_prefix",
    ]


def test_quality_decisions_record_first_removal_reason():
    frame = pd.DataFrame(
        [
            {"id": 1, "filenames": ["src/cache.py"], "title": "perf: cache"},
            {"id": 2, "filenames": [], "title": "perf: missing"},
            {"id": 3, "filenames": [".github/workflows/ci.yml"], "title": "perf: ci"},
            {"id": 4, "filenames": ["src/a.py"], "title": "Merge branch main"},
        ]
    )
    result, removed_counts, _ = add_quality_decisions(frame)
    assert result["quality_reason"].tolist() == ["passed", "empty_filename", "config_only", "merge_only"]
    assert result["quality_passed"].tolist() == [True, False, False, False]
    assert removed_counts == {
        "empty_filename": 1,
        "config_only": 1,
        "deleted_repo": 0,
        "merge_only": 1,
    }


def test_author_arm_assignment_prefers_aidev_url_matches():
    frame = pd.DataFrame(
        [
            {
                "html_url": "https://github.com/org/agent/pull/1",
                "aidev_source_html_url": None,
                "user": "maintainer",
                "user_type": "User",
                "body": "",
                "title": "perf: cache",
            },
            {
                "html_url": "https://github.com/org/renamed/pull/2",
                "aidev_source_html_url": "https://github.com/org/human/pull/2",
                "user": "claude-code-bot",
                "user_type": "Bot",
                "body": "Generated with [Claude Code]",
                "title": "perf: cache",
            },
            {
                "html_url": "https://github.com/org/local-agent/pull/3",
                "aidev_source_html_url": None,
                "user": "claude-code-bot",
                "user_type": "Bot",
                "body": "Generated with [Claude Code]",
                "title": "perf: cache",
            },
            {
                "html_url": "https://github.com/org/candidate/pull/4",
                "aidev_source_html_url": None,
                "user": "maintainer",
                "user_type": "User",
                "body": "",
                "title": "perf: cache",
            },
        ]
    )
    result = assign_author_arms(
        frame,
        {"https://github.com/org/agent/pull/1"},
        {
            "https://github.com/org/agent/pull/1",
            "https://github.com/org/human/pull/2",
        },
    )
    assert result["arm"].tolist() == [
        "agentic",
        "human_non_agentic_candidate",
        "agentic",
        "human_non_agentic_candidate",
    ]
    assert result["arm_source"].tolist() == [
        "aidev_pull_request",
        "aidev_human_pull_request",
        "author_heuristic_agentic",
        "author_heuristic_non_agentic",
    ]


def test_empty_author_arm_assignment_keeps_string_schema():
    result = assign_author_arms(pd.DataFrame(columns=["html_url"]), set(), set())

    assert str(result["arm"].dtype) == "string"
    assert str(result["arm_source"].dtype) == "string"


def test_file_stages_preserve_the_requested_order(tmp_path):
    raw = pd.DataFrame(
        [
            {
                "html_url": "https://github.com/org/repo/pull/1",
                "created_at": "2024-12-24T00:23:09Z",
                "title": "perf: reduce allocations",
                "body": "",
                "filenames": ["src/cache.py"],
                "commit_messages": [],
                "deleted_repo": False,
            },
            {
                "html_url": "https://github.com/org/repo/pull/2",
                "created_at": "2024-12-24T00:23:09Z",
                "title": "perf: tune CI",
                "body": "",
                "filenames": [".github/workflows/ci.yml"],
                "commit_messages": [],
                "deleted_repo": False,
            },
            {
                "html_url": "https://github.com/org/repo/pull/3",
                "created_at": "2025-07-30T19:36:14Z",
                "title": "perf: outside window",
                "body": "",
                "filenames": ["src/cache.py"],
                "commit_messages": [],
                "deleted_repo": False,
            },
        ]
    )
    raw_path = tmp_path / "raw.parquet"
    date_path = tmp_path / "date.parquet"
    heuristic_path = tmp_path / "heuristic.parquet"
    quality_path = tmp_path / "quality.parquet"
    raw.to_parquet(raw_path, index=False)

    date_stats = filter_dates(
        raw_path,
        date_path,
        pd.Timestamp("2024-12-24T00:23:09+00:00"),
        pd.Timestamp("2025-07-30T19:36:13+00:00"),
    )
    heuristic_stats = filter_heuristic(date_path, heuristic_path)
    quality_stats = run_quality_filter(heuristic_path, quality_path)

    assert date_stats["output_rows"] == 2
    assert validate_pr_urls(date_path)["unique_html_url_rows"] == 2
    assert heuristic_stats["output_rows"] == 2
    assert quality_stats["output_rows"] == 1
    assert pd.read_parquet(quality_path)["html_url"].tolist() == [
        "https://github.com/org/repo/pull/1"
    ]


def test_completed_stage_validation_detects_artifact_tampering(tmp_path):
    path = tmp_path / "stage.parquet"
    pd.DataFrame({"id": [1]}).to_parquet(path, index=False)
    stage = parquet_artifact(path)

    verify_parquet_artifact("date", path, stage)

    pd.DataFrame({"id": [2]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_parquet_artifact("date", path, stage)


def test_resume_requires_the_exact_recorded_signature(tmp_path):
    input_path = tmp_path / "input.parquet"
    pd.DataFrame({"id": [1]}).to_parquet(input_path, index=False)
    output_dir = tmp_path / "selection"
    args = SimpleNamespace(
        aidev_dataset="dysavepeople/AIDev",
        aidev_revision="b6d1b8af952053f20fd9f9aa03e66bddd12228fe",
        device="cpu",
        end="2025-07-30T19:36:13+00:00",
        input=input_path,
        max_input_chars=20000,
        model="annon-123/PerfAnnotator-mini",
        model_artifact_sha256="fake-model-sha256",
        model_batch_size=16,
        model_revision="7d7ba362c257c3ea8c69d52b4a736ae4f182e68c",
        model_row_batch_size=512,
        resume=False,
        seed=20260720,
        start="2024-12-24T00:23:09+00:00",
    )
    output_dir.mkdir()
    stale_manifest_temp = output_dir / ".run_manifest.json.interrupted.tmp"
    stale_manifest_temp.write_text("partial", encoding="utf-8")
    paths = output_paths(output_dir)
    manifest = load_or_create_manifest(paths, args)
    assert manifest["run_signature"]["code_sha256"]
    assert not stale_manifest_temp.exists()

    manifest["run_signature"]["code_sha256"]["build_official_selection.py"] = "0" * 64
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args.resume = True

    with pytest.raises(ValueError, match="does not match the exact"):
        load_or_create_manifest(paths, args)


def test_checksum_lock_covers_completed_artifacts(tmp_path):
    output_dir = tmp_path / "selection"
    output_dir.mkdir()
    summary = output_dir / "selection_summary.json"
    population = output_dir / "official_population.parquet"
    summary.write_text("{}\n", encoding="utf-8")
    pd.DataFrame({"html_url": ["https://github.com/a/b/pull/1"]}).to_parquet(
        population, index=False
    )

    checksum_path = write_checksum_lock(
        output_dir,
        {"summary": summary, "official_population": population},
    )

    checksum_text = checksum_path.read_text(encoding="utf-8")
    assert "selection_summary.json" in checksum_text
    assert "official_population.parquet" in checksum_text
