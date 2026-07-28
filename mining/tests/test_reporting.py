import json
from pathlib import Path
import sys

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import official_selection_report
import rebalancing_report
from build_official_selection import write_checksum_lock


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_parquet(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": range(rows)}).to_parquet(path, index=False)


def write_official_parquet(path: Path, rows: int, columns: set[str]) -> None:
    data = {}
    for column in columns:
        if column in {
            "quality_passed",
            "heuristic_match",
            "perfannotator_metadata_is_performance_improving",
        }:
            data[column] = [True] * rows
        elif column == "perfannotator_metadata_label_id":
            data[column] = [1] * rows
        elif column == "created_at":
            data[column] = ["2025-01-01T00:00:00Z"] * rows
        elif column == "arm":
            data[column] = ["agentic"] * rows
        else:
            data[column] = [f"value-{index}" for index in range(rows)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(path, index=False)


def write_rebalancing_outputs(
    output_dir: Path,
    *,
    agent_rows: int = 3,
    human_rows: int = 0,
    raw_rows: int = 4,
) -> None:
    write_parquet(output_dir / "final" / "agent_perf_prs_matched_criteria.parquet", agent_rows)
    write_parquet(output_dir / "final" / "human_perf_prs_matched_criteria.parquet", human_rows)
    write_parquet(output_dir / "intermediate" / "human_prs_task_typed.parquet", human_rows)
    write_parquet(output_dir / "intermediate" / "human_prs_filtered.parquet", human_rows)
    write_parquet(output_dir / "raw" / "github_human_prs.parquet", raw_rows)
    write_json(
        output_dir / "final" / "rebalancing_summary.json",
        {
            "agent_perf_prs_before_quality_filters": agent_rows,
            "agent_perf_prs_after_quality_filters": agent_rows,
            "human_perf_prs_before_quality_filters": human_rows,
            "human_perf_prs_after_quality_filters": human_rows,
            "stage_counts": {"raw_github_human_prs": raw_rows},
        },
    )
    write_json(
        output_dir / "final" / "rebalancing_detailed_report.json",
        {
            "source_counts": {},
            "stage_counts": {"raw_github_human_prs": raw_rows},
            "repo_mining": {"reports": []},
        },
    )


def write_official_outputs(
    output_dir: Path,
    *,
    human_rows: int = 2,
    manifest_status: str = "completed",
) -> None:
    write_json(
        output_dir / "selection_summary.json",
        {
            "window": {
                "start": "2024-12-24T00:23:09+00:00",
                "end": "2025-07-30T19:36:13+00:00",
                "inclusive": True,
            },
            "sampling": {"performed": False, "reason": "Not selected."},
            "model": {"id": "example/model", "revision": "abc123"},
            "stage_counts": {
                "input": 10,
                "after_date": 9,
                "after_heuristic": 7,
                "after_quality": 5,
                "after_model": 3,
                "agentic": 1,
                "human_non_agentic_candidate": 2,
            },
        },
    )
    parquet_paths = {
        name: output_dir / relative_path
        for name, relative_path in official_selection_report.PARQUET_PATHS.items()
    }
    parquet_rows = {
        "candidate_decisions": 7,
        "official_population": 3,
        "official_agentic": 1,
        "official_human_candidates": human_rows,
        "date_filtered": 9,
        "heuristic_candidates": 7,
        "quality_passed": 5,
        "model_predictions": 5,
    }
    for name, path in parquet_paths.items():
        write_official_parquet(
            path,
            parquet_rows[name],
            official_selection_report.REQUIRED_COLUMNS[name],
        )
    summary_path = output_dir / "selection_summary.json"
    write_json(
        output_dir / "run_manifest.json",
        {
            "status": manifest_status,
            "started_at": "2026-07-20T00:00:00+00:00",
            "completed_at": (
                "2026-07-20T00:05:00+00:00" if manifest_status == "completed" else None
            ),
            "run_signature": {"input_sha256": "abc123"},
            "reproducibility": {"seed": 20260720},
            "summary_artifact": {
                "sha256": official_selection_report.sha256_file(summary_path),
            },
            "outputs": {
                name: {
                    "output_rows": int(len(pd.read_parquet(path))),
                    "sha256": official_selection_report.sha256_file(path),
                    "schema_sha256": official_selection_report.schema_sha256(path),
                }
                for name, path in parquet_paths.items()
            },
        },
    )
    write_checksum_lock(
        output_dir,
        {
            "selection_summary": output_dir / "selection_summary.json",
            "run_manifest": output_dir / "run_manifest.json",
            **parquet_paths,
        },
    )


def test_rebalancing_report_rejects_directory_without_expected_outputs(tmp_path, capsys):
    with pytest.raises(FileNotFoundError, match="No rebalancing outputs found"):
        rebalancing_report.build_report(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        rebalancing_report.main(["--outputs-dir", str(tmp_path)])

    assert exc_info.value.code == 2
    assert "No rebalancing outputs found" in capsys.readouterr().err

    write_parquet(tmp_path / "raw" / "github_human_prs.parquet", 1)
    with pytest.raises(FileNotFoundError, match="Incomplete rebalancing outputs"):
        rebalancing_report.build_report(tmp_path)


def test_rebalancing_report_uses_parquet_metadata_for_row_counts(
    tmp_path, monkeypatch, capsys
):
    write_rebalancing_outputs(tmp_path)

    def fail_if_full_parquet_is_loaded(*args, **kwargs):
        raise AssertionError("report attempted to load a full parquet with pandas")

    monkeypatch.setattr(rebalancing_report.pd, "read_parquet", fail_if_full_parquet_is_loaded)
    report = rebalancing_report.build_report(tmp_path)

    assert report["counts"]["agent_perf_prs"] == 3
    assert report["counts"]["raw_github_human_prs"] == 4
    assert report["counts"]["total_perf_prs"] == 3

    monkeypatch.setattr(rebalancing_report, "DEFAULT_OUTPUTS_DIR", tmp_path)
    assert rebalancing_report.main(["--json"]) == 0
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["counts"]["raw_github_human_prs"] == 4


def test_rebalancing_report_rejects_conflicting_json_counts(tmp_path):
    write_rebalancing_outputs(tmp_path)
    detailed_path = tmp_path / "final" / "rebalancing_detailed_report.json"
    detailed = json.loads(detailed_path.read_text(encoding="utf-8"))
    detailed["stage_counts"]["raw_github_human_prs"] = 999
    write_json(detailed_path, detailed)

    with pytest.raises(rebalancing_report.ReportInputError, match="differs"):
        rebalancing_report.build_report(tmp_path)

    detailed["stage_counts"] = {}
    write_json(detailed_path, detailed)
    with pytest.raises(rebalancing_report.ReportInputError, match="keys differ"):
        rebalancing_report.build_report(tmp_path)


def test_official_selection_report_validates_counts_and_supports_text_and_json(
    tmp_path, capsys
):
    write_official_outputs(tmp_path)

    report = official_selection_report.build_report(tmp_path)

    assert report["validation"] == {"valid": True, "errors": []}
    assert report["parquet_row_counts"] == {
        "candidate_decisions": 7,
        "official_population": 3,
        "official_agentic": 1,
        "official_human_candidates": 2,
        "date_filtered": 9,
        "heuristic_candidates": 7,
        "quality_passed": 5,
        "model_predictions": 5,
    }
    assert report["partition"]["summary"]["matches"] is True
    assert report["partition"]["parquet"]["matches"] is True

    assert official_selection_report.main(["--output-dir", str(tmp_path)]) == 0
    assert "Validation: passed" in capsys.readouterr().out

    assert official_selection_report.main(["--output-dir", str(tmp_path), "--json"]) == 0
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["validation"]["valid"] is True


def test_official_selection_report_rejects_count_partition_mismatch(tmp_path):
    write_official_outputs(tmp_path, human_rows=1)

    report = official_selection_report.build_report(tmp_path)

    assert report["validation"]["valid"] is False
    assert report["partition"]["summary"]["matches"] is True
    assert report["partition"]["parquet"]["matches"] is False
    assert any(
        "Parquet count partition is invalid" in error
        for error in report["validation"]["errors"]
    )
    assert official_selection_report.main(["--output-dir", str(tmp_path), "--json"]) == 1


def test_official_selection_report_rejects_same_row_count_tampering(tmp_path):
    write_official_outputs(tmp_path)
    population_path = tmp_path / "official_population.parquet"
    pd.DataFrame({"row_id": [10, 11, 12]}).to_parquet(population_path, index=False)

    report = official_selection_report.build_report(tmp_path)

    assert report["validation"]["valid"] is False
    assert any("SHA-256 mismatch" in error for error in report["validation"]["errors"])


def test_official_checksum_round_trip_supports_nested_output_path(tmp_path):
    output_dir = tmp_path / "results" / "2026" / "selection"
    write_official_outputs(output_dir)

    report = official_selection_report.build_report(output_dir)

    assert report["validation"]["valid"] is True


def test_official_selection_report_is_safe_for_missing_and_incomplete_outputs(
    tmp_path, capsys
):
    missing_report = official_selection_report.build_report(tmp_path)

    assert missing_report["validation"]["valid"] is False
    assert all(value is None for value in missing_report["parquet_row_counts"].values())
    assert any(
        "selection_summary.json" in error for error in missing_report["validation"]["errors"]
    )
    assert official_selection_report.main(["--output-dir", str(tmp_path), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["validation"]["valid"] is False

    incomplete_dir = tmp_path / "incomplete"
    write_official_outputs(incomplete_dir, manifest_status="running")
    incomplete_report = official_selection_report.build_report(incomplete_dir)

    assert incomplete_report["validation"]["valid"] is False
    assert any(
        "status must be 'completed'" in error
        for error in incomplete_report["validation"]["errors"]
    )
