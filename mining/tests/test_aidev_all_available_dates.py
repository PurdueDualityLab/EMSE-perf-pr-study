from datetime import date
import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import classify_agentic_prs as incremental  # noqa: E402
from aidev_attribution import ATTRIBUTION_FIELDS, output_schema  # noqa: E402


def raw_row(number: int, created_at: str = "2025-08-01T00:00:00Z") -> dict[str, object]:
    return {
        "repo_id": 123,
        "repo_full_name": "owner/repo",
        "number": number,
        "html_url": f"https://github.com/owner/repo/pull/{number}",
        "created_at": created_at,
        "title": f"PR {number}",
    }


def attribution_row(row: dict[str, object], label: str = "human_candidate") -> dict[str, object]:
    values = dict(row)
    values.update({field.name: "" for field in ATTRIBUTION_FIELDS})
    values.update(
        aidev_attribution_label=label,
        aidev_attribution_status="complete_no_match" if label != "agentic" else "matched",
        aidev_attribution_method="aidev_published_searches_v1",
        aidev_attribution_detail_status="not_requested",
    )
    return values


def write_existing_state(path: Path, output_path: Path, raw_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "finalized": True,
                "output_sha256": incremental.sha256_file(output_path),
                "run_signature": {
                    "input_sha256": incremental.sha256_file(raw_path),
                    "end_date": "2025-07-30",
                    "code_sha256": {},
                },
            }
        )
    )


def test_prepare_pending_reuses_existing_identity_and_keeps_new_rows(tmp_path):
    raw_path = tmp_path / "raw.parquet"
    existing_path = tmp_path / "existing.parquet"
    pending_path = tmp_path / "pending.parquet"
    raw = pd.DataFrame([raw_row(1), raw_row(2)])
    raw.to_parquet(raw_path, index=False)
    pd.DataFrame([attribution_row(raw_row(1))]).to_parquet(existing_path, index=False)

    counts = incremental.prepare_pending(raw_path, existing_path, pending_path, batch_size=1)

    pending = pd.read_parquet(pending_path)
    assert counts == {
        "raw_rows": 2,
        "reused_rows": 1,
        "pending_rows": 1,
        "existing_identity_count": 1,
        "raw_identity_count": 2,
    }
    assert pending["number"].tolist() == [2]


def test_prepare_pending_rejects_duplicate_raw_identity(tmp_path):
    raw_path = tmp_path / "raw.parquet"
    existing_path = tmp_path / "existing.parquet"
    pending_path = tmp_path / "pending.parquet"
    pd.DataFrame([raw_row(1), raw_row(1)]).to_parquet(raw_path, index=False)
    pd.DataFrame([attribution_row(raw_row(2))]).to_parquet(existing_path, index=False)

    try:
        incremental.prepare_pending(raw_path, existing_path, pending_path, batch_size=2)
    except ValueError as error:
        assert "duplicate identity" in str(error)
    else:
        raise AssertionError("Expected duplicate raw identity rejection")


def test_historical_override_only_changes_non_agentic_labels():
    unmatched = {
        field.name: "" for field in ATTRIBUTION_FIELDS
    }
    unmatched.update(
        aidev_attribution_label="human_candidate",
        aidev_attribution_method="aidev_published_searches_v1",
        aidev_attribution_status="complete_no_match",
    )

    overridden = incremental.historical_override(unmatched, "openai_codex")

    assert overridden["aidev_attribution_label"] == "agentic"
    assert overridden["aidev_attribution_agent"] == "openai_codex"
    assert overridden["aidev_attribution_rule"] == "aidev_pinned_historical"
    assert overridden["aidev_attribution_status"] == "historical_aidev_precedence"
    already_agentic = dict(overridden)
    assert incremental.historical_override(already_agentic, "cursor") == already_agentic


def test_historical_aidev_agents_records_stable_identity_hash(tmp_path):
    source = tmp_path / "aidev_agentic.parquet"
    pd.DataFrame(
        {
            "repo_id": [123, 456],
            "number": [1, 2],
            "agent": ["OpenAI_Codex", "Cursor"],
        }
    ).to_parquet(source, index=False)

    agents, first_hash = incremental.historical_aidev_agents(str(source))
    _, second_hash = incremental.historical_aidev_agents(str(source))

    assert agents == {"123:1": "openai_codex", "456:2": "cursor"}
    assert first_hash == second_hash


def test_execute_validation_rejects_changed_existing_state(tmp_path):
    raw_path = tmp_path / "raw.parquet"
    existing_path = tmp_path / "existing.parquet"
    state_path = tmp_path / "existing.state.json"
    pending_path = tmp_path / "pending.parquet"
    pd.DataFrame([raw_row(1), raw_row(2)]).to_parquet(raw_path, index=False)
    pd.DataFrame([attribution_row(raw_row(1))]).to_parquet(existing_path, index=False)
    write_existing_state(state_path, existing_path, raw_path)
    plan = incremental.build_plan(
        raw_path,
        existing_path,
        state_path,
        date(2026, 6, 1),
        pending_path,
        batch_size=2,
    )

    state = json.loads(state_path.read_text())
    state["run_signature"]["end_date"] = "2025-07-31"
    state_path.write_text(json.dumps(state))

    try:
        incremental.validate_existing_against_plan(plan, raw_path, existing_path, state_path)
    except ValueError as error:
        assert "changed since" in str(error)
    else:
        raise AssertionError("Expected changed existing state rejection")


def test_merge_output_reuses_and_applies_historical_precedence(tmp_path):
    raw_path = tmp_path / "raw.parquet"
    existing_path = tmp_path / "existing.parquet"
    pending_path = tmp_path / "pending.parquet"
    output_path = tmp_path / "output.parquet"
    raw = pd.DataFrame([raw_row(1), raw_row(2)])
    raw.to_parquet(raw_path, index=False)
    pd.DataFrame([attribution_row(raw_row(1), "human_candidate")]).to_parquet(existing_path, index=False)
    pd.DataFrame([attribution_row(raw_row(2), "agentic")]).to_parquet(pending_path, index=False)

    counts = incremental.merge_output(
        raw_path,
        existing_path,
        pending_path,
        output_path,
        {"123:1": "openai_codex"},
        batch_size=2,
    )

    result = pq.read_table(output_path).to_pandas()
    assert result["number"].tolist() == [1, 2]
    assert result["aidev_attribution_label"].tolist() == ["agentic", "agentic"]
    assert result.loc[0, "aidev_attribution_status"] == "historical_aidev_precedence"
    assert counts["historical_overrides"] == 1
    assert counts["agentic"] == 2
    assert pq.ParquetFile(output_path).schema_arrow == output_schema(pq.ParquetFile(raw_path).schema_arrow)
