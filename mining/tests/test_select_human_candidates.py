from pathlib import Path
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from select_human_candidates import (
    METHOD_NAME,
    build_human_candidate_dataset,
    classify_human_candidates,
)


def row(**updates):
    value = {
        "repo_id": 1,
        "number": 1,
        "html_url": "https://github.com/owner/repo/pull/1",
        "title": "Improve performance",
        "body": "This change reduces allocations.",
        "user": "developer",
        "user_type": "User",
        "created_at": "2025-06-01T00:00:00Z",
        "aidev_attribution_label": "human_candidate",
        "aidev_attribution_status": "complete_no_match",
    }
    value.update(updates)
    return value


@pytest.mark.parametrize(
    ("updates", "expected_rule"),
    [
        ({"user": "codeflash-ai[bot]", "user_type": "Bot"}, "known_agent_login"),
        ({"user": "dependabot[bot]", "user_type": "Bot"}, "nonhuman_bot_author"),
        ({"user": "tempoxyz-bot"}, "suspicious_user_login"),
        (
            {"body": "[Codex Task](https://chatgpt.com/codex/cloud/tasks/task_123)"},
            "coding_agent_task_url",
        ),
        (
            {"body": "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"},
            "ai_coauthor_trailer",
        ),
        ({"body": "Generated with [Claude Code](https://claude.com/claude-code)"}, "explicit_ai_authorship"),
        ({"body": "Made with [Cursor](https://cursor.com)"}, "explicit_ai_authorship"),
        ({"body": "> _Written by Claude Code — human supervised._"}, "explicit_ai_authorship"),
        ({"body": "Reviewed by [Cursor Bugbot](https://cursor.com/bugbot)."}, "ai_review_signal"),
        ({"body": "<!-- BUGBOT_STATUS -->Cursor Bugbot reviewed this.<!-- /BUGBOT_STATUS -->"}, "ai_review_signal"),
        ({"body": "pr_agent:summary\nThis section was filled by PR-Agent."}, "ai_review_signal"),
        ({"body": "<!-- CURSOR_SUMMARY -->\nAutomated summary."}, "ai_generated_metadata"),
        ({"body": "<!-- End of auto-generated description by cubic. -->"}, "ai_generated_metadata"),
    ],
)
def test_strict_signals_exclude_human_candidates(updates, expected_rule):
    decision = classify_human_candidates(pd.DataFrame([row(**updates)])).iloc[0]

    assert decision["human_filter_label"] == "excluded_from_human_candidates"
    assert expected_rule in decision["human_filter_reasons"].split("|")
    assert decision["human_filter_method"] == METHOD_NAME


def test_product_mentions_and_unchecked_templates_remain_candidates():
    frame = pd.DataFrame(
        [
            row(body="Update the Claude SDK and document the Cursor pagination API."),
            row(body="- [ ] AI-assisted: explain any generated code"),
            row(user="AgentEnder", body="Implemented manually."),
            row(user="junagent", body="Implemented manually."),
        ]
    )

    decisions = classify_human_candidates(frame)

    assert decisions["human_filter_label"].tolist() == ["human_candidate"] * 4
    assert decisions["human_filter_primary_reason"].eq("no_observable_agent_signal").all()


def test_multiple_reasons_are_preserved_in_stable_order():
    decision = classify_human_candidates(
        pd.DataFrame(
            [
                row(
                    user="claude[bot]",
                    user_type="Bot",
                    body=(
                        "Generated with [Claude Code](https://claude.com/claude-code)\n"
                        "<!-- CURSOR_SUMMARY -->"
                    ),
                )
            ]
        )
    ).iloc[0]

    assert decision["human_filter_reasons"] == (
        "known_agent_login|nonhuman_bot_author|explicit_ai_authorship|ai_generated_metadata"
    )
    assert decision["human_filter_primary_reason"] == "known_agent_login"


def test_incomplete_baseline_evidence_is_not_retained():
    decision = classify_human_candidates(
        pd.DataFrame([row(aidev_attribution_status="search_incomplete")])
    ).iloc[0]

    assert decision["human_filter_label"] == "excluded_from_human_candidates"
    assert decision["human_filter_primary_reason"] == "insufficient_evidence"


def test_organization_author_is_not_retained_as_human_candidate():
    decision = classify_human_candidates(
        pd.DataFrame([row(user="project-org", user_type="Organization")])
    ).iloc[0]

    assert decision["human_filter_label"] == "excluded_from_human_candidates"
    assert decision["human_filter_primary_reason"] == "insufficient_evidence"


def test_build_dataset_selects_aidev_human_performance_rows(tmp_path):
    task_type_path = tmp_path / "task_type.parquet"
    attribution_path = tmp_path / "attribution.parquet"
    output_dir = tmp_path / "output"
    task_type = pd.DataFrame(
        [
            {**row(number=1), "aidev_task_type": "perf"},
            {**row(number=2, user="dependabot[bot]", user_type="Bot"), "aidev_task_type": "perf"},
            {**row(number=3), "aidev_task_type": "fix"},
            {**row(number=4), "aidev_task_type": "perf"},
        ]
    ).drop(columns=["aidev_attribution_label", "aidev_attribution_status"])
    attribution = pd.DataFrame(
        [
            {
                "repo_id": 1,
                "number": number,
                "aidev_attribution_label": "agentic" if number == 4 else "human_candidate",
                "aidev_attribution_agent": "openai_codex" if number == 4 else "",
                "aidev_attribution_rule": "openai_codex_head" if number == 4 else "",
                "aidev_attribution_evidence": "test",
                "aidev_attribution_status": "matched" if number == 4 else "complete_no_match",
                "aidev_attribution_method": "test",
            }
            for number in range(1, 5)
        ]
    )
    pq.write_table(pa.Table.from_pandas(task_type, preserve_index=False), task_type_path)
    pq.write_table(pa.Table.from_pandas(attribution, preserve_index=False), attribution_path)

    summary = build_human_candidate_dataset(
        task_type_path, attribution_path, output_dir, batch_size=2
    )
    decisions = pd.read_parquet(output_dir / "decisions.parquet")
    retained = pd.read_parquet(output_dir / "human_candidates.parquet")
    excluded = pd.read_parquet(output_dir / "excluded_from_human_candidates.parquet")

    assert decisions["number"].tolist() == [1, 2]
    assert retained["number"].tolist() == [1]
    assert excluded["number"].tolist() == [2]
    assert summary["counts"]["performance_prs"] == 3
    assert summary["counts"]["aidev_agentic_performance_prs"] == 1
    assert summary["counts"]["aidev_human_candidate_performance_prs"] == 2

    with pytest.raises(FileExistsError):
        build_human_candidate_dataset(task_type_path, attribution_path, output_dir)
