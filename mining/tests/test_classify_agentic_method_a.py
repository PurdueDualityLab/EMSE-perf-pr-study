from pathlib import Path
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classify_agentic_method_a import METHOD_NAME, classify_frame, classify_parquet


def row(**updates):
    value = {
        "repo_id": 1,
        "number": 1,
        "html_url": "https://github.com/owner/repo/pull/1",
        "title": "Improve implementation",
        "body": "Implemented manually.",
        "user": "developer",
        "user_type": "User",
        "created_at": "2025-06-01T00:00:00Z",
        "aidev_attribution_label": "human_candidate",
        "aidev_attribution_rule": "",
        "aidev_attribution_head_ref": "feature/change",
        "aidev_attribution_author_login": "developer",
        "aidev_attribution_author_type": "User",
    }
    value.update(updates)
    return value


def test_method_a_classifies_each_supported_detection_mechanism():
    frame = pd.DataFrame(
        [
            row(number=1, user="devin-ai-integration[bot]"),
            row(
                number=2,
                user="copilot",
                aidev_attribution_author_login="copilot",
                aidev_attribution_head_ref="copilot/fix-2",
            ),
            row(
                number=3,
                aidev_attribution_label="agentic",
                aidev_attribution_rule="openai_codex_head",
            ),
            row(
                number=4,
                aidev_attribution_label="agentic",
                aidev_attribution_rule="cursor_head",
            ),
            row(
                number=5,
                body="[Codex Task](https://chatgpt.com/codex/cloud/tasks/task_123)",
            ),
            row(number=6, body="Generated with [Claude Code](https://claude.ai/code)"),
            row(number=7, body="Co-Authored-By: Claude <noreply@anthropic.com>"),
            row(
                number=8,
                aidev_attribution_label="agentic",
                aidev_attribution_rule="aidev_pinned_historical",
            ),
            row(number=9, user="dependabot[bot]", user_type="Bot"),
        ]
    )

    decisions = classify_frame(frame)

    assert decisions["method_a_label"].tolist() == ["agentic"] * len(frame)
    assert decisions["method_a_rules"].tolist() == [
        "devin_author",
        "copilot_author_branch",
        "codex_branch",
        "cursor_branch",
        "coding_agent_task_url",
        "explicit_agent_generation",
        "claude_coauthor",
        "aidev_historical_agent",
        "generic_bot_author",
    ]
    assert decisions["method_a_method"].eq(METHOD_NAME).all()


def test_method_a_keeps_review_only_and_ambiguous_user_accounts_non_agentic():
    frame = pd.DataFrame(
        [
            row(number=1, body="Reviewed by Cursor Bugbot for commit abc123."),
            row(number=2, user="useful-ai-agent", body="Implemented manually."),
            row(number=3),
        ]
    )

    decisions = classify_frame(frame)

    assert decisions["method_a_label"].tolist() == ["non_agentic"] * 3
    assert decisions["method_a_reason"].tolist() == [
        "review_only_signal",
        "no_agent_signal",
        "no_agent_signal",
    ]


def test_any_bot_author_is_agentic_even_for_review_automation():
    decision = classify_frame(
        pd.DataFrame(
            [
                row(
                    user="cursor-review[bot]",
                    user_type="Bot",
                    body="Reviewed by Cursor Bugbot for commit abc123.",
                )
            ]
        )
    ).iloc[0]

    assert decision["method_a_label"] == "agentic"
    assert decision["method_a_rules"] == "generic_bot_author"


def test_direct_branch_rules_respect_aidev_start_dates():
    frame = pd.DataFrame(
        [
            row(created_at="2025-05-15T23:59:59Z", aidev_attribution_head_ref="codex/task"),
            row(created_at="2025-05-16T00:00:00Z", aidev_attribution_head_ref="codex/task"),
            row(created_at="2024-12-31T23:59:59Z", aidev_attribution_head_ref="cursor/task"),
            row(created_at="2025-01-01T00:00:00Z", aidev_attribution_head_ref="cursor/task"),
        ]
    )

    decisions = classify_frame(frame)

    assert decisions["method_a_label"].tolist() == [
        "non_agentic",
        "agentic",
        "non_agentic",
        "agentic",
    ]


def test_multiple_positive_rules_are_preserved_in_stable_order():
    decision = classify_frame(
        pd.DataFrame(
            [
                row(
                    user="devin-ai-integration[bot]",
                    user_type="Bot",
                    body="Co-Authored-By: Claude <noreply@anthropic.com>",
                    aidev_attribution_label="agentic",
                    aidev_attribution_rule="devin_author|claude_code_coauthor",
                )
            ]
        )
    ).iloc[0]

    assert decision["method_a_rules"] == "devin_author|claude_coauthor|generic_bot_author"


def test_incomplete_aidev_search_rules_are_not_positive_matches():
    decision = classify_frame(
        pd.DataFrame(
            [
                row(
                    aidev_attribution_label="unresolved",
                    aidev_attribution_rule=(
                        "devin_author|github_copilot_head|openai_codex_head|cursor_head|"
                        "claude_code_coauthor"
                    ),
                )
            ]
        )
    ).iloc[0]

    assert decision["method_a_label"] == "non_agentic"
    assert decision["method_a_rules"] == ""


def test_classify_parquet_writes_binary_decisions_and_sensitivity_summary(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_dir = tmp_path / "output"
    frame = pd.DataFrame(
        [
            row(number=1),
            row(number=2, user="dependabot[bot]", user_type="Bot"),
            row(number=3, body="Generated with [Claude Code](https://claude.ai/code)"),
        ]
    )
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), input_path)

    summary = classify_parquet(input_path, output_dir, batch_size=2)
    output = pd.read_parquet(output_dir / "decisions.parquet")

    assert output["method_a_label"].tolist() == ["non_agentic", "agentic", "agentic"]
    assert summary["counts"]["labels"] == {"agentic": 2, "non_agentic": 1}
    assert summary["sensitivity"]["generic_bot_only_agentic"] == 1
    assert summary["sensitivity"]["agentic_without_generic_bot_rule"] == 1
    assert summary["input"]["rows"] == summary["output"]["rows"] == 3

    with pytest.raises(FileExistsError):
        classify_parquet(input_path, output_dir)
