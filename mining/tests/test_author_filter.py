from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from author_filter import is_agent_authored, keep_agent_arm, keep_human_arm


def test_agent_signature_is_excluded_from_human_arm():
    row = {
        "user": "claude-code-bot",
        "user_type": "Bot",
        "body": "Generated with [Claude Code]",
        "agent": "Claude Code",
    }
    assert is_agent_authored(row)
    assert keep_agent_arm(row)
    assert not keep_human_arm(row)


def test_human_author_is_kept_in_human_arm():
    row = {
        "user": "maintainer",
        "user_type": "User",
        "body": "I profiled this locally.",
        "agent": "",
    }
    assert not is_agent_authored(row)
    assert not keep_agent_arm(row)
    assert keep_human_arm(row)
