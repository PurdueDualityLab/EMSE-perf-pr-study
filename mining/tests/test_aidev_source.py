from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_rebalanced_dataset import anti_join_agentic_urls
from load_aidev import read_aidev_table


def test_read_aidev_table_pins_configured_remote_revision(monkeypatch):
    paths = []

    def fake_read_parquet(path):
        paths.append(path)
        return pd.DataFrame()

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    read_aidev_table(
        "repository",
        {
            "source": {
                "aidev_dataset": "dysavepeople/AIDev",
                "aidev_revision": "abc123",
            }
        },
    )

    assert paths == ["hf://datasets/dysavepeople/AIDev@abc123/repository.parquet"]


def test_human_arm_anti_joins_known_agentic_urls_across_url_formats():
    human_prs = pd.DataFrame(
        [
            {
                "id": 1,
                "url": "https://api.github.com/repos/Owner/Repo/pulls/7",
            },
            {
                "id": 2,
                "html_url": "https://github.com/owner/repo/pull/8",
            },
            {"id": 3, "html_url": None},
        ]
    )
    agentic_prs = pd.DataFrame(
        [
            {
                "html_url": "https://github.com/owner/repo/pull/7/?source=aidev",
            }
        ]
    )

    result = anti_join_agentic_urls(human_prs, agentic_prs)

    assert result["id"].tolist() == [2, 3]
