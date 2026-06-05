from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from classify_task_type import attach_task_type


def test_attach_task_type_prefers_aidev_parquet_labels():
    prs = pd.DataFrame(
        [
            {"id": 1, "title": "fix: bug", "body": ""},
            {"id": 2, "title": "Improve cache performance", "body": "Benchmark evidence"},
        ]
    )
    task_types = pd.DataFrame([{"id": 1, "type": "perf"}])
    result = attach_task_type(prs, task_types).sort_values("id")
    assert result.loc[result["id"] == 1, "task_type"].item() == "perf"
    assert result.loc[result["id"] == 1, "task_type_source"].item() == "aidev_parquet"
    assert result.loc[result["id"] == 2, "task_type"].item() == "perf"
    assert result.loc[result["id"] == 2, "task_type_source"].item() == "compatible_classifier"


def test_attach_task_type_marks_conventional_commit_prefix():
    prs = pd.DataFrame([{"id": 3, "title": "docs: update readme", "body": ""}])
    result = attach_task_type(prs, pd.DataFrame())
    assert result["task_type"].item() == "docs"
    assert result["task_type_source"].item() == "compatible_classifier"
