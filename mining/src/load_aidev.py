from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


TABLES = {
    "pull_request": "pull_request.parquet",
    "repository": "repository.parquet",
    "pr_task_type": "pr_task_type.parquet",
    "human_pull_request": "human_pull_request.parquet",
    "human_pr_task_type": "human_pr_task_type.parquet",
}


def read_aidev_table(table: str, config: dict[str, Any]) -> pd.DataFrame:
    source = config.get("source", {})
    data_root = source.get("data_root")
    filename = TABLES[table]
    if data_root:
        path = Path(data_root) / filename
    else:
        dataset = source.get("aidev_dataset", "dysavepeople/AIDev")
        path = f"hf://datasets/{dataset}/{filename}"
    return pd.read_parquet(path)


def load_aidev_tables(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {name: read_aidev_table(name, config) for name in TABLES}
