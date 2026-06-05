from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_rebalanced_dataset import (  # type: ignore
    build_pipeline_signature,
    initial_resume_state,
    prepare_resume_checkpoint,
    resume_state_path,
    save_resume_state,
)
from schema import TimeWindow, ensure_output_dirs


def _signature() -> dict:
    return build_pipeline_signature(
        {
            "source": {
                "mode": "real",
                "aidev_dataset": "dysavepeople/AIDev",
                "data_root": None,
            },
            "criteria": {"star_floor": 100, "task_type": "perf"},
            "github": {"enabled": True, "token_env": "GITHUB_TOKEN"},
        },
        TimeWindow(
            start=pd.Timestamp("2025-01-01T00:00:00Z"),
            end=pd.Timestamp("2025-01-31T23:59:59Z"),
        ),
        3,
        None,
    )


def test_prepare_resume_checkpoint_recovers_existing_raw_checkpoint(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dirs = ensure_output_dirs(output_dir)
    raw_path = output_dirs["raw"] / "github_human_prs.parquet"
    state_path = resume_state_path(output_dir)

    signature = _signature()
    state = initial_resume_state(signature, 3)
    state["completed_repo_names"] = ["a/b", "c/d"]
    state["completed_repo_count"] = 2
    save_resume_state(state_path, state)

    pd.DataFrame(
        {
            "repo_full_name": ["a/b", "c/d"],
            "number": [1, 2],
            "filenames": [["src/a.py"], ["src/b.py"]],
        }
    ).to_parquet(raw_path, index=False)

    raw_df, new_state, completed, returned_state_path = prepare_resume_checkpoint(
        output_dir=output_dir,
        output_dirs=output_dirs,
        resume_cfg={"enabled": True, "reset": False},
        signature=signature,
        repo_count=3,
    )

    assert len(raw_df) == 2
    assert completed == {"a/b", "c/d"}
    assert new_state["completed_repo_count"] == 2
    assert new_state["raw_rows_count"] == 2
    assert returned_state_path == state_path


def test_prepare_resume_checkpoint_reset_clears_previous_outputs(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dirs = ensure_output_dirs(output_dir)
    raw_path = output_dirs["raw"] / "github_human_prs.parquet"
    state_path = resume_state_path(output_dir)

    signature = _signature()
    state = initial_resume_state(signature, 3)
    state["completed_repo_names"] = ["a/b"]
    state["completed_repo_count"] = 1
    save_resume_state(state_path, state)

    pd.DataFrame(
        {
            "repo_full_name": ["a/b"],
            "number": [1],
            "filenames": [["src/a.py"]],
        }
    ).to_parquet(raw_path, index=False)

    raw_df, new_state, completed, returned_state_path = prepare_resume_checkpoint(
        output_dir=output_dir,
        output_dirs=output_dirs,
        resume_cfg={"enabled": True, "reset": True},
        signature=signature,
        repo_count=3,
    )

    assert raw_df.empty
    assert completed == set()
    assert new_state["completed_repo_count"] == 0
    assert new_state["raw_rows_count"] == 0
    assert not raw_path.exists()
    assert returned_state_path == state_path
