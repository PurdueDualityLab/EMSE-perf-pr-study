from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import schema
import build_rebalanced_dataset
from build_rebalanced_dataset import (  # type: ignore
    build_pipeline_signature,
    initial_resume_state,
    prepare_resume_checkpoint,
    repository_identity_hash,
    resume_state_path,
    save_checkpoint_dataframe,
    save_resume_state,
)
from resume import load_resume_state
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
            "github": {"enabled": True, "token_file": "mining/github_tokens.txt"},
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
        resume_cfg={
            "enabled": True,
            "reset": False,
            "trust_legacy_checkpoint": True,
        },
        signature=signature,
        repo_count=3,
    )

    assert len(raw_df) == 2
    assert completed == {"a/b", "c/d"}
    assert new_state["completed_repo_count"] == 2
    assert new_state["raw_rows_count"] == 2
    assert returned_state_path == state_path
    assert new_state["raw_checkpoint_sha256"]


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


def test_existing_state_does_not_infer_partial_repo_as_completed(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dirs = ensure_output_dirs(output_dir)
    raw_path = output_dirs["raw"] / "github_human_prs.parquet"
    state_path = resume_state_path(output_dir)
    signature = _signature()
    state = initial_resume_state(signature, 1)
    state["repo_mining_reports"] = [
        {"repo_full_name": "a/b", "status": "partial", "pr_fetch_failed_count": 1}
    ]
    save_resume_state(state_path, state)
    save_checkpoint_dataframe(
        pd.DataFrame([{"repo_full_name": "a/b", "number": 1}]),
        raw_path,
    )

    _, new_state, completed, _ = prepare_resume_checkpoint(
        output_dir=output_dir,
        output_dirs=output_dirs,
        resume_cfg={"enabled": True, "reset": False},
        signature=signature,
        repo_count=1,
    )

    assert completed == set()
    assert new_state["completed_repo_names"] == []
    assert new_state["raw_rows_count"] == 1


def test_prepare_resume_checkpoint_rejects_orphan_raw_checkpoint(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dirs = ensure_output_dirs(output_dir)
    save_checkpoint_dataframe(
        pd.DataFrame([{"repo_full_name": "a/b", "number": 1}]),
        output_dirs["raw"] / "github_human_prs.parquet",
    )

    with pytest.raises(ValueError, match="without its resume state"):
        prepare_resume_checkpoint(
            output_dir=output_dir,
            output_dirs=output_dirs,
            resume_cfg={"enabled": True, "reset": False},
            signature=_signature(),
            repo_count=1,
        )


def test_prepare_resume_checkpoint_requires_opt_in_for_unhashed_completed_state(
    tmp_path,
):
    output_dir = tmp_path / "outputs"
    output_dirs = ensure_output_dirs(output_dir)
    state = initial_resume_state(_signature(), 1)
    state["completed_repo_names"] = ["a/b"]
    state["completed_repo_count"] = 1
    save_resume_state(resume_state_path(output_dir), state)
    save_checkpoint_dataframe(
        pd.DataFrame([{"repo_full_name": "a/b", "number": 1}]),
        output_dirs["raw"] / "github_human_prs.parquet",
    )

    with pytest.raises(ValueError, match="trust_legacy_checkpoint"):
        prepare_resume_checkpoint(
            output_dir=output_dir,
            output_dirs=output_dirs,
            resume_cfg={"enabled": True, "reset": False},
            signature=_signature(),
            repo_count=1,
        )


def test_prepare_resume_checkpoint_removes_orphaned_staged_parquet(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dirs = ensure_output_dirs(output_dir)
    staged = output_dirs["raw"] / ".github_human_prs.parquet.dead.pending.parquet"
    staged.write_bytes(b"partial")

    prepare_resume_checkpoint(
        output_dir=output_dir,
        output_dirs=output_dirs,
        resume_cfg={"enabled": True, "reset": False},
        signature=_signature(),
        repo_count=1,
    )

    assert not staged.exists()


def test_prepare_resume_checkpoint_accepts_parquet_written_before_state(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dirs = ensure_output_dirs(output_dir)
    state = initial_resume_state(_signature(), 1)
    state["raw_rows_count"] = 2
    save_resume_state(resume_state_path(output_dir), state)
    save_checkpoint_dataframe(
        pd.DataFrame([{"repo_full_name": "a/b", "number": 1}]),
        output_dirs["raw"] / "github_human_prs.parquet",
    )

    raw, resumed_state, completed, _ = prepare_resume_checkpoint(
        output_dir=output_dir,
        output_dirs=output_dirs,
        resume_cfg={"enabled": True, "reset": False},
        signature=_signature(),
        repo_count=1,
    )

    assert len(raw) == 1
    assert completed == set()
    assert resumed_state["raw_rows_count"] == 1


def test_pending_checkpoint_journal_recovers_state_after_interruption(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "outputs"
    output_dirs = ensure_output_dirs(output_dir)
    raw_path = output_dirs["raw"] / "github_human_prs.parquet"
    state_path = resume_state_path(output_dir)
    signature = _signature()
    state = initial_resume_state(signature, 1)
    real_save_state = build_rebalanced_dataset.save_resume_state

    def fail_main_state(path, payload):
        if path == state_path:
            raise OSError("interrupted after raw replacement")
        real_save_state(path, payload)

    monkeypatch.setattr(build_rebalanced_dataset, "save_resume_state", fail_main_state)
    with pytest.raises(OSError, match="interrupted"):
        build_rebalanced_dataset.save_repo_checkpoint(
            pd.DataFrame([{"repo_full_name": "a/b", "number": 1}]),
            raw_path,
            state_path,
            state,
            {"a/b"},
            {"repo_reports": [], "pr_failures": []},
            "a/b",
        )

    assert raw_path.is_file()
    assert state_path.with_name("checkpoint_pending.json").is_file()
    monkeypatch.setattr(build_rebalanced_dataset, "save_resume_state", real_save_state)

    raw, recovered_state, completed, _ = prepare_resume_checkpoint(
        output_dir=output_dir,
        output_dirs=output_dirs,
        resume_cfg={"enabled": True, "reset": False},
        signature=signature,
        repo_count=1,
    )

    assert len(raw) == 1
    assert completed == {"a/b"}
    assert recovered_state["raw_checkpoint_sha256"]
    assert not state_path.with_name("checkpoint_pending.json").exists()


def test_resume_rejects_checkpoint_that_no_longer_matches_recorded_hash(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dirs = ensure_output_dirs(output_dir)
    raw_path = output_dirs["raw"] / "github_human_prs.parquet"
    state_path = resume_state_path(output_dir)
    signature = _signature()
    build_rebalanced_dataset.save_repo_checkpoint(
        pd.DataFrame([{"repo_full_name": "a/b", "number": 1}]),
        raw_path,
        state_path,
        initial_resume_state(signature, 1),
        {"a/b"},
        {"repo_reports": [], "pr_failures": []},
        "a/b",
    )
    save_checkpoint_dataframe(
        pd.DataFrame([{"repo_full_name": "a/b", "number": 999}]),
        raw_path,
    )

    with pytest.raises(ValueError, match="integrity mismatch"):
        prepare_resume_checkpoint(
            output_dir=output_dir,
            output_dirs=output_dirs,
            resume_cfg={"enabled": True, "reset": False},
            signature=signature,
            repo_count=1,
        )


def test_resume_state_write_is_atomic_on_replace_failure(tmp_path, monkeypatch):
    state_path = tmp_path / "state" / "state.json"
    save_resume_state(state_path, {"value": "old"})

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(schema.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        save_resume_state(state_path, {"value": "new"})

    assert load_resume_state(state_path) == {"value": "old"}
    assert list(state_path.parent.glob(f".{state_path.name}.*.tmp")) == []


def test_checkpoint_parquet_write_is_atomic_on_replace_failure(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "raw" / "checkpoint.parquet"
    save_checkpoint_dataframe(pd.DataFrame([{"id": 1}]), checkpoint_path)

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(schema.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        save_checkpoint_dataframe(pd.DataFrame([{"id": 2}]), checkpoint_path)

    assert pd.read_parquet(checkpoint_path)["id"].tolist() == [1]
    assert list(checkpoint_path.parent.glob(f".{checkpoint_path.name}.*.tmp")) == []


def test_pipeline_signature_changes_with_search_limit_revision_and_repositories():
    base_config = {
        "source": {
            "mode": "real",
            "aidev_dataset": "dysavepeople/AIDev",
            "aidev_revision": "revision-one",
        },
        "criteria": {"star_floor": 100, "task_type": "perf"},
        "github": {
            "enabled": True,
            "token_file": "tokens.txt",
            "per_repo_search_limit": 10,
        },
    }
    window = TimeWindow(
        start=pd.Timestamp("2025-01-01T00:00:00Z"),
        end=pd.Timestamp("2025-01-02T00:00:00Z"),
    )
    first_hash = repository_identity_hash([{"repo_id": 1, "repo_full_name": "a/b"}])
    second_hash = repository_identity_hash([{"repo_id": 2, "repo_full_name": "c/d"}])
    base = build_pipeline_signature(base_config, window, 1, None, first_hash)

    changed_limit = {
        **base_config,
        "github": {**base_config["github"], "per_repo_search_limit": 20},
    }
    changed_revision = {
        **base_config,
        "source": {**base_config["source"], "aidev_revision": "revision-two"},
    }

    assert build_pipeline_signature(changed_limit, window, 1, None, first_hash) != base
    assert build_pipeline_signature(changed_revision, window, 1, None, first_hash) != base
    assert build_pipeline_signature(base_config, window, 1, None, second_hash) != base
    assert base["per_repo_search_limit"] == 10
    assert base["aidev_revision"] == "revision-one"
    assert base["repository_identity_hash"] == first_hash
