import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aggregate_perfminer_prs as aggregator
import classify_perfminer_commits as classifier
import extract_perfminer_commit_evidence as evidence
import fetch_pr_commit_manifest as manifest
from perfminer_reproduction import (
    PERFMINER_MODEL_ARTIFACT_SHA256,
    PERFMINER_MODEL_WEIGHTS_SHA256,
    commit_manifest_sha256,
)


def run_git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def test_commit_manifest_freezes_head_and_commit_list(monkeypatch, tmp_path):
    input_path = tmp_path / "prs.parquet"
    output_dir = tmp_path / "manifest"
    sha = "a" * 40
    pq.write_table(
        pa.table(
            {
                "repo_id": [42],
                "repo_full_name": ["owner/project"],
                "number": [7],
                "html_url": ["https://github.com/owner/project/pull/7"],
            }
        ),
        input_path,
    )

    class FakeClient:
        def request(self, _path):
            return {"id": 42, "full_name": "owner/project"}

        def fetch_pull_request(self, _name, _number):
            return {
                "base": {"repo": {"id": 42}, "sha": "c" * 40},
                "head": {"sha": sha},
                "commits": 1,
                "html_url": "https://github.com/owner/project/pull/7",
            }

        def fetch_pull_commits(self, _name, _number):
            return [
                {
                    "sha": sha,
                    "commit": {"message": "Improve cache lookup performance now"},
                    "parents": [{"sha": "b" * 40}],
                    "html_url": f"https://github.com/owner/project/commit/{sha}",
                }
            ]

    fake_client = FakeClient()
    monkeypatch.setattr(
        manifest.GitHubClient,
        "from_token_file",
        classmethod(lambda _cls, _path: fake_client),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fetch_pr_commit_manifest.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--token-file",
            str(tmp_path / "tokens.txt"),
            "--workers",
            "1",
        ],
    )

    manifest.main()

    prs = pq.read_table(output_dir / "pull_requests.parquet").to_pylist()
    commits = pq.read_table(output_dir / "commits.parquet").to_pylist()
    assert prs[0]["commit_manifest_status"] == "complete"
    assert prs[0]["pr_head_sha"] == sha
    assert commits[0]["commit_sha"] == sha
    assert commits[0]["pr_manifest_complete"] is True
    state = json.loads((output_dir / "commit_manifest.state.json").read_text(encoding="utf-8"))
    assert state["finalized"] is True


def test_unstable_pr_snapshot_never_emits_commit_rows(monkeypatch):
    target = manifest.RepositoryTarget(
        repo_id=42,
        source_names={"owner/project"},
        pull_requests={
            7: manifest.PullRequestTarget(
                7, "https://github.com/owner/project/pull/7"
            )
        },
    )

    class MovingHeadClient:
        calls = 0

        def fetch_pull_request(self, _name, _number):
            self.calls += 1
            head = "a" * 40 if self.calls % 2 else "b" * 40
            return {
                "base": {"repo": {"id": 42}, "sha": "c" * 40},
                "head": {"sha": head},
                "commits": 1,
                "html_url": "https://github.com/owner/project/pull/7",
            }

        def fetch_pull_commits(self, _name, _number):
            return [
                {
                    "sha": "b" * 40,
                    "commit": {"message": "Improve cache lookup performance now"},
                    "parents": [{"sha": "c" * 40}],
                }
            ]

    monkeypatch.setattr(manifest.time, "sleep", lambda _seconds: None)

    pr_row, commit_rows, retryable = manifest.fetch_pull_request_manifest(
        target,
        target.pull_requests[7],
        "owner/project",
        MovingHeadClient(),
    )

    assert pr_row["commit_manifest_status"] == "unstable_pull_request_snapshot"
    assert commit_rows == []
    assert retryable is True


def test_manifest_checkpoint_units_bound_large_repositories():
    target = manifest.RepositoryTarget(repo_id=42, source_names={"owner/project"})
    target.pull_requests = {
        number: manifest.PullRequestTarget(number, f"https://example.test/{number}")
        for number in range(1, 122)
    }

    batches = manifest.checkpoint_targets({42: target}, 50)

    assert [len(batch.pull_requests) for batch in batches] == [50, 50, 21]
    assert [batch.state_key for batch in batches] == ["42:1", "42:2", "42:3"]


def test_extract_repository_uses_pydriller_commit_diff(monkeypatch, tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    run_git(repository, "init", "--quiet")
    run_git(repository, "config", "user.email", "test@example.com")
    run_git(repository, "config", "user.name", "Test User")
    source = repository / "cache.py"
    source.write_text("def hot_path(items):\n    return sum(items)\n", encoding="utf-8")
    run_git(repository, "add", "cache.py")
    run_git(repository, "commit", "--quiet", "-m", "Create initial cache implementation now")
    source.write_text(
        "def hot_path(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total += item\n"
        "    return total\n",
        encoding="utf-8",
    )
    run_git(repository, "add", "cache.py")
    message = "Improve cache lookup performance now"
    run_git(repository, "commit", "--quiet", "-m", message)
    sha = run_git(repository, "rev-parse", "HEAD")
    commit_target = evidence.CommitTarget(
        repo_id=42,
        source_repo_full_name="owner/project",
        resolved_repo_full_name="owner/project",
        pr_number=7,
        pr_html_url="https://github.com/owner/project/pull/7",
        pr_base_sha="c" * 40,
        pr_head_sha=sha,
        commit_manifest_sha256=commit_manifest_sha256([(1, sha)]),
        pr_manifest_complete=True,
        commit_index=1,
        commit_sha=sha,
        commit_message_api=message,
        parent_count_api=1,
        commit_html_url=f"https://github.com/owner/project/commit/{sha}",
    )
    target = evidence.RepositoryTarget(42, "owner/project", [commit_target])
    monkeypatch.setattr(evidence, "ensure_repository", lambda *_args, **_kwargs: repository)
    monkeypatch.setattr(evidence, "fetch_pull_refs", lambda *_args, **_kwargs: set())

    result = evidence.extract_repository(
        target,
        tmp_path / "cache",
        git_timeout=30,
        commit_timeout=10,
        commit_attempts=3,
    )

    assert result["status"] == "complete"
    row = result["rows"][0]
    assert row["perfminer_evidence_status"] == "eligible"
    assert row["api_message_matches_git"] is True
    assert row["changed_method_name"] == "hot_path"
    assert row["code_diff"].startswith("@@")


def test_evidence_rejects_commit_manifest_from_another_head(tmp_path):
    prs_path = tmp_path / "prs.parquet"
    commits_path = tmp_path / "commits.parquet"
    old_sha = "a" * 40
    new_sha = "b" * 40
    pr_row = {
        "repo_id": 42,
        "source_repo_full_name": "owner/project",
        "resolved_repo_full_name": "owner/project",
        "pr_number": 7,
        "pr_html_url": "https://github.com/owner/project/pull/7",
        "pr_base_sha": "c" * 40,
        "pr_head_sha": new_sha,
        "commit_manifest_sha256": commit_manifest_sha256([(1, new_sha)]),
        "expected_commit_count": 1,
        "listed_commit_count": 1,
        "commit_manifest_status": "complete",
        "commit_manifest_error": "",
    }
    commit_row = {
        "repo_id": 42,
        "source_repo_full_name": "owner/project",
        "resolved_repo_full_name": "owner/project",
        "pr_number": 7,
        "pr_html_url": "https://github.com/owner/project/pull/7",
        "pr_base_sha": "c" * 40,
        "pr_head_sha": old_sha,
        "commit_manifest_sha256": commit_manifest_sha256([(1, old_sha)]),
        "commit_index": 1,
        "commit_sha": old_sha,
        "commit_message_api": "Improve cache lookup performance now",
        "parent_count_api": 1,
        "commit_html_url": "",
        "pr_manifest_complete": True,
    }
    pq.write_table(pa.Table.from_pylist([pr_row], schema=manifest.PR_SCHEMA), prs_path)
    pq.write_table(pa.Table.from_pylist([commit_row], schema=manifest.COMMIT_SCHEMA), commits_path)

    with pytest.raises(ValueError, match="disagree on pr_head_sha"):
        evidence.collect_targets(commits_path, prs_path)


def test_evidence_checkpoint_never_splits_a_pull_request():
    commits = [
        evidence.CommitTarget(
            repo_id=42,
            source_repo_full_name="owner/project",
            resolved_repo_full_name="owner/project",
            pr_number=7,
            pr_html_url="https://github.com/owner/project/pull/7",
            pr_base_sha="c" * 40,
            pr_head_sha="f" * 40,
            commit_manifest_sha256="digest",
            pr_manifest_complete=True,
            commit_index=index,
            commit_sha=f"{index:040x}",
            commit_message_api="Improve cache lookup performance now",
            parent_count_api=1,
            commit_html_url="",
        )
        for index in range(1, 121)
    ]
    commits.append(
        evidence.CommitTarget(
            repo_id=42,
            source_repo_full_name="owner/project",
            resolved_repo_full_name="owner/project",
            pr_number=8,
            pr_html_url="https://github.com/owner/project/pull/8",
            pr_base_sha="c" * 40,
            pr_head_sha="e" * 40,
            commit_manifest_sha256="other-digest",
            pr_manifest_complete=True,
            commit_index=1,
            commit_sha="e" * 40,
            commit_message_api="Improve cache lookup performance now",
            parent_count_api=1,
            commit_html_url="",
        )
    )
    target = evidence.RepositoryTarget(42, "owner/project", commits)

    batches = evidence.checkpoint_targets({42: target}, 100)

    assert [len(batch.commits) for batch in batches] == [120, 1]
    assert {commit.pr_number for commit in batches[0].commits} == {7}


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 4 if pair else 2

    def __call__(self, text, text_pair=None, **kwargs):
        if kwargs.get("return_tensors") == "pt":
            rows = len(text)
            return {
                "input_ids": torch.ones((rows, 4), dtype=torch.long),
                "attention_mask": torch.ones((rows, 4), dtype=torch.long),
            }
        first = str(text).split()
        if text_pair is None:
            return {"input_ids": list(range(len(first)))}
        second = str(text_pair).split()
        return {"input_ids": list(range(len(first) + len(second) + 4))}


class PositiveModel:
    config = SimpleNamespace(
        num_labels=2,
        id2label={0: "LABEL_0", 1: "LABEL_1"},
        label2id={"LABEL_0": 0, "LABEL_1": 1},
    )

    def __call__(self, input_ids, **_inputs):
        rows = input_ids.shape[0]
        return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]] * rows))


def test_classifier_rejects_non_operational_batch_size(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify_perfminer_commits.py",
            "--input",
            str(tmp_path / "input.parquet"),
            "--output-dir",
            str(tmp_path / "output"),
            "--model-dir",
            str(tmp_path / "model"),
            "--batch-size",
            "2",
        ],
    )

    with pytest.raises(SystemExit):
        classifier.main()


def evidence_row(sha: str, status: str) -> dict:
    row = evidence.unavailable_evidence(sha, "filtered")
    row.update(
        {
            "repo_id": 42,
            "source_repo_full_name": "owner/project",
            "resolved_repo_full_name": "owner/project",
            "pr_number": 7,
            "pr_html_url": "https://github.com/owner/project/pull/7",
            "pr_base_sha": "c" * 40,
            "pr_head_sha": "b" * 40,
            "commit_manifest_sha256": commit_manifest_sha256(
                [(1, "a" * 40), (2, "b" * 40)]
            ),
            "pr_manifest_complete": True,
            "commit_index": 1 if sha.startswith("a") else 2,
            "commit_sha": sha,
            "commit_message_api": "Improve cache lookup performance now",
            "parent_count_api": 1,
            "commit_html_url": f"https://github.com/owner/project/commit/{sha}",
            "api_message_matches_git": True,
            "api_parent_count_matches_git": True,
            "commit_message_raw": "Improve cache lookup performance now",
            "commit_message": "Improve cache lookup performance now",
            "commit_message_word_count": 5,
            "parent_count": 1,
            "changed_file_count": 1,
            "filename": "cache.py",
            "change_type": "MODIFY",
            "changed_method_count": 1,
            "changed_method_name": "hot_path",
            "code_diff": "@@ -1 +1 @@\n-old\n+new",
            "diff_chars": 23,
            "source_pair_md5": "f" * 32,
            "perfminer_evidence_status": status,
            "perfminer_exclusion_reason": "" if status == "eligible" else "merge_commit",
            "perfminer_evidence_error": "",
        }
    )
    return row


def test_commit_classifier_and_pr_aggregation(monkeypatch, tmp_path):
    evidence_path = tmp_path / "evidence.parquet"
    prediction_dir = tmp_path / "predictions"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    rows = [evidence_row("a" * 40, "eligible"), evidence_row("b" * 40, "excluded")]
    pq.write_table(pa.Table.from_pylist(rows, schema=evidence.EVIDENCE_SCHEMA), evidence_path)
    monkeypatch.setattr(
        classifier,
        "verify_figshare_model",
        lambda _path: PERFMINER_MODEL_WEIGHTS_SHA256,
    )
    monkeypatch.setattr(
        classifier,
        "model_artifact_sha256",
        lambda _path: PERFMINER_MODEL_ARTIFACT_SHA256,
    )
    monkeypatch.setattr(
        classifier,
        "load_model",
        lambda *_args, **_kwargs: (FakeTokenizer(), PositiveModel(), "cpu"),
    )
    monkeypatch.setattr(classifier, "configure_reproducibility", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify_perfminer_commits.py",
            "--input",
            str(evidence_path),
            "--output-dir",
            str(prediction_dir),
            "--model-dir",
            str(model_dir),
            "--row-batch-size",
            "1",
        ],
    )

    classifier.main()

    prediction_path = prediction_dir / "commit_predictions.parquet"
    predictions = pq.read_table(prediction_path).to_pylist()
    assert predictions[0]["perfminer_classification_status"] == "classified"
    assert predictions[0]["perfminer_is_performance"] is True
    assert predictions[1]["perfminer_classification_status"] == "not_eligible"
    assert predictions[1]["perfminer_label_id"] is None

    pr_path = tmp_path / "pull_requests.parquet"
    pr_row = {
        "repo_id": 42,
        "source_repo_full_name": "owner/project",
        "resolved_repo_full_name": "owner/project",
        "pr_number": 7,
        "pr_html_url": "https://github.com/owner/project/pull/7",
        "pr_base_sha": "c" * 40,
        "pr_head_sha": "b" * 40,
        "commit_manifest_sha256": commit_manifest_sha256(
            [(1, "a" * 40), (2, "b" * 40)]
        ),
        "expected_commit_count": 2,
        "listed_commit_count": 2,
        "commit_manifest_status": "complete",
        "commit_manifest_error": "",
    }
    pq.write_table(pa.Table.from_pylist([pr_row], schema=manifest.PR_SCHEMA), pr_path)

    aggregated, counts = aggregator.aggregate(pr_path, prediction_path)

    result = aggregated.to_pylist()[0]
    assert counts == {"performance_positive": 1}
    assert result["perfminer_pr_is_performance"] is True
    assert result["perfminer_pr_evidence_complete"] is True
    assert result["perfminer_positive_commit_shas"] == ["a" * 40]


def test_pr_without_eligible_commits_is_out_of_scope_not_negative():
    pr = {
        "resolved_repo_full_name": "owner/project",
        "pr_base_sha": "c" * 40,
        "pr_head_sha": "a" * 40,
        "commit_manifest_sha256": commit_manifest_sha256([(1, "a" * 40)]),
        "expected_commit_count": 1,
        "listed_commit_count": 1,
        "commit_manifest_status": "complete",
    }
    commit = {
        "resolved_repo_full_name": "owner/project",
        "pr_base_sha": "c" * 40,
        "pr_head_sha": "a" * 40,
        "commit_manifest_sha256": commit_manifest_sha256([(1, "a" * 40)]),
        "pr_manifest_complete": True,
        "commit_index": 1,
        "commit_sha": "a" * 40,
        "source_pair_md5": "",
        "perfminer_evidence_status": "excluded",
        "perfminer_classification_status": "not_eligible",
        "perfminer_is_performance": None,
        "perfminer_score": None,
        "perfminer_model_weights_sha256": "weights-sha",
    }

    result = aggregator.aggregate_row(pr, [commit])

    assert result["perfminer_pr_status"] == "out_of_scope_no_eligible_commit"
    assert result["perfminer_pr_is_performance"] is None
    assert result["perfminer_pr_evidence_complete"] is True


def test_aggregation_rejects_predictions_from_another_contract(tmp_path):
    path = tmp_path / "predictions.parquet"
    row = {
        "repo_id": 42,
        "resolved_repo_full_name": "owner/project",
        "pr_number": 7,
        "pr_base_sha": "c" * 40,
        "pr_head_sha": "a" * 40,
        "commit_manifest_sha256": commit_manifest_sha256([(1, "a" * 40)]),
        "pr_manifest_complete": True,
        "commit_index": 1,
        "commit_sha": "a" * 40,
        "source_pair_md5": "f" * 32,
        "perfminer_evidence_status": "eligible",
        "perfminer_classification_status": "classified",
        "perfminer_label_id": 1,
        "perfminer_is_performance": True,
        "perfminer_score": 0.9,
        "perfminer_model_weights_sha256": PERFMINER_MODEL_WEIGHTS_SHA256,
        "perfminer_model_artifact_sha256": PERFMINER_MODEL_ARTIFACT_SHA256,
        "perfminer_inference_batch_size": 1,
        "perfminer_max_length": 512,
        "perfminer_truncation": "longest_first",
        "perfminer_threshold": 0.5,
    }
    pq.write_table(pa.Table.from_pylist([row]), path)

    with pytest.raises(ValueError, match="truncation"):
        aggregator.commit_groups(path)
