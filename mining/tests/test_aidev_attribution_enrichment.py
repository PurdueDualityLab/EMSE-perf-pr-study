from datetime import date
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aidev_attribution as aidev  # noqa: E402
from aidev_attribution import (  # noqa: E402
    PullRequestIdentity,
    RepositoryTarget,
    attribution_for_row,
    collect_targets,
    search_repository,
    sha256_file,
    write_enriched_parquet,
)
from github_client import GitHubClientError, GitHubNotFoundError  # noqa: E402


class FakeAIDevClient:
    def __init__(self, failed_qualifier=None, resolved_repo="owner/repo", resolved_repo_id=123):
        self.failed_qualifier = failed_qualifier
        self.resolved_repo = resolved_repo
        self.resolved_repo_id = resolved_repo_id
        self.repository_fetches = []
        self.searches = []
        self.detail_fetches = []

    def request(self, path):
        self.repository_fetches.append(path)
        return {"full_name": self.resolved_repo, "id": self.resolved_repo_id}

    def search_pull_requests_for_range(
        self,
        repo_full_name,
        start,
        end,
        *,
        search_qualifiers=None,
        strict=False,
    ):
        assert strict
        self.searches.append((repo_full_name, start, end, search_qualifiers))
        if search_qualifiers == self.failed_qualifier:
            raise GitHubClientError("simulated search failure")
        return {
            "head:codex/": [{"number": 1, "user": {"login": "developer"}}],
            "author:devin-ai-integration[bot]": [],
            "head:copilot/": [
                {"number": 2, "user": {"login": "copilot"}},
                {"number": 3, "user": {"login": "not-copilot"}},
            ],
            "head:cursor/": [],
            '"Co-Authored-By: Claude"': [{"number": 4, "user": {"login": "developer"}}],
        }[search_qualifiers]

    def fetch_pull_request(self, repo_full_name, number):
        self.detail_fetches.append((repo_full_name, number))
        return {
            "head": {"ref": f"agent/pr-{number}", "label": f"owner:agent/pr-{number}"},
            "user": {"login": f"author-{number}", "type": "User"},
        }


def repository_target(count=5):
    identities = [PullRequestIdentity("owner/repo", number) for number in range(1, count + 1)]
    return RepositoryTarget(
        "owner/repo",
        {identity.key: identity for identity in identities},
        repository_ids={123},
    )


def row(number, created_at="2025-06-01T00:00:00Z"):
    return {
        "html_url": f"https://github.com/owner/repo/pull/{number}",
        "repo_id": 123,
        "number": number,
        "created_at": created_at,
    }


def test_published_aidev_rules_are_queried_and_enriched():
    client = FakeAIDevClient()

    repository = search_repository(repository_target(), date(2025, 7, 30), client)

    assert repository["status"] == "complete"
    assert client.repository_fetches == ["/repositories/123"]
    assert [search[3] for search in client.searches] == [
        "head:codex/",
        "author:devin-ai-integration[bot]",
        "head:copilot/",
        "head:cursor/",
        '"Co-Authored-By: Claude"',
    ]
    assert [search[1] for search in client.searches] == [
        date(2025, 5, 16),
        date(2024, 12, 24),
        date(2025, 1, 1),
        date(2025, 1, 1),
        date(2025, 2, 24),
    ]
    assert client.detail_fetches == [
        ("owner/repo", 1),
        ("owner/repo", 2),
        ("owner/repo", 4),
    ]

    repositories = {"owner/repo": repository}
    codex = attribution_for_row(row(1), repositories, date(2025, 7, 30))
    copilot = attribution_for_row(row(2), repositories, date(2025, 7, 30))
    rejected_copilot = attribution_for_row(row(3), repositories, date(2025, 7, 30))
    claude = attribution_for_row(row(4), repositories, date(2025, 7, 30))
    unmatched = attribution_for_row(row(5), repositories, date(2025, 7, 30))

    assert (codex["aidev_attribution_label"], codex["aidev_attribution_agent"]) == (
        "agentic",
        "openai_codex",
    )
    assert codex["aidev_attribution_head_ref"] == "agent/pr-1"
    assert copilot["aidev_attribution_agent"] == "github_copilot"
    assert claude["aidev_attribution_agent"] == "claude_code"
    assert rejected_copilot["aidev_attribution_label"] == "human_candidate"
    assert unmatched["aidev_attribution_label"] == "human_candidate"


def test_failed_search_does_not_turn_applicable_rows_into_human_candidates():
    client = FakeAIDevClient(failed_qualifier="head:cursor/")
    repository = search_repository(repository_target(), date(2025, 7, 30), client)
    repositories = {"owner/repo": repository}

    before_cursor = attribution_for_row(
        row(5, "2024-12-25T00:00:00Z"),
        repositories,
        date(2025, 7, 30),
    )
    after_cursor = attribution_for_row(row(5), repositories, date(2025, 7, 30))

    assert repository["status"] == "incomplete"
    assert before_cursor["aidev_attribution_label"] == "human_candidate"
    assert after_cursor["aidev_attribution_label"] == "unresolved"
    assert after_cursor["aidev_attribution_status"] == "search_incomplete"


def test_result_without_pr_number_keeps_rows_unresolved():
    client = FakeAIDevClient()
    original_search = client.search_pull_requests_for_range

    def malformed_search(repo_full_name, start, end, *, search_qualifiers=None, strict=False):
        if search_qualifiers == "head:cursor/":
            return [{"user": {"login": "developer"}}]
        return original_search(
            repo_full_name,
            start,
            end,
            search_qualifiers=search_qualifiers,
            strict=strict,
        )

    client.search_pull_requests_for_range = malformed_search
    repository = search_repository(repository_target(), date(2025, 7, 30), client)
    decision = attribution_for_row(
        row(5),
        {"owner/repo": repository},
        date(2025, 7, 30),
    )

    assert repository["searches"]["cursor_head"]["status"] == "failed"
    assert decision["aidev_attribution_label"] == "unresolved"


def test_copilot_result_without_valid_login_keeps_rows_unresolved():
    client = FakeAIDevClient()
    original_search = client.search_pull_requests_for_range

    def missing_login_search(repo_full_name, start, end, *, search_qualifiers=None, strict=False):
        if search_qualifiers == "head:copilot/":
            return [{"number": 5, "user": {"login": 123}}]
        return original_search(
            repo_full_name,
            start,
            end,
            search_qualifiers=search_qualifiers,
            strict=strict,
        )

    client.search_pull_requests_for_range = missing_login_search
    repository = search_repository(repository_target(), date(2025, 7, 30), client)
    decision = attribution_for_row(row(5), {"owner/repo": repository}, date(2025, 7, 30))

    assert repository["searches"]["github_copilot_head"]["status"] == "failed"
    assert decision["aidev_attribution_label"] == "unresolved"


def test_completed_repository_resume_reuses_searches_and_details():
    target = repository_target()
    completed = search_repository(target, date(2025, 7, 30), FakeAIDevClient())
    resume_client = FakeAIDevClient(failed_qualifier="head:codex/")

    resumed = search_repository(target, date(2025, 7, 30), resume_client, completed)

    assert resumed == completed
    assert resume_client.repository_fetches == ["/repositories/123"]
    assert resume_client.searches == []
    assert resume_client.detail_fetches == []


def test_repository_rename_is_resolved_before_search_and_detail_fetch():
    identity = PullRequestIdentity("old-owner/old-repo", 1)
    target = RepositoryTarget(
        "old-owner/old-repo",
        {identity.key: identity},
        repository_ids={123},
    )
    client = FakeAIDevClient(resolved_repo="owner/repo")

    repository = search_repository(target, date(2025, 7, 30), client)

    assert repository["query_repo_full_name"] == "owner/repo"
    assert repository["repository_resolution"]["renamed"] is True
    assert {search[0] for search in client.searches} == {"owner/repo"}
    assert client.detail_fetches == [("owner/repo", 1)]


def test_reclaimed_repository_name_is_not_searched():
    identity = PullRequestIdentity("old-owner/old-repo", 1)
    target = RepositoryTarget(
        "old-owner/old-repo",
        {identity.key: identity},
        repository_ids={123},
    )
    client = FakeAIDevClient(resolved_repo="old-owner/old-repo", resolved_repo_id=999)

    repository = search_repository(target, date(2025, 7, 30), client)

    assert repository["status"] == "incomplete"
    assert repository["repository_resolution"]["status"] == "failed"
    assert client.searches == []


def test_deleted_repository_is_terminal_and_rows_remain_unresolved():
    class DeletedRepositoryClient(FakeAIDevClient):
        def request(self, path):
            self.repository_fetches.append(path)
            raise GitHubNotFoundError("deleted")

    repository = search_repository(
        repository_target(),
        date(2025, 7, 30),
        DeletedRepositoryClient(),
    )
    decision = attribution_for_row(row(1), {"owner/repo": repository}, date(2025, 7, 30))

    assert repository["status"] == "complete"
    assert repository["repository_resolution"]["status"] == "unavailable"
    assert decision["aidev_attribution_label"] == "unresolved"
    assert decision["aidev_attribution_status"] == "search_incomplete"


def test_repository_resolution_upgrade_adoption_is_hash_limited():
    current = {
        "input_sha256": "input",
        "code_sha256": {
            "aidev_attribution.py": "replacement",
            "github_client.py": "client",
        },
    }
    existing = {
        "input_sha256": "input",
        "code_sha256": {
            "aidev_attribution.py": aidev.REPOSITORY_RESOLUTION_PREDECESSOR_SHA256,
            "github_client.py": "client",
        },
    }

    assert aidev.can_adopt_repository_resolution_upgrade(existing, current) is True
    existing["code_sha256"]["aidev_attribution.py"] = "unexpected"
    assert aidev.can_adopt_repository_resolution_upgrade(existing, current) is False


def test_enriched_parquet_preserves_input_rows_and_values(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    source = pd.DataFrame(
        {
            "html_url": [f"https://github.com/owner/repo/pull/{number}" for number in (1, 5)],
            "repo_id": [123, 123],
            "repo_full_name": ["owner/repo", "owner/repo"],
            "number": [1, 5],
            "created_at": ["2025-06-01T00:00:00Z", "2025-06-02T00:00:00Z"],
            "title": ["First", "Second"],
            "filenames": [["src/a.py"], ["src/b.py"]],
        }
    )
    source.to_parquet(input_path, index=False)
    input_hash = sha256_file(input_path)
    repository = search_repository(repository_target(), date(2025, 7, 30), FakeAIDevClient())

    row_count, labels = write_enriched_parquet(
        input_path,
        output_path,
        {"owner/repo": repository},
        date(2025, 7, 30),
        batch_size=1,
    )

    enriched = pd.read_parquet(output_path)
    assert row_count == 2
    assert labels == {"agentic": 1, "human_candidate": 1}
    assert sha256_file(input_path) == input_hash
    pd.testing.assert_frame_equal(enriched[source.columns], source)
    assert enriched["aidev_attribution_label"].tolist() == ["agentic", "human_candidate"]


def test_target_collection_prefers_current_url_after_repository_rename(tmp_path):
    input_path = tmp_path / "input.parquet"
    pd.DataFrame(
        {
            "html_url": ["https://github.com/new-owner/new-repo/pull/7"],
            "aidev_source_html_url": ["https://github.com/old-owner/old-repo/pull/7"],
            "repo_id": [123],
            "number": [7],
            "created_at": ["2025-06-01T00:00:00Z"],
        }
    ).to_parquet(input_path, index=False)

    targets, counts = collect_targets(input_path, date(2025, 7, 30))

    assert set(targets) == {"new-owner/new-repo"}
    assert counts["eligible_unique_prs"] == 1


def test_target_without_repository_id_remains_unresolved(tmp_path):
    input_path = tmp_path / "input.parquet"
    input_row = row(1)
    input_row.pop("repo_id")
    pd.DataFrame([input_row]).to_parquet(input_path, index=False)

    targets, counts = collect_targets(input_path, date(2025, 7, 30))

    assert targets == {}
    assert counts["unverifiable_repository_rows"] == 1


def test_row_without_repository_id_is_not_inferred_from_other_rows(tmp_path):
    input_path = tmp_path / "input.parquet"
    rows = [row(1), row(2)]
    rows[1]["repo_id"] = None
    pd.DataFrame(rows).to_parquet(input_path, index=False)

    targets, _ = collect_targets(input_path, date(2025, 7, 30))
    repository = search_repository(targets["owner/repo"], date(2025, 7, 30), FakeAIDevClient())
    decision = attribution_for_row(rows[1], {"owner/repo": repository}, date(2025, 7, 30))

    assert decision["aidev_attribution_label"] == "unresolved"
    assert decision["aidev_attribution_status"] == "invalid_repository_identity"


def test_output_is_not_replaced_when_input_hash_changed(tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    pd.DataFrame([row(1)]).to_parquet(input_path, index=False)
    repository = search_repository(repository_target(), date(2025, 7, 30), FakeAIDevClient())

    try:
        write_enriched_parquet(
            input_path,
            output_path,
            {"owner/repo": repository},
            date(2025, 7, 30),
            batch_size=1,
            expected_input_sha256="not-the-current-hash",
        )
    except ValueError as error:
        assert "Input changed" in str(error)
    else:
        raise AssertionError("Expected an input hash mismatch")

    assert not output_path.exists()


def test_run_searches_checkpoints_completed_repository_groups(tmp_path, monkeypatch):
    targets = {
        f"owner/repo-{number}": RepositoryTarget(f"owner/repo-{number}", {})
        for number in range(3)
    }
    state = {"repositories": {}}
    checkpoint_path = tmp_path / "state.json"

    def fake_worker(target, end_date, token_file, previous):
        return {
            "repo_full_name": target.repo_full_name,
            "query_repo_full_name": target.repo_full_name,
            "repository_resolution": {
                "status": "complete",
                "resolved_repo_full_name": target.repo_full_name,
                "renamed": False,
            },
            "status": "complete",
            "searches": {},
            "details": {},
            "matched_input_count": 0,
        }

    monkeypatch.setattr(aidev, "repository_worker", fake_worker)

    aidev.run_searches(
        targets,
        date(2025, 7, 30),
        Path("unused-token-file"),
        2,
        state,
        checkpoint_path,
        checkpoint_every=2,
    )

    persisted = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert set(persisted["repositories"]) == set(targets)


def test_empty_cli_run_and_resume_do_not_construct_github_client(tmp_path, monkeypatch):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    pd.DataFrame(
        {
            "html_url": pd.Series(dtype="string"),
            "number": pd.Series(dtype="int64"),
            "created_at": pd.Series(dtype="string"),
        }
    ).to_parquet(input_path, index=False)
    monkeypatch.setattr(
        aidev.GitHubClient,
        "from_token_file",
        classmethod(lambda cls, token_file: (_ for _ in ()).throw(AssertionError("network client used"))),
    )

    base_args = [
        "aidev_attribution.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--end-date",
        "2025-07-30",
    ]
    monkeypatch.setattr(sys, "argv", base_args)
    aidev.main()
    monkeypatch.setattr(sys, "argv", [*base_args, "--resume"])
    aidev.main()

    assert output_path.exists()
    assert aidev.state_path(output_path).exists()
    assert aidev.summary_path(output_path).exists()
