"""
Shared data loading utilities for the GitHub Performance Patch Study.

Centralizes loading of the main dataset, parquet bundles, and common
joins so that all RQ notebooks/scripts use consistent data.
"""

import os
import re
from pathlib import Path

import pandas as pd

# ============================================================
# Paths (relative to repo root)
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

MAIN_CSV = REPO_ROOT / "datasets" / "performance_prs_ai_vs_human.csv"
RAW_CSV = REPO_ROOT / "datasets" / "performance_prs_ai_vs_human_raw.csv"

AI_PR_DIR = REPO_ROOT / "datasets" / "ai_pr"
HUMAN_PR_DIR = REPO_ROOT / "datasets" / "human_pr"

VALIDATION_FINAL = (
    REPO_ROOT
    / "RQ2_test_and_validation"
    / "final_data"
    / "rq4_validation_evidence_final.parquet"
)

# HuggingFace dataset URIs
HF_BASE = "hf://datasets/hao-li/AIDev"

# ============================================================
# Core loaders
# ============================================================


def load_main_dataset() -> pd.DataFrame:
    """Load the central AI-vs-human performance PR dataset."""
    return pd.read_csv(MAIN_CSV)


def load_raw_dataset() -> pd.DataFrame:
    """Load the raw (pre-enrichment) dataset."""
    return pd.read_csv(RAW_CSV)


def load_parquet(directory: Path, filename: str) -> pd.DataFrame:
    """Load a parquet file from a dataset directory."""
    return pd.read_parquet(directory / filename)


# --- AI PR parquets ---

def load_ai_workflow_runs() -> pd.DataFrame:
    return load_parquet(AI_PR_DIR, "ai_pr_workflow_runs.parquet")


def load_ai_issue_comments() -> pd.DataFrame:
    return load_parquet(AI_PR_DIR, "ai_pr_issue_comments.parquet")


def load_ai_review_comments() -> pd.DataFrame:
    return load_parquet(AI_PR_DIR, "ai_pr_review_comments.parquet")


def load_ai_commits() -> pd.DataFrame:
    return load_parquet(AI_PR_DIR, "ai_pr_commits.parquet")


def load_ai_commit_details() -> pd.DataFrame:
    return load_parquet(AI_PR_DIR, "ai_pr_commit_details.parquet")


# --- Human PR parquets ---

def load_human_workflow_runs() -> pd.DataFrame:
    return load_parquet(HUMAN_PR_DIR, "human_pr_workflow_runs.parquet")


def load_human_issue_comments() -> pd.DataFrame:
    return load_parquet(HUMAN_PR_DIR, "human_pr_issue_comments.parquet")


def load_human_review_comments() -> pd.DataFrame:
    return load_parquet(HUMAN_PR_DIR, "human_pr_review_comments.parquet")


def load_human_commits() -> pd.DataFrame:
    return load_parquet(HUMAN_PR_DIR, "human_pr_commits.parquet")


def load_human_commit_details() -> pd.DataFrame:
    return load_parquet(HUMAN_PR_DIR, "human_pr_commit_details.parquet")


# --- Combined loaders ---

def load_all_workflow_runs() -> pd.DataFrame:
    """Load and concatenate AI + human workflow runs."""
    ai = load_ai_workflow_runs()
    ai["author_type"] = "AI Agent"
    human = load_human_workflow_runs()
    human["author_type"] = "Human"
    return pd.concat([ai, human], ignore_index=True)


def load_all_issue_comments() -> pd.DataFrame:
    """Load and concatenate AI + human issue comments."""
    ai = load_ai_issue_comments()
    ai["author_type"] = "AI Agent"
    human = load_human_issue_comments()
    human["author_type"] = "Human"
    return pd.concat([ai, human], ignore_index=True)


def load_all_review_comments() -> pd.DataFrame:
    """Load and concatenate AI + human review comments."""
    ai = load_ai_review_comments()
    ai["author_type"] = "AI Agent"
    human = load_human_review_comments()
    human["author_type"] = "Human"
    return pd.concat([ai, human], ignore_index=True)


def load_all_commit_details() -> pd.DataFrame:
    """Load and concatenate AI + human commit details."""
    ai = load_ai_commit_details()
    ai["author_type"] = "AI Agent"
    human = load_human_commit_details()
    human["author_type"] = "Human"
    return pd.concat([ai, human], ignore_index=True)


def load_validation_evidence() -> pd.DataFrame:
    """Load the final consolidated validation evidence from RQ2."""
    return pd.read_parquet(VALIDATION_FINAL)


# --- HuggingFace loaders ---

def load_hf_pr_task_types() -> pd.DataFrame:
    """Load AI PR task type classifications from HuggingFace."""
    return pd.read_parquet(f"{HF_BASE}/pr_task_type.parquet")


def load_hf_human_pr_task_types() -> pd.DataFrame:
    """Load human PR task type classifications from HuggingFace."""
    return pd.read_parquet(f"{HF_BASE}/human_pr_task_type.parquet")


def load_hf_pull_requests() -> pd.DataFrame:
    """Load AI pull requests from HuggingFace."""
    return pd.read_parquet(f"{HF_BASE}/pull_request.parquet")


def load_hf_human_pull_requests() -> pd.DataFrame:
    """Load human pull requests from HuggingFace."""
    return pd.read_parquet(f"{HF_BASE}/human_pull_request.parquet")


def load_hf_all_repositories() -> pd.DataFrame:
    """Load repository metadata from HuggingFace."""
    return pd.read_parquet(f"{HF_BASE}/all_repository.parquet")


# ============================================================
# Utility helpers
# ============================================================


def parse_github_url(html_url: str):
    """Parse a GitHub PR URL and return (owner, repo, number)."""
    m = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", html_url)
    if not m:
        raise ValueError(f"Cannot parse GitHub PR URL: {html_url}")
    owner, repo, number_str = m.groups()
    return owner, repo, int(number_str)


def get_merged_prs(df: pd.DataFrame = None) -> pd.DataFrame:
    """Filter to merged PRs only."""
    if df is None:
        df = load_main_dataset()
    return df[df["is_merged"] == True].copy()


def get_rejected_prs(df: pd.DataFrame = None) -> pd.DataFrame:
    """Filter to closed-but-not-merged PRs."""
    if df is None:
        df = load_main_dataset()
    return df[(df["state"] == "closed") & (df["is_merged"] == False)].copy()


def get_comments_for_pr(pr_id, issue_comments: pd.DataFrame,
                         review_comments: pd.DataFrame) -> pd.DataFrame:
    """Get all comments (issue + review) for a single PR, sorted by time."""
    ic = issue_comments[issue_comments["pr_id"] == pr_id][
        ["pr_id", "user_login", "user_type", "body", "created_at"]
    ].copy()
    ic["comment_source"] = "issue"

    rc = review_comments[review_comments["pr_id"] == pr_id][
        ["pr_id", "user_login", "user_type", "body", "created_at"]
    ].copy()
    rc["comment_source"] = "review"

    combined = pd.concat([ic, rc], ignore_index=True)
    combined["created_at"] = pd.to_datetime(combined["created_at"], errors="coerce")
    return combined.sort_values("created_at").reset_index(drop=True)
