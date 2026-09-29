"""Shared selection, corpus, prompt, schema, and manifest logic for RQ3 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PR_LEVEL = ROOT / "analysis/rq3_pattern_and_validation/current/data/rq3_pr_level.csv"
DEFAULT_MATCHES = ROOT / "analysis/rq3_pattern_and_validation/current/data/rq3_metric_matches.csv"
DEFAULT_EVIDENCE = ROOT / "mining/sample_evidence/final"
SELECTION_SEED = "rq3-regex-audit-sha256-v1"
PROMPT_VERSION = "rq3-llm-validation-v1"
TEXT_LIMIT = 15_000
DIFF_LIMIT = 30_000
TRUNCATION_MARKER = "\n\n... [content truncated for length] ..."
KEYS = ["repo_id", "number"]
ARMS = ("agentic", "human_candidate")
TASKS = ("regex_audit", "tradeoff_audit")
EVIDENCE_SOURCES = (
    "title_description", "issue_comments", "review_comments", "reviews",
    "commit_messages", "ci", "code_diff",
)
EVIDENCE_FILES = (
    "pull_requests.parquet", "pull_request_files.parquet", "commits.parquet",
    "issue_comments.parquet", "review_comments.parquet", "reviews.parquet",
    "workflow_runs.parquet", "check_runs.parquet", "collection_status.parquet",
)
SYSTEM_INSTRUCTION = (
    "You audit explicit quantitative performance claims in complete GitHub pull-request "
    "artifacts. Use only supplied evidence, quote it verbatim, and classify the PR as a whole."
)


class RegexAuditLabel(BaseModel):
    label: Literal["true_positive", "false_positive"]
    evidence_sources: list[Literal[*EVIDENCE_SOURCES]]
    evidence_quotes: list[str]
    rationale: str

    @field_validator("evidence_sources", mode="before")
    @classmethod
    def normalize_sources(cls, value: list[str]) -> list[str]:
        if value is None:
            return value
        return list(dict.fromkeys(list(value)))

    @model_validator(mode="after")
    def validate_semantics(self) -> "RegexAuditLabel":
        if not self.evidence_sources or not self.evidence_quotes:
            raise ValueError("every regex audit requires evidence sources and quotes")
        return self


class TradeoffAuditLabel(BaseModel):
    label: Literal["tradeoff", "joint_improvement"]
    gain_direction: Literal["improved"]
    memory_direction: Literal["increased", "reduced"]
    evidence_sources: list[Literal[*EVIDENCE_SOURCES]]
    evidence_quotes: list[str]
    rationale: str

    @field_validator("evidence_sources", mode="before")
    @classmethod
    def unique_sources(cls, value: list[str]) -> list[str]:
        if value is None or len(value) == 0:
            raise ValueError("evidence_sources must be nonempty")
        return list(dict.fromkeys(list(value)))

    @field_validator("evidence_quotes")
    @classmethod
    def quotes_required(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("evidence_quotes must be nonempty")
        return value

    @model_validator(mode="after")
    def validate_directions(self) -> "TradeoffAuditLabel":
        expected = "increased" if self.label == "tradeoff" else "reduced"
        if self.memory_direction != expected:
            raise ValueError(f"{self.label} requires memory_direction={expected}")
        return self


def label_model(task: str) -> type[BaseModel]:
    if task == "regex_audit":
        return RegexAuditLabel
    if task == "tradeoff_audit":
        return TradeoffAuditLabel
    raise ValueError(f"Unknown task: {task}")


def semantic_schema(task: str) -> dict[str, Any]:
    schema = label_model(task).model_json_schema()
    schema["additionalProperties"] = False
    return schema


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def select_regex_sample(
    frame: pd.DataFrame,
    *,
    per_arm: int = 44,
    expected_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Select the lowest versioned SHA-256 identities independently in each arm."""
    expected_counts = expected_counts or {"agentic": 437, "human_candidate": 441}
    population = frame[_bool(frame["validation_present"]) & frame["n_dims"].gt(0)].copy()
    counts = population["sample_arm"].value_counts().to_dict()
    if counts != expected_counts:
        raise ValueError(f"Regex-audit population counts are {counts}; expected {expected_counts}.")
    if population.duplicated(KEYS).any():
        raise ValueError("Regex-audit population contains duplicate PR identities.")
    population["selection_hash"] = population.apply(
        lambda row: hashlib.sha256(
            f"{SELECTION_SEED}:{int(row.repo_id)}:{int(row.number)}".encode()
        ).hexdigest(),
        axis=1,
    )
    selected = (
        population.sort_values(["sample_arm", "selection_hash", *KEYS], kind="mergesort")
        .groupby("sample_arm", sort=True, group_keys=False)
        .head(per_arm)
    )
    selected_counts = selected["sample_arm"].value_counts().to_dict()
    if selected_counts != {arm: per_arm for arm in ARMS}:
        raise ValueError(f"Could not select {per_arm} rows from every arm.")
    return selected.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def select_tradeoff_cases(frame: pd.DataFrame, *, expected_rows: int = 29) -> pd.DataFrame:
    flags = {column: _bool(frame[column]) for column in ("D1", "D2", "D3", "D5")}
    selected = frame[
        frame["validation_type"].isin(["benchmark", "profiling"])
        & frame["sub_pattern"].isin(["Caching", "Buffering"])
        & flags["D3"]
        & (flags["D1"] | flags["D2"] | flags["D5"])
    ].copy()
    if len(selected) != expected_rows or selected.duplicated(KEYS).any():
        raise ValueError(f"Tradeoff selector returned {len(selected)} unique rows; expected {expected_rows}.")
    return selected.sort_values(KEYS, kind="mergesort").reset_index(drop=True)


def prepare_samples(pr_level_path: Path, output_dir: Path) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Sample output directory is not empty.")
    source = pd.read_csv(pr_level_path)
    samples = {
        "regex_audit": select_regex_sample(source),
        "tradeoff_audit": select_tradeoff_cases(source),
    }
    output_dir.mkdir(parents=True)
    paths = {}
    for task, sample in samples.items():
        path = output_dir / f"{task}_sample.parquet"
        sample.to_parquet(path, index=False)
        paths[task] = path
    atomic_write_json(output_dir / "prepare_metadata.json", {
        "selection_seed": SELECTION_SEED,
        "source_sha256": sha256_file(pr_level_path),
        "samples": {task: {"rows": len(samples[task]), "sha256": sha256_file(path)} for task, path in paths.items()},
    })
    return paths


def _text(value: Any) -> str:
    return "" if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)) else str(value).strip()


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit] + TRUNCATION_MARKER


def _group(frame: pd.DataFrame, columns: list[str], order: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*KEYS, "text"])
    rows = []
    available = [column for column in order if column in frame]
    for key, group in frame.sort_values([*KEYS, *available], kind="mergesort").groupby(KEYS, sort=False):
        parts = []
        for row in group.to_dict("records"):
            values = [f"{column}={_text(row.get(column))}" for column in columns if _text(row.get(column))]
            if values:
                parts.append(" | ".join(values))
        rows.append({"repo_id": key[0], "number": key[1], "text": "\n\n".join(parts)})
    return pd.DataFrame(rows)


def _highlight_matches(matches: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "tradeoff_audit":
        matches = matches[matches["dimension"].isin(["D1", "D2", "D3", "D5"])]
    return _group(
        matches,
        ["dimension", "source", "cue", "quant_kind", "quant", "rule", "snippet"],
        ["dimension", "source", "cue", "quant", "snippet"],
    )


def build_input(sample_path: Path, matches_path: Path, evidence_dir: Path, task: str) -> pd.DataFrame:
    sample = pd.read_parquet(sample_path)
    if sample.empty or sample.duplicated(KEYS).any():
        raise ValueError("RQ3 requires a nonempty sample with unique PR identities.")
    tables = {name.removesuffix(".parquet"): pd.read_parquet(evidence_dir / name) for name in EVIDENCE_FILES}
    ids = set(map(tuple, sample[KEYS].to_numpy()))
    status = tables["collection_status"]
    if status.duplicated(KEYS).any() or not ids.issubset(set(map(tuple, status[KEYS].to_numpy()))):
        raise ValueError("Evidence status does not cover every selected PR exactly once.")
    status = status[status[KEYS].apply(tuple, axis=1).isin(ids)]
    status_columns = [
        "pull_request_files_status", "commits_status", "issue_comments_status",
        "review_comments_status", "reviews_status", "workflow_runs_status", "check_runs_status",
    ]
    incomplete = status.loc[~status[status_columns].eq("complete").all(axis=1), KEYS]
    if not incomplete.empty:
        raise ValueError("RQ3 refuses incomplete evidence for: " + json.dumps(incomplete.to_dict("records")))

    pull_requests = tables["pull_requests"]
    if pull_requests.duplicated(KEYS).any():
        raise ValueError("Evidence contains duplicate pull-request rows.")
    result = sample.merge(
        pull_requests[[*KEYS, "repo_full_name", "html_url", "title", "body"]],
        on=KEYS, how="left", suffixes=("", "_evidence"), validate="one_to_one",
    )
    for column in ("repo_full_name", "html_url", "title", "body"):
        evidence_column = f"{column}_evidence"
        if evidence_column in result:
            result[column] = result[evidence_column].combine_first(result.get(column))
            result = result.drop(columns=evidence_column)
    grouped = {
        "code_diff": _group(tables["pull_request_files"], ["filename", "patch"], ["file_index"]),
        "commit_messages": _group(tables["commits"], ["sha", "message"], ["commit_index"]),
        "issue_comments": _group(tables["issue_comments"], ["author_login", "created_at", "body"], ["created_at", "comment_id"]),
        "review_comments": _group(tables["review_comments"], ["author_login", "created_at", "path", "body"], ["created_at", "comment_id"]),
        "reviews": _group(tables["reviews"], ["author_login", "submitted_at", "state", "body"], ["submitted_at", "review_id"]),
    }
    workflows = _group(tables["workflow_runs"], ["name", "display_title", "status", "conclusion", "created_at"], ["created_at", "workflow_run_id"])
    checks = _group(tables["check_runs"], ["name", "app_slug", "status", "conclusion", "output_title", "output_summary"], ["started_at", "check_run_id"])
    ci = workflows.merge(checks, on=KEYS, how="outer", suffixes=("_workflows", "_checks"))
    ci["text"] = ci.get("text_workflows", "").fillna("") + "\n\n" + ci.get("text_checks", "").fillna("")
    grouped["ci"] = ci[[*KEYS, "text"]]
    matches = pd.read_csv(matches_path)
    matches = matches[matches[KEYS].apply(tuple, axis=1).isin(ids)]
    matched_ids = set(map(tuple, matches[KEYS].to_numpy()))
    if task == "regex_audit" and matched_ids != ids:
        raise ValueError("Every regex-audit PR must have at least one highlighted match row.")
    if task == "tradeoff_audit":
        dimensions = matches.groupby(KEYS)["dimension"].agg(set)
        invalid = [key for key in ids if key not in dimensions.index
                   or "D3" not in dimensions.loc[key]
                   or not dimensions.loc[key].intersection({"D1", "D2", "D5"})]
        if invalid:
            raise ValueError(f"Tradeoff PRs lack highlighted gain or memory matches: {sorted(invalid)}")
    grouped["highlighted_matches"] = _highlight_matches(matches, task)
    for name, frame in grouped.items():
        result = result.merge(frame.rename(columns={"text": name}), on=KEYS, how="left", validate="one_to_one")
        result[name] = result[name].fillna("")
    if result[["repo_full_name", "html_url", "title"]].isna().any().any():
        raise ValueError("Pull-request evidence is missing required identity or title fields.")
    result["evidence_complete"] = True
    return result


def prompt_for(row: dict[str, Any], task: str) -> str:
    if not bool(row.get("evidence_complete", False)):
        raise ValueError("RQ3 cannot classify a PR with incomplete evidence.")
    sections = {
        "TITLE": _text(row.get("title")),
        "DESCRIPTION": _truncate(_text(row.get("body")), TEXT_LIMIT),
        "ISSUE COMMENTS": _truncate(_text(row.get("issue_comments")), TEXT_LIMIT),
        "REVIEW COMMENTS": _truncate(_text(row.get("review_comments")), TEXT_LIMIT),
        "REVIEWS": _truncate(_text(row.get("reviews")), TEXT_LIMIT),
        "COMMIT MESSAGES": _truncate(_text(row.get("commit_messages")), TEXT_LIMIT),
        "CI WORKFLOWS AND CHECKS": _truncate(_text(row.get("ci")), TEXT_LIMIT),
        "CODE DIFF": _truncate(_text(row.get("code_diff")), DIFF_LIMIT),
        "REGEX MATCHES TO AUDIT": _truncate(_text(row.get("highlighted_matches")), TEXT_LIMIT),
    }
    context = "\n\n".join(f"{name}:\n{value or '[NONE]'}" for name, value in sections.items())
    if task == "regex_audit":
        instructions = """Classify this PR as exactly true_positive or false_positive. The PR is the classification unit.

A true positive requires at least one regex-detected metric genuinely supported by an explicit quantitative performance claim in the supplied PR artifacts. Incidental constants, configuration or code values, boilerplate, prospective/expected claims, and unrelated numbers do not qualify. A false positive has no qualifying detected metric. Provide short verbatim quotes and their sources: quote the supporting claim for true_positive, or the strongest non-qualifying regex match for false_positive. Keep rationale concise."""
    elif task == "tradeoff_audit":
        instructions = """Classify this PR as exactly tradeoff or joint_improvement; there is no eligibility or unknown class. Use the highlighted gain dimensions (D1 time, D2 throughput, D5 I/O) and memory dimension D3, checked against the full corpus.

Tradeoff means an improvement in time/throughput/I/O accompanied by increased or worse memory. Joint improvement means the performance gain is accompanied by reduced or improved memory. Set gain_direction=improved and memory_direction consistently, provide short verbatim quotes and their sources, and keep rationale concise."""
    else:
        raise ValueError(f"Unknown task: {task}")
    return f"{instructions}\n\n{context}\n"


def input_row_sha256(row: dict[str, Any], task: str) -> str:
    return sha256_json({"task": task, "prompt": prompt_for(row, task)})


def study_contract(task: str, sample_sha256: str, matches_sha256: str, evidence_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "study": f"rq3_{task}",
        "classification_unit": "pull_request",
        "prompt_version": PROMPT_VERSION,
        "system_instruction_sha256": hashlib.sha256(SYSTEM_INSTRUCTION.encode()).hexdigest(),
        "semantic_schema_sha256": sha256_json(semantic_schema(task)),
        "sample_sha256": sample_sha256,
        "metric_matches_sha256": matches_sha256,
        "evidence_sha256": evidence_hashes,
        "input_construction": {
            "identity_columns": KEYS,
            "independent_text_character_limit": TEXT_LIMIT,
            "diff_character_limit": DIFF_LIMIT,
            "truncation_marker": TRUNCATION_MARKER,
            "stable_ordering": True,
        },
    }


def prepare_run(
    task: str,
    sample_path: Path,
    matches_path: Path,
    evidence_dir: Path,
    output_dir: Path,
    provider_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; refusing to overwrite a run.")
    frame = build_input(sample_path, matches_path, evidence_dir, task).sort_values(KEYS, kind="mergesort")
    evidence_hashes = {name: sha256_file(evidence_dir / name) for name in EVIDENCE_FILES}
    contract = study_contract(task, sha256_file(sample_path), sha256_file(matches_path), evidence_hashes)
    contract_hash = sha256_json(contract)
    provider_hash = sha256_json(provider_config)
    rows = []
    for row in frame.to_dict("records"):
        prompt = prompt_for(row, task)
        rows.append({
            "repo_id": int(row["repo_id"]), "number": int(row["number"]),
            "custom_id": f"{int(row['repo_id'])}:{int(row['number'])}",
            "repo_full_name": row["repo_full_name"], "html_url": row["html_url"],
            "sample_arm": row["sample_arm"], "task": task, "prompt_version": PROMPT_VERSION,
            "evidence_complete": True, "prompt_chars": len(prompt),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_row_sha256": input_row_sha256(row, task),
            "study_contract_sha256": contract_hash,
            "provider_config_sha256": provider_hash,
        })
    manifest = pd.DataFrame(rows)
    metadata = {
        "task": task, "requests": len(manifest), "prompt_version": PROMPT_VERSION,
        "sample_sha256": sha256_file(sample_path), "metric_matches_sha256": sha256_file(matches_path),
        "evidence_sha256": evidence_hashes, "study_contract": contract,
        "study_contract_sha256": contract_hash, "provider_config": provider_config,
        "provider_config_sha256": provider_hash,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "batch_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    metadata["manifest_sha256"] = sha256_file(manifest_path)
    atomic_write_json(output_dir / "prepare_metadata.json", metadata)
    return frame, manifest, metadata


def validate_prepared_inputs(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    metadata: dict[str, Any],
    task: str,
    sample_path: Path,
    matches_path: Path,
    evidence_dir: Path,
) -> None:
    """Ensure dynamically rendered requests still match the prepared manifest."""
    evidence_hashes = {name: sha256_file(evidence_dir / name) for name in EVIDENCE_FILES}
    if sha256_file(sample_path) != metadata["sample_sha256"]:
        raise ValueError("Prepared sample changed.")
    if sha256_file(matches_path) != metadata["metric_matches_sha256"]:
        raise ValueError("Prepared metric matches changed.")
    if evidence_hashes != metadata["evidence_sha256"]:
        raise ValueError("Prepared evidence changed.")
    contract = study_contract(task, metadata["sample_sha256"], metadata["metric_matches_sha256"], evidence_hashes)
    if sha256_json(contract) != metadata["study_contract_sha256"]:
        raise ValueError("Prepared study contract changed.")

    rows = frame.assign(
        custom_id=lambda value: value["repo_id"].astype(str) + ":" + value["number"].astype(str)
    ).set_index("custom_id")
    if set(rows.index) != set(manifest["custom_id"]):
        raise ValueError("Prepared manifest identities changed.")
    for item in manifest.to_dict("records"):
        row = rows.loc[item["custom_id"]].to_dict()
        prompt = prompt_for(row, task)
        if hashlib.sha256(prompt.encode()).hexdigest() != item["prompt_sha256"]:
            raise ValueError(f"Prepared prompt changed for {item['custom_id']}.")
        if input_row_sha256(row, task) != item["input_row_sha256"]:
            raise ValueError(f"Prepared input row changed for {item['custom_id']}.")


def parse_sample_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic RQ3 LLM validation samples.")
    parser.add_argument("--pr-level", type=Path, default=DEFAULT_PR_LEVEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_sample_args()
    print({key: str(value) for key, value in prepare_samples(args.pr_level, args.output_dir).items()})
