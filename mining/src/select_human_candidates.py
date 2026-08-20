"""Select strict human candidates from AIDev-negative performance PRs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from schema import atomic_write_text, write_parquet


METHOD_NAME = "aidev_performance_human_filter_v1"
METHOD_VERSION = 1
IDENTITY_COLUMNS = ("repo_id", "number")
ATTRIBUTION_COLUMNS = (
    *IDENTITY_COLUMNS,
    "aidev_attribution_label",
    "aidev_attribution_agent",
    "aidev_attribution_rule",
    "aidev_attribution_evidence",
    "aidev_attribution_status",
    "aidev_attribution_method",
)
RULE_IDS = (
    "known_agent_login",
    "nonhuman_bot_author",
    "suspicious_user_login",
    "coding_agent_task_url",
    "ai_coauthor_trailer",
    "explicit_ai_authorship",
    "ai_review_signal",
    "ai_generated_metadata",
    "insufficient_evidence",
)
RULE_DESCRIPTIONS = {
    "known_agent_login": "Exact known coding-agent account observed in the performance cohort.",
    "nonhuman_bot_author": "GitHub reports the PR author type as Bot.",
    "suspicious_user_login": "A User login contains a delimited automation or coding-agent token.",
    "coding_agent_task_url": "The PR links to a known coding-agent task, session, or trace URL.",
    "ai_coauthor_trailer": "The PR contains a line-anchored coauthor trailer for a known AI agent.",
    "explicit_ai_authorship": "The PR explicitly attributes authorship or generation to AI.",
    "ai_review_signal": "The PR contains a structured AI review marker.",
    "ai_generated_metadata": "The PR contains a structured AI-generated summary or description marker.",
    "insufficient_evidence": "Required identity, author, date, or complete AIDev-negative evidence is missing.",
}

KNOWN_AGENT_LOGINS = frozenset(
    {
        "claude",
        "claude[bot]",
        "claude-nightshift[bot]",
        "codeflash-ai[bot]",
        "codegen-sh[bot]",
        "devin-ai-integration[bot]",
        "google-labs-jules[bot]",
        "kilo-code-bot[bot]",
        "chrisrackauckas-claude",
        "ravwojdyla-agent",
    }
)
SUSPICIOUS_USER_LOGIN_PATTERN = re.compile(
    r"(?:"
    r"\[bot\]$|"
    r"(?:^|[-_])(?:ai|agent|bot|robot|anthropic|claude|copilot|codex|cursor|devin|"
    r"gemini|jules|openai)(?:[-_]|\[bot\]|$)|"
    r"(?:bot|robot)$"
    r")",
    re.IGNORECASE,
)
TASK_URL_PATTERN = re.compile(
    r"https?://(?:"
    r"chatgpt\.com/codex/(?:cloud/)?tasks?/[^\s<>)\]]+|"
    r"app\.devin\.ai/sessions?/[^\s<>)\]]+|"
    r"jules\.google\.com/(?:tasks?|sessions?)/[^\s<>)\]]+|"
    r"claude\.ai/code/(?:session_[A-Za-z0-9]+|(?:tasks?|sessions?)/[^\s<>)\]]+)|"
    r"(?:www\.)?cursor\.com/(?:"
    r"(?:agents?|tasks?)/[^\s<>)\]]+|"
    r"agents\?id=bc-[0-9a-f-]+|"
    r"background-agent\?bcId=bc-[0-9a-f-]+"
    r")|"
    r"codegen\.com/agent/trace/[0-9]+|"
    r"hub\.continue\.dev/inbox(?:/pr)?(?:[/?][^\s<>)\]]*)?"
    r")",
    re.IGNORECASE,
)
AI_COAUTHOR_PATTERN = re.compile(
    r"(?:^|\n)\s*co-authored-by:\s*(?:"
    r"claude(?:\s+code)?(?:\s*<[^>\r\n]+>)?|"
    r"copilot\s*<223556219\+copilot@users\.noreply\.github\.com>|"
    r"cursor\s*<cursoragent@cursor\.com>|"
    r"amp\s*<amp@ampcode\.com>|"
    r"modular-kernel-agent\s*<modular@speedtrain\.co>|"
    r"gemini\s+2\.5\s+pro\b[^\r\n]*"
    r")\s*(?:\n|$)",
    re.IGNORECASE,
)
AGENT_NAME_PATTERN = (
    r"(?:openai\s+codex|codex|devin(?:\s+ai)?|github\s+copilot(?:\s+coding\s+agent)?|"
    r"copilot(?:\s+coding\s+agent)?|cursor(?:\s+(?:background\s+)?agent)?|"
    r"claude(?:\s+code)?|google\s+jules|jules|gemini|windsurf|cline|aider|amp|"
    r"codegen|terry|terragon|(?:an?\s+)?(?:generative\s+)?ai(?:\s+coding\s+agent)?|"
    r"vs\s+perf\s+rel\s+ai\s+agent)"
)
EXPLICIT_AUTHORSHIP_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in (
        rf"(?<![A-Za-z0-9])(?:implemented|generated|created|written|authored|drafted|built|made)\s+"
        rf"(?:entirely\s+)?(?:by|with|using)\s+\[?{AGENT_NAME_PATTERN}\]?\b",
        rf"\b(?:this\s+(?:pull\s+request|pr)|the\s+(?:implementation|code(?:\s+changes)?))\s+"
        rf"(?:was|were|is|are)?\s*(?:implemented|generated|created|written|authored|drafted|"
        rf"vibe-coded)\s+(?:by|with|using|with\s+(?:the\s+)?(?:assistance|help)\s+"
        rf"(?:of|from))\s+\[?{AGENT_NAME_PATTERN}\]?\b",
        rf"\bai\s+disclosure:\s*(?:drafted|generated|implemented|written|created)\s+with\s+"
        rf"\[?{AGENT_NAME_PATTERN}\]?\b",
        rf"\b(?:generated|made)\s*[- ]by:\s*\[?{AGENT_NAME_PATTERN}\]?\b",
        rf"\bmade\s*[- ]with:\s*\[?{AGENT_NAME_PATTERN}\]?\b",
        rf"\bactual\s+implementation\s+is\s+vibe-coded\s+with\s+{AGENT_NAME_PATTERN}\b",
        rf"\bwith\s+the\s+help\s+of\s+ai-agents?:\s*{AGENT_NAME_PATTERN}\b",
        r"_by\s+openai\s+codex_",
        r"generated\s+by\s+\[terry\]\(https?://(?:www\.)?terragonlabs\.com\)",
        r"\bai-generated\s+pull\s+request\b",
        r"(?:^|\n)\s*-\s*\[[xX]\]\s*[^\r\n]*\bai-assisted\s*:\s*[^\r\n]+",
    )
)
AI_REVIEW_PATTERN = re.compile(
    r"(?:"
    r"\bcoderabbit(?:\.?ai)?\b|"
    r"reviewed\s+by\s+\[?cursor\s+bugbot\b|"
    r"written\s+by\s+\[?cursor\s+bugbot\b|"
    r"<!--\s*/?bugbot_status\s*-->|"
    r"<!--\s*devin-review-badge-(?:begin|end)\s*-->|"
    r"https?://app\.devin\.ai/review/|"
    r"\bgreptile\b|"
    r"<!--\s*/?greptile_(?:comment|failed_comments|other_comments_section)\s*-->|"
    r"(?:<h3>|##\s*)greptile\s+summary|"
    r"<!--\s*(?:start|end)\s+pr-codex\s*-->|"
    r"##\s*pr-codex\s+overview|"
    r"<!--\s*(?:qodo|pr[-_ ]?agent)[^>]*-->|"
    r"\bqodo\s+(?:merge|review)\b|"
    r"qodo-merge-docs\.qodo\.ai|"
    r"\bpr_agent:(?:summary|walkthrough)\b|"
    r"\bsharing-pr-agent-artifacts\b"
    r")",
    re.IGNORECASE,
)
AI_METADATA_PATTERN = re.compile(
    r"(?:"
    r"<!--\s*this\s+is\s+an\s+auto-generated\s+comment:\s*"
    r"release\s+notes\s+by\s+coderabbit\.ai\s*-->|"
    r"##\s*summary\s+by\s+coderabbit|"
    r"<!--\s*korbit\s+ai\s+pr\s+description\s+(?:start|end)\s*-->|"
    r"##\s*description\s+by\s+korbit\s+ai|"
    r"<!--\s*/?cursor_summary\s*-->|"
    r"<!--\s*ellipsis_hidden\s*-->|"
    r"<!--\s*/?cubic[^>]*-->|"
    r"auto-generated\s+description\s+by\s+cubic|"
    r"https?://(?:www\.)?cubic\.dev/(?:pr|buttons)/|"
    r"(?:<h3>|##\s*)greptile\s+summary|"
    r"(?:<details>\s*)?<summary>\s*greptile\s+summary\s*</summary>|"
    r"##\s*pr-codex\s+overview|"
    r"\*this\s+summary\s+was\s+automatically\s+generated\s+by\s+@propel-code-bot\*|"
    r"this\s+pr\s+(?:body|description)\s+was\s+generated\s+by\s+"
    r"an?\s+ai\s+coding\s+assistant|"
    r"this\s+pr\s+description\s+was\s+generated\s+with\s+"
    r"(?:the\s+)?assistance\s+(?:of|from)\s+github\s+copilot"
    r")",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_unique_identities(frame: pd.DataFrame, label: str) -> None:
    missing = [column for column in IDENTITY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing identity columns: {', '.join(missing)}")
    if frame[list(IDENTITY_COLUMNS)].isna().any().any():
        raise ValueError(f"{label} contains null immutable identities.")
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"{label} contains duplicate immutable identities.")


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[column].fillna("").astype("string")


def _matches_explicit_authorship(value: str) -> bool:
    return any(pattern.search(value) is not None for pattern in EXPLICIT_AUTHORSHIP_PATTERNS)


def _join_rule_ids(rule_masks: Iterable[tuple[str, pd.Series]], index: pd.Index) -> pd.Series:
    result = pd.Series("", index=index, dtype="string")
    for rule_id, mask in rule_masks:
        first = mask & result.eq("")
        additional = mask & result.ne("")
        result = result.mask(first, rule_id)
        result = result.mask(additional, result + "|" + rule_id)
    return result


def classify_human_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    title = _text(frame, "title")
    body = _text(frame, "body")
    combined_text = title + "\n" + body
    login = _text(frame, "user").str.strip().str.lower()
    user_type = _text(frame, "user_type").str.strip().str.lower()
    created_at = pd.to_datetime(_text(frame, "created_at"), utc=True, errors="coerce")
    attribution_label = _text(frame, "aidev_attribution_label").str.strip().str.lower()
    attribution_status = _text(frame, "aidev_attribution_status").str.strip().str.lower()

    masks = {
        "known_agent_login": login.isin(KNOWN_AGENT_LOGINS),
        "nonhuman_bot_author": user_type.eq("bot"),
        "suspicious_user_login": (
            user_type.eq("user") & login.str.contains(SUSPICIOUS_USER_LOGIN_PATTERN, na=False)
        ),
        "coding_agent_task_url": combined_text.str.contains(TASK_URL_PATTERN, na=False),
        "ai_coauthor_trailer": combined_text.str.contains(AI_COAUTHOR_PATTERN, na=False),
        "explicit_ai_authorship": combined_text.map(_matches_explicit_authorship),
        "ai_review_signal": combined_text.str.contains(AI_REVIEW_PATTERN, na=False),
        "ai_generated_metadata": combined_text.str.contains(AI_METADATA_PATTERN, na=False),
        "insufficient_evidence": (
            login.eq("")
            | ~user_type.isin({"user", "bot"})
            | created_at.isna()
            | ~attribution_label.eq("human_candidate")
            | ~attribution_status.eq("complete_no_match")
        ),
    }
    ordered_masks = [(rule_id, masks[rule_id].fillna(False)) for rule_id in RULE_IDS]
    reasons = _join_rule_ids(ordered_masks, frame.index)
    excluded = reasons.ne("")

    result = pd.DataFrame(index=frame.index)
    result["human_filter_label"] = excluded.map(
        {True: "excluded_from_human_candidates", False: "human_candidate"}
    ).astype("string")
    result["human_filter_reasons"] = reasons
    result["human_filter_primary_reason"] = reasons.str.split("|").str[0].mask(
        ~excluded, "no_observable_agent_signal"
    )
    result["human_filter_method"] = METHOD_NAME
    return result


def performance_rows(path: Path) -> pd.DataFrame:
    source = pq.ParquetFile(path)
    required = {*IDENTITY_COLUMNS, "aidev_task_type"}
    missing = sorted(required - set(source.schema_arrow.names))
    if missing:
        raise ValueError("Task-type input is missing columns: " + ", ".join(missing))
    frame = pd.read_parquet(path, filters=[("aidev_task_type", "==", "perf")])
    require_unique_identities(frame, "Performance task-type input")
    return frame


def matching_attribution_rows(
    path: Path, identities: set[tuple[int, int]], batch_size: int
) -> pd.DataFrame:
    source = pq.ParquetFile(path)
    missing = [column for column in ATTRIBUTION_COLUMNS if column not in source.schema_arrow.names]
    if missing:
        raise ValueError("AIDev attribution input is missing columns: " + ", ".join(missing))
    matches: list[pd.DataFrame] = []
    for batch in source.iter_batches(batch_size=batch_size, columns=list(ATTRIBUTION_COLUMNS)):
        frame = pa.Table.from_batches([batch]).to_pandas()
        keys = pd.Series(
            list(zip(frame["repo_id"], frame["number"])), index=frame.index, dtype="object"
        )
        selected = frame.loc[keys.isin(identities)]
        if not selected.empty:
            matches.append(selected)
    if not matches:
        return pd.DataFrame(columns=ATTRIBUTION_COLUMNS)
    result = pd.concat(matches, ignore_index=True)
    require_unique_identities(result, "AIDev attribution matches")
    return result


def _write_outputs(frame: pd.DataFrame, output_dir: Path) -> dict[str, dict[str, object]]:
    outputs = {
        "decisions": (output_dir / "decisions.parquet", frame),
        "human_candidates": (
            output_dir / "human_candidates.parquet",
            frame.loc[frame["human_filter_label"].eq("human_candidate")].copy(),
        ),
        "excluded": (
            output_dir / "excluded_from_human_candidates.parquet",
            frame.loc[
                frame["human_filter_label"].eq("excluded_from_human_candidates")
            ].copy(),
        ),
    }
    metadata: dict[str, dict[str, object]] = {}
    for name, (path, values) in outputs.items():
        write_parquet(values, path)
        metadata[name] = {
            "path": str(path.resolve()),
            "rows": len(values),
            "sha256": sha256_file(path),
        }
    return metadata


def build_human_candidate_dataset(
    task_type_path: Path,
    attribution_path: Path,
    output_dir: Path,
    *,
    batch_size: int = 50_000,
    overwrite: bool = False,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for path in (task_type_path, attribution_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_paths = (
        output_dir / "decisions.parquet",
        output_dir / "human_candidates.parquet",
        output_dir / "excluded_from_human_candidates.parquet",
        output_dir / "summary.json",
    )
    if not overwrite and any(path.exists() for path in output_paths):
        raise FileExistsError("Human-candidate outputs already exist; pass --overwrite to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)

    performance = performance_rows(task_type_path)
    identities = set(
        (int(repo_id), int(number))
        for repo_id, number in performance[list(IDENTITY_COLUMNS)].itertuples(index=False, name=None)
    )
    attribution = matching_attribution_rows(attribution_path, identities, batch_size)
    if len(attribution) != len(performance):
        raise ValueError(
            "Performance and AIDev attribution inputs have nonmatching immutable identities."
        )
    added_columns = [column for column in ATTRIBUTION_COLUMNS if column not in IDENTITY_COLUMNS]
    population = performance.merge(
        attribution[[*IDENTITY_COLUMNS, *added_columns]],
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    candidates = population.loc[
        population["aidev_attribution_label"].eq("human_candidate")
    ].copy()
    decisions = classify_human_candidates(candidates)
    result = pd.concat([candidates.reset_index(drop=True), decisions.reset_index(drop=True)], axis=1)
    require_unique_identities(result, "Human-filter decisions")

    outputs = _write_outputs(result, output_dir)
    human_rows = outputs["human_candidates"]["rows"]
    excluded_rows = outputs["excluded"]["rows"]
    if human_rows + excluded_rows != len(result):
        raise ValueError("Human-candidate output partitions do not reconcile to the input cohort.")

    reason_counts: Counter[str] = Counter()
    combination_counts: Counter[str] = Counter(result["human_filter_reasons"].replace("", "none"))
    matched_rule_count: Counter[int] = Counter()
    for value in result.loc[
        result["human_filter_label"].eq("excluded_from_human_candidates"),
        "human_filter_reasons",
    ]:
        reasons = value.split("|")
        reason_counts.update(reasons)
        matched_rule_count[len(reasons)] += 1

    summary: dict[str, object] = {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "inputs": {
            "task_type": {
                "path": str(task_type_path.resolve()),
                "sha256": sha256_file(task_type_path),
            },
            "aidev_attribution": {
                "path": str(attribution_path.resolve()),
                "sha256": sha256_file(attribution_path),
            },
        },
        "counts": {
            "performance_prs": len(performance),
            "aidev_agentic_performance_prs": int(
                population["aidev_attribution_label"].eq("agentic").sum()
            ),
            "aidev_human_candidate_performance_prs": len(result),
            "human_candidates": int(human_rows),
            "excluded_from_human_candidates": int(excluded_rows),
            "rules": {rule_id: reason_counts.get(rule_id, 0) for rule_id in RULE_IDS},
            "reason_combinations": dict(sorted(combination_counts.items())),
            "matched_rule_count": {
                str(count): matches for count, matches in sorted(matched_rule_count.items())
            },
        },
        "rules": [
            {"rule_id": rule_id, "description": RULE_DESCRIPTIONS[rule_id]}
            for rule_id in RULE_IDS
        ],
        "outputs": outputs,
        "code_sha256": sha256_file(Path(__file__)),
        "notes": [
            "The retained label is human_candidate, not confirmed human authorship.",
            "AI review and generated PR metadata are exclusion signals under the selected strict policy.",
            "Delimited automation or agent tokens in User logins are exclusion signals under the selected strict policy.",
        ],
    }
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter AIDev performance human candidates using strict observable-agent signals."
    )
    parser.add_argument("--task-type", required=True, type=Path)
    parser.add_argument("--attribution", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_human_candidate_dataset(
        args.task_type,
        args.attribution,
        args.output_dir,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
