from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from schema import atomic_write_text


METHOD_NAME = "method_a_binary_agentic_v1"
METHOD_VERSION = 1
IDENTITY_COLUMNS = ("repo_id", "number", "html_url")
INPUT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "title",
    "body",
    "user",
    "user_type",
    "created_at",
    "aidev_attribution_label",
    "aidev_attribution_rule",
    "aidev_attribution_head_ref",
    "aidev_attribution_author_login",
    "aidev_attribution_author_type",
)
DECISION_COLUMNS = (
    "method_a_label",
    "method_a_reason",
    "method_a_rules",
    "method_a_method",
)
POSITIVE_RULE_IDS = (
    "devin_author",
    "copilot_author_branch",
    "codex_branch",
    "cursor_branch",
    "coding_agent_task_url",
    "explicit_agent_generation",
    "claude_coauthor",
    "aidev_historical_agent",
    "aidev_agentic_signal",
    "generic_bot_author",
)
AIDEV_RULE_MAP = {
    "devin_author": "devin_author",
    "github_copilot_head": "copilot_author_branch",
    "openai_codex_head": "codex_branch",
    "cursor_head": "cursor_branch",
    "claude_code_coauthor": "claude_coauthor",
    "aidev_pinned_historical": "aidev_historical_agent",
}
RULE_DESCRIPTIONS = {
    "devin_author": "Exact devin-ai-integration[bot] author or the corresponding AIDev rule.",
    "copilot_author_branch": "Exact Copilot author combined with a copilot/ branch.",
    "codex_branch": "A codex/ branch within the valid Codex activity period.",
    "cursor_branch": "A cursor/ branch within the valid Cursor activity period.",
    "coding_agent_task_url": "An official coding-agent task or session URL.",
    "explicit_agent_generation": "An explicit implemented/generated/created/written/made/built by or with coding-agent statement.",
    "claude_coauthor": "A Co-Authored-By: Claude marker.",
    "aidev_historical_agent": "A pinned historical AIDev agentic attribution.",
    "aidev_agentic_signal": "An AIDev agentic attribution not represented by another Method A rule ID.",
    "generic_bot_author": "GitHub author type Bot, including conventional automation.",
}

TASK_URL_PATTERN = re.compile(
    r"https?://(?:"
    r"chatgpt\.com/codex/(?:cloud/)?tasks?/[^\s)\]>]+|"
    r"app\.devin\.ai/sessions?/[^\s)\]>]+|"
    r"jules\.google\.com/(?:tasks?|sessions?)/[^\s)\]>]+|"
    r"cursor\.com/(?:agents?|tasks?)/[^\s)\]>]+|"
    r"claude\.ai/code/(?:tasks?|sessions?)/[^\s)\]>]+"
    r")",
    re.IGNORECASE,
)
GENERATION_PATTERN = re.compile(
    r"\b(?:implemented|generated|created|written|made|built)\s+"
    r"(?:entirely\s+)?(?:by|with)\s+\[?"
    r"(?:openai\s+codex|codex|devin(?:\s+ai)?|github\s+copilot(?:\s+coding\s+agent)?|"
    r"copilot\s+coding\s+agent|cursor(?:\s+agent)?|claude\s+code|google\s+jules|jules)\]?\b",
    re.IGNORECASE,
)
CLAUDE_COAUTHOR_PATTERN = re.compile(
    r"(?:^|\n)\s*co-authored-by:\s*claude(?:\s+code)?(?:\s*<[^>\n]+>)?",
    re.IGNORECASE,
)
REVIEW_ONLY_PATTERN = re.compile(
    r"(?:reviewed\s+by\s+[^\n]*(?:cursor|devin|copilot|claude|codex)|"
    r"open\s+in\s+devin\s+review|cursor\s+bugbot|"
    r"(?:summary|description)\s+generated\s+by\s+[^\n]*(?:ai|agent|copilot|claude|codex))",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[column].fillna("").astype("string")


def _normalized(frame: pd.DataFrame, *columns: str) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype="string")
    for column in columns:
        values = _text(frame, column).str.strip().str.lower()
        result = result.mask(result.eq("") & values.ne(""), values)
    return result


def _contains_rule(rules: pd.Series, rule_id: str) -> pd.Series:
    return rules.str.contains(rf"(?:^|\|){re.escape(rule_id)}(?:\||$)", regex=True, na=False)


def _on_or_after(timestamps: pd.Series, start_date: str) -> pd.Series:
    return timestamps.notna() & timestamps.ge(pd.Timestamp(start_date, tz="UTC"))


def _join_rule_ids(rule_masks: Iterable[tuple[str, pd.Series]], index: pd.Index) -> pd.Series:
    result = pd.Series("", index=index, dtype="string")
    for rule_id, mask in rule_masks:
        first = mask & result.eq("")
        additional = mask & result.ne("")
        result = result.mask(first, rule_id)
        result = result.mask(additional, result + "|" + rule_id)
    return result


def classify_frame(frame: pd.DataFrame) -> pd.DataFrame:
    title = _text(frame, "title")
    body = _text(frame, "body")
    combined_text = title + "\n" + body
    created_at = pd.to_datetime(_text(frame, "created_at"), utc=True, errors="coerce")
    aidev_label = _normalized(frame, "aidev_attribution_label")
    aidev_rules = _text(frame, "aidev_attribution_rule").str.strip().str.lower()
    author_login = _normalized(frame, "user", "aidev_attribution_author_login")
    author_type = _normalized(frame, "user_type", "aidev_attribution_author_type")
    head_ref = _normalized(frame, "aidev_attribution_head_ref")

    mapped_aidev = pd.Series(False, index=frame.index)
    masks: dict[str, pd.Series] = {}
    for aidev_rule, method_rule in AIDEV_RULE_MAP.items():
        matched = aidev_label.eq("agentic") & _contains_rule(aidev_rules, aidev_rule)
        masks[method_rule] = masks.get(method_rule, pd.Series(False, index=frame.index)) | matched
        mapped_aidev |= matched

    masks["devin_author"] |= author_login.eq("devin-ai-integration[bot]") & _on_or_after(
        created_at, "2024-12-24"
    )
    masks["copilot_author_branch"] |= (
        author_login.eq("copilot")
        & head_ref.str.startswith("copilot/", na=False)
        & _on_or_after(created_at, "2025-01-01")
    )
    masks["codex_branch"] |= head_ref.str.startswith("codex/", na=False) & _on_or_after(
        created_at, "2025-05-16"
    )
    masks["cursor_branch"] |= head_ref.str.startswith("cursor/", na=False) & _on_or_after(
        created_at, "2025-01-01"
    )
    masks["coding_agent_task_url"] = combined_text.str.contains(TASK_URL_PATTERN, na=False)
    masks["explicit_agent_generation"] = combined_text.str.contains(GENERATION_PATTERN, na=False)
    masks["claude_coauthor"] |= (
        combined_text.str.contains(CLAUDE_COAUTHOR_PATTERN, na=False)
        & _on_or_after(created_at, "2025-02-24")
    )
    masks["aidev_agentic_signal"] = aidev_label.eq("agentic") & ~mapped_aidev
    masks["generic_bot_author"] = author_type.eq("bot")

    ordered_masks = [(rule_id, masks[rule_id].fillna(False)) for rule_id in POSITIVE_RULE_IDS]
    rules = _join_rule_ids(ordered_masks, frame.index)
    agentic = rules.ne("")
    review_only = combined_text.str.contains(REVIEW_ONLY_PATTERN, na=False)

    result = pd.DataFrame(index=frame.index)
    result["method_a_label"] = agentic.map({True: "agentic", False: "non_agentic"}).astype("string")
    result["method_a_rules"] = rules
    result["method_a_reason"] = rules.mask(~agentic & review_only, "review_only_signal")
    result["method_a_reason"] = result["method_a_reason"].mask(
        ~agentic & ~review_only, "no_agent_signal"
    )
    result["method_a_method"] = METHOD_NAME
    return result


def _output_schema(input_schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [input_schema.field(column) for column in IDENTITY_COLUMNS]
        + [pa.field(column, pa.string()) for column in DECISION_COLUMNS]
    )


def _output_table(frame: pd.DataFrame, decisions: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    values: dict[str, pa.Array] = {}
    for column in IDENTITY_COLUMNS:
        values[column] = pa.array(frame[column], type=schema.field(column).type)
    for column in DECISION_COLUMNS:
        values[column] = pa.array(decisions[column], type=pa.string())
    return pa.Table.from_pydict(values, schema=schema)


def classify_parquet(
    input_path: Path,
    output_dir: Path,
    *,
    batch_size: int = 50_000,
    overwrite: bool = False,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    source = pq.ParquetFile(input_path)
    missing = [column for column in INPUT_COLUMNS if column not in source.schema_arrow.names]
    if missing:
        raise ValueError("Method A input is missing columns: " + ", ".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "decisions.parquet"
    summary_path = output_dir / "summary.json"
    if not overwrite and (output_path.exists() or summary_path.exists()):
        raise FileExistsError("Method A output already exists; pass --overwrite to replace it.")

    schema = _output_schema(source.schema_arrow)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir, prefix=".decisions.", suffix=".parquet.tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    label_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    matched_rule_count: Counter[int] = Counter()
    generic_bot_only = 0
    rows = 0

    try:
        writer = pq.ParquetWriter(temporary_path, schema, compression="zstd")
        try:
            for batch in source.iter_batches(batch_size=batch_size, columns=list(INPUT_COLUMNS)):
                frame = pa.Table.from_batches([batch]).to_pandas()
                decisions = classify_frame(frame)
                writer.write_table(_output_table(frame, decisions, schema))
                rows += len(frame)
                label_counts.update(decisions["method_a_label"].tolist())
                reason_counts.update(decisions["method_a_reason"].tolist())
                for value in decisions.loc[
                    decisions["method_a_label"].eq("agentic"), "method_a_rules"
                ]:
                    rule_ids = value.split("|")
                    rule_counts.update(rule_ids)
                    matched_rule_count[len(rule_ids)] += 1
                    if rule_ids == ["generic_bot_author"]:
                        generic_bot_only += 1
        finally:
            writer.close()
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    output_rows = pq.ParquetFile(output_path).metadata.num_rows
    if rows != source.metadata.num_rows or output_rows != rows:
        output_path.unlink(missing_ok=True)
        raise ValueError("Method A output row count does not match the input.")

    summary: dict[str, object] = {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "input": {
            "path": str(input_path.resolve()),
            "rows": int(source.metadata.num_rows),
            "sha256": sha256_file(input_path),
        },
        "output": {
            "path": str(output_path.resolve()),
            "rows": rows,
            "sha256": sha256_file(output_path),
        },
        "counts": {
            "labels": dict(sorted(label_counts.items())),
            "reasons": dict(sorted(reason_counts.items())),
            "rules": {rule_id: rule_counts.get(rule_id, 0) for rule_id in POSITIVE_RULE_IDS},
            "matched_positive_rule_count": {
                str(count): matches for count, matches in sorted(matched_rule_count.items())
            },
        },
        "sensitivity": {
            "generic_bot_only_agentic": generic_bot_only,
            "agentic_without_generic_bot_rule": label_counts.get("agentic", 0) - generic_bot_only,
            "note": "The official Method A label includes every GitHub Bot author as agentic.",
        },
        "rules": [
            {"rule_id": rule_id, "description": RULE_DESCRIPTIONS[rule_id]}
            for rule_id in POSITIVE_RULE_IDS
        ],
        "unsupported_input_evidence": [
            "GitHub App ID/slug is unavailable in the current input. Bot-authored app PRs are covered by generic_bot_author."
        ],
        "code_sha256": sha256_file(Path(__file__)),
    }
    atomic_write_text(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify PRs as agentic or non-agentic using binary Method A."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = classify_parquet(
        args.input,
        args.output_dir,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
