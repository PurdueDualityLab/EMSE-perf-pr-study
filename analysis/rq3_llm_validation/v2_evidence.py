"""Lossless activation export for the frozen RQ3 validation samples."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from analysis.rq3_llm_validation import experiment as v1

EXTRACTOR_DIR = Path(__file__).resolve().parents[1] / "rq3_pattern_and_validation"
if str(EXTRACTOR_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACTOR_DIR))
import metric_patterns as mp
from extract_metrics import extract_dimensions
from run_current import assemble_corpus

SPECS = (
    ("issue_comments", "body", "issue_comments", "comment_id"),
    ("review_comments", "body", "review_comments", "comment_id"),
    ("commits", "message", "commit_messages", "sha"),
    ("workflow_runs", "name", "ci_metadata", "workflow_run_id"),
    ("pull_request_files", "patch", "code_diff", "file_index"),
)


def normalize_with_offsets(raw: str) -> tuple[str, list[tuple[int, int]]]:
    """Reproduce normalization while tracing each output character to raw text.

    Replacement characters map to the whole replaced span. Thus raw locators
    conservatively enclose folded units, removed markup and CJK spacing.
    """
    text = raw
    offsets = [(i, i + 1) for i in range(len(raw))]

    def substitute(pattern, replacement):
        nonlocal text, offsets
        parts, mapped, last = [], [], 0
        for match in pattern.finditer(text):
            start, end = match.span()
            parts.append(text[last:start])
            mapped.extend(offsets[last:start])
            value = match.expand(replacement)
            if start < end:
                span = (offsets[start][0], offsets[end - 1][1])
            else:
                point = offsets[start][0] if start < len(offsets) else len(raw)
                span = (point, point)
            parts.append(value)
            mapped.extend([span] * len(value))
            last = end
        parts.append(text[last:])
        mapped.extend(offsets[last:])
        text, offsets = "".join(parts), mapped

    for pattern in (mp._LONG_TOKEN_RE, mp._HTML_COMMENT_RE, mp._URL_RE,
                    mp._EMAIL_RE, mp._HTML_TAG_RE):
        substitute(pattern, " ")
    offsets = [span for char, span in zip(text, offsets) for _ in char.lower()]
    text = text.lower()
    for pattern in [*mp._BOILERPLATE_RES, mp._ISO_DATE_RE, mp._SEMVER_RE, mp._HEX_RE]:
        substitute(pattern, " ")
    for pattern, replacement in mp._UNIT_SYNONYMS:
        substitute(pattern, replacement)
    if text != mp.normalize(raw) or len(text) != len(offsets):
        raise ValueError("Offset-preserving normalization differs from the original extractor.")
    return text, offsets


def source_records(sample: pd.DataFrame, tables: dict) -> dict:
    """Preserve the exact v1 concatenation order and expose record boundaries."""
    result = {key: [] for key in sample[v1.KEYS].itertuples(index=False, name=None)}

    def append(key, source, table, field, locator, value, row):
        if value is None or pd.isna(value):
            return
        prior = [record for record in result[key] if record["source"] == source]
        start = prior[-1]["end"] + 1 if prior else 0
        text = str(value)
        metadata = {name: str(row[name]) for name in
                    ("author_login", "author_type", "created_at", "submitted_at", "sha", "filename")
                    if name in row and pd.notna(row[name])}
        result[key].append({"record_id": f"r{len(result[key]) + 1:04d}", "source": source,
                            "table": table, "field": field, "locator": locator,
                            "metadata": metadata, "start": start, "end": start + len(text),
                            "text": text})

    for row in tables["pull_requests"].to_dict("records"):
        key = tuple(row[name] for name in v1.KEYS)
        if key in result:
            for field in ("title", "body"):
                append(key, "description", "pull_requests", field, field, row.get(field), row)
    for table, field, source, identifier in SPECS:
        for row in tables[table].to_dict("records"):
            key = tuple(row[name] for name in v1.KEYS)
            if key in result:
                append(key, source, table, field, str(row[identifier]), row.get(field), row)
    return result


def group_activations(activations: list[dict]) -> list[dict]:
    """Group overlapping quantity spans within a source and dimension only."""
    groups = []
    ordered = sorted(activations, key=lambda a: (
        a["source"], a["dimension"], a["quant_raw_start"], a["quant_raw_end"], a["activation_id"]))
    for activation in ordered:
        if (groups and groups[-1]["source"] == activation["source"]
                and groups[-1]["dimension"] == activation["dimension"]
                and activation["quant_raw_start"] < groups[-1]["quant_raw_end"]):
            group = groups[-1]
            group["quant_raw_end"] = max(group["quant_raw_end"], activation["quant_raw_end"])
            group["activations"].append(activation)
        else:
            groups.append({"source": activation["source"], "dimension": activation["dimension"],
                           "quant_raw_start": activation["quant_raw_start"],
                           "quant_raw_end": activation["quant_raw_end"], "activations": [activation]})
    for i, group in enumerate(groups, 1):
        group["occurrence_id"] = f"o{i:04d}"
    assigned = [a["activation_id"] for group in groups for a in group["activations"]]
    if len(assigned) != len(activations) or len(set(assigned)) != len(assigned):
        raise ValueError("Every activation must belong to exactly one occurrence.")
    return groups


def extract_occurrences(sample: pd.DataFrame, tables: dict, old_matches: pd.DataFrame) -> dict:
    corpus, _ = assemble_corpus(sample, tables)
    records = source_records(sample, tables)
    output = {}
    for key in sorted(corpus):
        activations, counts = [], {}
        for source, text in corpus[key].items():
            rebuilt = "\n".join(r["text"] for r in records[key] if r["source"] == source)
            if rebuilt != text:
                raise ValueError(f"Source reconstruction changed for {key}, {source}.")
            found = extract_dimensions(text, include_positions=True)
            if not found:
                continue
            normalized, offsets = normalize_with_offsets(text)
            for dimension, matches in found.items():
                counts[(source, dimension)] = len(matches)
                for match in matches:
                    item = {**match, "activation_id": f"a{len(activations) + 1:05d}",
                            "source": source, "dimension": dimension}
                    for name in ("cue", "quant"):
                        start, end = match[f"{name}_start"], match[f"{name}_end"]
                        raw_start, raw_end = offsets[start][0], offsets[end - 1][1]
                        item[f"{name}_raw_start"] = raw_start
                        item[f"{name}_raw_end"] = raw_end
                        item[f"{name}_raw_text"] = text[raw_start:raw_end]
                        item[f"{name}_normalized_text"] = normalized[start:end]
                        item[f"{name}_records"] = [
                            {"record_id": r["record_id"], "start": max(raw_start, r["start"]) - r["start"],
                             "end": min(raw_end, r["end"]) - r["start"]}
                            for r in records[key] if r["source"] == source
                            and r["start"] < raw_end and r["end"] > raw_start]
                        if not item[f"{name}_records"]:
                            raise ValueError(f"Unmapped {name} in {key}.")
                    activations.append(item)
        previous = old_matches[(old_matches.repo_id == key[0]) & (old_matches.number == key[1])]
        expected = {(r.source, r.dimension): int(r.n_matches) for r in previous.itertuples()}
        if counts != expected:
            raise ValueError(f"v1 activation counts changed for {key}: {counts} != {expected}")
        output[key] = {"records": records[key], "occurrences": group_activations(activations),
                       "activation_count": len(activations)}
    return output
