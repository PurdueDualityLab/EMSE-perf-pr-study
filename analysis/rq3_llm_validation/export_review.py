"""Export RQ3 consensus results as human-reviewable CSVs and case dossiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.rq3_llm_validation import experiment


DEFAULT_GENERATED = Path("analysis/rq3_llm_validation/generated")
DEFAULT_OUTPUT = DEFAULT_GENERATED / "review"


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    if pd.isna(value):
        return ""
    return str(value)


def _review_frame(task: str, generated_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_path = generated_dir / "samples" / f"{task}_sample.parquet"
    consensus_path = generated_dir / "consensus" / task / f"{task}_consensus.parquet"
    evidence = experiment.build_input(
        sample_path, experiment.DEFAULT_MATCHES, experiment.DEFAULT_EVIDENCE, task
    )
    consensus = pd.read_parquet(consensus_path)
    frame = evidence.merge(
        consensus.drop(columns=["sample_arm", "task"]),
        on=experiment.KEYS,
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(consensus):
        raise ValueError(f"Review export lost {task} consensus rows.")

    frame["agreement"] = frame["vote_count"].map({3: "unanimous", 2: "majority"})
    frame["provider_votes"] = frame.apply(
        lambda row: " | ".join(
            f"{provider}={row[f'{provider}_label']}"
            for provider in ("openai", "gemini", "qwen")
        ),
        axis=1,
    )
    frame["manual_review_status"] = "pending"
    frame["manual_review_label"] = ""
    frame["manual_review_notes"] = ""

    matches = pd.read_csv(experiment.DEFAULT_MATCHES)
    identities = set(map(tuple, frame[experiment.KEYS].to_numpy()))
    matches = matches[matches[experiment.KEYS].apply(tuple, axis=1).isin(identities)].copy()
    counts = matches.groupby(experiment.KEYS).size().rename("regex_match_rows")
    frame = frame.merge(counts, on=experiment.KEYS, how="left", validate="one_to_one")
    frame["regex_match_rows"] = frame["regex_match_rows"].fillna(0).astype(int)
    return frame.sort_values(["sample_arm", *experiment.KEYS], kind="mergesort"), matches


def _columns(task: str) -> list[str]:
    columns = [
        "repo_id", "number", "repo_full_name", "html_url", "sample_arm", "title",
        "author_type", "agent", "pattern", "sub_pattern", "validation_type", "dims",
        "consensus_label", "agreement", "vote_count", "coalition", "provider_votes",
        "regex_match_rows", "highlighted_matches",
    ]
    for provider in ("openai", "gemini", "qwen"):
        columns.extend([
            f"{provider}_label", f"{provider}_evidence_sources",
            f"{provider}_evidence_quotes", f"{provider}_rationale",
        ])
        if task == "tradeoff_audit":
            columns.extend([f"{provider}_gain_direction", f"{provider}_memory_direction"])
    columns.extend(["manual_review_status", "manual_review_label", "manual_review_notes"])
    return columns


def _write_csv(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    result = frame.reindex(columns=columns).copy()
    for column in result.columns:
        result[column] = result[column].map(_cell)
    result.to_csv(path, index=False)


def _dossier(row: dict[str, Any], matches: pd.DataFrame) -> str:
    lines = [
        f"# {row['repo_full_name']} #{row['number']}",
        "",
        f"- URL: {row['html_url']}",
        f"- Sample arm: `{row['sample_arm']}`",
        f"- Consensus: `{row['consensus_label']}` ({row['agreement']})",
        f"- Coalition: `{row['coalition']}`",
        "",
        "## Model Decisions",
        "",
    ]
    for provider in ("openai", "gemini", "qwen"):
        lines.extend([
            f"### {provider.title()}",
            "",
            f"- Label: `{row[f'{provider}_label']}`",
            f"- Sources: `{_cell(row[f'{provider}_evidence_sources'])}`",
            f"- Quotes: `{_cell(row[f'{provider}_evidence_quotes'])}`",
            "",
            row[f"{provider}_rationale"],
            "",
        ])
    lines.extend(["## Regex Matches", ""])
    for match in matches.sort_values(["dimension", "source", "snippet"], kind="mergesort").to_dict("records"):
        lines.extend([
            f"- **{match['dimension']} / {match['source']} / {match['rule']}**",
            f"  - Cue: `{_cell(match['cue'])}`",
            f"  - Quantity: `{_cell(match['quant'])}` (`{_cell(match['quant_kind'])}`)",
            f"  - Snippet: {_cell(match['snippet'])}",
        ])
    lines.extend([
        "",
        "## Exact Classification Prompt",
        "",
        "````text",
        experiment.prompt_for(row, "regex_audit").rstrip(),
        "````",
        "",
        "## Manual Review",
        "",
        "- Decision:",
        "- Notes:",
        "",
    ])
    return "\n".join(lines)


def export_review(generated_dir: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=False)
    regex, regex_matches = _review_frame("regex_audit", generated_dir)
    tradeoff, _ = _review_frame("tradeoff_audit", generated_dir)

    _write_csv(regex, output_dir / "regex_audit_all_results.csv", _columns("regex_audit"))
    _write_csv(tradeoff, output_dir / "tradeoff_audit_all_results.csv", _columns("tradeoff_audit"))

    false_positives = regex[regex["consensus_label"].eq("false_positive")].copy()
    dossier_dir = output_dir / "false_positive_dossiers"
    dossier_dir.mkdir()
    dossier_paths = []
    false_positive_ids = set(map(tuple, false_positives[experiment.KEYS].to_numpy()))
    false_positive_matches = regex_matches[
        regex_matches[experiment.KEYS].apply(tuple, axis=1).isin(false_positive_ids)
    ].sort_values([*experiment.KEYS, "dimension", "source", "snippet"], kind="mergesort")
    for row in false_positives.to_dict("records"):
        name = f"{row['repo_full_name'].replace('/', '__')}__{row['number']}.md"
        path = dossier_dir / name
        case_matches = false_positive_matches[
            false_positive_matches[experiment.KEYS].eq(
                pd.Series({"repo_id": row["repo_id"], "number": row["number"]})
            ).all(axis=1)
        ]
        path.write_text(_dossier(row, case_matches))
        dossier_paths.append(str(path.relative_to(output_dir)))
    false_positives["dossier_path"] = dossier_paths
    _write_csv(
        false_positives,
        output_dir / "regex_false_positive_review.csv",
        [*_columns("regex_audit"), "dossier_path"],
    )
    false_positive_matches.to_csv(output_dir / "regex_false_positive_matches.csv", index=False)

    summary = {
        "regex_rows": len(regex),
        "regex_false_positives": len(false_positives),
        "regex_false_positive_match_rows": len(false_positive_matches),
        "tradeoff_rows": len(tradeoff),
        "dossiers": len(dossier_paths),
    }
    (output_dir / "README.md").write_text(
        "# RQ3 manual-review export\n\n"
        "Start with `regex_false_positive_review.csv`. Each row is one consensus false "
        "positive and includes all three votes, rationales, quotes, highlighted matches, "
        "and empty manual-review columns. `regex_false_positive_matches.csv` expands the "
        "detected matches to one row per match. The linked dossiers contain the exact "
        "classification prompt and complete model decisions.\n\n"
        f"Export summary: `{json.dumps(summary, sort_keys=True)}`\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export_review(args.generated_dir, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
