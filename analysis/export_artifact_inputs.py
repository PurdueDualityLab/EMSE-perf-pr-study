"""Export compact, text-free inputs for the offline paper and audit replay.

Run only when deliberately refreshing the artifact from archived local inputs.
This command does not call GitHub, model providers, or Hugging Face.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["repo_id", "number"]
PROVIDERS = ("openai", "gemini", "qwen")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_inputs(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": 1, "files": {}}

    def save(name, frame, source, description, sort_keys=KEYS):
        if frame.duplicated(sort_keys).any():
            raise ValueError(f"Duplicate identities in {name}")
        target = args.output_dir / name
        frame.sort_values(sort_keys, kind="stable").to_csv(target, index=False)
        manifest["files"][name] = {
            "rows": len(frame), "columns": list(frame.columns),
            "sha256": sha256(target), "source_file": source.name,
            "source_sha256": sha256(source), "description": description,
        }

    source = args.regex_dir / "consensus/regex_audit_occurrence_votes.csv"
    columns = KEYS + ["sample_arm", "occurrence_id", "dimension"] + [f"{p}_valid" for p in PROVIDERS]
    votes = pd.read_csv(source)[columns]
    save("regex_occurrence_votes.csv", votes, source,
         "RQ4 metric-extractor audit v3: majority per occurrence, then any valid occurrence per PR.",
         KEYS + ["occurrence_id"])
    manifest["files"]["regex_occurrence_votes.csv"]["run"] = "rq4-regex-positive-audit-v3"

    source = args.tradeoff_dir / "consensus/tradeoff_audit/tradeoff_audit_consensus.parquet"
    columns = KEYS + ["sample_arm"] + [f"{p}_label" for p in PROVIDERS]
    frame = pd.read_parquet(source)
    save("tradeoff_votes.csv", frame[columns], source,
         "Initial binary trade-off audit underlying the submitted paper's 17/12 result; model labels, not verified measurements.")
    manifest["files"]["tradeoff_votes.csv"]["run"] = "initial-binary-tradeoff-audit"
    manifest["files"]["tradeoff_votes.csv"]["models"] = {
        p: sorted(frame[f"{p}_model"].dropna().unique().tolist()) for p in PROVIDERS
    }

    source = args.metric_source
    columns = KEYS + ["id", "agent", "is_merged", "time_to_merge_days"]
    metadata = pd.read_csv(source)[columns]
    expected = pd.read_csv(ROOT / "analysis/classification_labels/rq3_labels.csv")
    if set(map(tuple, metadata[KEYS].to_numpy())) != set(map(tuple, expected[KEYS].to_numpy())):
        raise ValueError("RQ4 metadata does not match the compact label identities")
    save("rq4_metadata.csv", metadata, source,
         "Outcome and agent metadata from the original frozen metric-extraction run; no PR text or model responses.")

    source = args.agent_sample
    sample = pd.read_parquet(source, columns=KEYS + ["aidev_attribution_agent"])
    if len(sample) != 1130 or sample.duplicated(KEYS).any():
        raise ValueError("Expected the 1,130 distinct agentic sample PRs")
    counts = sample.aidev_attribution_agent.value_counts().rename_axis("agent").reset_index(name="count")
    counts.insert(0, "cohort", "sample")
    save("agent_distribution.csv", counts, source,
         "Agent counts aggregated from the frozen final agentic sample.", ["cohort", "agent"])

    # The original workbook contains PR bodies. Export only labels and sampling metadata.
    manual = pd.read_excel(args.manual_workbook, sheet_name="REVIEW")
    meta = pd.read_excel(args.manual_workbook, sheet_name="METADATA").rename(columns={"luna_output": "aidev_task_type"})
    columns = ["case_id", *KEYS, "audit_arm", "audit_stratum", "stratum_population",
               "stratum_sample_size", "inclusion_probability", "sampling_weight", "aidev_task_type", "selection_hash"]
    combined = meta[columns].merge(manual[["case_id", "manual_label"]], on="case_id", validate="one_to_one")
    if len(combined) != 100 or not combined.manual_label.isin(["performance", "not_performance", "uncertain"]).all():
        raise ValueError("Expected 100 completed manual audit labels")
    save("manual_classifier_audit.csv", combined, args.manual_workbook,
         "Blinded human labels with disproportionate-stratified sampling metadata; raw and weighted estimates have different interpretations.")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-workbook", type=Path, required=True)
    parser.add_argument("--metric-source", type=Path, required=True)
    parser.add_argument("--agent-sample", type=Path, default=ROOT / "data/data/sample/agentic_sample.parquet")
    parser.add_argument("--regex-dir", type=Path, default=ROOT / "analysis/rq3_llm_validation/generated_regex_v3")
    parser.add_argument("--tradeoff-dir", type=Path, default=ROOT / "analysis/rq3_llm_validation/generated")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/artifact_inputs")
    result = export_inputs(parser.parse_args())
    print(json.dumps({name: info["rows"] for name, info in result["files"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
