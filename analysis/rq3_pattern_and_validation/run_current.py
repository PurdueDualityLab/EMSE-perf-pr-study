"""Run the historical RQ3 extractor on the current consensus sample and snapshot."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from extract_metrics import extract_dimensions, DIMS, SOURCES

KEYS = ["repo_id", "number"]
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def join_consensus(sample, rq1, rq2, status):
    """Keep resolved RQ1 and RQ2 presence; retain unresolved types explicitly."""
    for frame in (sample, rq1, rq2, status):
        if frame.duplicated(KEYS).any():
            raise ValueError("Duplicate PR identity in inputs")
    expected = set(sample[KEYS].itertuples(index=False, name=None))
    for frame in (rq1, status):
        if set(frame[KEYS].itertuples(index=False, name=None)) != expected:
            raise ValueError("RQ1 or evidence identities differ from the sample")
    if not set(rq2[KEYS].itertuples(index=False, name=None)) <= expected:
        raise ValueError("RQ2 contains identities outside the sample")
    base = sample.merge(rq1[KEYS + ["sample_arm", "included_in_analysis",
        "consensus_high_level_pattern", "consensus_sub_pattern", "consensus_status"]],
        on=KEYS, validate="one_to_one", suffixes=("", "_rq1"))
    base = base.merge(rq2[KEYS + ["sample_arm", "consensus_validation_present",
        "included_in_stage2_analysis", "consensus_primary_validation_type"]],
        on=KEYS, validate="one_to_one", suffixes=("", "_rq2"))
    for column in ("sample_arm_rq1", "sample_arm_rq2"):
        if not base.sample_arm.eq(base[column]).all():
            raise ValueError("Consensus arm mismatch")
    base = base.merge(status[KEYS + ["status"]], on=KEYS, how="left", validate="one_to_one")
    keep = base.included_in_analysis.eq(True) & base.consensus_validation_present.notna()
    base = base.loc[keep].copy()
    if not base.status.eq("complete").all():
        raise ValueError("Analytic PRs have missing or incomplete evidence")
    if base[["consensus_high_level_pattern", "consensus_sub_pattern"]].isna().any().any():
        raise ValueError("Included RQ1 row has missing labels")
    base["validation_present"] = base.consensus_validation_present.astype(bool)
    base["in_metric_layer"] = base.validation_present
    base["in_type_layer"] = base.included_in_stage2_analysis.eq(True)
    if (base.in_type_layer & ~base.validation_present).any():
        raise ValueError("Resolved positive type without positive presence")
    types = {"benchmark", "profiling", "static-reasoning", "anecdotal"}
    if not base.loc[base.in_type_layer, "consensus_primary_validation_type"].isin(types).all():
        raise ValueError("Invalid resolved validation type")
    base["validation_type"] = base.consensus_primary_validation_type.where(
        base.in_type_layer, "unresolved").where(base.validation_present, "none")
    return base.sort_values(KEYS).reset_index(drop=True)


def assemble_corpus(sample, tables):
    """Use full snapshot text, without the LLM prompt truncation limits."""
    identities = set(sample[KEYS].itertuples(index=False, name=None))
    corpus = {key: {source: "" for source in SOURCES} for key in identities}
    bots = {}
    specs = [("issue_comments", "body", "issue_comments"),
             ("review_comments", "body", "review_comments"),
             ("commits", "message", "commit_messages"),
             ("workflow_runs", "name", "ci_metadata"),
             ("pull_request_files", "patch", "code_diff")]
    for table, column, source in specs:
        for key, group in tables[table].groupby(KEYS, sort=True):
            if key in corpus:
                corpus[key][source] = "\n".join(group[column].dropna().astype(str))
                if table == "issue_comments":
                    bots[key] = "\n".join(group.loc[group.author_type.eq("Bot"), column].dropna().astype(str))
    pr = tables["pull_requests"].set_index(KEYS)
    if pr.index.has_duplicates:
        raise ValueError("Duplicate snapshot PR identity")
    for row in sample.itertuples():
        key = (row.repo_id, row.number)
        record = pr.loc[key]
        corpus[key]["description"] = "\n".join(str(value) for value in
            (record.title, record.body) if pd.notna(value))
    return corpus, bots


def extract_sample(sample, tables):
    corpus, bots = assemble_corpus(sample, tables)
    metadata = tables["pull_requests"].set_index(KEYS)
    rows, matches, sizes = [], [], []
    for pr in sample.itertuples():
        key = (pr.repo_id, pr.number)
        texts = corpus[key]
        found = {source: extract_dimensions(text) for source, text in texts.items()}
        dims = set().union(*(set(value) for value in found.values()))
        nodiff = set().union(*(set(value) for source, value in found.items() if source != "code_diff"))
        meta = metadata.loc[key]
        created = pd.to_datetime(meta.created_at, utc=True)
        merged = pd.to_datetime(meta.merged_at, utc=True)
        row = dict(repo_id=pr.repo_id, number=pr.number, id=pr.id, sample_arm=pr.sample_arm,
            html_url=pr.html_url, title=pr.title, author_type=pr.sample_arm,
            agent=pr.aidev_attribution_agent, pattern=pr.consensus_high_level_pattern,
            sub_pattern=pr.consensus_sub_pattern, rq1_label_source=pr.consensus_status,
            validation_present=pr.validation_present, validation_type=pr.validation_type,
            in_metric_layer=pr.in_metric_layer, in_type_layer=pr.in_type_layer, is_merged=bool(meta.merged),
            time_to_merge_days=(merged-created).total_seconds()/86400 if pd.notna(merged) else None,
            n_dims=len(dims), n_dims_specific=len(dims-{"D0"}), n_dims_nodiff=len(nodiff),
            dims="|".join(sorted(dims)), n_dims_description_only=len(found["description"]),
            dims_from_bot_comments="|".join(sorted(extract_dimensions(bots.get(key, "")))),
            dim_sources=json.dumps({d: sorted(s for s in SOURCES if d in found[s]) for d in sorted(dims)}))
        row.update({d: d in dims for d in DIMS})
        row.update({f"{d}_nodiff": d in nodiff for d in DIMS})
        rows.append(row)
        sizes.append(dict(repo_id=pr.repo_id, number=pr.number,
            **{f"chars_{s}": len(t) for s, t in texts.items()}))
        for source, dimensions in found.items():
            for dimension, claims in dimensions.items():
                matches.append(dict(repo_id=pr.repo_id, number=pr.number, id=pr.id,
                    author_type=pr.sample_arm, pattern=pr.consensus_high_level_pattern,
                    validation_present=pr.validation_present, validation_type=pr.validation_type,
                    dimension=dimension, source=source, n_matches=len(claims),
                    html_url=pr.html_url, **claims[0]))
    return pd.DataFrame(rows), pd.DataFrame(matches), pd.DataFrame(sizes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=ROOT / "data/data/sample/balanced_sample.parquet")
    parser.add_argument("--rq1", type=Path, default=HERE.parent / "rq1_optimization_patterns/consensus/rq1_consensus.parquet")
    parser.add_argument("--rq2", type=Path, default=HERE.parent / "rq2_validation/consensus/rq2_consensus.parquet")
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "mining/sample_evidence/final")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    files = [args.sample, args.rq1, args.rq2] + [args.evidence_dir / f"{name}.parquet" for name in
        ("collection_status", "pull_requests", "pull_request_files", "commits", "issue_comments", "review_comments", "workflow_runs")]
    tables = {path.stem: pd.read_parquet(path) for path in files[3:]}
    sample = pd.read_parquet(args.sample)
    rq1, rq2 = pd.read_parquet(args.rq1), pd.read_parquet(args.rq2)
    if len(sample) != 2260 or len(rq1) != 2260 or len(rq2) != 2258:
        raise ValueError("Unexpected official sample or consensus size")
    snapshots = tables["collection_status"].snapshot_at.dropna().unique()
    if len(snapshots) != 1:
        raise ValueError("Expected one evidence snapshot")
    for table in tables.values():
        checked = table.merge(sample[KEYS + ["selection_hash"]], on=KEYS,
            how="left", validate="many_to_one", suffixes=("", "_sample"))
        if not checked.selection_hash.eq(checked.selection_hash_sample).all():
            raise ValueError("Evidence selection hash does not match sample")
        if not checked.snapshot_at.eq(snapshots[0]).all():
            raise ValueError("Mixed evidence snapshots")
    selected = join_consensus(sample, rq1, rq2, tables["collection_status"])
    print(f"Extracting metrics for {len(selected)} analytic PRs ...", flush=True)
    frames = extract_sample(selected, tables)
    out = args.output_dir.resolve()
    (out / "data").mkdir(parents=True, exist_ok=True)
    audit = sample[KEYS + ["sample_arm"]].merge(rq1[KEYS + ["included_in_analysis", "inclusion_reason"]],
        on=KEYS, validate="one_to_one").merge(rq2[KEYS + ["consensus_validation_present", "included_in_stage2_analysis"]],
        on=KEYS, how="left", validate="one_to_one")
    audit["included_in_rq3"] = pd.MultiIndex.from_frame(audit[KEYS]).isin(pd.MultiIndex.from_frame(selected[KEYS]))
    audit.to_csv(out / "data/sample_inclusion.csv", index=False)
    for frame, name in zip(frames, ("rq3_pr_level", "rq3_metric_matches", "rq3_corpus_stats")):
        frame.to_csv(out / "data" / f"{name}.csv", index=False)
    summary = {"sample_rows": len(sample), "analytic_rows": len(selected),
        "snapshot_at": snapshots[0],
        "metric_matches": len(frames[1]),
        "validation_positive_rows": int(selected.validation_present.sum()),
        "metric_layer_rows": int(selected.in_metric_layer.sum()),
        "type_layer_rows": int(selected.in_type_layer.sum()),
        "unresolved_positive_types": int((selected.validation_type == "unresolved").sum()),
        "arms": selected.sample_arm.value_counts().to_dict(),
        "collection_status": tables["collection_status"].status.value_counts().to_dict(),
        "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import rq3_statistics as stats
    import make_figures as figures
    stats.DATA = figures.DATA = out / "data/rq3_pr_level.csv"
    stats.RES, stats.TAB = out / "results", out / "results/tables"
    figures.FIG = out / "figures"
    stats.AUTHORS = figures.AUTHORS = ["agentic", "human_candidate"]
    stats.VTYPES = figures.VTYPES = ["benchmark", "profiling", "static-reasoning", "anecdotal", "none", "unresolved"]
    figures.VTYPE_COLORS["static-reasoning"] = figures.VTYPE_COLORS["static-analysis"]
    figures.VTYPE_COLORS["unresolved"] = "#eeeeee"
    stats.main()
    figures.main()
    print(json.dumps({k: v for k, v in summary.items() if k != "inputs_sha256"}, indent=2))


if __name__ == "__main__":
    main()
