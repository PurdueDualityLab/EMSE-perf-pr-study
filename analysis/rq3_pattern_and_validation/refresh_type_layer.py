"""Rebuild the RQ3 resolved-type layer against the current RQ2 consensus.

The published RQ3 run (summary.json) used RQ2 consensus
a29be9ab...; RQ2 has since been rerun on the primary evidence type and the
published consensus is 5e31d0c7.... Validation *presence* is unchanged by that
rerun, so the D0-D9 extraction and every presence-based result stand; only
`validation_type` / `in_type_layer` and the statistics that condition on them
are stale.

This script does not re-extract. It applies the `run_current.build_base` type
rules to the already-published PR-level extraction output and re-drives the
type-dependent sections of `rq3_statistics` so the refreshed numbers are
produced by the same statistical code as the original run.

Refreshed  : sections 0, 1 and 2 (the sample table, category x validation, and
             the metric profile), and the BH family.
Carried over: section 3 (merge) and section 5 (diff sensitivity) do not
             reference validation type; their committed values remain correct.
Not covered: section 2.5 (per-agent) needs the `agent` column and section 3
             needs `is_merged`, neither of which is in the published compact
             labels. Rerun `run_current.py` once the evidence snapshot and the
             consensus parquets are available locally to regenerate those.

Usage (from the repository root):
    python analysis/rq3_pattern_and_validation/refresh_type_layer.py
"""

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

KEYS = ["repo_id", "number"]
TYPES = {"benchmark", "profiling", "static-reasoning", "anecdotal"}


def read_any(path):
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)


def rebuild_types(labels, rq2):
    """Apply run_current.build_base's type rules to already-extracted rows."""
    cols = [c for c in ["consensus_validation_present", "included_in_stage2_analysis",
                        "consensus_primary_validation_type"] if c in rq2.columns]
    df = labels.merge(rq2[KEYS + cols], on=KEYS, how="left", validate="one_to_one")
    if df[cols].isna().all(axis=1).any():
        raise SystemExit("some RQ3 rows have no row in the RQ2 consensus")

    present = df.consensus_validation_present.astype(bool)
    if not present.equals(df.validation_present.astype(bool)):
        raise SystemExit("validation presence changed; a full rerun of run_current.py is required, "
                         "not a type-layer refresh")

    in_type = df.included_in_stage2_analysis.eq(True)
    if (in_type & ~present).any():
        raise ValueError("Resolved positive type without positive presence")
    if not df.loc[in_type, "consensus_primary_validation_type"].isin(TYPES).all():
        raise ValueError("Invalid resolved validation type")

    old_type, old_layer = df.validation_type.copy(), df.in_type_layer.astype(bool).copy()
    df["in_type_layer"] = in_type
    df["validation_type"] = (df.consensus_primary_validation_type
                             .where(in_type, "unresolved").where(present, "none"))
    moved = df[old_type.ne(df.validation_type)]
    relabelled = moved[old_type.loc[moved.index].ne("unresolved")]
    if len(relabelled):
        raise SystemExit(f"{len(relabelled)} PRs changed an already-resolved type; "
                         "this is not an additive refresh, inspect RQ2 before proceeding")
    return df.drop(columns=cols), old_layer, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(ROOT, "analysis/classification_labels/rq3_labels.csv"))
    ap.add_argument("--rq2", default=os.path.join(ROOT, "analysis/rq2_validation/consensus/rq2_consensus.parquet"))
    ap.add_argument("--rq2-fallback", default=os.path.join(ROOT, "analysis/classification_labels/rq2_labels.csv"))
    ap.add_argument("--output-dir", default=os.path.join(HERE, "results_type_refresh"))
    ap.add_argument("--write-labels", action="store_true",
                    help="also rewrite the published compact labels in place")
    args = ap.parse_args()

    rq2_path = args.rq2 if os.path.exists(args.rq2) else args.rq2_fallback
    print(f"[INFO] RQ2 consensus: {rq2_path}")
    labels = read_any(args.labels)
    df, old_layer, moved = rebuild_types(labels, read_any(rq2_path))

    print(f"[INFO] resolved type layer: {int(old_layer.sum())} -> {int(df.in_type_layer.sum())}")
    print(f"[INFO] previously-unresolved PRs now resolved: {len(moved)}")
    print(moved.validation_type.value_counts().to_string())
    print(f"[INFO] still unresolved: {int((df.validation_type == 'unresolved').sum())}")

    os.makedirs(os.path.join(args.output_dir, "tables"), exist_ok=True)

    import rq3_statistics as stats
    stats.DATA = args.labels
    stats.RES = __import__("pathlib").Path(args.output_dir)
    stats.TAB = stats.RES / "tables"
    stats.AUTHORS = ["agentic", "human_candidate"]
    stats.VTYPES = ["benchmark", "profiling", "static-reasoning", "anecdotal", "none", "unresolved"]
    stats.TESTS.clear()
    stats.LINES.clear()

    # stats.load() reads DATA and needs columns the compact labels lack; feed it the
    # refreshed frame directly, with the same derived columns load() would add.
    d = df.copy()
    d["author_type"] = d.sample_arm
    for col in stats.DIMS:
        d[col] = d[col].astype(bool)
        d[f"{col}_nodiff"] = d[f"{col}_nodiff"].astype(bool)
    d["validation_present"] = d.validation_present.astype(bool)
    d["in_metric_layer"] = d.in_metric_layer.astype(bool)
    counts = d["pattern"].value_counts()
    rare = list(counts[counts < stats.MIN_CATEGORY_N].index)
    d["category"] = d["pattern"].replace({p: "Other" for p in rare})
    d["cat"] = d["category"].map(stats.SHORT)
    d["any_dim"] = d["n_dims"] > 0

    stats.w("Type layer refreshed against the current RQ2 consensus; extraction and validation "
            "presence are unchanged. Sections 3 (merge) and 5 (diff sensitivity) do not condition "
            "on validation type and are carried over from the committed run. Section 2.5 "
            "(per-agent) is not regenerated here: it needs the `agent` column.")
    stats.w("# RQ3 — refreshed resolved-type layer",
            f"Source: `{args.labels}` + `{rq2_path}` (n = {len(d)} PRs; "
            f"{int(d.in_metric_layer.sum())} in the positive metric layer; "
            f"{int(d.in_type_layer.sum())} in the resolved type layer). "
            f"Rare categories pooled as 'Other' for tests: {', '.join(rare) or 'none'}.")
    stats.section_sample(d)
    stats.section_A(d)
    stats.section_B(d)

    # the six merge tests are type-independent; reinstate them so BH runs over the full family
    committed = os.path.join(HERE, "results", "rq3_tests.csv")
    if os.path.exists(committed):
        c = pd.read_csv(committed)
        carried = c[c.family.str.startswith("C.")]
        for _, r in carried.iterrows():
            stats.TESTS.append({k: r.get(k) for k in c.columns if k not in ("p_bh", "significant_bh")})
        print(f"[INFO] carried over {len(carried)} type-independent merge tests into the BH family")
    stats.section_tests()

    out_md = os.path.join(args.output_dir, "rq3_results_type_refresh.md")
    with open(out_md, "w") as fh:
        fh.write("\n".join(stats.LINES) + "\n")
    print(f"[INFO] wrote {out_md} ({len(stats.TESTS)} tests in the family)")

    if args.write_labels:
        df.to_csv(args.labels, index=False)
        print(f"[INFO] rewrote {args.labels}")
        man_path = os.path.join(ROOT, "analysis/classification_labels/manifest.json")
        man = json.load(open(man_path))
        import hashlib
        man["files"]["rq3_labels.csv"]["sha256"] = hashlib.sha256(open(args.labels, "rb").read()).hexdigest()
        man["files"]["rq3_labels.csv"]["notes"] = (
            "validation_type and in_type_layer refreshed against the RQ2 consensus named in "
            "rq2_labels.csv (source_sha256 " + man["files"]["rq2_labels.csv"]["source_sha256"] + "); "
            "D0-D9 extraction and validation presence are unchanged from source_sha256 below.")
        man["summary"]["rq3_resolved_type_layer"] = int(df.in_type_layer.sum())
        man["summary"]["rq3_unresolved_positive_types"] = int((df.validation_type == "unresolved").sum())
        json.dump(man, open(man_path, "w"), indent=2)
        open(man_path, "a").write("\n")
        print(f"[INFO] updated {man_path}")


if __name__ == "__main__":
    main()
