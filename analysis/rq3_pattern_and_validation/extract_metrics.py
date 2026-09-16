"""
RQ3 Step 1 — Extract the performance-metric dimensions (D1–D9, plus the D0 fallback) reported in
each perf PR, and assemble the PR-level analytic dataset for Step 2.

Analytic sample
  * 407 valid perf PRs minus the 50 labelled "No Meaningful Change or Not
    Performance PR" in RQ1  ->  357 PRs (the category x validation layer).
  * The metric layer is defined only where RQ2 found validation evidence
    (validation_present == True; expected n = 177: 128 agent / 49 human).
    PRs without evidence form the "no metric reported" stratum.

Corpus per PR (each source extracted separately so its contribution is known)
  description        PR title + body
  issue_comments     issue-thread comments (bots included; flagged separately)
  review_comments    inline review comments
  commit_messages    commit messages of the PR's commits
  ci_metadata        workflow names of CI runs on the PR + RQ2 pipeline names
  code_diff          unified-diff patch text of all files in the PR

Matching rule (see metric_patterns.py)
  a dimension cue and a quantitative claim within WINDOW_TOKENS of each other,
  with no hypothetical/prospective construction within EXCLUSION_TOKENS of the
  cue. Multi-label per PR.

Outputs (RQ3_metric_targeting/data/)
  rq3_pr_level.csv        one row per PR (357) with labels, outcomes, D0–D9 flags
  rq3_metric_matches.csv  one row per (PR, dimension, source) with an audit snippet
  rq3_corpus_stats.csv    corpus size per PR and source

Run from the repo root:
  python RQ3_metric_targeting/extract_metrics.py
"""

import json
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.data_loader import (  # noqa: E402
    load_main_dataset, load_all_issue_comments, load_all_review_comments,
    load_ai_commits, load_human_commits, load_all_commit_details,
    load_all_workflow_runs, load_validation_evidence,
)
import metric_patterns as mp  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "data"
RQ1_DIR = ROOT / "RQ1_pattern_analysis" / "results"
VALID_IDS = ROOT / "datasets" / "pr_filtering" / "valid_perf_pr_ids.csv"

NO_MEANINGFUL = "No Meaningful Change or Not Performance PR"
DIMS = list(mp.DIMENSIONS)
SOURCES = ["description", "issue_comments", "review_comments",
           "commit_messages", "ci_metadata", "code_diff"]
SNIPPET_CHARS = 70


# ---------------------------------------------------------------------------
# Extraction core
# ---------------------------------------------------------------------------

def _token_starts(text):
    """Character offsets of whitespace-delimited tokens."""
    return [m.start() for m in mp.re.finditer(r"\S+", text)]


def _tok(starts, pos):
    """Token index containing character position ``pos``."""
    return max(bisect_right(starts, pos) - 1, 0)


def _any_within(sorted_idx, center, window):
    """True if any index in ``sorted_idx`` lies within ±window of ``center``."""
    lo = bisect_left(sorted_idx, center - window)
    return lo < len(sorted_idx) and sorted_idx[lo] <= center + window


def _nearest_within(items, item_idx, center, window):
    """Return the item (tok, payload) closest to ``center`` within ±window, else None.

    ``items`` is sorted by token index and ``item_idx`` holds those indices.
    """
    if not items:
        return None
    i = bisect_left(item_idx, center)
    cands = [items[j] for j in (i - 1, i) if 0 <= j < len(items)]
    best = min(cands, key=lambda it: abs(it[0] - center))
    return best if abs(best[0] - center) <= window else None


def _mask_spans(text, spans):
    """Replace matched spans with a filler of equal length (keeps offsets)."""
    if not spans:
        return text
    out, last = [], 0
    for s, e in sorted(spans):
        if s < last:
            continue
        out.append(text[last:s])
        out.append("".join(c if c.isspace() else "█" for c in text[s:e]))
        last = e
    out.append(text[last:])
    return "".join(out)


def _snippet(text, pos):
    s, e = max(0, pos - SNIPPET_CHARS), min(len(text), pos + SNIPPET_CHARS)
    return " ".join(text[s:e].split())


def extract_dimensions(raw_text):
    """
    Return {dim: [match dicts]} for one text. Each match records the cue, the
    quantitative claim it co-occurs with, and a snippet for auditing.
    """
    text = mp.normalize(raw_text)
    if not text.strip():
        return {}
    starts = _token_starts(text)

    # quantitative claims, per kind: sorted [(token index, (kind, matched text))]
    quant_by_kind = {}
    for kind, pat in mp.QUANT_PATTERNS:
        items = sorted((_tok(starts, m.start()), (kind, m.group(0))) for m in pat.finditer(text))
        if items:
            quant_by_kind[kind] = (items, [t for t, _ in items])

    def nearest_quant(center, kinds):
        best = None
        # Resolve equal-distance claims consistently across Python hash seeds.
        for k in sorted(kinds):
            if k not in quant_by_kind:
                continue
            items, idx = quant_by_kind[k]
            q = _nearest_within(items, idx, center, mp.WINDOW_TOKENS)
            if q is not None and (best is None or abs(q[0] - center) < abs(best[0] - center)):
                best = q
        return best

    excl_idx = [_tok(starts, m.start()) for m in mp.EXCLUSION_RE.finditer(text)]
    # a heading such as "## Expected performance improvements" makes the whole
    # following block prospective; exclude its tokens too
    for m in mp.PROSPECTIVE_HEADING_RE.finditer(text):
        end = mp.block_end(text, m.end())
        excl_idx.extend(range(_tok(starts, m.start()), _tok(starts, end - 1) + 1))
    excl_idx = sorted(set(excl_idx))

    # raw cue spans per dimension (unmasked) — used for masking and suppression
    cue_spans = {d: [(m.start(), m.end()) for m in pat.finditer(text)]
                 for d, pat in mp.CUE_PATTERNS.items()}
    cue_tok = {d: sorted(_tok(starts, s) for s, _ in spans) for d, spans in cue_spans.items()}

    # effective cues per dimension: matched on the text with higher-priority
    # phrases masked out ("build time" is not a D1 "time" cue)
    eff_cues = {}
    for dim, pat in mp.CUE_PATTERNS.items():
        t_dim = text
        masked = [sp for other in mp.MASK_BEFORE.get(dim, []) for sp in cue_spans[other]]
        if masked:
            t_dim = _mask_spans(text, masked)
        eff_cues[dim] = [(_tok(starts, m.start()), _tok(starts, m.end() - 1), m.group(0), m.start())
                         for m in pat.finditer(t_dim)]
    eff_cue_tok = {d: sorted(st for st, _, _, _ in cues) for d, cues in eff_cues.items()}

    def span_dist(qt, st, et):
        return 0 if st <= qt <= et else min(abs(qt - st), abs(qt - et))

    def nearer_cue_elsewhere(dim, qt, kind, dist):
        """True if another dimension has a cue strictly closer to the claim at ``qt``."""
        for d2, cues in eff_cues.items():
            if d2 == dim or kind not in mp.QUANT_KINDS[d2]:
                continue
            idx = eff_cue_tok[d2]
            i = bisect_left(idx, qt)
            for j in (i - 1, i):
                if 0 <= j < len(cues):
                    st, et, _, _ = cues[j]
                    if span_dist(qt, st, et) < dist:
                        return True
        return False

    found = {}

    def record(dim, cue, cue_pos, quant_item, rule):
        cue = " ".join(cue.split())
        found.setdefault(dim, []).append({
            "cue": cue if len(cue) <= 60 else cue[:28] + " … " + cue[-28:],
            "quant_kind": quant_item[0], "quant": " ".join(str(quant_item[1]).split())[:60],
            "rule": rule, "snippet": _snippet(text, cue_pos),
        })

    # (a) cue + quantitative claim (of a kind appropriate to the dimension) within window;
    #     the claim is credited to the nearest cue only
    for dim, cues in eff_cues.items():
        suppress = mp.SUPPRESS_IF_NEAR.get(dim, [])
        for st, et, cue, pos in cues:
            if _any_within(excl_idx, st, mp.EXCLUSION_TOKENS):
                continue
            if any(_any_within(cue_tok[s], st, mp.WINDOW_TOKENS) for s in suppress):
                continue
            q = nearest_quant(st, mp.QUANT_KINDS[dim])
            if q is None:
                continue
            qt, (kind, _) = q
            if mp.NEAREST_CUE_WINS and nearer_cue_elsewhere(dim, qt, kind, span_dist(qt, st, et)):
                continue
            if dim in mp.FALLBACK_DIMS:
                # fallback only when no named dimension is cued in the window and the
                # numerals carry no unit (a unit would itself name the dimension)
                if any(d2 != dim and _any_within(eff_cue_tok[d2], qt, mp.WINDOW_TOKENS) for d2 in eff_cues):
                    continue
                if any(_any_within(quant_by_kind[k][1], qt, 2) for k in mp.UNIT_KINDS if k in quant_by_kind):
                    continue
            record(dim, cue, pos, q[1], "cue+quant")

    # (b) self-sufficient cues
    for dim, rules in mp.SELF_SUFFICIENT.items():
        for pat, required_ctx, suppress_dims in rules:
            for m in pat.finditer(text):
                ct = _tok(starts, m.start())
                if _any_within(excl_idx, ct, mp.EXCLUSION_TOKENS):
                    continue
                if any(_any_within(cue_tok[s], ct, mp.WINDOW_TOKENS) for s in suppress_dims):
                    continue
                if required_ctx is not None and nearest_quant(ct, required_ctx) is None:
                    continue
                record(dim, m.group(0), m.start(), ("self", m.group(0)), "self-sufficient")

    return found


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------

def _concat(series):
    return "\n".join(str(x) for x in series.dropna() if str(x).strip())


def build_corpus(main):
    """Return {pr_id: {source: text}} and a bot-flag map for issue comments."""
    ids = set(main["id"])

    ic = load_all_issue_comments()
    ic = ic[ic["pr_id"].isin(ids)]
    rc = load_all_review_comments()
    rc = rc[rc["pr_id"].isin(ids)]

    ai_c, hu_c = load_ai_commits(), load_human_commits()
    commits = pd.concat([ai_c[["pr_id", "commit_message"]], hu_c[["pr_id", "commit_message"]]])
    commits = commits[commits["pr_id"].isin(ids)]

    details = load_all_commit_details()
    details = details[details["pr_id"].isin(ids)]

    runs = load_all_workflow_runs()
    runs = runs[runs["pr_id"].isin(ids)]
    rq2 = load_validation_evidence()
    pipe_names = {r.pr_id: " | ".join(map(str, r.pipeline_names)) if len(r.pipeline_names) else ""
                  for r in rq2.itertuples()}

    desc = {r.id: f"{r.title or ''}\n{r.body if isinstance(r.body, str) else ''}"
            for r in main.itertuples()}
    issue_txt = ic.groupby("pr_id")["body"].apply(_concat).to_dict()
    issue_bot_txt = ic[ic["user_type"] == "Bot"].groupby("pr_id")["body"].apply(_concat).to_dict()
    review_txt = rc.groupby("pr_id")["body"].apply(_concat).to_dict()
    commit_txt = commits.groupby("pr_id")["commit_message"].apply(_concat).to_dict()
    diff_txt = details.groupby("pr_id")["patch"].apply(_concat).to_dict()
    ci_txt = runs.groupby("pr_id")["workflow_name"].apply(lambda s: " | ".join(sorted(set(s.dropna())))).to_dict()

    # patch-size metadata
    size = details.groupby("pr_id").agg(
        patch_lines_changed=("changes", "sum"),
        patch_additions=("additions", "sum"),
        patch_deletions=("deletions", "sum"),
        files_changed=("filename", "nunique"),
    )

    corpus = {}
    for pid in ids:
        corpus[pid] = {
            "description": desc.get(pid, ""),
            "issue_comments": issue_txt.get(pid, ""),
            "review_comments": review_txt.get(pid, ""),
            "commit_messages": commit_txt.get(pid, ""),
            "ci_metadata": " | ".join(x for x in [ci_txt.get(pid, ""), pipe_names.get(pid, "")] if x),
            "code_diff": diff_txt.get(pid, ""),
        }
    return corpus, issue_bot_txt, size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_labelled_sample():
    """357 real perf PRs with RQ1 pattern, RQ2 validation, and outcome metadata."""
    main = load_main_dataset()
    valid = pd.read_csv(VALID_IDS)
    main = main[main["id"].isin(valid["id"])].copy()

    rq1 = pd.concat([pd.read_csv(RQ1_DIR / "Agent_PRs.csv"), pd.read_csv(RQ1_DIR / "Human_PRs.csv")])
    rq1 = rq1.rename(columns={"PR ID": "id", "Final Pattern": "pattern",
                              "Final Sub Pattern": "sub_pattern", "Label Source": "rq1_label_source"})
    rq2 = load_validation_evidence().rename(columns={"pr_id": "id"})

    m = main.merge(rq1[["id", "pattern", "sub_pattern", "rq1_label_source"]], on="id", how="left")
    m = m.merge(rq2[["id", "validation_present", "validation_type", "evidence_sources"]], on="id", how="left")
    assert m["pattern"].notna().all(), "RQ1 label missing for some PRs"
    assert m["validation_present"].notna().all(), "RQ2 label missing for some PRs"

    n_all = len(m)
    m = m[m["pattern"] != NO_MEANINGFUL].copy()
    print(f"[INFO] {n_all} valid perf PRs - {n_all - len(m)} 'No Meaningful' = {len(m)} analytic PRs")

    m["evidence_sources_norm"] = m["evidence_sources"].apply(_norm_sources)
    m["author_type"] = m["author_type"].replace({"AI Agent": "AI Agent", "Human": "Human"})
    return m


def _norm_sources(v):
    """RQ2 evidence_sources is heterogeneous (ndarray-like strings and bare labels)."""
    s = str(v).lower() if v is not None else ""
    out = []
    for key, label in [("description", "description"), ("comment", "comments"),
                       ("pipeline", "pipeline"), ("code diff", "code_diff")]:
        if key in s:
            out.append(label)
    return "|".join(out)


def main():
    print("[INFO] Loading labelled sample ...")
    sample = load_labelled_sample()
    print("[INFO] Assembling corpus ...")
    corpus, issue_bot_txt, size = build_corpus(sample)

    pr_rows, match_rows, corpus_rows = [], [], []
    for r in sample.itertuples():
        pid = r.id
        texts = corpus[pid]
        per_source = {src: extract_dimensions(txt) for src, txt in texts.items()}
        bot_dims = set(extract_dimensions(issue_bot_txt.get(pid, "")))

        dims_all = set().union(*[set(d) for d in per_source.values()])
        dims_nodiff = set().union(*[set(d) for s, d in per_source.items() if s != "code_diff"])
        dims_desc = set(per_source["description"])
        dim_sources = {d: sorted(s for s, found in per_source.items() if d in found) for d in sorted(dims_all)}

        row = {
            "id": pid, "number": r.number, "html_url": r.html_url, "title": r.title,
            "author_type": r.author_type, "agent": r.agent, "primary_language": r.primary_language,
            "state": r.state, "is_merged": bool(r.is_merged), "time_to_merge_days": r.time_to_merge_days,
            "pattern": r.pattern, "sub_pattern": r.sub_pattern, "rq1_label_source": r.rq1_label_source,
            "validation_present": bool(r.validation_present), "validation_type": r.validation_type,
            "evidence_sources_norm": r.evidence_sources_norm,
            "in_metric_layer": bool(r.validation_present),
        }
        for c in ["patch_lines_changed", "patch_additions", "patch_deletions", "files_changed"]:
            row[c] = int(size.loc[pid, c]) if pid in size.index else np.nan
        for d in DIMS:
            row[d] = d in dims_all
        row["n_dims"] = len(dims_all)
        row["n_dims_specific"] = len(dims_all - {"D0"})      # named dimensions D1–D9 only
        row["dims"] = "|".join(sorted(dims_all, key=lambda d: (d == "D0", d)))
        for d in DIMS:
            row[f"{d}_nodiff"] = d in dims_nodiff
        row["n_dims_nodiff"] = len(dims_nodiff)
        row["n_dims_description_only"] = len(dims_desc)
        row["dims_from_bot_comments"] = "|".join(sorted(bot_dims))
        row["dim_sources"] = json.dumps(dim_sources)
        pr_rows.append(row)

        for src, found in per_source.items():
            for d, matches in found.items():
                first = matches[0]
                match_rows.append({
                    "id": pid, "author_type": r.author_type, "pattern": r.pattern,
                    "validation_present": bool(r.validation_present), "validation_type": r.validation_type,
                    "dimension": d, "source": src, "n_matches": len(matches),
                    "rule": first["rule"], "cue": first["cue"], "quant_kind": first["quant_kind"],
                    "quant": first["quant"], "snippet": first["snippet"], "html_url": r.html_url,
                })
        corpus_rows.append({"id": pid, **{f"chars_{s}": len(t) for s, t in texts.items()}})

    pr_df = pd.DataFrame(pr_rows)
    matches_df = pd.DataFrame(match_rows)
    corpus_df = pd.DataFrame(corpus_rows)

    OUT_DIR.mkdir(exist_ok=True)
    pr_df.to_csv(OUT_DIR / "rq3_pr_level.csv", index=False)
    matches_df.to_csv(OUT_DIR / "rq3_metric_matches.csv", index=False)
    corpus_df.to_csv(OUT_DIR / "rq3_corpus_stats.csv", index=False)
    print(f"[INFO] Saved {len(pr_df)} PR rows, {len(matches_df)} match rows -> {OUT_DIR}")

    # ---- quick summary -----------------------------------------------------
    val = pr_df[pr_df["in_metric_layer"]]
    print(f"\n[INFO] Metric layer (validation present): n={len(val)} "
          f"({(val.author_type == 'AI Agent').sum()} agent / {(val.author_type == 'Human').sum()} human)")
    print(f"       window={mp.WINDOW_TOKENS} tokens, exclusion window={mp.EXCLUSION_TOKENS} tokens")
    print("\n=== Dimension coverage among validated PRs (% reporting) ===")
    for d in DIMS:
        a = val[val.author_type == "AI Agent"][d].mean() * 100
        h = val[val.author_type == "Human"][d].mean() * 100
        print(f"  {d} {mp.DIMENSIONS[d]:<28} agent={a:5.1f}%  human={h:5.1f}%  "
              f"(no-diff: agent={val[val.author_type == 'AI Agent'][f'{d}_nodiff'].mean()*100:5.1f}% "
              f"human={val[val.author_type == 'Human'][f'{d}_nodiff'].mean()*100:5.1f}%)")
    print("\n=== n_dims distribution among validated PRs ===")
    print(val.groupby("author_type")["n_dims"].describe()[["count", "mean", "50%", "max"]])
    print("\n=== Dimensions detected among NON-validated PRs (should be low; sanity check) ===")
    nv = pr_df[~pr_df["in_metric_layer"]]
    print(f"  {len(nv)} PRs; any dimension: {(nv.n_dims > 0).sum()} ({(nv.n_dims > 0).mean()*100:.1f}%)"
          f"; excluding diff: {(nv.n_dims_nodiff > 0).sum()}")
    if len(matches_df):
        print("\n=== Match rows by source (validated PRs) ===")
        print(matches_df[matches_df.validation_present].groupby("source").size())


if __name__ == "__main__":
    main()
