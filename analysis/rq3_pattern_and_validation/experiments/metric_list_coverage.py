"""
Experiment: is the D1–D9 dimension list comprehensive?

Runs the Step-1 extractor under alternative dimension lists and compares the
coverage each list achieves on the same corpus:

  baseline        D0–D9 as in metric_patterns.py (D0 and the CJK handling were adopted
                  from an earlier run of this experiment; the remaining candidates are kept
                  so the comparison can be re-run)
  +<candidate>    baseline plus one candidate dimension / enrichment
  all             baseline plus every candidate

Candidates (from orphan-claim mining on the corpus + Gregg/Jain resource taxonomies):
  D10  Storage / disk footprint  database/index/disk size, on-disk cache, log volume
  D11  GPU / accelerator         GPU utilisation, VRAM, CUDA kernel time
  D12  Errors / timeouts         error, failure, timeout, retry, rate-limit rates
  D13  UI / rendering            re-renders, frame drops, jank, Web Vitals, Lighthouse
  D14  Work volume processed     rows scanned, records processed, search space, iterations
  D15  LLM tokens / context      prompt/completion tokens, context length
  D16  Cache effectiveness       hit/miss rate, evictions (currently folded into D5)
  size-tables                    D6 byte-unit table cells under a "size" header (bundle-size bots)

Coverage = % of validated PRs with ≥ 1 dimension (all / agent / human); the
share of *non-validated* PRs that fire is reported as a noise proxy (a list that
"covers" more by firing on PRs without validation evidence is picking up noise).

Output: results/metric_list_coverage.md (+ snippets of every new-dimension hit
for auditing). Run from the repo root:
  python RQ3_metric_targeting/experiments/metric_list_coverage.py
"""

import copy
import re
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RQ3 = HERE.parent
sys.path.insert(0, str(RQ3.parent))
sys.path.insert(0, str(RQ3))
import metric_patterns as mp  # noqa: E402
import extract_metrics as em  # noqa: E402

OUT = RQ3 / "results" / "metric_list_coverage.md"
NUM, GAP = mp.NUM, mp._GAP
CMP = {"pct", "mult", "from_to", "before_after"}

# ---------------------------------------------------------------------------
# Candidate dimensions: id -> dict(label, cue, kinds, self=[(regex, ctx, suppress)], fallback)
# ---------------------------------------------------------------------------

CANDIDATES = {
    "D10": dict(
        label="Storage / disk footprint",
        cue=re.compile(r"\b(?:disk[- ](?:space|usage|footprint|size)|storage(?:[- ](?:size|usage|footprint|cost))?|"
                       r"(?:database|db|index|table|collection|snapshot|backup|log|wal|blob)[- ]size|"
                       r"on[- ]disk|log[- ]volume|(?:disk|persistent)[- ]cache[- ]size|"
                       r"file[- ]count|inodes?|s3[- ](?:size|objects))\b"),
        kinds={"bytes"} | CMP),
    "D11": dict(
        label="GPU / accelerator",
        cue=re.compile(r"\b(?:gpu(?:[- ](?:memory|utili[sz]ation|usage|time|load))?|vram|cuda(?:[- ](?:kernels?|time|graphs?))?|"
                       r"tpu|npu|accelerators?|hbm|tflops|kernel[- ](?:time|launch(?:es)?)|"
                       r"device[- ]memory|sm[- ]utili[sz]ation|tensor[- ]cores?)\b"),
        kinds={"time", "bytes", "rate"} | CMP),
    "D12": dict(
        label="Errors / timeouts under load",
        cue=re.compile(r"\b(?:error[- ]rates?|errors|failure[- ]rates?|failures|timeouts?|timed[- ]out|"
                       r"retr(?:y|ies)[- ]rate|retries|dropped[- ](?:requests?|frames?|connections?|messages?)|"
                       r"rate[- ]limit(?:ed|s|ing)?|429s?|5xx|oom[- ]kills?|crash(?:es)?|"
                       r"success[- ]rate|availability|uptime|sla)\b"),
        kinds=CMP),
    "D13": dict(
        label="UI / rendering responsiveness",
        cue=re.compile(r"\b(?:re-?renders?|re-?rendering|renders?|frame[- ](?:drops?|rate|time|budget)|"
                       r"dropped[- ]frames?|jank(?:y)?|stutter(?:s|ing)?|lighthouse|web[- ]vitals|"
                       r"cls|inp|fid|tbt|tti|lcp|fcp|layout[- ]shifts?|paint(?:ing)?|"
                       r"scroll(?:ing)?[- ]performance|smooth(?:ness)?|animation[- ]performance|"
                       r"time[- ]to[- ]interactive|input[- ]delay)\b"),
        kinds={"time", "rate"} | CMP,
        self=[(re.compile(rf"\b{NUM}\s?fps\b"), None, [])]),
    "D14": dict(
        label="Work volume processed",
        cue=re.compile(r"\b(?:rows?[- ](?:scanned|examined|processed|read|returned|fetched)|"
                       r"records?[- ](?:processed|scanned|read)|items?[- ]processed|documents?[- ](?:processed|scanned)|"
                       r"elements?[- ]processed|entries[- ]processed|search[- ]space|candidates?|"
                       r"comparisons|iterations?|passes|steps|nodes?[- ](?:visited|expanded)|"
                       r"operations?|ops|lookups|evaluations|function[- ]evaluations|calls?[- ]made|"
                       r"files?[- ](?:processed|scanned|read))\b"),
        kinds=CMP),
    "D15": dict(
        label="LLM tokens / context",
        cue=re.compile(r"\b(?:tokens?|prompt[- ](?:size|length|tokens)|context[- ](?:length|window|size)|"
                       r"completion[- ]tokens|input[- ]tokens|output[- ]tokens|token[- ](?:usage|count|budget)|"
                       r"tokens?[- ]per[- ](?:request|call|prompt))\b"),
        kinds=CMP,
        self=[(re.compile(rf"\b{NUM}\s?k?\s?tokens\b"), None, [])]),
    "D16": dict(
        label="Cache effectiveness",
        cue=re.compile(r"\b(?:cache[- ]hits?|hit[- ]rates?|hit[- ]ratios?|miss[- ]rates?|cache[- ]miss(?:es)?|"
                       r"evictions?|eviction[- ]rate|cache[- ]efficiency|hit/miss)\b"),
        kinds=CMP),
}

# ---------------------------------------------------------------------------
# Enrichments of the existing list
# ---------------------------------------------------------------------------

SIZE_TABLE_RULE = (re.compile(
    rf"(?:size|bundle|gzip|brotli|minified)[^\n]{{0,400}}\n(?:[^\n]*\n){{0,40}}?[^\n]*\|\s*-?~?\**{NUM}\s?{mp.BYTE_UNIT}\**\s*(?:\(|\|)"), None, [])
SIZE_CUES = r"(?:raw|compressed|uncompressed|gzipped|minified|total|final|output)[- ]size"


# ---------------------------------------------------------------------------
# Configuration plumbing (monkey-patches metric_patterns for one run)
# ---------------------------------------------------------------------------

BASE = dict(
    DIMENSIONS=dict(mp.DIMENSIONS), CUE_PATTERNS=dict(mp.CUE_PATTERNS), QUANT_KINDS=copy.deepcopy(mp.QUANT_KINDS),
    SELF_SUFFICIENT=copy.deepcopy(mp.SELF_SUFFICIENT), MASK_BEFORE=copy.deepcopy(mp.MASK_BEFORE),
    SUPPRESS_IF_NEAR=copy.deepcopy(mp.SUPPRESS_IF_NEAR), FALLBACK_DIMS=set(mp.FALLBACK_DIMS), QUANT_PATTERNS=list(mp.QUANT_PATTERNS),
    normalize=mp.normalize,
)


def apply_config(extra=(), size_tables=False):
    for k, v in BASE.items():
        setattr(mp, k, copy.deepcopy(v) if k != "normalize" else v)
    for cid in extra:
        c = CANDIDATES[cid]
        mp.DIMENSIONS[cid] = c["label"]
        mp.CUE_PATTERNS[cid] = c["cue"]
        mp.QUANT_KINDS[cid] = set(c["kinds"])
        if c.get("self"):
            mp.SELF_SUFFICIENT[cid] = list(c["self"])
        if c.get("fallback"):
            mp.FALLBACK_DIMS.add(cid)
    if size_tables:
        mp.SELF_SUFFICIENT.setdefault("D6", []).append(SIZE_TABLE_RULE)
        mp.CUE_PATTERNS["D6"] = re.compile(mp.CUE_PATTERNS["D6"].pattern[:-3] + "|" + SIZE_CUES + r")\b")


def run(sample, corpus, name, **cfg):
    apply_config(**cfg)
    t0 = time.time()
    rows, snippets = [], []
    new_dims = set(cfg.get("extra", ()))
    for r in sample.itertuples():
        dims, dims_nodiff = set(), set()
        for src, txt in corpus[r.id].items():
            found = em.extract_dimensions(txt)
            dims |= set(found)
            if src != "code_diff":
                dims_nodiff |= set(found)
            for d in found:
                if d in new_dims or cfg.get("size_tables"):
                    m = found[d][0]
                    snippets.append(dict(config=name, dim=d, src=src, author=r.author_type, validated=bool(r.validation_present),
                                         cue=m["cue"], quant=m["quant"], snippet=m["snippet"], url=r.html_url))
        rows.append(dict(id=r.id, author=r.author_type, validated=bool(r.validation_present), vtype=r.validation_type,
                         dims=dims, n=len(dims), n_nodiff=len(dims_nodiff)))
    df = pd.DataFrame(rows)
    print(f"  {name:<18} {time.time() - t0:5.1f}s")
    return df, pd.DataFrame(snippets)


def coverage_row(name, df, base_df=None, new_dims=()):
    v, nv = df[df.validated], df[~df.validated]
    row = {"configuration": name,
           "validated ≥1 dim %": 100 * (v.n > 0).mean(),
           "agent %": 100 * (v[v.author == "AI Agent"].n > 0).mean(),
           "human %": 100 * (v[v.author == "Human"].n > 0).mean(),
           "mean #dims (validated)": v.n.mean(),
           "non-validated ≥1 dim % (noise)": 100 * (nv.n > 0).mean()}
    if base_df is not None:
        merged = v.merge(base_df[["id", "n"]], on="id", suffixes=("", "_base"))
        row["newly covered validated PRs"] = int(((merged.n > 0) & (merged.n_base == 0)).sum())
        row["validated PRs hit by new dims"] = int(v.dims.apply(lambda s: bool(s & set(new_dims))).sum()) if new_dims else ""
        row["non-validated PRs hit by new dims"] = int(nv.dims.apply(lambda s: bool(s & set(new_dims))).sum()) if new_dims else ""
    return row


def main():
    print("[INFO] loading sample + corpus ...")
    sample = em.load_labelled_sample()
    corpus, _, _ = em.build_corpus(sample)

    configs = [("baseline", {})]
    configs += [(f"+{cid} {CANDIDATES[cid]['label']}", {"extra": (cid,)}) for cid in CANDIDATES]
    configs += [("+size-tables (D6)", {"size_tables": True})]
    configs += [("all candidates", {"extra": tuple(CANDIDATES), "size_tables": True})]

    results, all_snips = {}, []
    for name, cfg in configs:
        df, snips = run(sample, corpus, name, **cfg)
        results[name] = (df, cfg)
        if len(snips):
            all_snips.append(snips)
    apply_config()   # restore baseline

    base_df = results["baseline"][0]
    rows = [coverage_row("baseline", base_df)]
    for name, (df, cfg) in results.items():
        if name == "baseline":
            continue
        rows.append(coverage_row(name, df, base_df, cfg.get("extra", ())))
    table = pd.DataFrame(rows).set_index("configuration")

    # per-dimension frequency under "all candidates"
    all_df = results["all candidates"][0]
    v = all_df[all_df.validated]
    dims_all = sorted(set().union(*v.dims), key=lambda d: (len(d), d))
    freq = pd.DataFrame({
        "label": [mp.DIMENSIONS.get(d, CANDIDATES.get(d, {}).get("label", "")) for d in dims_all],
        "agent %": [100 * v[v.author == "AI Agent"].dims.apply(lambda s: d in s).mean() for d in dims_all],
        "human %": [100 * v[v.author == "Human"].dims.apply(lambda s: d in s).mean() for d in dims_all],
        "all %": [100 * v.dims.apply(lambda s: d in s).mean() for d in dims_all],
        "non-validated % (noise)": [100 * all_df[~all_df.validated].dims.apply(lambda s: d in s).mean() for d in dims_all],
    }, index=dims_all)

    snips = pd.concat(all_snips, ignore_index=True) if all_snips else pd.DataFrame()

    lines = ["# Metric-list coverage experiment", "",
             f"Same corpus and extractor settings as Step 1 (window {mp.WINDOW_TOKENS}, exclusion {mp.EXCLUSION_TOKENS}, "
             f"nearest-cue attribution). n = {len(sample)} PRs, {int(sample.validation_present.sum())} validated.", "",
             "## Coverage by configuration", "",
             table.round(1).to_markdown(), "",
             "*newly covered* = validated PRs with 0 dimensions under the baseline that gain ≥ 1 under the configuration; "
             "*noise* = share of PRs **without** RQ2 validation evidence in which the list fires.", "",
             "## Dimension frequency under 'all candidates' (% of validated PRs)", "",
             freq.round(1).to_markdown(), "",
             "## Audit snippets for new dimensions / enrichments", ""]
    if len(snips):
        snips = snips.drop_duplicates(subset=["config", "dim", "url", "src"])
        for (cfgname, d), g in snips.groupby(["config", "dim"], sort=False):
            if cfgname == "all candidates":
                continue
            lines.append(f"### {cfgname} — {d} ({len(g)} PR-source hits; {int(g.validated.sum())} in validated PRs)")
            for _, s in g.head(25).iterrows():
                lines.append(f"- [{s.author[:5]}|{'val' if s.validated else 'NOV'}|{s.src}] `{s.cue}` ⟷ `{s.quant}` — {s.snippet[:150]} ({s.url})")
            lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"\n[INFO] wrote {OUT}\n")
    print(table.round(1).to_string())
    print()
    print(freq.round(1).to_string())


if __name__ == "__main__":
    main()
