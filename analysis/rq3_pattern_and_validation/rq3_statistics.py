"""
RQ3 Step 2 — Mapping optimization patterns to validation evidence and to the
performance metrics reported as that evidence.

Input : data/rq3_pr_level.csv            (from extract_metrics.py)
Output: results/rq3_results.md           (all tables + tests, narrative-free)
        results/tables/*.csv             (every table as CSV)
        results/rq3_tests.csv            (the RQ3 test family with BH-adjusted p)
        results/extremes_*.csv           (PRs at the extremes, for qualitative reading)

Statistics
  * Independence of two categorical variables: chi-square when Cochran's rule
    holds (>= 80 % of expected counts >= 5 and none < 1); otherwise Fisher's
    exact test for 2x2 tables and a Monte-Carlo Fisher–Freeman–Halton test
    (fixed margins, B = 20,000, seeded) for larger tables.  Effect size:
    Cramér's V (odds ratio with Haldane–Anscombe correction for 2x2).
  * Ordinal outcome (number of dimensions): Mann–Whitney U (Cliff's delta) for
    two groups; Kruskal–Wallis (epsilon-squared) across categories.
  * The false-discovery rate across the whole RQ3 test family is controlled
    with Benjamini–Hochberg; raw and adjusted p-values are both reported.

Run from the repo root:
  python RQ3_metric_targeting/rq3_statistics.py
"""

import math
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu, random_table
from scipy.stats import false_discovery_control

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metric_patterns as mp  # noqa: E402

DATA = HERE / "data" / "rq3_pr_level.csv"
METADATA = None
RES = HERE / "results"
TAB = RES / "tables"

DIMS = list(mp.DIMENSIONS)
DIM_LABEL = {d: f"{d} {mp.DIMENSIONS[d]}" for d in DIMS}
AUTHORS = ["AI Agent", "Human"]
VTYPES = ["benchmark", "profiling", "static-analysis", "anecdotal", "none"]
MIN_CATEGORY_N = 10          # categories below this (on the 357) are pooled as "Other" for tests
MIN_DIM_PREVALENCE = 8       # a dimension is tested only if at least this many validated PRs report it
MC_B = 20_000
SEED = 20250911

SHORT = {
    "Algorithm-Level Optimizations": "Algorithm",
    "Build & Compilation & Infrastructure Optimization": "Build/Infra",
    "Code Smells and Structural Simplification": "Code smells",
    "Control-Flow and Branching Optimizations": "Control-flow",
    "Data Structure Selection and Adaptation": "Data structure",
    "I/O and Synchronization": "I/O & sync",
    "Loop Transformations": "Loop",
    "Memory and Data Locality Optimizations": "Memory/locality",
    "Network, Database, and Data Access Optimization": "Network/DB",
    "Other": "Other",
}

TESTS = []      # the RQ3 test family


# ──────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ──────────────────────────────────────────────────────────────────────────────

def cramers_v(chi2, n, r, c):
    k = min(r, c) - 1
    return math.sqrt(chi2 / (n * k)) if n > 0 and k > 0 else float("nan")


def odds_ratio(a, b, c, d):
    """OR with Haldane–Anscombe correction when any cell is zero; 95 % CI."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    orr = (a / b) / (c / d)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return orr, math.exp(math.log(orr) - 1.96 * se), math.exp(math.log(orr) + 1.96 * se)


def independence_test(ct: pd.DataFrame, label: str, stratum: str, family: str):
    """Test independence of the rows and columns of a contingency table."""
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    n = int(ct.values.sum())
    r, c = ct.shape
    out = dict(family=family, test_label=label, stratum=stratum, n=n, shape=f"{r}x{c}")
    if r < 2 or c < 2:
        out.update(test="n/a (degenerate table)", statistic=np.nan, dof=np.nan, p_raw=np.nan,
                   effect_name="Cramér's V", effect=np.nan, note="")
        TESTS.append(out)
        return out
    chi2, p_asym, dof, expected = chi2_contingency(ct.values, correction=False)
    v = cramers_v(chi2, n, r, c)
    cochran = (expected >= 5).mean() >= 0.8 and expected.min() >= 1
    note = f"min expected={expected.min():.2f}"
    if cochran:
        test, p = "chi-square", p_asym
    elif (r, c) == (2, 2):
        test, p = "Fisher's exact", fisher_exact(ct.values)[1]
    else:
        rng = np.random.default_rng(SEED)
        sims = random_table(ct.sum(axis=1).values, ct.sum(axis=0).values, seed=rng).rvs(MC_B)
        stat_sim = ((sims - expected) ** 2 / expected).sum(axis=(1, 2))
        p = (1 + (stat_sim >= chi2 - 1e-9).sum()) / (MC_B + 1)
        test = f"Conditional Pearson chi-square (Monte Carlo, B={MC_B:,})"
    effect_name, effect = "Cramér's V", v
    if (r, c) == (2, 2):
        a, b = ct.values[0]
        cc, d = ct.values[1]
        orr, lo, hi = odds_ratio(a, b, cc, d)
        note += f"; OR={orr:.2f} [{lo:.2f}, {hi:.2f}]"
    out.update(test=test, statistic=chi2, dof=dof, p_raw=p, effect_name=effect_name, effect=effect, note=note)
    TESTS.append(out)
    w(test_line(out))
    return out


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def mwu_test(a, b, label, stratum, family, names=("A", "B")):
    a, b = pd.Series(a).dropna().values, pd.Series(b).dropna().values
    out = dict(family=family, test_label=label, stratum=stratum, n=len(a) + len(b),
               shape=f"{names[0]} n={len(a)}, {names[1]} n={len(b)}")
    if len(a) < 2 or len(b) < 2:
        out.update(test="Mann–Whitney U (skipped: n<2)", statistic=np.nan, dof=np.nan, p_raw=np.nan,
                   effect_name="Cliff's delta", effect=np.nan, note="")
        TESTS.append(out)
        return out
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    d = cliffs_delta(a, b)
    note = (f"median {names[0]}={np.median(a):.2f} (mean {a.mean():.2f}); "
            f"median {names[1]}={np.median(b):.2f} (mean {b.mean():.2f})")
    out.update(test="Mann–Whitney U", statistic=u, dof=np.nan, p_raw=p,
               effect_name="Cliff's delta", effect=d, note=note)
    TESTS.append(out)
    w(test_line(out))
    return out


def kw_test(groups: dict, label, stratum, family):
    groups = {k: pd.Series(v).dropna().values for k, v in groups.items() if len(pd.Series(v).dropna()) >= 2}
    n = sum(len(v) for v in groups.values())
    out = dict(family=family, test_label=label, stratum=stratum, n=n, shape=f"k={len(groups)}")
    if len(groups) < 2:
        out.update(test="Kruskal–Wallis (skipped)", statistic=np.nan, dof=np.nan, p_raw=np.nan,
                   effect_name="epsilon²", effect=np.nan, note="")
        TESTS.append(out)
        return out
    if len(np.unique(np.concatenate(list(groups.values())))) == 1:
        h, p = 0.0, 1.0
    else:
        h, p = kruskal(*groups.values())
    k = len(groups)
    eps2 = (h - k + 1) / (n - k) if n > k else float("nan")
    note = "; ".join(f"{SHORT.get(g, g)} md={np.median(v):.1f}" for g, v in groups.items())
    out.update(test="Kruskal–Wallis", statistic=h, dof=k - 1, p_raw=p, effect_name="epsilon²",
               effect=eps2, note=note)
    TESTS.append(out)
    w(test_line(out))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Markdown helpers
# ──────────────────────────────────────────────────────────────────────────────

LINES = []


def w(*args):
    LINES.extend(args)
    LINES.append("")


def md(df: pd.DataFrame, floatfmt="{:.2f}", index=True) -> str:
    d = df.copy()
    if index:
        d = d.reset_index()
    cols = list(d.columns)
    rows = []
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        rows.append(cells)
    head = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([head, sep, body])


def save(df: pd.DataFrame, name: str):
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv")


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def pct(num, den, nd=1):
    """Share of a group as a percentage. Groups differ in size (280 agent vs 77 human PRs),
    so cells report percentages only; the group size is given in an `n` column/row."""
    return f"{100 * num / den:.{nd}f}%" if den else "–"


def col_pct(ct: pd.DataFrame, nd=1) -> pd.DataFrame:
    """Column-normalised percentages with an `n` row appended."""
    n = ct.sum(axis=0)
    out = (100 * ct / n.replace(0, np.nan)).round(nd).astype(object)
    out.loc["n"] = [int(v) for v in n.values]
    return out


def row_pct(ct: pd.DataFrame, nd=1) -> pd.DataFrame:
    """Row-normalised percentages with an `n` column appended."""
    n = ct.sum(axis=1)
    out = (100 * ct.div(n.replace(0, np.nan), axis=0)).round(nd)
    out["n"] = n
    return out


def test_line(t):
    eff = "" if np.isnan(t["effect"]) else f", {t['effect_name']}={t['effect']:.3f}"
    st = "" if np.isnan(t["statistic"]) else f", stat={t['statistic']:.2f}"
    return f"- **{t['test_label']}** [{t['stratum']}], n={t['n']}: {t['test']}{st}, p={fmt_p(t['p_raw'])}{eff} ({t['note']})"


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────

def load():
    df = pd.read_csv(DATA)
    if METADATA is not None:
        metadata = pd.read_csv(METADATA)
        extra = [c for c in metadata if c not in df or c in ("repo_id", "number")]
        df = df.merge(metadata[extra], on=["repo_id", "number"], how="left", validate="one_to_one")
        if df["is_merged"].isna().any():
            raise ValueError("Missing archived outcome metadata for metric labels")
    if "author_type" not in df and "sample_arm" in df:
        df["author_type"] = df["sample_arm"]
    for d in DIMS:
        df[d] = df[d].astype(bool)
        df[f"{d}_nodiff"] = df[f"{d}_nodiff"].astype(bool)
    df["is_merged"] = df["is_merged"].astype(bool)
    df["validation_present"] = df["validation_present"].astype(bool)
    df["validation_type"] = df["validation_type"].fillna("none")
    if "in_type_layer" not in df:
        df["in_type_layer"] = df.in_metric_layer & ~df.validation_type.isin(["none", "unresolved"])
    counts = df["pattern"].value_counts()
    rare = counts[counts < MIN_CATEGORY_N].index
    df["category"] = df["pattern"].replace({p: "Other" for p in rare})
    df["cat"] = df["category"].map(SHORT)
    df["any_dim"] = df["n_dims"] > 0
    return df, list(rare)


def section_sample(df):
    w("## 0. Analytic sample")
    val = df[df.validation_present]
    t = pd.DataFrame({
        "PRs (category layer)": df.groupby("author_type").size(),
        "with validation evidence": val.groupby("author_type").size(),
        "metric layer (all positive)": df[df.in_metric_layer].groupby("author_type").size(),
        "resolved type layer": df[df.in_type_layer].groupby("author_type").size(),
    })
    t.loc["All"] = t.sum()
    w(md(t, "{:.0f}"))
    w(f"Extractor settings: window = {mp.WINDOW_TOKENS} tokens; exclusion window = {mp.EXCLUSION_TOKENS} tokens; "
      f"nearest-cue attribution = {mp.NEAREST_CUE_WINS}. Categories with n < {MIN_CATEGORY_N} on the {len(df)} PRs are pooled as "
      f"'Other' for inferential tests only.")
    vt = pd.crosstab(df["validation_type"], df["author_type"]).reindex(VTYPES).fillna(0).astype(int)
    vt["All"] = vt.sum(axis=1)
    vt = col_pct(vt)
    w(f"Validation type (RQ2) by author type, % of the author's PRs ({len(df)} PRs):", md(vt, "{:.1f}"))
    save(t, "T0_sample"); save(vt, "T0_validation_type_by_author")


def section_A(df):
    w(f"## 1. Optimization category × validation evidence (n = {len(df)})")
    # ---- A1 presence
    rows = []
    for cat, g in df.groupby("pattern"):
        r = {"Category": cat, "n": len(g)}
        for a in AUTHORS:
            ga = g[g.author_type == a]
            r[f"{a} n"] = len(ga)
            r[f"{a} validated"] = pct(int(ga.validation_present.sum()), len(ga))
        r["All validated"] = pct(int(g.validation_present.sum()), len(g))
        rows.append(r)
    A1 = pd.DataFrame(rows).set_index("Category").sort_values("n", ascending=False)
    w("### 1.1 Validation presence by category (% of the category's PRs for that author)", md(A1, index=True))
    save(A1, "T1_1_category_x_validation_presence")

    for stratum, g in [("pooled", df)] + [(a, df[df.author_type == a]) for a in AUTHORS]:
        ct = pd.crosstab(g["category"], g["validation_present"])
        independence_test(ct, "Category × validation present", stratum, "A. category×validation")
    # per-category author difference in validation rate
    w("Validation rate by sample arm within each category (adaptive contingency test):")
    rows = []
    for cat, g in df.groupby("category"):
        ct = pd.crosstab(g["author_type"], g["validation_present"]).reindex(index=AUTHORS, columns=[True, False]).fillna(0).astype(int)
        t = independence_test(ct, f"Author × validation present — {SHORT[cat]}", "within category", "A. category×validation")
        rows.append({"Category": SHORT[cat], f"{AUTHORS[0]} validated": pct(int(ct.loc[AUTHORS[0], True]), int(ct.loc[AUTHORS[0]].sum())),
                     f"{AUTHORS[1]} validated": pct(int(ct.loc[AUTHORS[1], True]), int(ct.loc[AUTHORS[1]].sum())),
                     "test": t["test"], "p (raw)": fmt_p(t["p_raw"]), "note": t["note"]})
    A1b = pd.DataFrame(rows).set_index("Category")
    w(md(A1b)); save(A1b, "T1_1b_author_x_validation_within_category")

    # ---- A2 type
    w("### 1.2 Validation type by category (% of the category's PRs; 'none' = no validation evidence)")
    for a in AUTHORS + ["All"]:
        g = df if a == "All" else df[df.author_type == a]
        ct = pd.crosstab(g["pattern"], g["validation_type"]).reindex(columns=VTYPES).fillna(0).astype(int)
        ct = row_pct(ct).sort_values("n", ascending=False)
        w(f"**{a}**", md(ct, "{:.1f}"))
        save(ct, f"T1_2_category_x_validation_type_{a.replace(' ', '_')}")
    val = df[df.in_type_layer]
    for stratum, g in [("pooled", val)] + [(a, val[val.author_type == a]) for a in AUTHORS]:
        ct = pd.crosstab(g["category"], g["validation_type"])
        independence_test(ct, "Category × validation type (validated PRs)", stratum, "A. category×validation")
    ct = pd.crosstab(val["category"], val["validation_type"] == "benchmark")
    independence_test(ct, "Category × benchmark-based vs other evidence (validated PRs)", "pooled", "A. category×validation")


def profile_table(g, label):
    """Category × D0–D9 incidence (% of validated PRs in the category)."""
    rows = []
    for cat, gg in g.groupby("pattern"):
        r = {"Category": cat, "n validated": len(gg)}
        for d in DIMS:
            r[d] = pct(int(gg[d].sum()), len(gg), 0)
        r["≥1 dim"] = pct(int(gg.any_dim.sum()), len(gg))
        r["mean #dims"] = f"{gg.n_dims.mean():.2f}"
        rows.append(r)
    t = pd.DataFrame(rows).set_index("Category").sort_values("n validated", ascending=False)
    tot = {"n validated": len(g), "≥1 dim": pct(int(g.any_dim.sum()), len(g)), "mean #dims": f"{g.n_dims.mean():.2f}"}
    for d in DIMS:
        tot[d] = pct(int(g[d].sum()), len(g), 0)
    t.loc["All"] = pd.Series(tot)
    w(f"**{label}** — % of the category's validated PRs reporting each dimension", md(t))
    return t


def section_B(df):
    val = df[df.in_metric_layer].copy()
    w("## 2. Metric profile: category × reported dimensions (validated PRs, n = %d)" % len(val))
    w("Dimensions: " + "; ".join(f"**{d}** {mp.DIMENSIONS[d]}" for d in DIMS)
      + ". D0 records a quantified gain whose dimension is not named and is credited only when no D1–D9 cue is in the claim's window.")
    for a in AUTHORS + ["All"]:
        g = val if a == "All" else val[val.author_type == a]
        t = profile_table(g, f"{a} (n = {len(g)})")
        save(t, f"T2_1_metric_profile_{a.replace(' ', '_')}")

    # tests: category × Dk for prevalent dimensions; author × Dk
    w("### 2.2 Tests on the metric profile")
    for d in DIMS:
        if val[d].sum() >= MIN_DIM_PREVALENCE:
            independence_test(pd.crosstab(val["category"], val[d]), f"Category × {d} reported", "pooled (validated)", "B. metric profile")
    for d in DIMS:
        if val[d].sum() >= 5:
            ct = pd.crosstab(val["author_type"], val[d]).reindex(index=AUTHORS, columns=[True, False]).fillna(0).astype(int)
            independence_test(ct, f"Author × {d} reported", "validated", "B. metric profile")
    independence_test(pd.crosstab(val["author_type"], val["any_dim"]), "Author × any dimension reported", "validated", "B. metric profile")
    independence_test(pd.crosstab(val["category"], val["any_dim"]), "Category × any dimension reported", "pooled (validated)", "B. metric profile")

    # dimensionality
    w("### 2.3 Dimensionality of the evidence (number of distinct dimensions per validated PR)")
    dist = pd.crosstab(val["n_dims"], val["author_type"]).reindex(columns=AUTHORS).fillna(0).astype(int)
    dist["All"] = dist.sum(axis=1)
    dist = col_pct(dist)
    w("% of the author's validated PRs (counting D0):", md(dist, "{:.1f}")); save(dist, "T2_3_dimensionality_distribution")
    d0_only = val[(val.n_dims > 0) & (val.n_dims_specific == 0)]
    w("PRs whose only reported 'dimension' is **D0** (a quantified gain with no named dimension): "
      + ", ".join(f"{a}: {pct(int((d0_only.author_type == a).sum()), int((val.author_type == a).sum()))}" for a in AUTHORS)
      + f"; all: {pct(len(d0_only), len(val))}. Excluding D0, the share of validated PRs with ≥1 named dimension is "
      + ", ".join(f"{a}: {pct(int((val[val.author_type == a].n_dims_specific > 0).sum()), int((val.author_type == a).sum()))}" for a in AUTHORS)
      + f"; all: {pct(int((val.n_dims_specific > 0).sum()), len(val))}.")
    mwu_test(val[val.author_type == AUTHORS[0]].n_dims, val[val.author_type == AUTHORS[1]].n_dims,
             "#dims: agent vs human", "validated", "B. metric profile", ("agent", "human"))
    kw_test({c: g.n_dims for c, g in val.groupby("category")}, "#dims across categories", "pooled (validated)", "B. metric profile")
    for a in AUTHORS:
        ga = val[val.author_type == a]
        kw_test({c: g.n_dims for c, g in ga.groupby("category")}, "#dims across categories", a, "B. metric profile")
    by_cat = val.groupby(["pattern", "author_type"])["n_dims"].agg(["count", "mean", "median", "max"]).round(2)
    w("Per category and author:", md(by_cat)); save(by_cat, "T2_3_dimensionality_by_category_author")

    # metric profile × validation type
    typed = val[val.in_type_layer]
    w(f"### 2.4 Metric profile × validation type (resolved positive types, n = {len(typed)})")
    rows = []
    for vt, g in typed.groupby("validation_type"):
        r = {"Validation type": vt, "n": len(g), "≥1 dim": pct(int(g.any_dim.sum()), len(g)), "mean #dims": f"{g.n_dims.mean():.2f}"}
        for d in DIMS:
            r[d] = pct(int(g[d].sum()), len(g), 0)
        rows.append(r)
    t = pd.DataFrame(rows).set_index("Validation type").reindex([v for v in VTYPES if v not in {"none", "unresolved"}])
    w("% of PRs with that validation type reporting each dimension:", md(t)); save(t, "T2_4_metric_profile_by_validation_type")
    bench = typed[typed.validation_type == "benchmark"]
    other = typed[typed.validation_type != "benchmark"]
    mwu_test(bench.n_dims, other.n_dims, "#dims: benchmark vs other evidence", "resolved types", "B. metric profile", ("benchmark", "other"))
    independence_test(pd.crosstab(typed["validation_type"] == "benchmark", typed["any_dim"]),
                       "Benchmark evidence × any dimension reported", "resolved types", "B. metric profile")
    for d in DIMS:
        if typed[d].sum() >= 5:
            independence_test(pd.crosstab(typed["validation_type"] == "benchmark", typed[d]),
                               f"Benchmark evidence × {d} reported", "resolved types", "B. metric profile")
    return val


def section_agent(df):
    """2.5 — metric reporting by author type and by individual agent, under three denominators."""
    w("### 2.5 Metric reporting by author type and by agent")
    w("Rates are % of the row's PRs. *all PRs* = the category layer; the metric layer contains all positive validation consensuses, including unresolved types; "
      "*benchmark* = PRs whose evidence is a benchmark; *no diff* = dimension flags recomputed with the code diff excluded. "
      "The *all PRs* quantification columns count only PRs that are validated **and** report a dimension, i.e. they are "
      "the product of the validation rate and the conditional quantification rate; PRs with a quantitative claim but no "
      "RQ2 validation label (`extremes_metric_without_validation_label.csv`) are not counted. "
      "Per-agent rows are descriptive and are not part of the RQ3 test family.")
    agents = df[df.author_type == AUTHORS[0]]
    groups = [(AUTHORS[1], df[df.author_type == AUTHORS[1]]), (f"{AUTHORS[0]} (all)", agents)]
    by_agent = sorted(agents.groupby("agent"), key=lambda kv: -len(kv[1]))
    groups += [(f"  {name}", g) for name, g in by_agent]
    rows = []
    for name, g in groups:
        val = g[g.in_metric_layer]
        bench = g[g.in_type_layer & g.validation_type.eq("benchmark")]
        rows.append({
            "author": name,
            "n (all PRs)": len(g),
            "validation present": pct(int(g.validation_present.sum()), len(g)),
            "metric layer": pct(len(val), len(g)),
            "n resolved type layer": int(g.in_type_layer.sum()),
            "benchmark evidence": pct(len(bench), len(g)),
            "metric layer & ≥1 dim, all PRs": pct(val.any_dim.sum(), len(g)),
            "metric layer & ≥1 dim, all PRs (no diff)": pct((val.n_dims_nodiff > 0).sum(), len(g)),
            "n metric layer": len(val),
            "≥1 dim, metric layer": pct(val.any_dim.sum(), len(val)),
            "mean #dims, metric layer": round(float(val.n_dims.mean()), 2) if len(val) else np.nan,
            "n benchmark": len(bench),
            "≥1 dim, benchmark": pct(bench.any_dim.sum(), len(bench)),
        })
    t = pd.DataFrame(rows).set_index("author")
    w(md(t, "{:.2f}"))
    save(t, "T2_5_metric_reporting_by_agent")
    return t


def section_merge_extremes(df, val):
    # 3 merge status
    w("## 3. Merge status")
    for stratum, g in [("pooled", val)] + [(a, val[val.author_type == a]) for a in AUTHORS]:
        mwu_test(g[g.is_merged].n_dims, g[~g.is_merged].n_dims, "#dims: merged vs not merged", f"{stratum} (validated)", "C. merge", ("merged", "not merged"))
        independence_test(pd.crosstab(g["is_merged"], g["any_dim"]), "Merged × any dimension reported", f"{stratum} (validated)", "C. merge")
    rows = []
    for (a, m), g in val.groupby(["author_type", "is_merged"]):
        rows.append({"Author": a, "Merged": m, "n": len(g), "≥1 dim": pct(int(g.any_dim.sum()), len(g)),
                     "mean #dims": f"{g.n_dims.mean():.2f}", "median #dims": f"{g.n_dims.median():.0f}",
                     "median time-to-merge (days)": f"{g.time_to_merge_days.median():.2f}" if m else ""})
    t = pd.DataFrame(rows).set_index(["Author", "Merged"])
    w(md(t)); save(t, "T3_merge_x_dimensionality")

    # 4 extremes
    w("## 4. PRs at the extremes (exported for qualitative reading)")
    cols = ["id", "html_url", "author_type", "agent", "pattern", "sub_pattern", "validation_type", "is_merged", "n_dims", "dims"]
    top = val.sort_values("n_dims", ascending=False).head(15)[cols]
    top.to_csv(RES / "extremes_top_dimensionality.csv", index=False)
    w(f"Highest dimensionality (top 15) → `results/extremes_top_dimensionality.csv`", md(top[["html_url", "author_type", "pattern", "validation_type", "n_dims", "dims"]], index=False))
    zero = val[val.n_dims == 0][cols]
    zero.to_csv(RES / "extremes_validated_no_metric.csv", index=False)
    z = (val.assign(none=val.n_dims == 0).groupby(["validation_type", "author_type"])["none"].mean() * 100).round(1).unstack()
    z["n (validated PRs)"] = val.groupby("validation_type").size()
    w(f"Validated PRs reporting **no** metric dimension ({pct(len(zero), len(val))} overall), as % of each validation type "
      f"→ `results/extremes_validated_no_metric.csv`", md(z, "{:.1f}"))
    leak = df[(~df.validation_present) & (df.n_dims > 0)][cols]
    leak.to_csv(RES / "extremes_metric_without_validation_label.csv", index=False)
    w(f"PRs *without* an RQ2 validation label in which the extractor still found a quantitative metric claim (n = {len(leak)}; "
      f"candidates for an RQ2 label recheck) → `results/extremes_metric_without_validation_label.csv`",
      md(leak[["html_url", "author_type", "pattern", "n_dims", "dims"]], index=False))
    return val


def section_sensitivity(val):
    w("## 5. Sensitivity: excluding the code diff from the corpus")
    rows = []
    for a in AUTHORS + ["All"]:
        g = val if a == "All" else val[val.author_type == a]
        for d in DIMS:
            rows.append({"Author": a, "n": len(g), "Dimension": d, "with diff": pct(int(g[d].sum()), len(g)),
                         "without diff": pct(int(g[f'{d}_nodiff'].sum()), len(g))})
        rows.append({"Author": a, "n": len(g), "Dimension": "≥1 dim", "with diff": pct(int((g.n_dims > 0).sum()), len(g)),
                     "without diff": pct(int((g.n_dims_nodiff > 0).sum()), len(g))})
    t = pd.DataFrame(rows).set_index(["Author", "n", "Dimension"])
    w("% of the author's validated PRs:", md(t)); save(t, "T5_sensitivity_diff")
    changed = (val.n_dims != val.n_dims_nodiff).sum()
    w(f"PRs whose dimension count changes when the diff is excluded: {changed} of {len(val)}.")
    w(f"Dimensions found in the description alone: {int((val.n_dims_description_only > 0).sum())} PRs have ≥1.")
    if "dims_from_bot_comments" in val:
        w(f"Bot-authored issue comments contribute a dimension in {int((val.dims_from_bot_comments.fillna('') != '').sum())} PRs.")


def section_tests():
    t = pd.DataFrame(TESTS)
    valid = t["p_raw"].notna()
    t["p_bh"] = np.nan
    if valid.any():
        p_adj = false_discovery_control(t.loc[valid, "p_raw"].values, method="bh")
        rej = p_adj < 0.05
        t.loc[valid, "p_bh"] = p_adj
        t.loc[valid, "significant_bh"] = rej
    t.to_csv(RES / "rq3_tests.csv", index=False)
    w("## 6. RQ3 test family (Benjamini–Hochberg across all %d tests)" % int(valid.sum()))
    show = t[["family", "test_label", "stratum", "n", "shape", "test", "statistic", "dof", "p_raw", "p_bh", "effect_name", "effect", "note"]].copy()
    show["p_raw"] = show["p_raw"].apply(fmt_p)
    show["p_bh"] = show["p_bh"].apply(fmt_p)
    show["statistic"] = show["statistic"].round(2)
    show["effect"] = show["effect"].round(3)
    w(md(show, index=False))
    sig = t[t.get("significant_bh", pd.Series(False, index=t.index)) == True]  # noqa: E712
    w("**Significant after BH (q < 0.05):**" if len(sig) else "**No test remains significant after BH correction.**")
    for _, r in sig.iterrows():
        w(test_line(r) + f" → q={fmt_p(r['p_bh'])}")


def main():
    TESTS.clear(); LINES.clear()
    RES.mkdir(parents=True, exist_ok=True); TAB.mkdir(exist_ok=True)
    df, rare = load()
    w("Metric profiles use all positive validation consensuses, including unresolved types. "
      "Only comparisons involving validation type require a resolved positive type; "
      "unresolved types are never treated as non-benchmark evidence. "
      "Tests are PR-level exploratory associations.")
    w("# RQ3 — Optimization, validation, and reported metrics",
      f"Source: `{DATA}` (n = {len(df)} PRs; {int(df.in_metric_layer.sum())} in the positive metric layer; {int(df.in_type_layer.sum())} in the resolved type layer). "
      f"Rare categories pooled as 'Other' for tests: {', '.join(rare) or 'none'}.")
    section_sample(df)
    section_A(df)
    val = section_B(df)
    section_agent(df)
    val = section_merge_extremes(df, val)
    section_sensitivity(val)
    section_tests()
    (RES / "rq3_results.md").write_text("\n".join(LINES))
    print(f"[INFO] wrote {RES / 'rq3_results.md'} ({len(TESTS)} tests), tables -> {TAB}")
    tdf = pd.read_csv(RES / "rq3_tests.csv")
    print(tdf[["test_label", "stratum", "test", "p_raw", "p_bh", "effect_name", "effect"]].to_string(max_colwidth=60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--metadata", type=Path, help="Compact archived metadata supplement for the public label export.")
    parser.add_argument("--output-dir", type=Path, default=RES)
    parser.add_argument("--current", action="store_true")
    args = parser.parse_args()
    DATA, METADATA, RES = args.data, args.metadata, args.output_dir
    TAB = RES / "tables"
    if args.current:
        AUTHORS = ["agentic", "human_candidate"]
        VTYPES = ["benchmark", "profiling", "static-reasoning", "anecdotal", "none", "unresolved"]
    main()
