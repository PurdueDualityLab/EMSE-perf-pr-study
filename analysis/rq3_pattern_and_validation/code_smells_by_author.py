"""RQ3 post-hoc — structural cleanups vs actual performance changes, by author.

Contrast the "Code Smells and Structural Simplification" category (structural
cleanups) with the eight remaining RQ1 categories (actual performance changes)
on three outcomes, separately for agentic and human PRs:

  (1) validation presence      — category layer, all analytic PRs (n = 2081)
  (2) any dimension reported   — metric layer, validated PRs (n = 1699)
  (3) number of dimensions     — metric layer (mean, median, Cliff's delta)

and test whether the Code-smells gap differs between the two arms (category x
author interaction). A combined outcome, validated AND >= 1 dimension on the
category layer, folds the two selection steps into one unconditional rate. A
final section checks whether Code smells PRs report *different* metrics or
merely *fewer*: each D0-D9 unconditionally and conditional on >= 1 dimension.

Statistics
  * 2x2 tables: chi-square when Cochran's rule holds, else Fisher's exact;
    odds ratio with Haldane-Anscombe correction and 95% CI.
  * Counts: Mann-Whitney U with Cliff's delta.
  * Interaction, binary outcomes: logistic regression outcome ~ cs + agent +
    cs:agent, likelihood-ratio test on the interaction; the Wald z on the
    difference of log odds ratios (= the saturated-model interaction Wald test)
    is reported alongside with the ratio of odds ratios and its CI.
  * Interaction, #dims: difference of Cliff's delta between arms, permutation
    test shuffling arm labels within each category group (B = 20,000, seeded),
    bootstrap 95% CI (5,000 resamples within the four cells), and a
    negative-binomial regression n_dims ~ cs + agent + cs:agent with an LRT as
    a parametric cross-check.
  * Benjamini-Hochberg across the pre-specified family: the 8 within-arm
    contrasts, the 8 author contrasts within category group, and the 4
    interaction tests (20 tests). The per-dimension, pairwise, and
    negative-binomial cross-check tests are descriptive follow-ups and are
    reported with raw p only (p_bh is empty for them).

Usage (from the repository root):
    python analysis/rq3_pattern_and_validation/code_smells_by_author.py \
        [--labels analysis/classification_labels/rq3_labels.csv] \
        [--output-dir analysis/rq3_pattern_and_validation/results]
"""

import argparse
import math
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import (chi2, chi2_contingency, false_discovery_control, fisher_exact,
                         mannwhitneyu, norm, rankdata)
import statsmodels.api as sm
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
CS = "Code Smells and Structural Simplification"
DIMS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D0"]
DIM_NAME = {"D1": "latency/exec time", "D2": "throughput", "D3": "memory", "D4": "CPU work",
            "D5": "I/O & network", "D6": "artifact size", "D7": "build/CI time",
            "D8": "energy & cost", "D9": "scalability/concurrency", "D0": "unspecified perf"}
ARMS = [("agentic", "Agent"), ("human_candidate", "Human")]
GROUPS = ["Code smells", "Other"]
SEED = 20250911
B_PERM = 20_000
B_BOOT = 5_000
PRESPECIFIED = {"within-arm", "within-group", "interaction"}   # the BH family

TESTS = []


# ── helpers ──────────────────────────────────────────────────────────────────

def fmt_p(p):
    return "n/a" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def odds_ratio(a, b, c, d):
    if min(a, b, c, d) == 0:
        a, b, c, d = a + .5, b + .5, c + .5, d + .5
    lor = math.log((a / b) / (c / d))
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return math.exp(lor), math.exp(lor - 1.96 * se), math.exp(lor + 1.96 * se), lor, se


def test_2x2(x1, x0, label, stratum, family):
    """Outcome rate in group 1 vs group 0 (boolean arrays)."""
    x1, x0 = np.asarray(x1, bool), np.asarray(x0, bool)
    a, b = int(x1.sum()), int(len(x1) - x1.sum())
    c, d = int(x0.sum()), int(len(x0) - x0.sum())
    ct = np.array([[a, b], [c, d]])
    o, lo, hi, lor, se = odds_ratio(a, b, c, d)
    res = dict(family=family, test_label=label, stratum=stratum, n=a + b + c + d,
               k1=a, n1=a + b, k0=c, n0=c + d, p1=a / (a + b), p0=c / (c + d),
               OR=o, OR_lo=lo, OR_hi=hi, lor=lor, se=se)
    if ct.sum(axis=0).min() == 0:
        res.update(test="n/a (degenerate)", statistic=np.nan, p_raw=np.nan,
                   effect_name="OR", effect=o, note="")
    else:
        stat, p, _, exp = chi2_contingency(ct, correction=False)
        if (exp >= 5).mean() >= 0.8 and exp.min() >= 1:
            test = "chi-square"
        else:
            test, p = "Fisher's exact", fisher_exact(ct)[1]
        res.update(test=test, statistic=stat, p_raw=p, effect_name="OR", effect=o,
                   note=f"min expected={exp.min():.2f}; OR={o:.2f} [{lo:.2f}, {hi:.2f}]")
    TESTS.append(res)
    return res


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    x = np.concatenate([a, b])
    r = rankdata(x)
    u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return 2 * u / (len(a) * len(b)) - 1


def test_mwu(a, b, label, stratum, family):
    a, b = np.asarray(a), np.asarray(b)
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    delta = cliffs_delta(a, b)
    res = dict(family=family, test_label=label, stratum=stratum, n=len(a) + len(b),
               n1=len(a), n0=len(b), m1=a.mean(), m0=b.mean(), md1=float(np.median(a)),
               md0=float(np.median(b)), test="Mann-Whitney U", statistic=u, p_raw=p,
               effect_name="Cliff's delta", effect=delta,
               note=f"group1 mean={a.mean():.2f} md={np.median(a):.0f}; "
                    f"group0 mean={b.mean():.2f} md={np.median(b):.0f}")
    TESTS.append(res)
    return res


def lrt_interaction(formula_full, formula_red, data, fit):
    full, red = fit(formula_full, data), fit(formula_red, data)
    lr = 2 * (full.llf - red.llf)
    return lr, 1 - chi2.cdf(lr, 1), full


# ── analysis ─────────────────────────────────────────────────────────────────

def load(labels_path):
    df = pd.read_csv(labels_path)
    df["cs"] = (df.pattern == CS).astype(int)
    df["agent"] = (df.sample_arm == "agentic").astype(int)
    df["grp"] = np.where(df.cs == 1, "Code smells", "Other")
    df["arm"] = np.where(df.agent == 1, "Agent", "Human")
    df["val"] = df.validation_present.astype(int)
    df["any_dim"] = (df.n_dims >= 1).astype(int)
    df["quant"] = ((df.val == 1) & (df.n_dims >= 1)).astype(int)
    return df


def descriptives(df, ml):
    rows = []
    for _, arm in ARMS:
        for g in GROUPS:
            c = df[(df.arm == arm) & (df.grp == g)]
            m = ml[(ml.arm == arm) & (ml.grp == g)]
            rows.append({
                "arm": arm, "group": g, "n_category_layer": len(c),
                "validated_n": int(c.val.sum()), "validated_pct": round(100 * c.val.mean(), 1),
                "n_metric_layer": len(m),
                "any_dim_n": int(m.any_dim.sum()), "any_dim_pct": round(100 * m.any_dim.mean(), 1),
                "mean_dims": round(m.n_dims.mean(), 3), "median_dims": float(m.n_dims.median()),
                "q1_dims": float(m.n_dims.quantile(.25)), "q3_dims": float(m.n_dims.quantile(.75)),
                "quantified_n": int(c.quant.sum()), "quantified_pct": round(100 * c.quant.mean(), 1),
            })
    return pd.DataFrame(rows)


def within_arm(df, ml):
    """Code smells vs Other within each arm."""
    rows, store = [], {}
    for _, arm in ARMS:
        c, m = df[df.arm == arm], ml[ml.arm == arm]
        r = test_2x2(c.loc[c.cs == 1, "val"], c.loc[c.cs == 0, "val"],
                     "Code smells vs Other: validation present", arm, "within-arm")
        store[("val", arm)] = r
        rows.append(dict(arm=arm, outcome="validation present", code_smells=f"{100*r['p1']:.1f}%",
                         other=f"{100*r['p0']:.1f}%", effect=f"OR {r['OR']:.2f} [{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]",
                         test=r["test"], p_raw=r["p_raw"]))
        r = test_2x2(m.loc[m.cs == 1, "any_dim"], m.loc[m.cs == 0, "any_dim"],
                     "Code smells vs Other: any dimension (validated)", arm, "within-arm")
        store[("any", arm)] = r
        rows.append(dict(arm=arm, outcome="any dimension (validated)", code_smells=f"{100*r['p1']:.1f}%",
                         other=f"{100*r['p0']:.1f}%", effect=f"OR {r['OR']:.2f} [{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]",
                         test=r["test"], p_raw=r["p_raw"]))
        r = test_mwu(m.loc[m.cs == 1, "n_dims"], m.loc[m.cs == 0, "n_dims"],
                     "Code smells vs Other: #dims (validated)", arm, "within-arm")
        store[("nd", arm)] = r
        rows.append(dict(arm=arm, outcome="#dims (validated)",
                         code_smells=f"mean {r['m1']:.2f}, md {r['md1']:.0f}",
                         other=f"mean {r['m0']:.2f}, md {r['md0']:.0f}",
                         effect=f"Cliff's δ {r['effect']:+.3f}; Δmean {r['m1']-r['m0']:+.2f}",
                         test=r["test"], p_raw=r["p_raw"]))
        r = test_2x2(c.loc[c.cs == 1, "quant"], c.loc[c.cs == 0, "quant"],
                     "Code smells vs Other: validated and >=1 dim (category layer)", arm, "within-arm")
        store[("quant", arm)] = r
        rows.append(dict(arm=arm, outcome="validated and ≥1 dim (category layer)",
                         code_smells=f"{100*r['p1']:.1f}%", other=f"{100*r['p0']:.1f}%",
                         effect=f"OR {r['OR']:.2f} [{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]",
                         test=r["test"], p_raw=r["p_raw"]))
    return pd.DataFrame(rows), store


def within_group(df, ml):
    """Agent vs Human within each category group."""
    rows = []
    for g in GROUPS:
        c, m = df[df.grp == g], ml[ml.grp == g]
        r = test_2x2(c.loc[c.agent == 1, "val"], c.loc[c.agent == 0, "val"],
                     "Agent vs Human: validation present", g, "within-group")
        rows.append(dict(group=g, outcome="validation present", agent=f"{100*r['p1']:.1f}%",
                         human=f"{100*r['p0']:.1f}%", effect=f"OR {r['OR']:.2f} [{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]",
                         test=r["test"], p_raw=r["p_raw"]))
        r = test_2x2(m.loc[m.agent == 1, "any_dim"], m.loc[m.agent == 0, "any_dim"],
                     "Agent vs Human: any dimension (validated)", g, "within-group")
        rows.append(dict(group=g, outcome="any dimension (validated)", agent=f"{100*r['p1']:.1f}%",
                         human=f"{100*r['p0']:.1f}%", effect=f"OR {r['OR']:.2f} [{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]",
                         test=r["test"], p_raw=r["p_raw"]))
        r = test_mwu(m.loc[m.agent == 1, "n_dims"], m.loc[m.agent == 0, "n_dims"],
                     "Agent vs Human: #dims (validated)", g, "within-group")
        rows.append(dict(group=g, outcome="#dims (validated)",
                         agent=f"mean {r['m1']:.2f}, md {r['md1']:.0f}",
                         human=f"mean {r['m0']:.2f}, md {r['md0']:.0f}",
                         effect=f"Cliff's δ {r['effect']:+.3f}; Δmean {r['m1']-r['m0']:+.2f}",
                         test=r["test"], p_raw=r["p_raw"]))
        r = test_2x2(c.loc[c.agent == 1, "quant"], c.loc[c.agent == 0, "quant"],
                     "Agent vs Human: validated and >=1 dim (category layer)", g, "within-group")
        rows.append(dict(group=g, outcome="validated and ≥1 dim (category layer)",
                         agent=f"{100*r['p1']:.1f}%", human=f"{100*r['p0']:.1f}%",
                         effect=f"OR {r['OR']:.2f} [{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]",
                         test=r["test"], p_raw=r["p_raw"]))
    return pd.DataFrame(rows)


def interaction(df, ml, store, rng):
    rows = []
    logit = lambda f, d: smf.logit(f, d).fit(disp=0)
    for key, y, data, label in [("val", "val", df, "validation present"),
                                ("any", "any_dim", ml, "any dimension (validated)"),
                                ("quant", "quant", df, "validated and ≥1 dim (category layer)")]:
        ra, rh = store[(key, "Agent")], store[(key, "Human")]
        dl = ra["lor"] - rh["lor"]
        se = math.sqrt(ra["se"] ** 2 + rh["se"] ** 2)
        z = dl / se
        p_wald = 2 * (1 - norm.cdf(abs(z)))
        lr, p_lr, _ = lrt_interaction(f"{y} ~ cs + agent + cs:agent", f"{y} ~ cs + agent", data, logit)
        TESTS.append(dict(family="interaction", test_label=f"category x author: {label}", stratum="pooled",
                          n=len(data), test="logistic regression LRT (cs:agent)", statistic=lr, p_raw=p_lr,
                          effect_name="ratio of ORs (Agent/Human)", effect=math.exp(dl),
                          note=f"ratio of ORs={math.exp(dl):.2f} [{math.exp(dl-1.96*se):.2f}, "
                               f"{math.exp(dl+1.96*se):.2f}]; Wald z={z:.2f}, p={p_wald:.3f}"))
        rows.append(dict(outcome=label, OR_agent=round(ra["OR"], 2), OR_human=round(rh["OR"], 2),
                         ratio_of_ORs=round(math.exp(dl), 2), ratio_lo=round(math.exp(dl - 1.96 * se), 2),
                         ratio_hi=round(math.exp(dl + 1.96 * se), 2), wald_z=round(z, 2),
                         p_wald=p_wald, LRT_chi2=round(lr, 2), p_LRT=p_lr))

    # #dims: difference of Cliff's delta, permutation + bootstrap; NB regression cross-check
    da, dh = store[("nd", "Agent")]["effect"], store[("nd", "Human")]["effect"]
    obs = da - dh
    cs_x, cs_a = ml.loc[ml.cs == 1, "n_dims"].values, ml.loc[ml.cs == 1, "agent"].values
    ot_x, ot_a = ml.loc[ml.cs == 0, "n_dims"].values, ml.loc[ml.cs == 0, "agent"].values
    perm = np.empty(B_PERM)
    for i in range(B_PERM):
        pa, po = rng.permutation(cs_a), rng.permutation(ot_a)
        perm[i] = (cliffs_delta(cs_x[pa == 1], ot_x[po == 1])
                   - cliffs_delta(cs_x[pa == 0], ot_x[po == 0]))
    p_perm = (1 + (np.abs(perm) >= abs(obs) - 1e-12).sum()) / (B_PERM + 1)
    cells = {(c, a): ml[(ml.cs == c) & (ml.agent == a)].n_dims.values for c in (0, 1) for a in (0, 1)}
    boot = np.empty(B_BOOT)
    for i in range(B_BOOT):
        s = {k: rng.choice(v, len(v), replace=True) for k, v in cells.items()}
        boot[i] = cliffs_delta(s[(1, 1)], s[(0, 1)]) - cliffs_delta(s[(1, 0)], s[(0, 0)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    means = {k: v.mean() for k, v in cells.items()}
    did = (means[(1, 1)] - means[(0, 1)]) - (means[(1, 0)] - means[(0, 0)])
    TESTS.append(dict(family="interaction", test_label="category x author: #dims (validated)", stratum="pooled",
                      n=len(ml), test=f"permutation on Δ Cliff's delta (arm shuffled within group, B={B_PERM:,})",
                      statistic=obs, p_raw=p_perm, effect_name="δ(Agent) − δ(Human)", effect=obs,
                      note=f"δ agent={da:+.3f}, human={dh:+.3f}; bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]; "
                           f"mean DiD={did:+.3f}"))
    nb = lambda f, d: smf.negativebinomial(f, d).fit(disp=0, maxiter=200)
    lr_nb, p_nb, full = lrt_interaction("n_dims ~ cs + agent + cs:agent", "n_dims ~ cs + agent", ml, nb)
    b, se_b = full.params["cs:agent"], full.bse["cs:agent"]
    rr_h, rr_a = math.exp(full.params["cs"]), math.exp(full.params["cs"] + b)
    TESTS.append(dict(family="interaction", test_label="category x author: #dims (validated), NB cross-check",
                      stratum="pooled", n=len(ml), test="negative-binomial regression LRT (cs:agent)",
                      statistic=lr_nb, p_raw=p_nb, effect_name="ratio of rate ratios (Agent/Human)",
                      effect=math.exp(b),
                      note=f"RR agent={rr_a:.2f}, human={rr_h:.2f}; ratio={math.exp(b):.2f} "
                           f"[{math.exp(b-1.96*se_b):.2f}, {math.exp(b+1.96*se_b):.2f}]; alpha={full.params['alpha']:.2f}"))
    dims_rows = pd.DataFrame([
        dict(quantity="Cliff's δ (Code smells vs Other), Agent", value=f"{da:+.3f}"),
        dict(quantity="Cliff's δ (Code smells vs Other), Human", value=f"{dh:+.3f}"),
        dict(quantity="δ(Agent) − δ(Human) [bootstrap 95% CI]", value=f"{obs:+.3f} [{lo:+.3f}, {hi:+.3f}]"),
        dict(quantity=f"permutation p (arm shuffled within group, B={B_PERM:,})", value=fmt_p(p_perm)),
        dict(quantity="mean #dims difference-in-differences (Agent gap − Human gap)", value=f"{did:+.3f}"),
        dict(quantity="NB regression: RR(Code smells vs Other) Agent / Human; ratio [95% CI]",
             value=f"{rr_a:.2f} / {rr_h:.2f}; {math.exp(b):.2f} [{math.exp(b-1.96*se_b):.2f}, {math.exp(b+1.96*se_b):.2f}]"),
        dict(quantity="NB regression: LRT χ²(1) on cs:agent, p", value=f"{lr_nb:.2f}, {fmt_p(p_nb)}"),
    ])
    return pd.DataFrame(rows), dims_rows


def per_dimension(ml):
    """Fewer metrics or different metrics? Pooled metric layer."""
    rows = []
    cs = ml.cs == 1
    for d in DIMS:
        r = test_2x2(ml.loc[cs, d], ml.loc[~cs, d], f"Code smells vs Other: {d} reported",
                     "pooled (validated)", "per-dimension")
        rows.append(dict(dimension=f"{d} {DIM_NAME[d]}", denominator="validated PRs",
                         code_smells=f"{100*r['p1']:.1f}% ({r['k1']}/{r['n1']})",
                         other=f"{100*r['p0']:.1f}% ({r['k0']}/{r['n0']})",
                         OR=f"{r['OR']:.2f} [{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]", test=r["test"], p_raw=r["p_raw"]))
    rep = ml[ml.n_dims >= 1]
    csr = rep.cs == 1
    for d in DIMS:
        r = test_2x2(rep.loc[csr, d], rep.loc[~csr, d], f"Code smells vs Other: {d} | >=1 dim",
                     "pooled (>=1 dim)", "per-dimension mix")
        rows.append(dict(dimension=f"{d} {DIM_NAME[d]}", denominator="PRs reporting ≥1 dim",
                         code_smells=f"{100*r['p1']:.1f}% ({r['k1']}/{r['n1']})",
                         other=f"{100*r['p0']:.1f}% ({r['k0']}/{r['n0']})",
                         OR=f"{r['OR']:.2f} [{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]", test=r["test"], p_raw=r["p_raw"]))
    return pd.DataFrame(rows)


def pairwise(ml):
    """Code smells vs each other category on #dims (validated, pooled)."""
    a = ml.loc[ml.cs == 1, "n_dims"].values
    rows = []
    for cat, sub in ml[ml.cs == 0].groupby("pattern"):
        b = sub.n_dims.values
        r = test_mwu(a, b, f"Code smells vs {cat}: #dims (validated)", "pooled (validated)", "pairwise")
        rows.append(dict(category=cat, n=len(b), mean_dims=round(b.mean(), 2), median_dims=float(np.median(b)),
                         any_dim_pct=round(100 * (b >= 1).mean(), 1), cliffs_delta=round(r["effect"], 3),
                         p_raw=r["p_raw"]))
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=os.path.join(HERE, "..", "classification_labels", "rq3_labels.csv"))
    ap.add_argument("--output-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(SEED)

    df = load(args.labels)
    ml = df[df.validation_present].copy()
    tab_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tab_dir, exist_ok=True)

    T1 = descriptives(df, ml)
    T2, store = within_arm(df, ml)
    T3 = within_group(df, ml)
    T4, T4_dims = interaction(df, ml, store, rng)
    T5 = per_dimension(ml)
    T6 = pairwise(ml)

    tests = pd.DataFrame(TESTS)
    tests["prespecified"] = (tests.family.isin(PRESPECIFIED)
                             & ~tests.test_label.str.contains("NB cross-check"))
    mask = tests.prespecified & tests.p_raw.notna()
    tests["p_bh"] = np.nan
    tests.loc[mask, "p_bh"] = false_discovery_control(tests.loc[mask, "p_raw"].values, method="bh")
    bh = dict(zip(zip(tests.test_label, tests.stratum), tests.p_bh))

    def attach_bh(tab, label_fn):
        tab = tab.copy()
        tab["p_bh"] = [bh[label_fn(r)] for _, r in tab.iterrows()]
        return tab

    lbl2 = {"validation present": "validation present", "any dimension (validated)": "any dimension (validated)",
            "#dims (validated)": "#dims (validated)",
            "validated and ≥1 dim (category layer)": "validated and >=1 dim (category layer)"}
    T2 = attach_bh(T2, lambda r: (f"Code smells vs Other: {lbl2[r.outcome]}", r.arm))
    T3 = attach_bh(T3, lambda r: (f"Agent vs Human: {lbl2[r.outcome]}", r.group))
    T4 = attach_bh(T4, lambda r: (f"category x author: {r.outcome}", "pooled"))
    T5 = attach_bh(T5, lambda r: (f"Code smells vs Other: {r.dimension.split()[0]} reported", "pooled (validated)")
                   if r.denominator == "validated PRs"
                   else (f"Code smells vs Other: {r.dimension.split()[0]} | >=1 dim", "pooled (>=1 dim)"))
    T6 = attach_bh(T6, lambda r: (f"Code smells vs {r.category}: #dims (validated)", "pooled (validated)"))
    p_dims_bh = bh[("category x author: #dims (validated)", "pooled")]

    keep = ["family", "test_label", "stratum", "n", "test", "statistic", "p_raw", "p_bh", "prespecified",
            "effect_name", "effect", "note"]
    tests[keep].to_csv(os.path.join(args.output_dir, "rq3_code_smells_tests.csv"), index=False)
    T1.to_csv(os.path.join(tab_dir, "T7_1_code_smells_descriptives.csv"), index=False)
    T2.to_csv(os.path.join(tab_dir, "T7_2_code_smells_vs_other_within_arm.csv"), index=False)
    T3.to_csv(os.path.join(tab_dir, "T7_3_agent_vs_human_within_group.csv"), index=False)
    T4.to_csv(os.path.join(tab_dir, "T7_4_category_x_author_interaction.csv"), index=False)
    T5.to_csv(os.path.join(tab_dir, "T7_5_code_smells_per_dimension.csv"), index=False)
    T6.to_csv(os.path.join(tab_dir, "T7_6_code_smells_pairwise_dims.csv"), index=False)

    # ── markdown ─────────────────────────────────────────────────────────────
    def fp(tab):
        tab = tab.copy()
        for c in ("p_raw", "p_bh", "p_wald", "p_LRT"):
            if c in tab:
                tab[c] = tab[c].map(fmt_p)
        return tab

    n_cs, n_ot = int((df.cs == 1).sum()), int((df.cs == 0).sum())
    n_cs_v, n_ot_v = int((ml.cs == 1).sum()), int((ml.cs == 0).sum())
    L = []
    w = L.append
    w("# RQ3 post-hoc — structural cleanups vs actual performance changes, by author")
    w(f"Source: `analysis/classification_labels/rq3_labels.csv`. *Code smells* = "
      f"\"{CS}\" (structural cleanups; n = {n_cs} on the category layer, {n_cs_v} validated); "
      f"*Other* = the eight remaining RQ1 categories (actual performance changes; n = {n_ot}, {n_ot_v} validated). "
      f"Validation presence and the combined outcome use the category layer (n = {len(df)}); any dimension and #dims "
      f"use the metric layer (validated PRs, n = {len(ml)}). Tests: chi-square / Fisher's exact with OR "
      f"(Haldane–Anscombe) for 2×2; Mann–Whitney U with Cliff's δ for counts; interaction by logistic-regression LRT "
      f"(binary) and a permutation test on the difference of Cliff's δ (#dims, arm shuffled within category group, "
      f"B = {B_PERM:,}, seed {SEED}) with a negative-binomial LRT as cross-check. Benjamini–Hochberg across the "
      f"{int(mask.sum())} pre-specified tests (§2–§4); the per-dimension (§5), pairwise (§6) and negative-binomial "
      f"tests are descriptive follow-ups reported with raw p only. All tests: `rq3_code_smells_tests.csv`.")
    w("")
    w("## 1. Descriptives")
    w("`quantified` = validated **and** ≥1 dimension reported, as a share of the category layer.")
    w(md_table(T1)); w("")
    w("## 2. Code smells vs Other, within each arm")
    w("OR < 1 and Cliff's δ < 0: Code smells PRs have the outcome less often / report fewer dimensions.")
    w(md_table(fp(T2))); w("")
    w("## 3. Agent vs Human, within each category group")
    w("OR > 1 and Cliff's δ > 0: agents have the outcome more often / report more dimensions than humans.")
    w(md_table(fp(T3))); w("")
    w("## 4. Interaction: does the Code-smells gap differ between agents and humans?")
    w("Ratio of ORs = OR(Agent) / OR(Human); a ratio > 1 means the Code-smells penalty is smaller for agents.")
    w(md_table(fp(T4))); w("")
    w(f"**#dims (validated PRs, n = {len(ml)})** — permutation test BH q = {fmt_p(p_dims_bh)}; the NB cross-check is descriptive (raw p only).")
    w(md_table(T4_dims)); w("")
    w("## 5. Fewer metrics or different metrics? (pooled metric layer; descriptive, raw p)")
    w("Unconditional incidence among validated PRs, then the *mix* among PRs that report at least one dimension.")
    w(md_table(fp(T5))); w("")
    w("## 6. Code smells vs each other category on #dims (validated PRs, pooled; descriptive, raw p)")
    cs_nd = ml.loc[ml.cs == 1, "n_dims"]
    w(f"Code smells: n = {len(cs_nd)}, mean = {cs_nd.mean():.2f}, median = {cs_nd.median():.0f}, "
      f"any dimension = {100*(cs_nd >= 1).mean():.1f}%. Cliff's δ < 0: Code smells reports fewer dimensions than that category.")
    w(md_table(fp(T6))); w("")
    w("## 7. Interpretation")
    w("Humans calibrate verification to the nature of the change: they validate and quantify actual performance "
      "changes at a high rate and structural cleanups at a markedly lower one. Agents apply the same verification "
      "behaviour regardless of category. The category × author interaction is therefore human selectivity, not agent "
      "over-reporting on cleanups: within the *Other* categories agents and humans are indistinguishable on every "
      "outcome, and the gap opens only where humans pull back. Code smells PRs that do quantify report the same metric "
      "mix as everyone else (§5); they are simply less likely to quantify at all. The two binary interactions "
      "(validation presence, any dimension) are significant before correction and borderline after it; the combined "
      "outcome is significant under any family. This family is separate from the RQ3 Step 1 and Step 2 families.")
    with open(os.path.join(args.output_dir, "rq3_code_smells_results.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
