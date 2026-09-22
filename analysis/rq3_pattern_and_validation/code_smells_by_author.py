"""RQ3 post-hoc — structural cleanups vs actual performance changes, by author.

Contrast the "Code Smells and Structural Simplification" category (structural
cleanups) with the eight remaining RQ1 categories (actual performance changes)
on three binary outcomes, separately for agentic and human PRs:

  (1) validation presence         — category layer, all analytic PRs (n = 2081)
  (2) any dimension reported      — metric layer, validated PRs (n = 1699)
  (3) validated AND >= 1 dimension — category layer; folds (1) and (2) into one
                                     unconditional rate

and test whether the Code-smells gap differs between the two arms (category x
author interaction). A final section checks whether Code smells PRs report
*different* metrics or merely *fewer*: each D0-D9 unconditionally and
conditional on >= 1 dimension.

Statistics
  * 2x2 tables: chi-square when Cochran's rule holds, else Fisher's exact;
    odds ratio with Haldane-Anscombe correction and 95% CI.
  * Interaction: logistic regression outcome ~ cs + agent + cs:agent,
    likelihood-ratio test on the interaction; the Wald z on the difference of
    log odds ratios (= the saturated-model interaction Wald test) is reported
    alongside with the ratio of odds ratios and its CI.
  * Benjamini-Hochberg in two pre-specified families: the 3 interaction tests
    (the claim of the analysis) and the 12 simple-effect contrasts (6 cleanups
    vs other within each arm, 6 agent vs human within each category group),
    which decompose the interaction. The per-dimension follow-ups are
    descriptive and are reported with raw p only (p_bh is empty for them).

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
from scipy.stats import chi2, chi2_contingency, false_discovery_control, fisher_exact, norm
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
CS = "Code Smells and Structural Simplification"
DIMS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D0"]
DIM_NAME = {"D1": "latency/exec time", "D2": "throughput", "D3": "memory", "D4": "CPU work",
            "D5": "I/O & network", "D6": "artifact size", "D7": "build/CI time",
            "D8": "energy & cost", "D9": "scalability/concurrency", "D0": "unspecified perf"}
ARMS = [("agentic", "Agent"), ("human_candidate", "Human")]
GROUPS = ["Code smells", "Other"]
OUTCOMES = [  # (key, column, layer, label)
    ("val", "val", "df", "validation present"),
    ("any", "any_dim", "ml", "any dimension (validated)"),
    ("quant", "quant", "df", "validated and >=1 dim (category layer)"),
]
FAMILIES = {"interaction": "interaction", "within-arm": "simple-effects", "within-group": "simple-effects"}

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


def lrt_interaction(y, data):
    full = smf.logit(f"{y} ~ cs + agent + cs:agent", data).fit(disp=0)
    red = smf.logit(f"{y} ~ cs + agent", data).fit(disp=0)
    lr = 2 * (full.llf - red.llf)
    return lr, 1 - chi2.cdf(lr, 1)


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
                "quantified_n": int(c.quant.sum()), "quantified_pct": round(100 * c.quant.mean(), 1),
            })
    return pd.DataFrame(rows)


def contrast_row(r, left, right, extra):
    return {**extra, left: f"{100*r['p1']:.1f}%", right: f"{100*r['p0']:.1f}%",
            "effect": f"OR {r['OR']:.2f} [{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]", "test": r["test"], "p_raw": r["p_raw"]}


def within_arm(df, ml):
    """Code smells vs Other within each arm."""
    rows, store = [], {}
    for _, arm in ARMS:
        layer = {"df": df[df.arm == arm], "ml": ml[ml.arm == arm]}
        for key, col, lay, label in OUTCOMES:
            d = layer[lay]
            r = test_2x2(d.loc[d.cs == 1, col], d.loc[d.cs == 0, col],
                         f"Code smells vs Other: {label}", arm, "within-arm")
            store[(key, arm)] = r
            rows.append(contrast_row(r, "code_smells", "other", dict(arm=arm, outcome=label)))
    return pd.DataFrame(rows), store


def within_group(df, ml):
    """Agent vs Human within each category group."""
    rows = []
    for g in GROUPS:
        layer = {"df": df[df.grp == g], "ml": ml[ml.grp == g]}
        for key, col, lay, label in OUTCOMES:
            d = layer[lay]
            r = test_2x2(d.loc[d.agent == 1, col], d.loc[d.agent == 0, col],
                         f"Agent vs Human: {label}", g, "within-group")
            rows.append(contrast_row(r, "agent", "human", dict(group=g, outcome=label)))
    return pd.DataFrame(rows)


def interaction(df, ml, store):
    rows = []
    layer = {"df": df, "ml": ml}
    for key, col, lay, label in OUTCOMES:
        ra, rh = store[(key, "Agent")], store[(key, "Human")]
        dl = ra["lor"] - rh["lor"]
        se = math.sqrt(ra["se"] ** 2 + rh["se"] ** 2)
        z = dl / se
        p_wald = 2 * (1 - norm.cdf(abs(z)))
        lr, p_lr = lrt_interaction(col, layer[lay])
        TESTS.append(dict(family="interaction", test_label=f"category x author: {label}", stratum="pooled",
                          n=len(layer[lay]), test="logistic regression LRT (cs:agent)", statistic=lr, p_raw=p_lr,
                          effect_name="ratio of ORs (Agent/Human)", effect=math.exp(dl),
                          note=f"ratio of ORs={math.exp(dl):.2f} [{math.exp(dl-1.96*se):.2f}, "
                               f"{math.exp(dl+1.96*se):.2f}]; Wald z={z:.2f}, p={p_wald:.3f}"))
        rows.append(dict(outcome=label, OR_agent=round(ra["OR"], 2), OR_human=round(rh["OR"], 2),
                         ratio_of_ORs=round(math.exp(dl), 2), ratio_lo=round(math.exp(dl - 1.96 * se), 2),
                         ratio_hi=round(math.exp(dl + 1.96 * se), 2), wald_z=round(z, 2),
                         p_wald=p_wald, LRT_chi2=round(lr, 2), p_LRT=p_lr))
    return pd.DataFrame(rows)


def per_dimension(ml):
    """Fewer metrics or different metrics? Pooled metric layer (descriptive)."""
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
        rows.append(dict(dimension=f"{d} {DIM_NAME[d]}", denominator="PRs reporting >=1 dim",
                         code_smells=f"{100*r['p1']:.1f}% ({r['k1']}/{r['n1']})",
                         other=f"{100*r['p0']:.1f}% ({r['k0']}/{r['n0']})",
                         OR=f"{r['OR']:.2f} [{r['OR_lo']:.2f}, {r['OR_hi']:.2f}]", test=r["test"], p_raw=r["p_raw"]))
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", default=os.path.join(HERE, "..", "classification_labels", "rq3_labels.csv"))
    ap.add_argument("--output-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    df = load(args.labels)
    ml = df[df.validation_present].copy()
    tab_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(tab_dir, exist_ok=True)

    T1 = descriptives(df, ml)
    T2, store = within_arm(df, ml)
    T3 = within_group(df, ml)
    T4 = interaction(df, ml, store)
    T5 = per_dimension(ml)

    tests = pd.DataFrame(TESTS)
    tests["bh_family"] = tests.family.map(FAMILIES).fillna("descriptive (raw p only)")
    tests["p_bh"] = np.nan
    for fam in ("interaction", "simple-effects"):
        m = (tests.bh_family == fam) & tests.p_raw.notna()
        tests.loc[m, "p_bh"] = false_discovery_control(tests.loc[m, "p_raw"].values, method="bh")
    bh = dict(zip(zip(tests.test_label, tests.stratum), tests.p_bh))

    def attach_bh(tab, label_fn):
        tab = tab.copy()
        tab["p_bh"] = [bh[label_fn(r)] for _, r in tab.iterrows()]
        return tab

    T2 = attach_bh(T2, lambda r: (f"Code smells vs Other: {r.outcome}", r.arm))
    T3 = attach_bh(T3, lambda r: (f"Agent vs Human: {r.outcome}", r.group))
    T4 = attach_bh(T4, lambda r: (f"category x author: {r.outcome}", "pooled"))

    keep = ["family", "bh_family", "test_label", "stratum", "n", "test", "statistic", "p_raw", "p_bh",
            "effect_name", "effect", "note"]
    tests[keep].to_csv(os.path.join(args.output_dir, "rq3_code_smells_tests.csv"), index=False)
    T1.to_csv(os.path.join(tab_dir, "T7_1_code_smells_descriptives.csv"), index=False)
    T2.to_csv(os.path.join(tab_dir, "T7_2_code_smells_vs_other_within_arm.csv"), index=False)
    T3.to_csv(os.path.join(tab_dir, "T7_3_agent_vs_human_within_group.csv"), index=False)
    T4.to_csv(os.path.join(tab_dir, "T7_4_category_x_author_interaction.csv"), index=False)
    T5.to_csv(os.path.join(tab_dir, "T7_5_code_smells_per_dimension.csv"), index=False)

    # ── markdown ─────────────────────────────────────────────────────────────
    def fp(tab):
        tab = tab.copy()
        for c in ("p_raw", "p_bh", "p_wald", "p_LRT"):
            if c in tab:
                tab[c] = tab[c].map(fmt_p)
        return tab

    n_cs, n_ot = int((df.cs == 1).sum()), int((df.cs == 0).sum())
    n_cs_v, n_ot_v = int((ml.cs == 1).sum()), int((ml.cs == 0).sum())
    n_int = int((tests.bh_family == "interaction").sum())
    n_simple = int((tests.bh_family == "simple-effects").sum())
    L = []
    w = L.append
    w("# RQ3 post-hoc — structural cleanups vs actual performance changes, by author")
    w(f"Source: `analysis/classification_labels/rq3_labels.csv`. *Code smells* = "
      f"\"{CS}\" (structural cleanups; n = {n_cs} on the category layer, {n_cs_v} validated); "
      f"*Other* = the eight remaining RQ1 categories (actual performance changes; n = {n_ot}, {n_ot_v} validated). "
      f"Validation presence and the combined outcome use the category layer (n = {len(df)}); any dimension uses the "
      f"metric layer (validated PRs, n = {len(ml)}). Tests: chi-square / Fisher's exact with OR (Haldane–Anscombe) "
      f"for 2×2; interaction by logistic-regression LRT. Benjamini–Hochberg in two families: the {n_int} interaction "
      f"tests (§4) and the {n_simple} simple-effect contrasts (§2–§3); the per-dimension follow-ups (§5) are "
      f"descriptive and carry raw p only. All tests: `rq3_code_smells_tests.csv`.")
    w("")
    w("## 1. Descriptives")
    w("`quantified` = validated **and** ≥1 dimension reported, as a share of the category layer.")
    w(md_table(T1)); w("")
    w("## 2. Code smells vs Other, within each arm")
    w("OR < 1: Code smells PRs have the outcome less often.")
    w(md_table(fp(T2))); w("")
    w("## 3. Agent vs Human, within each category group")
    w("OR > 1: agents have the outcome more often than humans.")
    w(md_table(fp(T3))); w("")
    w("## 4. Interaction: does the Code-smells gap differ between agents and humans?")
    w("Ratio of ORs = OR(Agent) / OR(Human); a ratio > 1 means the Code-smells penalty is smaller for agents.")
    w(md_table(fp(T4))); w("")
    w("## 5. Fewer metrics or different metrics? (pooled metric layer; descriptive, raw p)")
    w("Unconditional incidence among validated PRs, then the *mix* among PRs that report at least one dimension.")
    w(md_table(fp(T5))); w("")
    w("## 6. Interpretation")
    w("Humans calibrate verification to the nature of the change: they validate and quantify actual performance "
      "changes at a high rate and structural cleanups at a markedly lower one. Agents apply the same verification "
      "behaviour regardless of category. The category × author interaction is therefore human selectivity, not agent "
      "over-reporting on cleanups: within the *Other* categories agents and humans are indistinguishable on every "
      "outcome, and the gap opens only where humans pull back. Code smells PRs that do quantify report the same metric "
      "mix as everyone else (§5); they are simply less likely to quantify at all. The two component interactions "
      "(validation presence, any dimension) are significant before correction and borderline after it; the combined "
      "outcome is significant after correction. These families are separate from the RQ3 Step 1 and Step 2 families.")
    with open(os.path.join(args.output_dir, "rq3_code_smells_results.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
