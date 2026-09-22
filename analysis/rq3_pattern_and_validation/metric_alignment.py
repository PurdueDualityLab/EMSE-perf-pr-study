"""RQ3 Step 2 — do the reported metrics fit the optimization applied?

Two checks, neither of which needs a hand-made classification of the catalog.

1. Alignment. Each RQ1 sub-pattern's *expected* dimensions are derived
   mechanically from the catalog's `Optimized Metrics` column
   (catalog_expected_dims.py -> catalog_expected_dims.csv). We compare the share
   of validated PRs that report >= 1 expected dimension with a permutation null
   that reassigns the expected sets across sub-patterns while every PR keeps the
   dimensions it reports. This asks whether what is reported tracks the kind of
   change at all.

2. Memory-for-time. Caching and buffering spend memory to
   save time or I/O; that exchange is textbook and needs no annotation. Within
   these PRs we report how often a gain (D1/D2/D5) is reported, how often memory
   (D3) is, and how often both are, overall, by author type, and on measured
   evidence (benchmark or profiling primary type). Other patterns are not used
   as a comparison group: they differ in what they are expected to move, so
   their memory rate is not a baseline for this one.

p-values are Benjamini-Hochberg adjusted across this step's family.

Usage (from the repository root):
    python analysis/rq3_pattern_and_validation/metric_alignment.py \
        [--labels analysis/classification_labels/rq3_labels.csv] \
        [--output-dir analysis/rq3_pattern_and_validation/results] \
        [--figure-dir analysis/rq3_pattern_and_validation/figures]
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

DIMS = [f"D{i}" for i in range(10)]
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = []
MEASURED = {"benchmark", "profiling"}
TYPES = ["benchmark", "profiling", "static-reasoning", "anecdotal"]
MEM_FOR_TIME = ["Caching", "Buffering"]
GAIN = ["D1", "D2", "D5"]          # what a memory-for-time pattern is expected to improve
ARM_LABEL = {"agentic": "Agentic", "human_candidate": "Human"}


def parse_dims(v):
    return [d for d in str(v).split("|") if d in DIMS] if pd.notna(v) else []


def load(labels_path):
    df = pd.read_csv(labels_path)
    cat = pd.read_csv(os.path.join(HERE, "catalog_expected_dims.csv"))
    cat = cat.rename(columns={"Sub pattern": "sub_pattern", "catalog_expected_dims": "expected_dims"})
    cat["sub_pattern"] = cat.sub_pattern.str.strip()
    missing = set(df.sub_pattern.dropna().str.strip()) - set(cat.sub_pattern)
    if missing:
        raise SystemExit(f"sub-patterns absent from the catalog mapping: {sorted(missing)}")
    df["sub_pattern"] = df.sub_pattern.str.strip()
    df = df.merge(cat[["sub_pattern", "expected_dims"]], on="sub_pattern", how="left")
    if "sample_arm" in df and "author_type" not in df:
        df["author_type"] = df.sample_arm
    for d in DIMS:
        df[d] = df[d].astype(bool)
        if f"{d}_nodiff" in df:
            df[f"{d}_nodiff"] = df[f"{d}_nodiff"].astype(bool)
    df["validation_present"] = df.validation_present.astype(bool)
    df["validation_type"] = df.validation_type.fillna("none")
    df["measured"] = df.validation_type.isin(MEASURED)
    df["expected"] = df.expected_dims.apply(parse_dims)
    df["reports_expected"] = [any(bool(r[d]) for d in r.expected) for _, r in df.iterrows()]
    df["any_dim"] = df[DIMS].any(axis=1)
    df["gain"] = df[GAIN].any(axis=1)
    df["mem_for_time"] = df.sub_pattern.isin(MEM_FOR_TIME)
    return df


def record(label, stratum, n, test, stat, p, eff_name, eff, note=""):
    TESTS.append(dict(test_label=label, stratum=stratum, n=n, test=test, statistic=stat,
                      p_raw=p, effect_name=eff_name, effect=eff, note=note))
    print(f"  - {label} [{stratum}], n={n}: {test}, stat={stat:.2f}, "
          f"p={'<0.001' if p < 0.001 else f'{p:.3f}'}, {eff_name}={eff:.3f}{'; ' + note if note else ''}")


def two_by_two(ct, label, stratum):
    """Chi-square when Cochran's rule holds, otherwise Fisher's exact."""
    if ct.shape != (2, 2) or ct.values.sum() == 0 or ct.values.min(axis=0).sum() == 0:
        return
    exp = stats.chi2_contingency(ct, correction=False)[3]
    a, b, c, d = ct.values[0, 0], ct.values[0, 1], ct.values[1, 0], ct.values[1, 1]
    orr = (a * d) / (b * c) if b * c else np.inf
    if exp.min() >= 5:
        stat, p = stats.chi2_contingency(ct, correction=False)[:2]
        test = "chi-square"
    else:
        orr, p = stats.fisher_exact(ct.values)
        stat, test = orr, "Fisher's exact"
    n = int(ct.values.sum())
    record(label, stratum, n, test, stat, p, "Cramér's V",
           np.sqrt(stats.chi2_contingency(ct, correction=False)[0] / n),
           f"min expected={exp.min():.2f}; OR={orr:.2f}")


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    rows = []
    for _, r in df.iterrows():
        rows.append(["" if isinstance(r[c], float) and np.isnan(r[c])
                     else (floatfmt.format(r[c]) if isinstance(r[c], float) else str(r[c]))
                     for c in cols])
    head = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    return "\n".join([head, sep] + ["| " + " | ".join(r) + " |" for r in rows])


def pct(x):
    return f"{100 * x:.1f}%"


def pct_ci(k, n):
    lo, hi = wilson(k, n)
    return f"{pct(k / n)} [{pct(lo)}-{pct(hi)}]" if n else "-"


def alignment_permutation(frame, B, seed=20260921):
    """Observed share of PRs reporting >=1 expected dimension, and its distribution when the
    expected sets are permuted across the sub-patterns present in `frame` (PR-level reported
    dimensions held fixed). Returns (observed, null array, one-sided p for observed >= null)."""
    subs = sorted(frame.sub_pattern.unique())
    exp_sets = [tuple(sorted(frame.loc[frame.sub_pattern == s, "expected"].iloc[0])) for s in subs]
    distinct = sorted(set(exp_sets))
    col = {d: i for i, d in enumerate(DIMS)}
    M = frame[DIMS].values.astype(bool)
    codes, uniques = pd.factorize(frame.sub_pattern)
    sub_pos = {s: i for i, s in enumerate(uniques)}
    hits = np.zeros((len(subs), len(distinct)))
    for j, E in enumerate(distinct):
        any_e = M[:, [col[d] for d in E]].any(axis=1) if E else np.zeros(len(M), bool)
        counts = np.bincount(codes[any_e], minlength=len(sub_pos))
        for i, s in enumerate(subs):
            hits[i, j] = counts[sub_pos[s]]
    e_index = np.array([distinct.index(E) for E in exp_sets])
    n = len(frame)
    obs = hits[np.arange(len(subs)), e_index].sum() / n
    rng = np.random.default_rng(seed)
    null = np.array([hits[np.arange(len(subs)), e_index[rng.permutation(len(subs))]].sum() / n
                     for _ in range(B)])
    p = (np.sum(null >= obs) + 1) / (B + 1)
    return float(obs), null, float(p)


def rate_rows(frame, by):
    rows = []
    for name, g in list(frame.groupby(by)) + [("All", frame)]:
        gg = g[g.gain]
        rows.append({by: name, "n": len(g),
                     "reports a gain (D1/D2/D5)": f"{int(g.gain.sum())} ({pct(g.gain.mean())})",
                     "reports memory (D3)": f"{int(g.D3.sum())} ({pct_ci(int(g.D3.sum()), len(g))})",
                     "gain and memory": f"{int((g.gain & g.D3).sum())} ({pct((g.gain & g.D3).mean())})",
                     "gain without memory": int((g.gain & ~g.D3).sum()),
                     "of gain reporters, memory too": pct(gg.D3.mean()) if len(gg) else "-",
                     "any dimension": pct(g.any_dim.mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="analysis/classification_labels/rq3_labels.csv")
    ap.add_argument("--output-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--figure-dir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--permutations", type=int, default=20000)
    args = ap.parse_args()

    df = load(args.labels)
    val = df[df.validation_present].copy()
    typed = val[val.validation_type.isin(TYPES)]
    meas = val[val.measured]
    os.makedirs(os.path.join(args.output_dir, "tables"), exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    out = []

    def w(s=""):
        print(s)
        out.append(s)

    def save(t, name):
        t.to_csv(os.path.join(args.output_dir, "tables", f"{name}.csv"), index=False)

    w("# RQ3 Step 2 — do the reported metrics fit the optimization applied?")
    w(f"Source: `{args.labels}`. Population: the RQ3 metric layer (n = {len(val)} of {len(df)} "
      f"analytic PRs; {len(typed)} with a resolved primary evidence type, of which {len(meas)} "
      f"measured = benchmark or profiling).")
    w("Expected dimensions per sub-pattern: `catalog_expected_dims.csv`, derived mechanically "
      "from the catalog's Optimized Metrics column (`catalog_expected_dims.py`).")
    w()

    # ---- 1. alignment ----
    w("## 1. Alignment: do reported dimensions track the pattern?")
    w("Share of validated PRs reporting >= 1 dimension their pattern is expected to improve, "
      "against a null that permutes the expected sets across sub-patterns (each sub-pattern "
      "receives another sub-pattern's expected set; every PR keeps its reported dimensions). "
      "One-sided p: observed >= null.")
    rows = []
    for label, frame in [("All validated", val),
                         ("agentic", val[val.author_type == "agentic"]),
                         ("human_candidate", val[val.author_type == "human_candidate"]),
                         ("measured evidence", meas),
                         ("static reasoning", val[val.validation_type == "static-reasoning"])]:
        obs, null, p = alignment_permutation(frame, args.permutations)
        rows.append({"population": label, "n": len(frame), "observed": pct(obs),
                     "null mean": pct(null.mean()),
                     "null 95%": f"{pct(np.percentile(null, 2.5))}-{pct(np.percentile(null, 97.5))}",
                     "p (one-sided)": f"{p:.4f}"})
        record("Reported dim in pattern's expected set vs permutation null", label, len(frame),
               f"permutation (B={args.permutations})", obs - null.mean(), p,
               "observed - null", obs - null.mean(), f"observed={pct(obs)}; null={pct(null.mean())}")
    t = pd.DataFrame(rows)
    w(md(t))
    save(t, "T6_1_alignment_permutation")
    w()

    # ---- 2. memory-for-time ----
    w("## 2. Memory-for-time patterns: is the memory cost reported alongside the gain?")
    w(f"Memory-for-time = {', '.join(MEM_FOR_TIME)} (spend memory to save time or I/O). "
      "A gain is any of D1/D2/D5. Descriptive: rates within these PRs, by author type; no "
      "comparison group, since other patterns are not expected to move memory.")
    mft_all = val[val.mem_for_time]
    for label, frame, tag in [("### 2.1 All validated memory-for-time PRs", mft_all, "all"),
                              ("### 2.2 Measured evidence only", mft_all[mft_all.measured], "measured")]:
        w(label)
        t = rate_rows(frame, "author_type")
        w(md(t))
        save(t, f"T6_2_memory_for_time_{tag}")
        w()

    w("### 2.3 By pattern (all validated)")
    rows = []
    for sp, g in val[val.mem_for_time].groupby("sub_pattern"):
        rows.append({"sub_pattern": sp, "n": len(g), "reports a gain": pct(g.gain.mean()),
                     "reports memory": pct(g.D3.mean()), "gain and memory": pct((g.gain & g.D3).mean()),
                     "n measured": int(g.measured.sum()),
                     "reports memory (measured)": pct(g[g.measured].D3.mean()) if g.measured.any() else "-"})
    t = pd.DataFrame(rows).sort_values("n", ascending=False)
    w(md(t))
    save(t, "T6_4_memory_for_time_by_pattern")
    w()

    if all(f"{d}_nodiff" in val for d in DIMS):
        w("### 2.4 Sensitivity: code diff excluded from the corpus")
        mft = val[val.mem_for_time]
        w(f"Memory-for-time PRs reporting memory: {pct(mft.D3.mean())} -> {pct(mft.D3_nodiff.mean())} "
          f"without the diff.")
        w()

    # ---- 3. extremes ----
    w("## 3. Extremes")
    mft = val[val.mem_for_time].copy()
    mft["dims"] = mft[DIMS].apply(lambda r: "|".join(d for d in DIMS if r[d]), axis=1)
    cols = [c for c in ["html_url", "author_type", "sub_pattern", "validation_type", "dims", "n_dims"]
            if c in mft.columns]
    one_sided = mft[mft.gain & ~mft.D3]
    one_sided[cols].to_csv(os.path.join(args.output_dir, "extremes_memory_for_time_gain_no_memory.csv"),
                           index=False)
    w(f"{len(one_sided)} memory-for-time PRs report a gain and no memory figure "
      f"(`extremes_memory_for_time_gain_no_memory.csv`); {int((mft.gain & mft.D3).sum())} report both. "
      f"Sample of the latter:")
    w()
    w(md(mft[mft.gain & mft.D3][cols].head(10)))
    w()

    # ---- 4. test family ----
    w("## 4. Step 2 test family (Benjamini-Hochberg)")
    ft = pd.DataFrame(TESTS)
    ft["p_bh"] = bh(ft.p_raw.values)
    ft = ft[["test_label", "stratum", "n", "test", "statistic", "p_raw", "p_bh",
             "effect_name", "effect", "note"]]
    w(md(ft))
    ft.to_csv(os.path.join(args.output_dir, "rq3_step2_tests.csv"), index=False)
    sig = ft[ft.p_bh < 0.05]
    w()
    w(f"Significant after BH (q < 0.05): {len(sig)} of {len(ft)}.")
    for _, r in sig.iterrows():
        w(f"- {r.test_label} [{r.stratum}]: q={r.p_bh:.4f}, {r.note}")

    with open(os.path.join(args.output_dir, "rq3_step2_results.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nwrote {os.path.join(args.output_dir, 'rq3_step2_results.md')}")
    make_figure(val, os.path.join(args.figure_dir, "rq3_memory_for_time.pdf"))


def make_figure(val, path):
    """Within the memory-for-time patterns with measured evidence: share of PRs reporting a gain,
    and share reporting a gain together with memory, by author type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
    C = {"reports a gain (time, throughput, I/O)": "#9ec5f4",
         "reports a gain and memory": "#0d366b"}
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2,
        "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.labelcolor": INK, "pdf.fonttype": 42})
    mft = val[val.mem_for_time & val.measured]
    panels = [("Caching and buffering PRs with measured evidence", mft)]
    arms = ["human_candidate", "agentic"]          # y grows upward, so agentic lands on top
    fig, axes = plt.subplots(1, 1, figsize=(3.6, 2.0), layout="constrained")
    axes = [axes]
    for ax, (title, frame) in zip(axes, panels):
        for k, arm in enumerate(arms):
            g = frame[frame.author_type == arm]
            vals = [100 * g.gain.mean(), 100 * (g.gain & g.D3).mean()]
            for j, (key, v) in enumerate(zip(C, vals)):
                y = k + (j - 0.5) * 0.3
                ax.barh(y, v, height=0.28, color=C[key], label=key if k == 0 else None)
                ax.text(v + 1, y, f"{v:.0f}%", va="center", fontsize=6.5, color=INK2)
            ax.text(97, k, f"n={len(g)}", va="center", ha="right", fontsize=6.5, color=INK2)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([ARM_LABEL[a] for a in arms])
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of PRs (%)")
        ax.set_title(title, loc="left")
        ax.tick_params(axis="y", length=0)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.28))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
