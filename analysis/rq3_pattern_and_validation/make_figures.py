"""
RQ3 figures (PDF, paper-ready).

  figures/rq3_validation_by_category.pdf   validation type share per optimization category, agent vs human
  figures/rq3_metric_frequency.pdf         % of validated PRs reporting each dimension D0–D9, agent vs human
  figures/rq3_metric_profile_heatmap.pdf   category × D0–D9 incidence (plus no-metric column) among validated PRs, agent vs human
  figures/rq3_dimensionality.pdf           number of reported dimensions per validated PR, by author and by evidence type

Run from the repo root after extract_metrics.py:
  python RQ3_metric_targeting/make_figures.py
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metric_patterns as mp  # noqa: E402

DATA = HERE / "data" / "rq3_pr_level.csv"
FIG = HERE / "figures"
DIMS = list(mp.DIMENSIONS)

# palette: categorical slots 1–2 (agent blue, human orange); one-hue blue ramp for magnitude
AGENT, HUMAN = "#2a78d6", "#eb6834"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
VTYPE_COLORS = {          # ordinal by rigour, light -> dark; "none" is neutral
    "none": "#d9d8d3", "anecdotal": "#86b6ef", "static-analysis": "#5598e7",
    "profiling": "#256abf", "benchmark": "#104281",
}
VTYPES = ["benchmark", "profiling", "static-analysis", "anecdotal", "none"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

SHORT = {
    "Algorithm-Level Optimizations": "Algorithm-level",
    "Build & Compilation & Infrastructure Optimization": "Build & infra",
    "Code Smells and Structural Simplification": "Code smells",
    "Control-Flow and Branching Optimizations": "Control-flow",
    "Data Structure Selection and Adaptation": "Data structure",
    "I/O and Synchronization": "I/O & sync",
    "Loop Transformations": "Loop transf.",
    "Memory and Data Locality Optimizations": "Memory & locality",
    "Network, Database, and Data Access Optimization": "Network / DB",
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2,
    "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "pdf.fonttype": 42,
})


def _clean(ax, x_grid=True):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if x_grid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def fig_validation_by_category(df):
    order = df["pattern"].value_counts().index.tolist()          # by total n, largest on top
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, author in zip(axes, ["AI Agent", "Human"]):
        g = df[df.author_type == author]
        ct = pd.crosstab(g["pattern"], g["validation_type"]).reindex(index=order, columns=VTYPES).fillna(0)
        n = ct.sum(axis=1)
        share = ct.div(n.replace(0, np.nan), axis=0).fillna(0) * 100
        y = np.arange(len(order))[::-1]
        left = np.zeros(len(order))
        for vt in VTYPES:
            ax.barh(y, share[vt].values, left=left, color=VTYPE_COLORS[vt], height=0.68,
                    edgecolor="white", linewidth=1.0, label=vt)
            left += share[vt].values
        for yi, (cat, ni) in zip(y, n.items()):
            ax.text(101, yi, f"n={int(ni)}", va="center", ha="left", fontsize=6.5, color=INK2)
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("share of PRs (%)")
        ax.set_title(author, loc="left", color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT.get(c, c) for c in order])
        ax.tick_params(axis="y", length=0)
        _clean(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04), title="RQ2 validation type", title_fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "rq3_validation_by_category")
    plt.close(fig)


def fig_metric_profile_heatmap(df):
    val = df[df.validation_present].copy()
    val["NONE"] = val["n_dims"] == 0          # validated, but no metric dimension reported
    cols = DIMS + ["NONE"]
    col_labels = DIMS + ["none"]
    order = val["pattern"].value_counts().index.tolist()
    cmap = LinearSegmentedColormap.from_list("blue_ramp", RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.22})
    vmax = 100
    for ax, author in zip(axes, ["AI Agent", "Human"]):
        g = val[val.author_type == author]
        n = g.groupby("pattern").size().reindex(order).fillna(0).astype(int)
        cnt = g.groupby("pattern")[cols].sum().reindex(order).fillna(0)
        share = cnt.div(n.replace(0, np.nan), axis=0).fillna(0) * 100
        im = ax.imshow(share.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i in range(len(order)):
            for j, d in enumerate(cols):
                v = share.iloc[i, j]
                if cnt.iloc[i, j]:
                    ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=6,
                            color="white" if v > vmax * 0.55 else INK)
        # separate the "none" column from the dimensions
        ax.axvline(len(DIMS) - 0.5, color=INK2, linewidth=0.8)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{SHORT.get(c, c)} (n={n[c]})" if ax is axes[0] else f"n={n[c]}" for c in order])
        ax.tick_params(length=0)
        if ax is axes[1]:
            ax.tick_params(axis="y", pad=2)
        ax.set_title(f"{author} (validated n={len(g)})", loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        # thin white grid between cells
        ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("% of validated PRs in category")
    cbar.outline.set_visible(False)
    fig.text(0.5, -0.02, "  ".join(f"{d}: {mp.DIMENSIONS[d]}" for d in DIMS[:5]) + "\n" +
             "  ".join(f"{d}: {mp.DIMENSIONS[d]}" for d in DIMS[5:]) + "  none: no metric dimension reported",
             ha="center", va="top", fontsize=6.5, color=INK2)
    _save(fig, "rq3_metric_profile_heatmap")
    plt.close(fig)


def fig_metric_frequency(df):
    """% of validated PRs reporting each dimension, agent vs human (sorted by pooled frequency)."""
    val = df[df.validation_present]
    pooled = val[DIMS].mean().sort_values(ascending=False)
    order = pooled.index.tolist()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    y = np.arange(len(order))[::-1]
    height = 0.38
    xmax = 0
    for i, (author, color) in enumerate([("AI Agent", AGENT), ("Human", HUMAN)]):
        g = val[val.author_type == author]
        share = g[order].mean().values * 100
        yy = y + (0.5 - i) * height
        ax.barh(yy, share, height=height * 0.94, color=color, label=f"{author} (n={len(g)})")
        for yi, v in zip(yy, share):
            ax.text(v + 0.8, yi, f"{v:.0f}%", va="center", ha="left", fontsize=6.5, color=INK2)
        xmax = max(xmax, share.max())
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d}  {mp.DIMENSIONS[d]}" for d in order])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax * 1.18)
    ax.set_xlabel("% of validated PRs reporting the dimension")
    ax.legend(frameon=False, loc="lower right")
    _clean(ax)
    fig.tight_layout()
    _save(fig, "rq3_metric_frequency")
    plt.close(fig)


def fig_dimensionality(df):
    val = df[df.validation_present].copy()
    val["k"] = val["n_dims"].clip(upper=4)
    ks = [0, 1, 2, 3, 4]
    labels = ["0", "1", "2", "3", "4+"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ax = axes[0]
    width = 0.38
    for i, (author, color) in enumerate([("AI Agent", AGENT), ("Human", HUMAN)]):
        g = val[val.author_type == author]
        share = g["k"].value_counts(normalize=True).reindex(ks).fillna(0) * 100
        x = np.arange(len(ks)) + (i - 0.5) * width
        ax.bar(x, share.values, width=width * 0.94, color=color, label=f"{author} (n={len(g)})")
        for xi, v in zip(x, share.values):
            if v > 0:
                ax.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("distinct metric dimensions reported per validated PR")
    ax.set_ylabel("share of PRs (%)")
    ax.set_title("by author type", loc="left")
    ax.legend(frameon=False)
    _clean(ax, x_grid=False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    groups = [("benchmark", VTYPE_COLORS["benchmark"]), ("profiling", VTYPE_COLORS["profiling"]),
              ("static-analysis", VTYPE_COLORS["static-analysis"]), ("anecdotal", VTYPE_COLORS["anecdotal"])]
    width = 0.2
    for i, (vt, color) in enumerate(groups):
        g = val[val.validation_type == vt]
        share = g["k"].value_counts(normalize=True).reindex(ks).fillna(0) * 100
        x = np.arange(len(ks)) + (i - 1.5) * width
        ax.bar(x, share.values, width=width * 0.94, color=color, label=f"{vt} (n={len(g)})")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("distinct metric dimensions reported per validated PR")
    ax.set_title("by RQ2 validation type", loc="left")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    _clean(ax, x_grid=False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "rq3_dimensionality")
    plt.close(fig)


def _save(fig, name):
    """Save the PDF; also a PNG preview when RQ3_PNG_DIR is set (for quick inspection)."""
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    png_dir = os.environ.get("RQ3_PNG_DIR")
    if png_dir:
        Path(png_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(png_dir) / f"{name}.png", bbox_inches="tight", dpi=170)


def main():
    FIG.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    df["validation_present"] = df["validation_present"].astype(bool)
    df["validation_type"] = df["validation_type"].fillna("none")
    for d in DIMS:
        df[d] = df[d].astype(bool)
    fig_validation_by_category(df)
    fig_metric_frequency(df)
    fig_metric_profile_heatmap(df)
    fig_dimensionality(df)
    print(f"[INFO] figures -> {FIG}")


if __name__ == "__main__":
    main()
