"""Render compact vector figures from published labels and structural deltas."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report/figures'
ARMS = ['agentic', 'human_candidate']
COLORS = ['#9BBCE8', '#B9DAB9']
LABELS = ['Agentic', 'Human-candidate']
SHORT = {
    'Memory and Data Locality Optimizations': 'Memory / data locality',
    'Algorithm-Level Optimizations': 'Algorithm-level',
    'Code Smells and Structural Simplification': 'Structural simplification',
    'I/O and Synchronization': 'I/O / synchronization',
    'Build & Compilation & Infrastructure Optimization': 'Build / infrastructure',
    'Network, Database, and Data Access Optimization': 'Network / database',
    'Data Structure Selection and Adaptation': 'Data structures',
    'Control-Flow and Branching Optimizations': 'Control-flow / branching',
    'Loop Transformations': 'Loop transformations',
}


def main():
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'legend.fontsize': 8,
                         'pdf.fonttype': 42})
    rq1 = pd.read_csv(ROOT / 'analysis/classification_labels/rq1_labels.csv')
    rq1 = rq1[rq1.included_in_analysis]
    assert len(rq1) == 2083
    order = rq1.consensus_high_level_pattern.value_counts().index
    fig, ax = plt.subplots(figsize=(6.1, 4.1), layout='constrained')
    for i, (arm, color, label) in enumerate(zip(ARMS, COLORS, LABELS)):
        group = rq1[rq1.sample_arm.eq(arm)]
        rates = group.consensus_high_level_pattern.value_counts().reindex(order, fill_value=0)*100/len(group)
        bars = ax.barh(np.arange(len(order)) + (i-.5)*.36, rates, height=.34,
                       color=color, label=f'{label} (n={len(group):,})')
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=8)
    ax.set_yticks(np.arange(len(order)), [SHORT.get(label, label) for label in order])
    ax.invert_yaxis(); ax.set_xlim(0, 41)
    ax.set_xlabel('Share of resolved optimization labels (%)')
    ax.legend(loc='lower right', frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(OUT / 'rq1_current.pdf', bbox_inches='tight'); plt.close(fig)

    rq2 = pd.read_csv(ROOT / 'analysis/classification_labels/rq2_labels.csv')
    rq2 = rq2[rq2.included_in_stage2_analysis]
    assert len(rq2) == 1707
    order = ['static-reasoning', 'benchmark', 'anecdotal', 'profiling']
    fig, ax = plt.subplots(figsize=(5.2, 3.2), layout='constrained')
    for i, (arm, color, label) in enumerate(zip(ARMS, COLORS, LABELS)):
        group = rq2[rq2.sample_arm.eq(arm)]
        rates = group.consensus_primary_validation_type.value_counts().reindex(order, fill_value=0)*100/len(group)
        bars = ax.bar(np.arange(4)+(i-.5)*.35, rates, width=.34, color=color,
                      label=f'{label} (n={len(group)})')
        ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=8)
    ax.set_xticks(np.arange(4), ['Static\nreasoning', 'Benchmark', 'Anecdotal', 'Profiling'])
    ax.set_ylabel('Share of resolved positive types (%)'); ax.set_ylim(0, 74)
    ax.legend(frameon=False, loc='upper right')
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(OUT / 'rq2_current.pdf', bbox_inches='tight'); plt.close(fig)

    data = pd.read_csv(ROOT / 'analysis/maintainability/current/analysis_deltas.csv')
    fig, axes = plt.subplots(1, 3, figsize=(5.2, 3.0), layout='constrained')
    for ax, metric, title in zip(axes, ['nloc', 'avg_ccn', 'function_count'], ['Mean file NLOC', 'Mean file AvgCCN', 'Mean file\nfunction count']):
        groups = [data.loc[data.metric.eq(metric) & data.sample_arm.eq(arm), 'delta_pct'] for arm in ARMS]
        assert [len(group) for group in groups] == [1005, 1024]
        boxes = ax.boxplot(groups, whis=(10, 90), showfliers=False, patch_artist=True,
                          tick_labels=['Agentic', 'Human-cand.'], medianprops={'color': 'black'})
        for patch, color in zip(boxes['boxes'], COLORS):
            patch.set_facecolor(color); patch.set_alpha(.7)
        ax.axhline(0, color='gray', linestyle='--', linewidth=.7)
        ax.set_title(title, fontsize=9); ax.tick_params(axis='x', labelrotation=35, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    axes[0].set_ylabel('Base-to-head change (%)', fontsize=9)
    fig.savefig(OUT / 'maintainability_current.pdf', bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    main()
