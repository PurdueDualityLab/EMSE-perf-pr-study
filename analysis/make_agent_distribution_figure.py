"""Plot the distribution of agentic performance PRs across AIDev agents."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POPULATION_INPUT = (
    ROOT / "mining/curated_labels_v1/pull_request_labels.parquet"
)
DEFAULT_SAMPLE_INPUT = ROOT / "data/data/sample/agentic_sample.parquet"
DEFAULT_OUTPUT = ROOT / "report/figures/aidev_agent_perf_distribution.pdf"

AGENTS = (
    "openai_codex",
    "github_copilot",
    "cursor",
    "devin",
    "claude_code",
)
AGENT_LABELS = {
    "openai_codex": "OpenAI Codex",
    "github_copilot": "GitHub Copilot",
    "cursor": "Cursor",
    "devin": "Devin",
    "claude_code": "Claude Code",
}
COHORT_LABELS = {
    "population": "Agentic performance population",
    "sample": "Final agentic sample",
}
COLORS = {
    "population": "#6b7280",
    "sample": "#9BBCE8",
}
AGENT_COLORS = ("#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00")
OFFICIAL_COUNTS = {
    "population": {
        "openai_codex": 527,
        "github_copilot": 513,
        "cursor": 167,
        "devin": 92,
        "claude_code": 57,
    },
    "sample": {
        "openai_codex": 392,
        "github_copilot": 458,
        "cursor": 160,
        "devin": 65,
        "claude_code": 55,
    },
}


def _read_columns(path: Path, columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path, columns=sorted(columns))
    if frame.duplicated(["repo_id", "number"]).any():
        raise ValueError(f"{path} contains duplicate PR identities.")
    return frame


def _count_agents(rows: pd.DataFrame, cohort: str) -> pd.Series:
    agents = rows["aidev_attribution_agent"]
    if agents.isna().any() or agents.eq("").any():
        raise ValueError(f"{cohort} contains missing agent labels.")
    if agents.str.contains("|", regex=False).any():
        raise ValueError(
            f"{cohort} contains multi-agent attributions; choose an explicit "
            "counting policy before rendering the figure."
        )
    unknown = sorted(set(agents) - set(AGENTS))
    if unknown:
        raise ValueError(f"{cohort} contains unknown agents: {unknown}")
    return agents.value_counts().reindex(AGENTS, fill_value=0).astype(int)


def load_cohort_counts(
    population_path: Path, sample_path: Path
) -> dict[str, pd.Series]:
    population_frame = _read_columns(
        population_path,
        {
            "repo_id",
            "number",
            "aidev_attribution_label",
            "aidev_attribution_agent",
            "is_performance",
        },
    )
    population = population_frame[
        population_frame["is_performance"].eq(True)
        & population_frame["aidev_attribution_label"].eq("agentic")
    ]

    sample = _read_columns(
        sample_path,
        {
            "repo_id",
            "number",
            "aidev_attribution_label",
            "aidev_attribution_agent",
            "aidev_task_type",
            "sample_arm",
            "selected",
        },
    )
    valid_sample = (
        sample["aidev_task_type"].eq("perf")
        & sample["aidev_attribution_label"].eq("agentic")
        & sample["sample_arm"].eq("agentic")
        & sample["selected"].eq(True)
    )
    if not valid_sample.all():
        raise ValueError("Final sample contains rows outside the agentic performance cohort.")

    return {
        "population": _count_agents(population, "population"),
        "sample": _count_agents(sample, "sample"),
    }


def validate_official_counts(counts: dict[str, pd.Series]) -> None:
    if not counts or set(counts) - set(OFFICIAL_COUNTS):
        raise ValueError("Unexpected cohorts")
    for cohort in counts:
        expected = OFFICIAL_COUNTS[cohort]
        actual = counts[cohort].to_dict()
        if actual != expected:
            raise ValueError(
                f"Official {cohort} counts changed: expected {expected}, got {actual}."
            )


def render_figure(
    counts: dict[str, pd.Series], output: Path, selected_cohorts: tuple[str, ...]
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.4), layout="constrained")
    positions = np.arange(len(AGENTS))
    height = 0.34 if len(selected_cohorts) == 2 else 0.58
    offsets = (-height / 2, height / 2) if len(selected_cohorts) == 2 else (0,)

    maximum = 0
    for cohort, offset in zip(selected_cohorts, offsets):
        values = counts[cohort]
        total = int(values.sum())
        maximum = max(maximum, int(values.max()))
        bars = ax.barh(
            positions + offset,
            values.to_numpy(),
            height=height,
            color=COLORS[cohort],
            label=f"{COHORT_LABELS[cohort]} (n={total:,})",
        )
        labels = [f"{value:,} ({value / total:.1%})" for value in values]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=8)

    ax.set_yticks(positions, [AGENT_LABELS[agent] for agent in AGENTS])
    ax.invert_yaxis()
    ax.set_xlim(0, maximum * 1.32)
    ax.set_xlabel("Number of agentic performance PRs")
    ax.xaxis.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def render_pie_chart(counts: dict[str, pd.Series], output: Path) -> None:
    values = counts["sample"]
    total = int(values.sum())
    fig, ax = plt.subplots(figsize=(6.2, 3.6), layout="constrained")
    wedges, _, _ = ax.pie(
        values.to_numpy(),
        colors=AGENT_COLORS,
        startangle=90,
        counterclock=False,
        autopct="%.1f%%",
        pctdistance=0.72,
        textprops={"fontsize": 8},
        wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    labels = [
        f"{AGENT_LABELS[agent]}: {value:,}"
        for agent, value in values.items()
    ]
    ax.legend(
        wedges,
        labels,
        title=f"Final agentic sample (n={total:,})",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    ax.set_aspect("equal")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population-input", type=Path, default=DEFAULT_POPULATION_INPUT
    )
    parser.add_argument("--sample-input", type=Path, default=DEFAULT_SAMPLE_INPUT)
    parser.add_argument("--counts", type=Path, help="Render from a frozen cohort,agent,count CSV, without reading Parquets.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cohort",
        choices=("population", "sample", "both"),
        default="both",
        help="Cohort to display; 'both' compares pre-filter population and final sample.",
    )
    parser.add_argument(
        "--chart",
        choices=("bars", "pie"),
        default="bars",
        help="Render the selected cohorts as bars or the final sample as a pie chart.",
    )
    parser.add_argument(
        "--skip-official-controls",
        action="store_true",
        help="Allow counts that differ from the official study artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.counts:
        frame = pd.read_csv(args.counts)
        if frame.duplicated(['cohort', 'agent']).any() or frame['count'].lt(0).any():
            raise ValueError("Invalid cohort counts")
        counts = {cohort: g.set_index('agent')['count'].reindex(AGENTS)
                  for cohort, g in frame.groupby('cohort')}
    else:
        counts = load_cohort_counts(args.population_input, args.sample_input)
    if not args.skip_official_controls:
        validate_official_counts(counts)

    selected_cohorts = (
        ("population", "sample") if args.cohort == "both" else (args.cohort,)
    )
    if set(selected_cohorts) - set(counts):
        raise ValueError("The count CSV does not contain every requested cohort")
    for cohort in selected_cohorts:
        print(f"{COHORT_LABELS[cohort]}:")
        for agent, value in counts[cohort].items():
            print(f"  {AGENT_LABELS[agent]}: {value:,}")
    if args.chart == "pie":
        if args.cohort != "sample":
            raise ValueError("The pie chart represents only the final sample cohort.")
        render_pie_chart(counts, args.output)
    else:
        render_figure(counts, args.output, selected_cohorts)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
