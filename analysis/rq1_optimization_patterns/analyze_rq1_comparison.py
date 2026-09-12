"""Compare high-level RQ1 consensus patterns between study arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import textwrap
from typing import Any

import numpy as np
import pandas as pd


ARMS = ("agentic", "human_candidate")
Z_975 = 1.959963984540054


def cramers_v(table: np.ndarray) -> float:
    values = np.asarray(table, dtype=float)
    if values.ndim != 2 or values.sum() <= 0 or min(values.shape) < 2:
        raise ValueError("Cramer's V requires a nonempty contingency table with at least two rows and columns.")
    expected = np.outer(values.sum(axis=1), values.sum(axis=0)) / values.sum()
    if (expected == 0).any():
        raise ValueError("Cramer's V requires nonzero row and column marginals.")
    chi2 = float(np.sum((values - expected) ** 2 / expected))
    return math.sqrt(chi2 / (values.sum() * min(values.shape[0] - 1, values.shape[1] - 1)))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


def category_effects(table: np.ndarray) -> dict[str, Any]:
    """Compute agentic minus human effects, applying 0.5 only if a cell is zero."""
    from scipy.stats import fisher_exact

    values = np.asarray(table, dtype=float)
    if values.shape != (2, 2) or (values < 0).any() or (values.sum(axis=1) == 0).any():
        raise ValueError("Category effects require a valid 2x2 table with both arms represented.")
    raw = values.copy()
    raw_a, raw_b = raw[0]
    raw_c, raw_d = raw[1]
    raw_n1, raw_n0 = raw_a + raw_b, raw_c + raw_d
    raw_p1, raw_p0 = raw_a / raw_n1, raw_c / raw_n0
    risk_difference = raw_p1 - raw_p0
    rd_se = math.sqrt(raw_p1 * (1 - raw_p1) / raw_n1 + raw_p0 * (1 - raw_p0) / raw_n0)
    corrected = bool((values == 0).any())
    if corrected:
        values += 0.5
    a, b = values[0]
    c, d = values[1]
    n1, n0 = a + b, c + d
    p1, p0 = a / n1, c / n0
    risk_ratio = p1 / p0
    rr_se = math.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    odds_ratio = (a * d) / (b * c)
    or_se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    fisher_or, fisher_p = fisher_exact(raw, alternative="two-sided")
    return {
        "fisher_odds_ratio": float(fisher_or),
        "fisher_p_value": float(fisher_p),
        "risk_difference": float(risk_difference),
        "risk_difference_ci_low": float(risk_difference - Z_975 * rd_se),
        "risk_difference_ci_high": float(risk_difference + Z_975 * rd_se),
        "risk_ratio": float(risk_ratio),
        "risk_ratio_ci_low": float(math.exp(math.log(risk_ratio) - Z_975 * rr_se)),
        "risk_ratio_ci_high": float(math.exp(math.log(risk_ratio) + Z_975 * rr_se)),
        "odds_ratio": float(odds_ratio),
        "odds_ratio_ci_low": float(math.exp(math.log(odds_ratio) - Z_975 * or_se)),
        "odds_ratio_ci_high": float(math.exp(math.log(odds_ratio) + Z_975 * or_se)),
        "continuity_corrected": corrected,
    }


def cluster_bootstrap_cramers_v(
    frame: pd.DataFrame,
    iterations: int = 10_000,
    seed: int = 20260911,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("Bootstrap iterations must be positive.")
    repositories = np.array(sorted(frame["repo_id"].unique()))
    if len(repositories) < 2:
        raise ValueError("Cluster bootstrap requires at least two repositories.")
    categories = sorted(frame["consensus_high_level_pattern"].unique())
    cluster_counts = (
        frame.groupby(["repo_id", "sample_arm", "consensus_high_level_pattern"])
        .size()
        .unstack(["sample_arm", "consensus_high_level_pattern"], fill_value=0)
        .reindex(index=repositories, columns=pd.MultiIndex.from_product([ARMS, categories]), fill_value=0)
        .to_numpy()
        .reshape(len(repositories), len(ARMS), len(categories))
    )
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        selected = rng.choice(repositories, size=len(repositories), replace=True)
        weights = np.bincount(np.searchsorted(repositories, selected), minlength=len(repositories))
        table = np.tensordot(weights, cluster_counts, axes=(0, 0))
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] >= 2 and (table.sum(axis=1) > 0).all():
            estimates.append(cramers_v(table))
    if not estimates:
        raise ValueError("No valid cluster-bootstrap replicates were produced.")
    low, high = np.percentile(estimates, [2.5, 97.5])
    return {
        "method": "repository_cluster_percentile_bootstrap",
        "seed": seed,
        "requested_iterations": iterations,
        "valid_iterations": len(estimates),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def subpattern_richness(frame: pd.DataFrame, iterations: int, seed: int) -> dict[str, Any]:
    labels = frame["consensus_sub_pattern"].to_numpy()
    arms = frame["sample_arm"].to_numpy()
    observed = {arm: int(frame.loc[frame["sample_arm"].eq(arm), "consensus_sub_pattern"].nunique()) for arm in ARMS}
    observed_delta = observed[ARMS[0]] - observed[ARMS[1]]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iterations):
        shuffled = rng.permutation(arms)
        deltas.append(len(set(labels[shuffled == ARMS[0]])) - len(set(labels[shuffled == ARMS[1]])))
    p_value = (sum(abs(value) >= abs(observed_delta) for value in deltas) + 1) / (iterations + 1)
    target = min(int((arms == arm).sum()) for arm in ARMS)

    def expected_rarefied(arm: str) -> float:
        counts = frame.loc[frame["sample_arm"].eq(arm), "consensus_sub_pattern"].value_counts()
        population = int(counts.sum())
        if target == population:
            return float(len(counts))
        denominator = math.comb(population, target)
        return sum(1 - math.comb(population - int(count), target) / denominator for count in counts)

    return {
        "role": "secondary_exploratory_analysis",
        "observed_richness": observed,
        "observed_agentic_minus_human": observed_delta,
        "permutation_iterations": iterations,
        "permutation_seed": seed,
        "permutation_p_value_two_sided": float(p_value),
        "rarefaction_target_rows": target,
        "expected_rarefied_richness": {arm: expected_rarefied(arm) for arm in ARMS},
    }


def analyze_comparison(
    consensus: pd.DataFrame,
    bootstrap_iterations: int = 10_000,
    permutation_iterations: int = 10_000,
    seed: int = 20260911,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from scipy.stats import chi2_contingency

    required = {
        "repo_id",
        "number",
        "sample_arm",
        "consensus_high_level_pattern",
        "consensus_sub_pattern",
        "included_in_analysis",
    }
    missing = required - set(consensus.columns)
    if missing:
        raise ValueError(f"Consensus artifact is missing columns: {sorted(missing)}")
    if consensus.duplicated(["repo_id", "number"]).any():
        raise ValueError("Consensus artifact contains duplicate PR identities.")
    included = consensus[consensus["included_in_analysis"].eq(True)].copy()  # noqa: E712
    if included.empty or set(included["sample_arm"]) != set(ARMS):
        raise ValueError("Included consensus rows must contain both study arms.")
    if included[["consensus_high_level_pattern", "consensus_sub_pattern"]].isna().any().any():
        raise ValueError("Included consensus rows contain missing labels.")
    categories = sorted(included["consensus_high_level_pattern"].unique())
    table = pd.crosstab(included["sample_arm"], included["consensus_high_level_pattern"]).reindex(
        index=ARMS, columns=categories, fill_value=0
    )
    if len(categories) < 2:
        raise ValueError("Primary analysis requires at least two high-level patterns.")
    chi2, p_value, dof, expected = chi2_contingency(table.to_numpy(), correction=False)
    expected_frame = pd.DataFrame(expected, index=ARMS, columns=categories).rename_axis("sample_arm").reset_index()

    distribution_rows = []
    for arm in ARMS:
        total = int(table.loc[arm].sum())
        for category in categories:
            count = int(table.loc[arm, category])
            distribution_rows.append(
                {"sample_arm": arm, "high_level_pattern": category, "count": count, "total": total, "proportion": count / total}
            )
    distribution = pd.DataFrame(distribution_rows)

    tests = []
    for category in categories:
        a = int(table.loc[ARMS[0], category])
        c = int(table.loc[ARMS[1], category])
        values = np.array([[a, int(table.loc[ARMS[0]].sum()) - a], [c, int(table.loc[ARMS[1]].sum()) - c]])
        tests.append({"high_level_pattern": category, "agentic_count": a, "human_candidate_count": c, **category_effects(values)})
    tests_frame = pd.DataFrame(tests)
    tests_frame["holm_adjusted_p_value"] = holm_adjust(tests_frame["fisher_p_value"].tolist())
    tests_frame["holm_reject_0_05"] = tests_frame["holm_adjusted_p_value"].le(0.05)

    summary = {
        "analysis_population": "included atomic 2-of-3 consensus rows",
        "included_rows": len(included),
        "included_arm_counts": included["sample_arm"].value_counts().sort_index().to_dict(),
        "pearson_chi_square": {
            "statistic": float(chi2),
            "degrees_of_freedom": int(dof),
            "p_value": float(p_value),
            "minimum_expected_cell": float(np.min(expected)),
        },
        "cramers_v": float(cramers_v(table.to_numpy())),
        "cramers_v_bootstrap_95_ci": cluster_bootstrap_cramers_v(included, bootstrap_iterations, seed),
        "category_effect_method": {
            "contrast": "agentic versus human_candidate",
            "confidence_level": 0.95,
            "confidence_intervals": "Wald risk difference and log-Wald risk/odds ratios",
            "zero_cell_handling": "Haldane-Anscombe 0.5 correction for risk and odds ratios only",
            "multiple_testing": "Holm family-wise correction over high-level patterns",
        },
        "category_fisher_tests": json.loads(tests_frame.to_json(orient="records")),
        "subpattern_richness": subpattern_richness(included, permutation_iterations, seed),
    }
    return distribution, expected_frame, tests_frame, summary


def plot_distribution(distribution: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    order = (
        distribution.groupby("high_level_pattern")["count"]
        .sum()
        .sort_values(ascending=False)
        .index
    )
    pivot = (
        distribution.pivot(
            index="high_level_pattern", columns="sample_arm", values="proportion"
        )
        .reindex(index=order, columns=ARMS, fill_value=0)
        * 100
    )
    labels = [textwrap.fill(label, 32) for label in pivot.index]
    fig, ax = plt.subplots(
        figsize=(13, max(6, 0.6 * len(pivot))), constrained_layout=True
    )
    positions = np.arange(len(pivot))
    height = 0.35
    agentic_bars = ax.barh(
        positions - height / 2,
        pivot[ARMS[0]],
        height,
        label="Agentic",
        color="#9BBCE8",
    )
    human_bars = ax.barh(
        positions + height / 2,
        pivot[ARMS[1]],
        height,
        label="Human candidate",
        color="#B9DAB9",
    )
    for bars in (agentic_bars, human_bars):
        ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(40, float(pivot.to_numpy().max()) * 1.25))
    ax.set_xlabel("Percentage of Optimization Patterns (%)", fontsize=14)
    ax.set_title(
        "Distribution of Performance Optimization Patterns in Agentic and Human-Candidate PRs",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )
    ax.legend(title="Study Arm", fontsize=13, title_fontsize=14)
    ax.grid(False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus", type=Path, default=base / "consensus" / "rq1_consensus.parquet")
    parser.add_argument("--output-dir", type=Path, default=base / "comparison")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260911)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distribution, expected, tests, summary = analyze_comparison(
        pd.read_parquet(args.consensus), args.bootstrap_iterations, args.permutation_iterations, args.seed
    )
    summary["source_sha256"] = {"consensus": sha256_file(args.consensus)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(args.output_dir / "high_level_pattern_distribution.csv", index=False)
    expected.to_csv(args.output_dir / "high_level_pattern_expected_cells.csv", index=False)
    tests.to_csv(args.output_dir / "high_level_pattern_fisher_tests.csv", index=False)
    (args.output_dir / "rq1_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_distribution(distribution, args.output_dir / "rq1_high_level_pattern_comparison.png")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
