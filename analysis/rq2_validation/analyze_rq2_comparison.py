"""Compare RQ2 consensus outcomes between the agentic and human arms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ARMS = ("agentic", "human_candidate")
VALIDATION_TYPES = ("benchmark", "profiling", "static-reasoning", "anecdotal")
Z_95 = 1.959963984540054


def wilson_interval(events: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    p = events / total
    denominator = 1 + Z_95**2 / total
    center = (p + Z_95**2 / (2 * total)) / denominator
    half = Z_95 * math.sqrt(p * (1 - p) / total + Z_95**2 / (4 * total**2)) / denominator
    return center - half, center + half


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _ratio_ci(a: int, n1: int, c: int, n0: int, *, odds: bool) -> tuple[float | None, float | None, float | None]:
    b, d = n1 - a, n0 - c
    if n1 == 0 or n0 == 0:
        return None, None, None
    if odds:
        estimate = None if c * b == 0 else (a * d) / (b * c)
        aa, bb, cc, dd = (a, b, c, d) if min(a, b, c, d) > 0 else (a + .5, b + .5, c + .5, d + .5)
        corrected = (aa * dd) / (bb * cc)
        se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    else:
        estimate = None if c == 0 else (a / n1) / (c / n0)
        aa, cc = (a, c) if min(a, c) > 0 else (a + .5, c + .5)
        nn1, nn0 = (n1, n0) if min(a, c) > 0 else (n1 + 1, n0 + 1)
        corrected = (aa / nn1) / (cc / nn0)
        se = math.sqrt(max(0.0, 1 / aa - 1 / nn1 + 1 / cc - 1 / nn0))
    return estimate, math.exp(math.log(corrected) - Z_95 * se), math.exp(math.log(corrected) + Z_95 * se)


def binary_comparison(frame: pd.DataFrame, event: pd.Series, label: str) -> dict[str, Any]:
    event = pd.Series(event, index=frame.index).astype(bool)
    counts = []
    for arm in ARMS:
        selected = event[frame["sample_arm"].eq(arm)]
        counts.append((int(selected.sum()), len(selected)))
    (a, n1), (c, n0) = counts
    if n1 == 0 or n0 == 0:
        raise ValueError(f"{label} comparison requires observations in both sample arms.")
    table = np.array([[a, n1 - a], [c, n0 - c]], dtype=int)
    if np.any(table.sum(axis=0) == 0):
        chi2, pearson_p = 0.0, 1.0
        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    else:
        chi2, pearson_p, _, expected = stats.chi2_contingency(table)
    odds_ratio, fisher_p = stats.fisher_exact(table)
    p1, p0 = a / n1, c / n0
    low1, high1 = wilson_interval(a, n1)
    low0, high0 = wilson_interval(c, n0)
    rr, rr_low, rr_high = _ratio_ci(a, n1, c, n0, odds=False)
    odds, odds_low, odds_high = _ratio_ci(a, n1, c, n0, odds=True)
    return {
        "label": label,
        "table": table.tolist(),
        "agentic_events": a,
        "agentic_total": n1,
        "human_candidate_events": c,
        "human_candidate_total": n0,
        "agentic_rate": p1,
        "human_candidate_rate": p0,
        "pearson_chi2": float(chi2),
        "pearson_p": float(pearson_p),
        "fisher_odds_ratio": None if not np.isfinite(odds_ratio) else float(odds_ratio),
        "fisher_p": float(fisher_p),
        "risk_difference": p1 - p0,
        "risk_difference_ci95": [low1 - high0, high1 - low0],
        "risk_ratio": rr,
        "risk_ratio_ci95": [rr_low, rr_high],
        "odds_ratio": odds,
        "odds_ratio_ci95": [odds_low, odds_high],
        "phi_cramers_v": math.sqrt(float(chi2) / table.sum()),
        "expected_cells": expected.tolist(),
    }


def cramers_v(table: np.ndarray) -> float:
    if table.sum() == 0 or np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    return math.sqrt(float(chi2) / (table.sum() * min(table.shape[0] - 1, table.shape[1] - 1)))


def cluster_bootstrap_cramers_v(frame: pd.DataFrame, *, iterations: int = 2000, seed: int = 20260911) -> list[float]:
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive.")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    clusters = list(frame["repo_id"].drop_duplicates())
    cluster_index = {repo_id: index for index, repo_id in enumerate(clusters)}
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    type_index = {label: index for index, label in enumerate(VALIDATION_TYPES)}
    cluster_tables = np.zeros((len(clusters), len(ARMS), len(VALIDATION_TYPES)), dtype=int)
    for row in frame.itertuples(index=False):
        cluster_tables[
            cluster_index[row.repo_id],
            arm_index[row.sample_arm],
            type_index[row.consensus_primary_validation_type],
        ] += 1
    for _ in range(iterations):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        weights = np.bincount(sampled, minlength=len(clusters))
        table = np.tensordot(weights, cluster_tables, axes=(0, 0))
        values.append(cramers_v(table))
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def _apply_holm(results: list[dict[str, Any]]) -> None:
    adjusted = holm_adjust([result["fisher_p"] for result in results])
    for result, value in zip(results, adjusted):
        result["holm_adjusted_p"] = value
        result["holm_reject_0_05"] = value <= .05


def analyze(frame: pd.DataFrame, *, bootstrap_iterations: int = 2000, bootstrap_seed: int = 20260911) -> dict[str, Any]:
    required = {
        "repo_id", "number", "sample_arm", "consensus_validation_present",
        "consensus_status", "consensus_primary_validation_type",
        "included_in_stage2_analysis", "multilabel_status",
        "consensus_validation_types", "included_in_multilabel_analysis",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Consensus is missing columns: {sorted(missing)}")
    if frame.duplicated(["repo_id", "number"]).any() or not frame["sample_arm"].isin(ARMS).all():
        raise ValueError("Consensus contains invalid identities or sample arms.")
    if not frame["consensus_status"].isin(("full_unanimous", "exact_majority", "unresolved")).all():
        raise ValueError("Consensus contains an unknown status.")
    if not frame["consensus_validation_present"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("consensus_validation_present must be boolean.")
    expected_inclusion = frame["consensus_validation_present"] & ~frame["consensus_status"].eq("unresolved")
    if not frame["included_in_stage2_analysis"].eq(expected_inclusion).all():
        raise ValueError("Stage 2 inclusion is inconsistent with consensus status.")
    if not frame["multilabel_status"].isin(("full_unanimous", "exact_majority", "unresolved", "not_applicable_absent")).all():
        raise ValueError("Consensus contains an unknown multi-label status.")
    expected_multilabel = frame["consensus_validation_present"] & frame["multilabel_status"].isin(("full_unanimous", "exact_majority"))
    if not frame["included_in_multilabel_analysis"].eq(expected_multilabel).all():
        raise ValueError("Multi-label inclusion is inconsistent with multi-label status.")
    absent = frame[~frame["consensus_validation_present"]]
    if not absent["consensus_primary_validation_type"].eq("none").all() or not absent[
        "consensus_validation_types"
    ].map(lambda values: isinstance(values, (list, tuple, np.ndarray)) and len(values) == 0).all():
        raise ValueError("Absent consensus rows must use the canonical absent label.")
    presence = binary_comparison(frame, frame["consensus_validation_present"], "validation_present")
    positive = frame[frame["consensus_validation_present"]].copy()
    unresolved = binary_comparison(positive, positive["consensus_status"].eq("unresolved"), "stage2_unresolved")
    stage2 = frame[frame["included_in_stage2_analysis"]].copy()
    if stage2.empty:
        raise ValueError("Stage 2 has no resolved positive consensuses.")
    if not stage2["consensus_primary_validation_type"].isin(VALIDATION_TYPES).all():
        raise ValueError("Stage 2 contains unknown primary types.")
    multilabel = frame[frame["included_in_multilabel_analysis"]].copy()
    for _, row in multilabel.iterrows():
        types = row["consensus_validation_types"]
        if not isinstance(types, (list, tuple, np.ndarray)):
            raise ValueError("Resolved multi-label type sets must be lists.")
        if len(types) != len(set(types)) or set(types) - set(VALIDATION_TYPES):
            raise ValueError("Multi-label analysis contains invalid validation type sets.")
        if row["consensus_primary_validation_type"] not in types:
            raise ValueError("Multi-label primary type must belong to its validation type set.")
    if len(frame) == 2258:
        controls = {
            "stage1 positive": (int(frame["consensus_validation_present"].sum()), 1839),
            "stage2 consensus": (len(stage2), 1819),
            "stage2 unresolved": (int(positive["consensus_status"].eq("unresolved").sum()), 20),
            "multilabel consensus": (len(multilabel), 1707),
            "multilabel unresolved": (int(positive["multilabel_status"].eq("unresolved").sum()), 132),
        }
        failures = {name: values for name, values in controls.items() if values[0] != values[1]}
        if failures:
            raise ValueError(f"Official RQ2 analysis controls failed: {failures}")
    table_frame = pd.crosstab(stage2["sample_arm"], stage2["consensus_primary_validation_type"]).reindex(index=ARMS, columns=VALIDATION_TYPES, fill_value=0)
    chi2, p_value, dof, expected = stats.chi2_contingency(table_frame.to_numpy(), correction=False)
    primary = {
        "table": table_frame.to_dict(orient="split"),
        "chi2": float(chi2),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_value),
        "expected_cells": expected.tolist(),
        "minimum_expected_cell": float(expected.min()),
        "cramers_v": cramers_v(table_frame.to_numpy()),
        "cramers_v_cluster_bootstrap_ci95": cluster_bootstrap_cramers_v(stage2, iterations=bootstrap_iterations, seed=bootstrap_seed),
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
    }
    per_primary = [binary_comparison(stage2, stage2["consensus_primary_validation_type"].eq(label), label) for label in VALIDATION_TYPES]
    per_type = [binary_comparison(multilabel, multilabel["consensus_validation_types"].map(lambda values, target=label: target in values), label) for label in VALIDATION_TYPES]
    _apply_holm(per_primary)
    _apply_holm(per_type)
    return {
        "rows": len(frame),
        "stage1_rows": len(frame),
        "stage2_positive_consensus_rows": len(stage2),
        "multilabel_positive_consensus_rows": len(multilabel),
        "stage1_presence": presence,
        "stage2_unresolved": unresolved,
        "stage2_primary": primary,
        "stage2_per_primary": per_primary,
        "stage2_per_validation_type": per_type,
    }


def _flat_results(result: dict[str, Any]) -> pd.DataFrame:
    def flatten(value: dict[str, Any]) -> dict[str, Any]:
        row = {key: item for key, item in value.items() if not isinstance(item, (list, dict))}
        for key, item in value.items():
            if key.endswith("ci95") and isinstance(item, list) and len(item) == 2:
                row[f"{key}_low"], row[f"{key}_high"] = item
        return row

    rows = []
    for family, values in (("stage1", [result["stage1_presence"]]), ("stage2_unresolved", [result["stage2_unresolved"]]), ("stage2_primary_binary", result["stage2_per_primary"]), ("stage2_multilabel", result["stage2_per_validation_type"])):
        for value in values:
            rows.append({"family": family, **flatten(value)})
    rows.append({"family": "stage2_primary_omnibus", "label": "primary_validation_type", **flatten(result["stage2_primary"])})
    return pd.DataFrame(rows)


def plot_comparison(frame: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    stage2 = frame[frame["included_in_stage2_analysis"]]
    order = ("static-reasoning", "benchmark", "anecdotal", "profiling")
    labels = ("Static Reasoning", "Benchmark-Based", "Anecdotal", "Profiling-Based")
    primary = (
        pd.crosstab(
            stage2["consensus_primary_validation_type"],
            stage2["sample_arm"],
            normalize="columns",
        )
        .reindex(index=order, columns=ARMS, fill_value=0)
        * 100
    )

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    positions = np.arange(len(order))
    width = 0.38
    agentic_bars = axis.bar(
        positions - width / 2,
        primary[ARMS[0]],
        width,
        label="Agentic",
        color="#9BBCE8",
    )
    human_bars = axis.bar(
        positions + width / 2,
        primary[ARMS[1]],
        width,
        label="Human candidate",
        color="#B9DAB9",
    )
    for bars in (agentic_bars, human_bars):
        axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)
    axis.set_xticks(positions, labels, rotation=30)
    axis.set_ylim(0, max(70, float(primary.to_numpy().max()) * 1.15))
    axis.set_ylabel("Percentage of PRs", fontsize=14)
    axis.set_title(
        "Comparison of Performance Validation Methods in Agentic vs. Human-Candidate PRs",
        fontsize=17,
        pad=12,
    )
    axis.legend(title="Study Arm", loc="upper right")
    axis.grid(axis="y", color="#CCCCCC")
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.consensus)
    result = analyze(frame, bootstrap_iterations=args.bootstrap_iterations, bootstrap_seed=args.bootstrap_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rq2_comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _flat_results(result).to_csv(args.output_dir / "rq2_comparison.csv", index=False)
    plot_comparison(frame, args.output_dir / "rq2_comparison.png")
    print(json.dumps({"rows": result["rows"], "stage2_rows": result["stage2_positive_consensus_rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
