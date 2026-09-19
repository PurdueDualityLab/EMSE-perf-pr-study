"""Track RQ2 validation-presence reporting over the sampling window.

The aggregate RQ2 comparison collapses 65 ISO weeks into one 2x2 table. This
script keeps the time dimension, which the weekly-balanced design supports
directly: every stratum holds equal numbers of agentic and human-candidate PRs,
so calendar time cannot confound the author-type contrast within a stratum.

It reports stratified and regression estimates of the author-type difference,
tests whether that difference is constant over the window, and tests the trend
within each author type. The reported primary-type mix is analyzed the same way as a secondary
outcome.
"""

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
OFFICIAL_CONTROLS = {
    "rows": 2258,
    "strata": 65,
    "agentic_positive": 929,
    "agentic_total": 1129,
    "human_candidate_positive": 910,
    "human_candidate_total": 1129,
    "stage2_rows": 1819,
}


def wilson_interval(events: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    p = events / total
    denominator = 1 + Z_95**2 / total
    center = (p + Z_95**2 / (2 * total)) / denominator
    half = Z_95 * math.sqrt(p * (1 - p) / total + Z_95**2 / (4 * total**2)) / denominator
    return center - half, center + half


def newcombe_interval(a: int, n1: int, c: int, n0: int) -> tuple[float | None, float | None]:
    """Newcombe's square-and-add interval for a difference of proportions."""
    if n1 == 0 or n0 == 0:
        return None, None
    low1, high1 = wilson_interval(a, n1)
    low0, high0 = wilson_interval(c, n0)
    difference = a / n1 - c / n0
    lower = difference - math.sqrt((a / n1 - low1) ** 2 + (high0 - c / n0) ** 2)
    upper = difference + math.sqrt((high1 - a / n1) ** 2 + (c / n0 - low0) ** 2)
    return lower, upper


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------


def prepare(labels: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    """Join labels with creation timestamps and rebuild the sampling strata."""
    keys = ["repo_id", "number"]
    for frame, name in ((labels, "labels"), (dates, "dates")):
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} contain duplicate PR identities.")
    frame = labels.merge(dates[[*keys, "created_at"]], on=keys, validate="one_to_one")
    if len(frame) != len(labels):
        raise ValueError("Creation timestamps do not cover every labeled PR.")
    if not frame["sample_arm"].isin(ARMS).all():
        raise ValueError("Labels contain an unknown sample arm.")
    timestamps = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Input contains invalid created_at values.")
    calendar = timestamps.dt.isocalendar()
    frame["created_at"] = timestamps
    frame["sampling_stratum"] = (
        calendar["year"].astype(int).astype(str)
        + "-W"
        + calendar["week"].astype(int).astype(str).str.zfill(2)
    )
    frame["month"] = timestamps.dt.strftime("%Y-%m")
    frame["quarter"] = timestamps.dt.year.astype(str) + "-Q" + timestamps.dt.quarter.astype(str)
    origin = timestamps.min()
    frame["years_since_start"] = (timestamps - origin).dt.total_seconds() / (365.25 * 24 * 3600)
    frame["validation_present"] = frame["consensus_validation_present"].astype(bool)
    return frame.sort_values(["created_at", *keys], kind="mergesort").reset_index(drop=True)


def check_controls(frame: pd.DataFrame) -> dict[str, Any]:
    """Verify the rebuilt strata reproduce the published sampling design."""
    table = pd.crosstab(frame["sampling_stratum"], frame["sample_arm"]).reindex(columns=ARMS, fill_value=0)
    unbalanced = table.loc[table["agentic"] != table["human_candidate"]]
    observed = {
        "rows": len(frame),
        "strata": len(table),
        "agentic_positive": int(frame.loc[frame["sample_arm"].eq("agentic"), "validation_present"].sum()),
        "agentic_total": int(frame["sample_arm"].eq("agentic").sum()),
        "human_candidate_positive": int(frame.loc[frame["sample_arm"].eq("human_candidate"), "validation_present"].sum()),
        "human_candidate_total": int(frame["sample_arm"].eq("human_candidate").sum()),
        "stage2_rows": int(frame["included_in_stage2_analysis"].astype(bool).sum()),
    }
    failures = {key: (observed[key], expected) for key, expected in OFFICIAL_CONTROLS.items() if observed[key] != expected}
    if failures:
        raise ValueError(f"Rebuilt temporal sample failed RQ2 controls: {failures}")
    if len(unbalanced):
        raise ValueError(f"Rebuilt strata are not 1:1 balanced: {unbalanced.to_dict(orient='index')}")
    return {
        **observed,
        "window_start": frame["created_at"].min().isoformat(),
        "window_end": frame["created_at"].max().isoformat(),
        "weekly_balance_verified": True,
    }


def period_table(frame: pd.DataFrame, event: pd.Series, period: str) -> pd.DataFrame:
    """Per-period 2x2 counts with Wilson intervals and the author-type difference."""
    event = pd.Series(event, index=frame.index).astype(bool)
    rows = []
    for name, group in frame.groupby(period, sort=True):
        selected = event.loc[group.index]
        counts = {arm: (int(selected[group["sample_arm"].eq(arm)].sum()), int(group["sample_arm"].eq(arm).sum())) for arm in ARMS}
        (a, n1), (c, n0) = counts["agentic"], counts["human_candidate"]
        low1, high1 = wilson_interval(a, n1)
        low0, high0 = wilson_interval(c, n0)
        rows.append({
            "period": name,
            "agentic_events": a,
            "agentic_total": n1,
            "agentic_rate": a / n1 if n1 else None,
            "agentic_ci95_low": low1,
            "agentic_ci95_high": high1,
            "human_candidate_events": c,
            "human_candidate_total": n0,
            "human_candidate_rate": c / n0 if n0 else None,
            "human_candidate_ci95_low": low0,
            "human_candidate_ci95_high": high0,
            "risk_difference": (a / n1 - c / n0) if n1 and n0 else None,
            "risk_difference_ci95_low": (low1 - high0) if n1 and n0 else None,
            "risk_difference_ci95_high": (high1 - low0) if n1 and n0 else None,
            "risk_difference_newcombe_ci95_low": newcombe_interval(a, n1, c, n0)[0],
            "risk_difference_newcombe_ci95_high": newcombe_interval(a, n1, c, n0)[1],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Stratified analysis
# ----------------------------------------------------------------------------


def _stratum_tables(frame: pd.DataFrame, event: pd.Series, stratum: str) -> list[tuple[int, int, int, int]]:
    event = pd.Series(event, index=frame.index).astype(bool)
    tables = []
    for _, group in frame.groupby(stratum, sort=True):
        selected = event.loc[group.index]
        agentic = group["sample_arm"].eq("agentic")
        a = int(selected[agentic].sum())
        b = int(agentic.sum()) - a
        c = int(selected[~agentic].sum())
        d = int((~agentic).sum()) - c
        tables.append((a, b, c, d))
    return tables


def mantel_haenszel(tables: list[tuple[int, int, int, int]]) -> dict[str, Any]:
    """Common odds ratio across strata with the Robins-Breslow-Greenland interval."""
    numerator = denominator = 0.0
    observed = expected = variance = 0.0
    pr = qr = ps = qs = 0.0
    used = 0
    for a, b, c, d in tables:
        total = a + b + c + d
        if total == 0 or (a + c) in (0, total) or (a + b) in (0, total):
            continue
        used += 1
        numerator += a * d / total
        denominator += b * c / total
        observed += a
        expected += (a + b) * (a + c) / total
        variance += (a + b) * (c + d) * (a + c) * (b + d) / (total**2 * (total - 1))
        p, q = (a + d) / total, (b + c) / total
        r, s = a * d / total, b * c / total
        pr += p * r
        qr += q * r
        ps += p * s
        qs += q * s
    if denominator == 0 or numerator == 0:
        raise ValueError("Mantel-Haenszel estimation requires informative strata.")
    odds_ratio = numerator / denominator
    se = math.sqrt(
        pr / (2 * numerator**2)
        + (qr + ps) / (2 * numerator * denominator)
        + qs / (2 * denominator**2)
    )
    chi2 = (abs(observed - expected) - 0.5) ** 2 / variance
    return {
        "informative_strata": used,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci95": [math.exp(math.log(odds_ratio) - Z_95 * se), math.exp(math.log(odds_ratio) + Z_95 * se)],
        "log_odds_ratio_se": se,
        "chi2_continuity_corrected": chi2,
        "p_value": float(stats.chi2.sf(chi2, 1)),
    }


def breslow_day(tables: list[tuple[int, int, int, int]], odds_ratio: float) -> dict[str, Any]:
    """Breslow-Day homogeneity test of the odds ratio, with Tarone correction."""
    statistic = 0.0
    residual_sum = variance_sum = 0.0
    used = 0
    for a, b, c, d in tables:
        total = a + b + c + d
        n1, m1 = a + b, a + c
        if total == 0 or m1 in (0, total) or n1 in (0, total):
            continue
        if abs(odds_ratio - 1.0) < 1e-12:
            fitted = n1 * m1 / total
        else:
            alpha = 1 - odds_ratio
            beta = (total - n1 - m1) + odds_ratio * (n1 + m1)
            gamma = -odds_ratio * n1 * m1
            root = math.sqrt(max(beta**2 - 4 * alpha * gamma, 0.0))
            candidates = [(-beta + root) / (2 * alpha), (-beta - root) / (2 * alpha)]
            low, high = max(0.0, n1 + m1 - total), min(n1, m1)
            fitted = next((value for value in candidates if low - 1e-9 <= value <= high + 1e-9), None)
            if fitted is None:
                continue
            fitted = min(max(fitted, low + 1e-9), high - 1e-9)
        cells = (fitted, n1 - fitted, m1 - fitted, total - n1 - m1 + fitted)
        if min(cells) <= 0:
            continue
        used += 1
        cell_variance = 1.0 / sum(1.0 / value for value in cells)
        statistic += (a - fitted) ** 2 / cell_variance
        residual_sum += a - fitted
        variance_sum += cell_variance
    degrees = max(used - 1, 0)
    tarone = statistic - (residual_sum**2 / variance_sum if variance_sum else 0.0)
    return {
        "informative_strata": used,
        "degrees_of_freedom": degrees,
        "breslow_day_chi2": statistic,
        "breslow_day_p": float(stats.chi2.sf(statistic, degrees)) if degrees else None,
        "tarone_chi2": tarone,
        "tarone_p": float(stats.chi2.sf(tarone, degrees)) if degrees else None,
    }


# ----------------------------------------------------------------------------
# Logistic regression
# ----------------------------------------------------------------------------


def fit_logistic(design: np.ndarray, response: np.ndarray, *, tolerance: float = 1e-10, iterations: int = 100) -> dict[str, Any]:
    """Newton-Raphson logistic fit with model-based and cluster-robust variances."""
    beta = np.zeros(design.shape[1])
    log_likelihood = -np.inf
    for _ in range(iterations):
        eta = design @ beta
        probability = 1.0 / (1.0 + np.exp(-eta))
        weights = np.clip(probability * (1 - probability), 1e-12, None)
        gradient = design.T @ (response - probability)
        hessian = design.T @ (design * weights[:, None])
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        eta = design @ beta
        probability = 1.0 / (1.0 + np.exp(-eta))
        current = float(np.sum(response * eta - np.log1p(np.exp(eta))))
        if abs(current - log_likelihood) < tolerance:
            log_likelihood = current
            break
        log_likelihood = current
    eta = design @ beta
    probability = 1.0 / (1.0 + np.exp(-eta))
    weights = np.clip(probability * (1 - probability), 1e-12, None)
    bread = np.linalg.inv(design.T @ (design * weights[:, None]))
    return {
        "beta": beta,
        "covariance": bread,
        "bread": bread,
        "residual": response - probability,
        "log_likelihood": log_likelihood,
        "observations": len(response),
    }


def cluster_covariance(fit: dict[str, Any], design: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Sandwich covariance with clustering on repository, plus a finite-sample factor."""
    scores = design * fit["residual"][:, None]
    unique = np.unique(clusters)
    meat = np.zeros((design.shape[1], design.shape[1]))
    for cluster in unique:
        total = scores[clusters == cluster].sum(axis=0)
        meat += np.outer(total, total)
    groups, observations, parameters = len(unique), design.shape[0], design.shape[1]
    correction = (groups / (groups - 1)) * ((observations - 1) / (observations - parameters))
    return fit["bread"] @ (correction * meat) @ fit["bread"]


def _term(name: str, index: int, beta: np.ndarray, covariance: np.ndarray, robust: np.ndarray) -> dict[str, Any]:
    def summarize(matrix: np.ndarray) -> dict[str, Any]:
        se = math.sqrt(matrix[index, index])
        z = beta[index] / se
        return {
            "standard_error": se,
            "z": z,
            "p_value": float(2 * stats.norm.sf(abs(z))),
            "odds_ratio": math.exp(beta[index]),
            "odds_ratio_ci95": [math.exp(beta[index] - Z_95 * se), math.exp(beta[index] + Z_95 * se)],
        }

    return {"term": name, "coefficient": float(beta[index]), "model_based": summarize(covariance), "cluster_robust": summarize(robust)}


def _linear_combination(weights: np.ndarray, beta: np.ndarray, covariance: np.ndarray, robust: np.ndarray) -> dict[str, Any]:
    value = float(weights @ beta)
    result: dict[str, Any] = {"coefficient": value, "odds_ratio": math.exp(value)}
    for name, matrix in (("model_based", covariance), ("cluster_robust", robust)):
        se = math.sqrt(float(weights @ matrix @ weights))
        result[name] = {
            "standard_error": se,
            "p_value": float(2 * stats.norm.sf(abs(value / se))),
            "odds_ratio_ci95": [math.exp(value - Z_95 * se), math.exp(value + Z_95 * se)],
        }
    return result


def trend_model(frame: pd.DataFrame, event: pd.Series, label: str) -> dict[str, Any]:
    """Logistic model of the outcome on author type, calendar time, and their interaction."""
    event = pd.Series(event, index=frame.index).astype(bool)
    response = event.to_numpy(dtype=float)
    agentic = frame["sample_arm"].eq("agentic").to_numpy(dtype=float)
    time = frame["years_since_start"].to_numpy(dtype=float)
    centre = float(time.mean())
    centred = time - centre
    clusters = frame["repo_id"].to_numpy()
    full_design = np.column_stack([np.ones_like(response), agentic, centred, agentic * centred])
    additive_design = full_design[:, :3]
    arm_only_design = full_design[:, :2]

    full = fit_logistic(full_design, response)
    additive = fit_logistic(additive_design, response)
    arm_only = fit_logistic(arm_only_design, response)
    robust_full = cluster_covariance(full, full_design, clusters)
    robust_additive = cluster_covariance(additive, additive_design, clusters)

    beta = full["beta"]
    covariance, robust = full["covariance"], robust_full
    span = float(time.max() - time.min())
    terms = {
        "author_type_at_window_midpoint": _term("agentic", 1, beta, covariance, robust),
        "time_human_candidate": _term("years", 2, beta, covariance, robust),
        "author_type_by_time": _term("agentic x years", 3, beta, covariance, robust),
    }
    interaction_lrt = 2 * (full["log_likelihood"] - additive["log_likelihood"])
    time_lrt = 2 * (additive["log_likelihood"] - arm_only["log_likelihood"])
    return {
        "outcome": label,
        "observations": full["observations"],
        "clusters": int(len(np.unique(clusters))),
        "time_unit": "years since the first PR in the window",
        "time_centre_years": centre,
        "window_span_years": span,
        "terms": terms,
        "agentic_time_slope": _linear_combination(np.array([0.0, 0.0, 1.0, 1.0]), beta, covariance, robust),
        "author_type_at_window_start": _linear_combination(np.array([0.0, 1.0, 0.0, -centre]), beta, covariance, robust),
        "author_type_at_window_end": _linear_combination(np.array([0.0, 1.0, 0.0, span - centre]), beta, covariance, robust),
        "interaction_likelihood_ratio": {
            "chi2": float(interaction_lrt),
            "degrees_of_freedom": 1,
            "p_value": float(stats.chi2.sf(interaction_lrt, 1)),
        },
        "time_likelihood_ratio": {
            "chi2": float(time_lrt),
            "degrees_of_freedom": 1,
            "p_value": float(stats.chi2.sf(time_lrt, 1)),
        },
        "time_adjusted_author_type_odds_ratio": _linear_combination(
            np.array([0.0, 1.0, 0.0]), additive["beta"], additive["covariance"], robust_additive
        ),
    }


# ----------------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------------


def outcome_analysis(frame: pd.DataFrame, event: pd.Series, label: str) -> dict[str, Any]:
    weekly = _stratum_tables(frame, event, "sampling_stratum")
    quarterly = _stratum_tables(frame, event, "quarter")
    common = mantel_haenszel(weekly)
    quarters = period_table(frame, event, "quarter")
    halves = period_table(frame, event, "window_half")
    half_tests = []
    for row in halves.itertuples(index=False):
        table = np.array([
            [row.agentic_events, row.agentic_total - row.agentic_events],
            [row.human_candidate_events, row.human_candidate_total - row.human_candidate_events],
        ])
        chi2, p_value = stats.chi2_contingency(table, correction=True)[:2]
        half_tests.append({
            "period": row.period,
            "table": table.tolist(),
            "agentic_rate": row.agentic_rate,
            "human_candidate_rate": row.human_candidate_rate,
            "risk_difference": row.risk_difference,
            "risk_difference_ci95": [row.risk_difference_ci95_low, row.risk_difference_ci95_high],
            "risk_difference_newcombe_ci95": [
                row.risk_difference_newcombe_ci95_low,
                row.risk_difference_newcombe_ci95_high,
            ],
            "yates_chi2": float(chi2),
            "p_value": float(p_value),
            "fisher_p": float(stats.fisher_exact(table)[1]),
        })
    for test, adjusted in zip(half_tests, holm_adjust([test["fisher_p"] for test in half_tests])):
        test["holm_adjusted_p"] = adjusted
        test["holm_reject_0_05"] = adjusted <= 0.05
    return {
        "outcome": label,
        "mantel_haenszel_by_week": common,
        "mantel_haenszel_by_quarter": mantel_haenszel(quarterly),
        "homogeneity_by_week": breslow_day(weekly, common["odds_ratio"]),
        "homogeneity_by_quarter": breslow_day(quarterly, mantel_haenszel(quarterly)["odds_ratio"]),
        "trend_model": trend_model(frame, event, label),
        "window_half_tests": half_tests,
        "quarterly": quarters.to_dict(orient="records"),
    }


def type_shares(frame: pd.DataFrame, period: str, inclusion: str = "included_in_stage2_analysis") -> pd.DataFrame:
    """Primary-type composition among resolved positive PRs, by author type and period.

    Defaults to the 1,819-PR resolved-primary-type subset, the denominator the
    paper's validation-type analysis uses, so the temporal view matches it.
    """
    resolved = frame[frame[inclusion].astype(bool)]
    rows = []
    for (name, arm), group in resolved.groupby([period, "sample_arm"], sort=True):
        record = {"period": name, "sample_arm": arm, "total": len(group)}
        for label in VALIDATION_TYPES:
            events = int(group["consensus_primary_validation_type"].eq(label).sum())
            low, high = wilson_interval(events, len(group))
            record[f"{label}_events"] = events
            record[f"{label}_share"] = events / len(group) if len(group) else None
            record[f"{label}_ci95_low"], record[f"{label}_ci95_high"] = low, high
        rows.append(record)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------


def plot_temporal(quarterly: pd.DataFrame, benchmark: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"agentic": "#9BBCE8", "human_candidate": "#B9DAB9"}
    labels = {"agentic": "Agentic", "human_candidate": "Human candidate"}
    periods = list(quarterly["period"])
    positions = np.arange(len(periods))

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True, sharex=True)

    axis = axes[0]
    for arm in ARMS:
        rate = quarterly[f"{arm}_rate"].to_numpy(dtype=float) * 100
        low = quarterly[f"{arm}_ci95_low"].to_numpy(dtype=float) * 100
        high = quarterly[f"{arm}_ci95_high"].to_numpy(dtype=float) * 100
        axis.errorbar(
            positions, rate, yerr=[rate - low, high - rate], marker="o", capsize=3,
            color=colors[arm], markeredgecolor="#444444", linewidth=2, label=labels[arm],
        )
    axis.set_ylabel("PRs reporting validation (%)", fontsize=12)
    axis.set_title("Reported performance validation over the sampling window", fontsize=15, pad=10)
    axis.set_ylim(30, 100)
    axis.legend(title="Author type", loc="lower right")
    axis.grid(axis="y", color="#CCCCCC")

    axis = axes[1]
    for arm in ARMS:
        selected = benchmark[benchmark["sample_arm"].eq(arm)].set_index("period").reindex(periods)
        share = selected["benchmark_share"].to_numpy(dtype=float) * 100
        low = selected["benchmark_ci95_low"].to_numpy(dtype=float) * 100
        high = selected["benchmark_ci95_high"].to_numpy(dtype=float) * 100
        axis.errorbar(
            positions, share, yerr=[share - low, high - share], marker="o", capsize=3,
            color=colors[arm], markeredgecolor="#444444", linewidth=2, label=labels[arm],
        )
    axis.set_ylabel("Benchmark as primary (%)", fontsize=12)
    axis.set_title("Benchmark as primary evidence among resolved positive PRs", fontsize=13, pad=8)
    axis.set_ylim(0, 100)
    axis.legend(title="Author type", loc="lower right")
    axis.grid(axis="y", color="#CCCCCC")

    # The two panels have different denominators (all PRs above, resolved
    # positives below), so no single per-quarter count belongs on the shared
    # axis. They are carried in the caption and in the exported tables.
    for axis in axes:
        axis.set_xticks(positions, periods, rotation=0, fontsize=10)

    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def add_window_half(frame: pd.DataFrame) -> pd.DataFrame:
    """Split the window at the median ISO week, so each half holds 32-33 strata."""
    strata = sorted(frame["sampling_stratum"].unique())
    midpoint = strata[len(strata) // 2]
    result = frame.copy()
    result["window_half"] = np.where(
        result["sampling_stratum"] < midpoint,
        f"{strata[0]}..{strata[len(strata) // 2 - 1]}",
        f"{midpoint}..{strata[-1]}",
    )
    return result


def analyze(frame: pd.DataFrame) -> dict[str, Any]:
    controls = check_controls(frame)
    presence = outcome_analysis(frame, frame["validation_present"], "validation_present")
    stage2 = frame[frame["included_in_stage2_analysis"].astype(bool)].copy()
    benchmark = outcome_analysis(stage2, stage2["consensus_primary_validation_type"].eq("benchmark"), "benchmark_primary_type")
    multilabel = frame[frame["included_in_multilabel_analysis"].astype(bool)].copy()
    return {
        "controls": controls,
        "validation_presence": presence,
        "benchmark_primary_type": benchmark,
        "benchmark_primary_type_multilabel_subset": outcome_analysis(
            multilabel,
            multilabel["consensus_primary_validation_type"].eq("benchmark"),
            "benchmark_primary_type_multilabel_subset",
        ),
    }


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=root / "analysis/classification_labels/rq2_labels.csv")
    parser.add_argument("--dates", type=Path, default=here / "pr_created_at.csv")
    parser.add_argument("--output-dir", type=Path, default=here)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = add_window_half(prepare(pd.read_csv(args.labels), pd.read_csv(args.dates)))
    result = analyze(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rq2_temporal.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")

    weekly = period_table(frame, frame["validation_present"], "sampling_stratum")
    monthly = period_table(frame, frame["validation_present"], "month")
    quarterly = period_table(frame, frame["validation_present"], "quarter")
    weekly.to_csv(args.output_dir / "presence_by_week.csv", index=False)
    monthly.to_csv(args.output_dir / "presence_by_month.csv", index=False)
    quarterly.to_csv(args.output_dir / "presence_by_quarter.csv", index=False)
    benchmark = type_shares(frame, "quarter")
    benchmark.to_csv(args.output_dir / "primary_type_by_quarter.csv", index=False)
    type_shares(frame, "quarter", inclusion="included_in_multilabel_analysis").to_csv(
        args.output_dir / "primary_type_by_quarter_multilabel.csv", index=False
    )
    plot_temporal(quarterly, benchmark, args.output_dir / "rq2_temporal")
    print(json.dumps({
        "rows": result["controls"]["rows"],
        "strata": result["controls"]["strata"],
        "interaction_p": result["validation_presence"]["trend_model"]["interaction_likelihood_ratio"]["p_value"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
