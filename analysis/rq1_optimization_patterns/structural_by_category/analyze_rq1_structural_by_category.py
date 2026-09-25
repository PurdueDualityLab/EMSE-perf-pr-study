"""Compare base-to-head structural change between study arms within each RQ1 pattern.

Section 4.1 reports one aggregate structural contrast per metric. That contrast pools
every optimization category, so it cannot say whether the arms differ uniformly or
whether the difference is concentrated in particular kinds of optimization work. This
script joins the RQ1 consensus labels to the per-PR structural deltas and repeats the
arm contrast inside each high-level pattern, with Benjamini-Hochberg correction over the
patterns of a metric. It also runs a Kruskal-Wallis test across patterns with the arms pooled, to
report whether the structural change depends on the kind of optimization at all.

Both the percentage change and the absolute change are tested. Percentage change is the
measure Section 4.1 reports, but it divides by the base value, so a pattern whose files
are small can show a large percentage shift from a small edit; reporting the absolute
change alongside it shows whether a percentage result survives in natural units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, norm

ARMS = ("agentic", "human_candidate")
ALPHA = 0.05
DEFAULT_METRICS = ("nloc", "function_count")
MEASURES = {"delta_pct": "percent change, base to head", "delta": "absolute change, base to head"}


def benjamini_hochberg_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg step-up adjustment; returns q-values in the input order."""
    total = len(p_values)
    order = sorted(range(total), key=lambda index: (p_values[index], index))
    adjusted = [0.0] * total
    running = 1.0
    for rank in range(total - 1, -1, -1):
        index = order[rank]
        running = min(running, min(1.0, total / (rank + 1) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


def dominance(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    below = (x[:, None] > y[None, :]).sum(axis=1)
    above = (x[:, None] < y[None, :]).sum(axis=1)
    y_below = (y[:, None] < x[None, :]).sum(axis=1)
    y_above = (y[:, None] > x[None, :]).sum(axis=1)
    ties = len(x) * len(y) - int(below.sum()) - int(above.sum())
    return (below - above) / len(y), (y_above - y_below) / len(x), ties


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Cliff's delta with the consistent asymmetric interval of Cliff (1993)."""
    per_x, per_y, ties = dominance(x, y)
    n1, n2 = len(x), len(y)
    delta = float(per_x.mean())
    if min(n1, n2) < 2:
        return delta, float("nan"), float("nan")
    spread = (n2**2 * (n1 - 1) * per_x.var(ddof=1) + n1**2 * (n2 - 1) * per_y.var(ddof=1)
              - (n1 * n2 - ties - n1 * n2 * delta**2))
    sigma = float(np.sqrt(max(spread, 0) / (n1 * n2 * (n1 - 1) * (n2 - 1))))
    z = norm.ppf(1 - ALPHA / 2)
    half = z * sigma * np.sqrt((1 - delta**2) ** 2 + z**2 * sigma**2)
    denominator = 1 - delta**2 + z**2 * sigma**2
    return delta, float((delta - delta**3 - half) / denominator), float((delta - delta**3 + half) / denominator)


def hodges_lehmann(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.median(np.subtract.outer(x, y).ravel()))


def magnitude(delta: float) -> str:
    """Romano et al. thresholds, the convention used elsewhere in the study."""
    size = abs(delta)
    if size < 0.147:
        return "negligible"
    if size < 0.330:
        return "small"
    if size < 0.474:
        return "medium"
    return "large"


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().eq("true")


def load_joined(labels_path: Path, deltas_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = pd.read_csv(labels_path)
    included = labels[as_bool(labels["included_in_analysis"])].copy()
    if included.empty or set(included["sample_arm"]) != set(ARMS):
        raise ValueError("Included RQ1 label rows must contain both study arms.")
    if included.duplicated(["repo_id", "number"]).any():
        raise ValueError("RQ1 label artifact contains duplicate PR identities.")

    deltas = pd.read_csv(deltas_path)
    deltas = deltas[as_bool(deltas["complete_pr"])].copy()
    for column in ("delta", "delta_pct"):
        deltas[column] = pd.to_numeric(deltas[column], errors="coerce")
    deltas = deltas.replace([np.inf, -np.inf], np.nan)

    joined = included[["repo_id", "number", "sample_arm", "consensus_high_level_pattern"]].merge(
        deltas[["repo_id", "number", "metric", "delta", "delta_pct"]],
        on=["repo_id", "number"],
        how="inner",
    )
    labelled_prs = set(map(tuple, included[["repo_id", "number"]].to_numpy()))
    joined_prs = set(map(tuple, joined[["repo_id", "number"]].drop_duplicates().to_numpy()))
    coverage = {
        "rq1_included_prs": len(labelled_prs),
        "prs_with_structural_data": len(joined_prs),
        "coverage_fraction": len(joined_prs) / len(labelled_prs),
        "note": "PRs without complete structural pairs carry no parseable function-level metrics and are absent from the contrasts",
    }
    return joined, coverage


def contrast(frame: pd.DataFrame, measure: str) -> dict[str, Any] | None:
    x = frame.loc[frame["sample_arm"].eq(ARMS[0]), measure].dropna().to_numpy(dtype=float)
    y = frame.loc[frame["sample_arm"].eq(ARMS[1]), measure].dropna().to_numpy(dtype=float)
    if len(x) == 0 or len(y) == 0:
        return None
    statistic, p_value = mannwhitneyu(x, y, alternative="two-sided")
    delta, delta_low, delta_high = cliffs_delta(x, y)
    return {
        "n_agentic": len(x),
        "n_human_candidate": len(y),
        "median_agentic": float(np.median(x)),
        "median_human_candidate": float(np.median(y)),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "cliffs_delta": delta,
        "cliffs_delta_ci_low": delta_low,
        "cliffs_delta_ci_high": delta_high,
        "cliffs_delta_magnitude": magnitude(delta),
        "hodges_lehmann_shift": hodges_lehmann(x, y),
    }


def analyze(joined: pd.DataFrame, metrics: tuple[str, ...], min_group_size: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    across_patterns: dict[str, Any] = {}

    for metric in metrics:
        metric_frame = joined[joined["metric"].eq(metric)]
        if metric_frame.empty:
            raise ValueError(f"No structural rows for metric {metric!r}.")
        patterns = sorted(metric_frame["consensus_high_level_pattern"].unique())

        for measure in MEASURES:
            tested: list[dict[str, Any]] = []
            for pattern in patterns:
                pattern_frame = metric_frame[metric_frame["consensus_high_level_pattern"].eq(pattern)]
                result = contrast(pattern_frame, measure)
                base = {"metric": metric, "measure": measure, "high_level_pattern": pattern}
                if result is None or min(result["n_agentic"], result["n_human_candidate"]) < min_group_size:
                    counts = pattern_frame["sample_arm"].value_counts()
                    rows.append({**base, "n_agentic": int(counts.get(ARMS[0], 0)),
                                 "n_human_candidate": int(counts.get(ARMS[1], 0)), "tested": False,
                                 "skip_reason": f"fewer than {min_group_size} PRs in an arm"})
                    continue
                entry = {**base, **result, "tested": True, "skip_reason": None}
                tested.append(entry)
                rows.append(entry)

            adjusted = benjamini_hochberg_adjust([entry["p_value"] for entry in tested])
            for entry, value in zip(tested, adjusted):
                entry["bh_adjusted_p_value"] = value
                entry["bh_reject_0_05"] = bool(value <= ALPHA)

            groups = [
                metric_frame.loc[metric_frame["consensus_high_level_pattern"].eq(pattern), measure].dropna().to_numpy(dtype=float)
                for pattern in patterns
            ]
            groups = [group for group in groups if len(group) > 0]
            statistic, p_value = kruskal(*groups)
            across_patterns[f"{metric}::{measure}"] = {
                "metric": metric,
                "measure": measure,
                "test": "Kruskal-Wallis across high-level patterns, study arms pooled",
                "n_patterns": len(groups),
                "statistic": float(statistic),
                "degrees_of_freedom": len(groups) - 1,
                "p_value": float(p_value),
            }

    frame = pd.DataFrame(rows)
    ordering = ["metric", "measure", "high_level_pattern", "tested", "skip_reason", "n_agentic",
                "n_human_candidate", "median_agentic", "median_human_candidate", "statistic", "p_value",
                "bh_adjusted_p_value", "bh_reject_0_05", "cliffs_delta", "cliffs_delta_ci_low",
                "cliffs_delta_ci_high", "cliffs_delta_magnitude", "hodges_lehmann_shift"]
    frame = frame.reindex(columns=[column for column in ordering if column in frame.columns])
    frame = frame.sort_values(["metric", "measure", "high_level_pattern"]).reset_index(drop=True)
    return frame, across_patterns


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    analysis_root = base.parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=analysis_root / "classification_labels" / "rq1_labels.csv")
    parser.add_argument("--deltas", type=Path,
                        default=analysis_root / "quantitative_analysis" / "results" / "structural" / "pr_deltas.csv")
    parser.add_argument("--output-dir", type=Path, default=base / "results")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS),
                        help="structural metrics to contrast; avg_ccn is available but excluded by default")
    parser.add_argument("--min-group-size", type=int, default=5,
                        help="minimum PRs per arm for a pattern to be tested rather than reported as counts only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    joined, coverage = load_joined(args.labels, args.deltas)
    tests, across_patterns = analyze(joined, tuple(args.metrics), args.min_group_size)

    summary = {
        "analysis_population": "RQ1 included consensus rows joined to complete structural pairs",
        "metrics": list(args.metrics),
        "measures": MEASURES,
        "min_group_size": args.min_group_size,
        "multiple_testing": "Benjamini-Hochberg false-discovery-rate correction over high-level patterns, separately within each metric and measure",
        "effect_size": "Cliff's delta, agentic minus human_candidate, with the Cliff (1993) interval",
        "coverage": coverage,
        "across_pattern_tests": across_patterns,
        "pattern_contrasts": json.loads(tests.to_json(orient="records")),
        "source_sha256": {"labels": sha256_file(args.labels), "deltas": sha256_file(args.deltas)},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tests.to_csv(args.output_dir / "category_structural_tests.csv", index=False)
    (args.output_dir / "rq1_structural_by_category_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
