# Structural Change by Optimization Category — Journal RQ2

This analysis examines whether the aggregate structural differences reported
for journal RQ1 are concentrated in particular optimization categories. It joins
the optimization-pattern labels used by journal RQ2 to the archived structural
measurements. Directory and file names retain the original `rq1` prefix.

## Method

The analysis joins the optimization consensus labels in
`analysis/classification_labels/rq1_labels.csv` to the per-PR structural deltas in
`analysis/quantitative_analysis/results/structural/pr_deltas.csv`, keeping rows
with resolved optimization labels and complete eligible structural pairs.
Undefined measurements are excluded from the corresponding contrasts. Join
coverage and per-comparison sample sizes are reported explicitly.

Within each high-level pattern, the two arms are compared with a two-sided
Mann-Whitney U test and Cliff's delta using the consistent asymmetric interval of
Cliff (1993), the same effect size and interval used by the quantitative
characterization. Benjamini-Hochberg false-discovery-rate correction is applied over
the patterns of a metric, separately for each measure, and the adjusted value is
reported as `q`. The correction family is distinct from the per-category Fisher
comparisons in `analyze_rq1_comparison.py`, which use Holm adjustment.
A category with fewer than `--min-group-size` PRs in
either arm is reported with its counts but not tested, because the contrast would
not be interpretable at that size.

Both the percentage change and the absolute change are tested. Percentage change
is the measure Section 4.1 reports, but it divides by the base value, so a pattern
whose files are small can show a large percentage shift from a small edit.
Reporting the absolute change alongside it shows whether a percentage result
survives in natural units.

A Kruskal-Wallis test across patterns, with the arms pooled, reports whether the
structural change depends on the kind of optimization at all. This is context for
the per-pattern contrasts, not a test of the arm difference.

`nloc` and `function_count` are contrasted by default. `avg_ccn` is available
through `--metrics` for additional analyses.

## Reproduction

From the repository root:

```bash
python analysis/rq1_optimization_patterns/structural_by_category/analyze_rq1_structural_by_category.py \
  --output-dir reproduction/rq2/structure
```

Both inputs, the output directory, the metric list, and the minimum tested group
size are overridable; see `--help`. The defaults resolve relative to this file, so
the script runs without arguments from a checkout.

## Outputs

- `results/category_structural_tests.csv`: one row per metric, measure, and
  pattern, carrying arm counts, medians, the Mann-Whitney statistic, raw and
  Benjamini-Hochberg-adjusted p-values (`q`), Cliff's delta with its interval and magnitude, and the
  Hodges-Lehmann shift. Untested patterns carry counts and a skip reason.
- `results/rq1_structural_by_category_summary.json`: the same contrasts with the
  join coverage, the across-pattern Kruskal-Wallis tests, the declared correction
  family, and SHA-256 digests of both inputs.
