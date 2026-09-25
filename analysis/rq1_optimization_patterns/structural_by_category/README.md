# RQ1 Structural Change by Optimization Category

This directory refines the aggregate structural contrast reported in Section 4.1
by repeating it inside each RQ1 high-level optimization pattern. Section 4.1
compares base-to-head structural change between the study arms with every
optimization category pooled, so it establishes that a difference exists but not
whether the arms differ uniformly across kinds of optimization work.

## Method

The analysis joins the RQ1 consensus labels in
`analysis/classification_labels/rq1_labels.csv` to the per-PR structural deltas in
`analysis/quantitative_analysis/results/structural/pr_deltas.csv`, keeping rows
that RQ1 included in its analysis and whose structural pairs are complete. PRs
without complete pairs carry no parseable function-level metrics and are absent
from the contrasts; the join coverage is reported in the summary rather than
imputed.

Within each high-level pattern, the two arms are compared with a two-sided
Mann-Whitney U test and Cliff's delta using the consistent asymmetric interval of
Cliff (1993), the same effect size and interval used by the quantitative
characterization. Benjamini-Hochberg false-discovery-rate correction is applied over
the patterns of a metric, separately for each measure, and the adjusted value is
reported as `q`. This exploratory follow-up uses the same procedure as the
quantitative characterization and RQ3 rather than the Holm correction that
`analyze_rq1_comparison.py` applies to the per-category Fisher tests. A pattern with fewer than `--min-group-size` PRs in
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
through `--metrics` but is excluded by default, because average cyclomatic
complexity does not move materially in either arm or in any pattern.

## Run

    python analyze_rq1_structural_by_category.py

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
