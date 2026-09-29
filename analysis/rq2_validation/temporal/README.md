# Temporal Analysis of Validation Evidence — Journal RQ3

This analysis examines changes in reported performance validation over the
sampling window. It uses the same classification labels as the aggregate RQ3
comparison and evaluates two outcomes: validation presence and benchmark-based
primary evidence among PRs with a resolved positive evidence type.

The temporal comparisons were specified after the aggregate analysis and are
interpreted as exploratory. Their inferential results are not adjusted jointly
with the aggregate comparisons.

## Reproduction

From the repository root:

```bash
python analysis/rq2_validation/temporal/analyze_rq2_temporal.py \
  --labels analysis/classification_labels/rq2_labels.csv \
  --dates analysis/rq2_validation/temporal/pr_created_at.csv \
  --output-dir reproduction/rq3/temporal
```

The versioned timestamp table supports offline reproduction. The separate
`fetch_pr_created_at.py` utility reconstructs creation timestamps through the
authenticated GitHub CLI when preparing a new input export.

## Population and Integrity Controls

The analysis includes 2,258 PRs across 65 ISO-week strata, with 1,129 PRs per
author group. Each retained weekly stratum is balanced 1:1. The recorded creation
dates span January 2, 2025 through June 1, 2026. Sampling strata use the ISO year
and week of `created_at` in UTC, consistent with the sampling procedure.

The input controls require 929 agentic and 910 human-authored positive validation
labels and 1,819 PRs in the resolved primary-type population. Stored
`human_candidate` identifiers correspond to the manuscript's human-authored
group.

## Statistical Methods

- **Logistic regression:** models author type, creation time in years centered
  at the mean observed creation time, and their interaction. Repository-clustered
  standard errors and Wald tests provide the principal temporal inference.
  Model-based standard errors and likelihood-ratio results are also reported.
- **Quarterly proportions:** report counts, rates, and Wilson confidence
  intervals. Newcombe intervals describe between-group differences in proportions.
- **Stratified comparisons:** Mantel–Haenszel estimates summarize odds ratios
  across weekly strata. Breslow–Day tests with Tarone's correction assess
  heterogeneity across weeks and quarters.
- **Period comparisons:** partition the 65 observed weekly strata at the median
  sampled week. Comparisons use contingency-table tests and Holm adjustment
  across the two resulting periods.

The output retains both the conservative bound-difference interval
(`risk_difference_ci95`) and Newcombe's square-and-add interval
(`risk_difference_newcombe_ci95`). An additional sensitivity analysis uses the
complete primary-type/type-set consensus subset and records its outputs
separately from the primary-type analysis.

## Interpretation

These observational analyses describe reported evidence. Weekly balancing
aligns the groups' sampling periods, but repository, language, and agent
composition can vary over time. Early quarters contain fewer observations and
have wider intervals. PRs also differ in their age at the evidence-collection
snapshot. Temporal changes in reported evidence do not independently establish
improvements in agent capability or the quality of the reported measurements.

## Outputs

| File | Contents |
| --- | --- |
| `rq2_temporal.json` | Population controls, regressions, stratified comparisons, and period estimates |
| `presence_by_week.csv` | Weekly counts, rates, and intervals |
| `presence_by_month.csv` | Monthly counts, rates, and intervals |
| `presence_by_quarter.csv` | Quarterly counts, rates, and intervals |
| `primary_type_by_quarter.csv` | Evidence-type proportions in the resolved primary-type population |
| `primary_type_by_quarter_multilabel.csv` | Evidence-type proportions in the complete type-set sensitivity population |
| `rq2_temporal.png`, `rq2_temporal.pdf` | Two-panel temporal visualization |

The journal-wide reproduction command exports the paper figure as
`validation_over_time.pdf`. Historical `rq2_*` output names remain available for
compatibility with the analysis interfaces.
