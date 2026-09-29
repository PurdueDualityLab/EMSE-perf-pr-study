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
(`risk_difference_newcombe_ci95`). The latter is used in the descriptive table
below.

## Reference Results

### Validation Presence

| Period | Agentic | Human-authored | Difference, percentage points | Newcombe 95% CI |
| --- | ---: | ---: | ---: | --- |
| 2025-Q1, 10 PRs per group | 70.0% | 70.0% | 0.0 | [-35.9, 35.9] |
| 2025-Q2 | 63.9% | 76.1% | -12.2 | [-23.4, -0.5] |
| 2025-Q3 | 74.7% | 72.2% | 2.5 | [-6.5, 11.5] |
| 2025-Q4 | 76.4% | 76.6% | -0.2 | [-9.2, 8.8] |
| 2026-Q1 | 86.3% | 79.6% | 6.7 | [0.9, 12.4] |
| 2026-Q2 | 92.9% | 90.6% | 2.3 | [-2.1, 6.6] |

| Regression term | Odds ratio per year | Clustered 95% CI | Clustered p |
| --- | ---: | --- | ---: |
| Time, human-authored PRs | 3.06 | [1.71, 5.49] | <0.001 |
| Time, agentic PRs | 6.47 | [3.30, 12.70] | <0.001 |
| Author type × time | 2.11 | [0.88, 5.06] | 0.093 |

Validation reporting increases in both groups. The interaction is not
statistically significant with repository-clustered standard errors; the
descriptive crossover does not establish different validation-presence trends.

The first period, 2025-W01 through 2025-W42, contains 347 PRs per group and has
validation rates of 69.5% and 74.1%. The second, 2025-W43 through 2026-W23,
contains 782 PRs per group and has rates of 88.0% and 83.5%, respectively.

### Benchmark-Based Primary Evidence

This outcome uses the 1,819 PRs with reported validation and a resolved primary
evidence type.

| Regression term | Odds ratio per year | Clustered 95% CI | Clustered p |
| --- | ---: | --- | ---: |
| Time, human-authored PRs | 0.98 | [0.56, 1.73] | 0.950 |
| Time, agentic PRs | 3.63 | [1.32, 9.96] | 0.012 |
| Author type × time | 3.69 | [1.24, 10.97] | 0.019 |

Benchmark-based primary evidence increases among agentic PRs, while the
human-authored group shows no statistically significant temporal change. The
2026-Q2 proportions are 60.1% and 58.7%, respectively. An additional sensitivity
analysis uses the 1,707-PR complete primary-type/type-set consensus subset; its
results are recorded separately in the JSON and CSV outputs.

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
