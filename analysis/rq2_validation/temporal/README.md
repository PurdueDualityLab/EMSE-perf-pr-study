# RQ2 validation reporting over time

The primary RQ2 comparison collapses the whole sampling window into one 2x2
table and finds no association between author type and validation presence
(82.3% agentic vs.\ 80.6% human-candidate, chi-square p = 0.330). The
preliminary study reported the opposite ordering with a large gap (45.7% vs.\
63.6%). The paper's Validation Presence paragraph attributes that earlier gap to
the asymmetric repository filters or to a change in agent behavior over time,
without separating them.

This directory keeps the time dimension instead of collapsing it. The weekly
balanced design supports the analysis directly: each of the 65 ISO-week strata
holds equal numbers of agentic and human-candidate PRs, so calendar time cannot
confound the author-type contrast within a stratum.

**This is a post-hoc, exploratory analysis.** It was not pre-declared, it uses
the same labels as the primary analysis, and its p-values are not corrected
against the primary RQ2 family.

## Inputs

`analysis/classification_labels/rq2_labels.csv` supplies the 2,258 RQ2
identities, author types, and consensus labels. It carries no timestamp, and the
`created_at` column lives in `data/data/sample/balanced_sample.parquet` inside
the private dataset submodule. `fetch_pr_created_at.py` therefore rebuilds that
immutable field from GitHub through the authenticated `gh` CLI, batching
GraphQL aliases and falling back to REST for identities GraphQL cannot resolve.

```bash
python analysis/rq2_validation/temporal/fetch_pr_created_at.py
python analysis/rq2_validation/temporal/analyze_rq2_temporal.py
```

The rebuilt timestamps reproduce the published sampling design exactly, which is
the check that they are the same values the sampler used. `check_controls`
fails the run unless all of the following hold, and all of them do:

- 2,258 rows over 65 ISO-week strata, matching `weekly_strata.parquet`;
- every one of the 65 strata 1:1 balanced between the two author types;
- 929/1,129 agentic and 910/1,129 human-candidate positives, and 1,819 rows in
  the stage-2 primary-type analysis, matching `rq2_consensus_summary.json`.

All 2,258 timestamps resolved through GraphQL; none needed the REST fallback.
The window runs from 2025-01-02 to 2026-06-01.

## Method

Strata are rebuilt with the sampler's own rule: ISO year and week of `created_at`
in UTC (`mining/src/build_balanced_sample.py`). The outcome is
`consensus_validation_present`; the secondary outcome is
`consensus_primary_validation_type == "benchmark"` among the 1,819 resolved
positive PRs. Both are analyzed identically.

- **Mantel-Haenszel** gives the common agentic-vs-human odds ratio across the 65
  weekly strata, with the Robins-Breslow-Greenland interval and the
  continuity-corrected MH chi-square. This is the time-adjusted version of the
  paper's presence test.
- **Breslow-Day, with Tarone's correction**, tests whether that odds ratio is
  constant across strata. It is reported by week and by calendar quarter; weeks
  are small, so the quarterly version is the more stable of the two.
- **Logistic regression** models the outcome on author type, calendar time in years since
  the first PR (centered at the window midpoint), and their interaction.
  The interaction is the direct test of whether the author-type difference changed.
  Every coefficient carries a model-based and a **repository-clustered sandwich**
  standard error (534 clusters over 2,258 PRs); the clustered one is the
  inferential statement, matching the repository-cluster treatment used
  elsewhere in RQ1 and RQ2. The likelihood-ratio tests assume independence and
  are reported alongside, not in place of, the clustered Wald tests.
- **Period splits** give the 2x2 comparison in each half of the window and in
  each calendar quarter, with Wilson intervals per author type, Newcombe intervals for
  the difference, Yates-corrected chi-square, Fisher's exact test, and Holm
  adjustment across the two halves.

Risk-difference intervals appear twice. `risk_difference_ci95` is the
conservative bound-difference form used in `analyze_rq2_comparison.py`, kept so
the numbers line up with the primary analysis; `risk_difference_newcombe_ci95`
is Newcombe's square-and-add interval, which agrees with the tests and is the
one quoted in the tables below.

## Results

### Validation presence

The aggregate finding survives time adjustment. The weekly-stratified
Mantel-Haenszel odds ratio is 1.122 [0.904, 1.393], p = 0.319, essentially the
crude 1.118 — which is what the weekly balancing is designed to produce.

What the aggregate hides is that **both author types rose steeply and agentic
PRs rose faster, crossing over in late 2025**.

| Period | Agentic | Human-candidate | Difference (pp) | Newcombe 95% CI |
| --- | ---: | ---: | ---: | --- |
| 2025-Q1 (n = 10 each) | 70.0% | 70.0% | 0.0 | [-35.9, 35.9] |
| 2025-Q2 | 63.9% | 76.1% | **-12.2** | [-23.4, -0.5] |
| 2025-Q3 | 74.7% | 72.2% | +2.5 | [-6.5, 11.5] |
| 2025-Q4 | 76.4% | 76.6% | -0.2 | [-9.2, 8.8] |
| 2026-Q1 | 86.3% | 79.6% | **+6.7** | [0.9, 12.4] |
| 2026-Q2 | 92.9% | 90.6% | +2.3 | [-2.1, 6.6] |

Split at the median week, the first half (2025-W01..2025-W42, 347 PRs per author type)
has agentic 69.5% against human 74.1%, a difference of -4.6 pp (Fisher
p = 0.206), reproducing the preliminary study's ordering. The second half
(2025-W43..2026-W23, 782 PRs per author type) has agentic 88.0% against human 83.5%,
+4.5 pp (Fisher p = 0.014, Holm 0.028).

Trend model, odds ratios per year:

| Term | OR | Clustered 95% CI | Clustered p | Model-based p |
| --- | ---: | --- | ---: | ---: |
| Time, human-candidate PRs | 3.06 | [1.71, 5.49] | <0.001 | <0.001 |
| Time, agentic PRs | 6.47 | [3.30, 12.70] | <0.001 | <0.001 |
| Author type x time interaction | 2.11 | [0.88, 5.06] | 0.093 | 0.020 |
| Author type at window start | 0.58 | [0.26, 1.29] | 0.182 | 0.074 |
| Author type at window end | 1.68 | [0.86, 3.26] | 0.128 | 0.012 |

The time trend is large and unambiguous in both author types. The evidence that the
*gap* changed is moderate and depends on how repository clustering is handled:
Breslow-Day rejects a constant odds ratio across weeks (chi-square(60) = 86.59,
p = 0.014; by quarter chi-square(5) = 9.65, p = 0.086), and the likelihood-ratio
test of the interaction gives p = 0.020, but the repository-clustered Wald test
gives p = 0.093. **We do not claim a significant change in the presence gap.**
The defensible statement is that reported validation rose sharply in both author
types, that the direction of the author-type difference reverses across the
window, and that the aggregate null is an average over a period in which the two
were ordered differently at each end.

### Primary evidence type (secondary outcome)

The same treatment applied to benchmark-as-primary-type is cleaner, and it does
survive clustering.

| Term | OR | Clustered 95% CI | Clustered p |
| --- | ---: | --- | ---: |
| Time, human-candidate PRs | 0.98 | [0.56, 1.73] | 0.950 |
| Time, agentic PRs | 3.63 | [1.32, 9.96] | 0.012 |
| Author type x time interaction | 3.69 | [1.24, 10.97] | 0.019 |

The human-candidate benchmark share is flat across the window. The agentic
share rises from 14.3% (2025-Q1, n = 7) and 31.6% (2025-Q2) to 44.7% (2026-Q1)
and 60.1% (2026-Q2), where it meets the human-candidate share (58.7%). The gap
narrows from -23.6 pp [-31.8, -14.8] in the first half to -6.6 pp [-11.9, -1.3]
in the second, and Breslow-Day rejects homogeneity across quarters
(chi-square(5) = 21.50, p < 0.001).

So the paper's headline type difference (benchmark 46.1% agentic vs.\ 57.2%
human, static reasoning 51.4% vs.\ 39.3%) is an average over a closing gap, not
a stable property of the window.

The denominator here is the 1,819 PRs with a resolved primary type, matching the
paper's validation-type analysis and the per-primary-type tests in
`analyze_rq2_comparison.py`. The stricter 1,707-PR multi-label subset gives the
same picture, slightly stronger: `rq2_temporal.json` carries it under
`benchmark_primary_type_multilabel_subset` (interaction OR 4.00, clustered
p = 0.015; human-candidate slope OR 1.00, p = 0.988; agentic slope OR 3.98,
p = 0.008; gap -25.1 pp to -7.1 pp), and its quarterly shares are exported to
`primary_type_by_quarter_multilabel.csv`.

## What this does and does not support

It supports the temporal half of that attribution: the preliminary study's
sample sits in the early part of this window, where the author-type ordering
matches what it reported, and the expanded sample is dominated by later weeks,
where the ordering is reversed or absent. A change over time is therefore a live
contributor to the revised finding, alongside the broader evidence corpus and
the changed coding rule. The analysis is written up in the paper's
`Validation over Time` paragraph in the RQ2 results section.

It does not establish that agents got better at validating.

- **Both author types rose.** Human-candidate reporting rose by an odds ratio of 3.06 per
  year, in a population defined by the absence of agentic signals. A rise that
  large among human-candidate PRs points to a period or composition effect that
  agentic PRs share — for example wider adoption of CI benchmarking bots, or
  later weeks drawing on repositories with more measurement infrastructure. The
  agentic trend cannot be read as agent improvement while the human trend is
  unexplained.
- **Composition is not held fixed.** Weekly balancing equalizes the two author
  types within a week; it does not make week *t* comparable to week *t'*. The agent mix,
  repository mix, and language mix all change across the window, and the early
  weeks are very thin (10 PRs per author type in 2025-Q1, 119 in 2025-Q2,
  against 322 in 2026-Q2). An agent-level breakdown would separate "agents improved" from "the
  mix of agents changed", but it needs `data/data/sample/agentic_sample.parquet`
  from the private submodule, which was not available when this ran.
- **PR age is a nuisance, in the conservative direction.** Older PRs had longer
  to accumulate discussion before the 2026-08-27 snapshot, which would inflate
  early evidence, not deflate it. The observed rise is therefore not an artifact
  of truncated early evidence.
- **The labels are model consensus**, with the same caveats as the primary RQ2
  analysis; nothing here re-audits them.

## Outputs

| File | Contents |
| --- | --- |
| `pr_created_at.csv` | 2,258 identities with the GitHub creation timestamp and its source |
| `rq2_temporal.json` | All controls, stratified tests, trend models, and period tables |
| `presence_by_week.csv` | Per-ISO-week counts, rates, and intervals (65 rows) |
| `presence_by_month.csv` | The same by calendar month |
| `presence_by_quarter.csv` | The same by calendar quarter |
| `primary_type_by_quarter.csv` | Primary-type shares by author type and quarter, over the 1,819 resolved primary types |
| `primary_type_by_quarter_multilabel.csv` | The same over the stricter 1,707-PR multi-label subset |
| `rq2_temporal.png`, `.pdf` | Two-panel figure by quarter: validation-presence rates, and benchmark share among resolved positives |
