# Section 4.1: Quantitative analysis

This directory ports the previous paper's quantitative characterization to the
current 2,260-PR balanced sample. It includes the original analysis sources,
current reproducible scripts, compact per-PR measurements, summary tables,
statistical results, and figures. Full PR bodies, comments, diffs, and cached
source files remain outside this GitHub export.

## What is included

- `analyze_quantitative.py`: archived merge outcomes, elapsed time to merge,
  patch-size summaries, and reproduction of structural comparisons.
- [`../maintainability/run_maintainability.py`](../maintainability/run_maintainability.py):
  resumable Lizard measurement of complete source-file pairs at stored base/head
  SHAs, followed by PR-level aggregation.
- `legacy/`: the upstream `ai.py`, `human.py`, and both quantitative-analysis
  notebooks, pinned to upstream revision
  `76374c11c0027492b3f4a7983fa22b583bbe1f92`. Python scripts are unchanged;
  notebook outputs, execution metadata, and attachments are removed. Original
  and exported hashes are recorded in `legacy/manifest.json`. These are historical
  references with old input paths, not the entrypoints for the current dataset.
- `results/pr_quantitative.csv`: one compact row for each of 2,260 sample PRs,
  with observed outcomes, elapsed time, and numeric patch metadata.
- `results/outcomes_summary.csv`, `outcomes_tests.csv`, and
  `merge_rate_and_time.pdf`: updated outcome/size comparison.
- `results/structural/`: per-PR metric deltas, the common complete-case subset,
  inclusion decisions, coverage, summaries, tests, and the structural figure.
- `results/manifest.json`: snapshot, source hashes, measurement definitions,
  test-family and interval methods, and outcome-output checksums.
  `source_sha256` records the inputs the run actually read, while
  `measurement_source_sha256` carries forward the snapshot parquets behind the
  published per-PR measurements when the run reproduces from the compact CSV.
  `checksums.sha256` covers the published derived bundle.

## Reproduce from the compact published measurements

From the repository root, in a virtual environment:

```bash
pip install -r analysis/quantitative_analysis/requirements.txt
python analysis/quantitative_analysis/analyze_quantitative.py \
  --labels analysis/quantitative_analysis/results/pr_quantitative.csv \
  --output-dir /tmp/quantitative-reproduction
```

This reproduces outcome and structural summaries, tests, and figures without
downloading the full corpus. CSV outcomes/metric values are the inputs to this
mode; it does not independently remeasure source-code complexity.
The scripts read numeric CSVs with `float_precision='round_trip'` to preserve
floating-point values and rank ties across repeated exports.

To rebuild the outcome measurements from the original local inputs:

```bash
python analysis/quantitative_analysis/analyze_quantitative.py \
  --sample data/data/sample/balanced_sample.parquet \
  --evidence mining/sample_evidence/final/pull_requests.parquet \
  --output-dir /tmp/quantitative-from-snapshot
```

To remeasure source complexity, use the sibling maintainability script and its
documented evidence inputs. It downloads source at the stored revisions and
caches results locally; these caches are not published to GitHub.

## Current results and denominators

The evidence snapshot is dated **2026-08-27**. It covers the official 2,260 PRs,
with one unavailable human-candidate PR. Missing observations are not failures
to merge and are excluded from the merge-rate denominator.

| Outcome | Agentic | Human-candidate |
| --- | ---: | ---: |
| Selected PRs | 1,130 | 1,130 |
| Observed PRs | 1,130 | 1,129 |
| Merged | 615 | 827 |
| Merge rate among observed PRs | 54.4% [51.5, 57.3] | 73.3% [70.6, 75.8] |
| Median creation-to-merge time, merged PRs | 4.99 h [2.95, 8.04] | 19.16 h [16.29, 21.87] |
| Median added + deleted lines | 130.5 [115, 148] | 81.0 [72, 96] |
| Median added lines | 89.5 [78, 102] | 54.0 [46, 63] |
| Median deleted lines | 22.0 [19, 25] | 18.0 [16, 21] |
| Median changed files | 3 [3, 3] | 3 [3, 3] |
| Median actual PR commits | 3 [2, 3] | 2 [2, 2] |

Bracketed ranges are two-sided 95% intervals for that arm alone: Wilson intervals
for the merge rates and binomial order-statistic intervals for the medians.

### Characterization tests

All seven between-arm comparisons form one Benjamini--Hochberg family; `q` is the
adjusted p-value. Effect sizes contrast agentic with human-candidate PRs, so a
negative value means the agentic arm is lower. The shift column is the
Hodges--Lehmann median of all pairwise differences with its Moses interval, which
is a location shift and is not the difference between the two medians above.

| Contrast | Test | p | q | Effect [95% CI] | Shift [95% CI] |
| --- | --- | ---: | ---: | --- | --- |
| Merge rate | chi-square (Yates) = 85.89 | <0.001 | <0.001 | V = 0.195 | -18.83 pp [-22.71, -14.94] |
| Time to merge (h) | Mann--Whitney U = 201663.5 | <0.001 | <0.001 | delta = -0.207 [-0.266, -0.147] | -3.59 h [-6.69, -1.73] |
| Added lines | Mann--Whitney U = 715192.5 | <0.001 | <0.001 | delta = 0.121 [0.074, 0.168] | 14 [8, 22] |
| Deleted lines | Mann--Whitney U = 669430.5 | 0.042 | 0.049 | delta = 0.049 [0.002, 0.097] | 1 [0, 3] |
| Added + deleted lines | Mann--Whitney U = 710186.0 | <0.001 | <0.001 | delta = 0.113 [0.066, 0.160] | 19 [10, 30] |
| Changed files | Mann--Whitney U = 656531.5 | 0.222 | 0.222 | delta = 0.029 [-0.018, 0.076] | 0 [0, 0] |
| Commits | Mann--Whitney U = 699634.0 | <0.001 | <0.001 | delta = 0.097 [0.050, 0.143] | 0 [0, 0] |

The merge-rate table also yields an odds ratio of 0.436 [0.366, 0.520] for merging
in the agentic arm. Cliff's delta intervals use the consistent asymmetric form of
Cliff (1993). Added, deleted, and added + deleted lines are nested measures rather
than independent comparisons, and the family is exploratory characterization, not
confirmatory hypothesis testing.

Structural analysis considers 10,606 eligible modified/renamed file pairs in
2,066 PRs. Twelve unavailable pairs affect four PRs. Requiring complete retrieval
and finite percentage changes for all three metrics leaves **2,029 PRs: 1,005
agentic and 1,024 human-candidate**. Median AvgCCN change is zero in both arms
(Mann--Whitney p = 0.624); AvgCCN increases in 44.9% and 42.8%, respectively.
The structural test family uses BH adjustment across its three comparisons.

## Interpretation and changes from the historical analysis

- `human_candidate` means no selected observable agentic signal, not confirmed
  human authorship. None of these comparisons demonstrates a causal effect.
- Outcome statistics describe the archived snapshot, not current live GitHub
  status. Open PRs are not labeled rejected. Elapsed time is wall-clock time
  between creation and merge, not active reviewer effort; only merged PRs enter
  this comparison. This is not survival analysis or a censoring correction.
  Because that comparison conditions on merging, and merge rates differ between
  the arms, its p-value, Cliff's delta, and shift describe the merged PRs only
  and are not an unconditional statement about how long a submitted PR takes.
- Rates are computed from exact counts before rounding, unlike the historical
  notebook's early rounding of group means.
- The characterization comparisons now carry effect sizes, 95% intervals, and BH
  adjustment across their own family, matching the treatment of the structural
  and RQ3 families. The earlier release tested only merge rate and elapsed time
  and reported raw p-values without intervals. Patch-size differences were
  previously described from medians alone, with no test.
- Patch size uses PR-level additions/deletions, not summed per-commit diffs that
  may count the same edits repeatedly. `commits_count` is the actual PR commit
  count, not the number of commit-file records.
- Source-level structural metrics are means across paired files, not PR-wide
  totals. Added/deleted files are outside that paired-file contrast. Missing
  AvgCCN values and zero relative-change baselines are not treated as zeros.
- The structural baseline is the stored base SHA, not a reconstructed merge
  base; base-branch changes may influence the contrast. Complete-case exclusions
  do not preserve exact weekly balance. Neither analysis adjusts for repository
  clustering, language mix, or differences in observation age.

Full study artifacts are maintained in the private
[Hugging Face dataset](https://huggingface.co/datasets/rcalvome/EMSE-perf-pr-study).
The published dataset does not currently include the entire enriched evidence
snapshot; its hashes identify the local measurement inputs. Figures and tables
in this directory are derived artifacts, not the full dataset.
