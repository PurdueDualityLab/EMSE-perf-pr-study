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
- `results/manifest.json`: snapshot, source hashes, measurement definitions, and
  outcome-output checksums. `checksums.sha256` covers the published derived bundle.

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
| Merge rate among observed PRs | 54.4% | 73.3% |
| Median creation-to-merge time, merged PRs | 4.99 h | 19.16 h |
| Median added + deleted lines | 130.5 | 81.0 |
| Median changed files | 3 | 3 |
| Median actual PR commits | 3 | 2 |

Merge-rate comparison: chi-square with Yates correction, chi-square = 85.89,
p < 0.001, Cramer's V = 0.195. Elapsed time comparison: two-sided Mann--Whitney
U = 201663.5, p < 0.001, Cliff's delta = -0.207 (agentic minus human-candidate).
The two outcome tests retain the original analysis's raw exploratory p-values.

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
- Rates are computed from exact counts before rounding, unlike the historical
  notebook's early rounding of group means.
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
