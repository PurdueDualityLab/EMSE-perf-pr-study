# Paired-File Structural Analysis — Journal RQ1

This module measures structural changes between recorded base and head file
versions using Lizard. Offline reproduction from the archived measurements is
described in [the quantitative analysis documentation](../quantitative_analysis/README.md).
Reconstructing the measurements requires the archived PR evidence and access to
the source revisions. Install the project requirements and
`analysis/maintainability/requirements.txt`, then run from the repository root:

```bash
python analysis/maintainability/run_maintainability.py
```

## Measurement Specification

- Analyze complete source files at the immutable `base_sha` and `head_sha`
  recorded in `mining/sample_evidence/final/pull_requests.parquet`.
- These are the stored base/head revisions, not reconstructed merge-base
  revisions. Base-branch changes can therefore influence the contrast; this is
  a structural comparison, not a causal estimate of the PR's isolated impact.
- Use modified and renamed PR files supported by Lizard 1.24.0. Renames pair the
  previous filename with the current filename. Added/removed files and unsupported
  languages are outside this paired-file estimand.
- Compute NLOC, function count, and mean function cyclomatic complexity per file.
  For each metric, average the same available file pairs on both sides, retaining
  the previous study's mean-of-file metrics. NLOC and function count at PR level
  are thus file means, not PR-wide totals; AvgCCN is not a function-weighted mean.
- AvgCCN is undefined for files without detected functions. Such pairs are
  omitted for AvgCCN on both sides, never assigned zero complexity.
- Report both absolute and percentage changes. A zero baseline has a valid
  absolute delta but an undefined percentage change.
- The primary comparison requires all eligible file pairs to be fetched and
  finite percentage changes for all three metrics, yielding a common PR cohort.
  Unlike the previous analysis, missing deltas do not count as non-increases.
- Compare percentage changes using two-sided Mann--Whitney U tests, Cliff's
  delta, and BH adjustment across three tests. These exploratory PR-level tests
  do not account for repository clustering or preserve exact weekly balance
  after complete-case exclusions. Lack of significance is not equivalence.

## Outputs and Execution

The local `current/` directory contains compressed source caches, per-file metric checkpoints,
input SHA-256 hashes, file eligibility and fetch-status audits, all PR deltas,
complete-case analysis deltas, cohort coverage, summary statistics, tests, and a
PDF boxplot. Subsequent executions reuse successful file measurements and repeat
retrieval for incomplete pairs.
The figure shows median/IQR with 10th/90th-percentile whiskers and hides outlier
markers; untrimmed values are used in tests and are available in the CSVs.

The reference measurement contains 10,606 eligible file pairs in 2,066 PRs; 12 unavailable pairs affect four
PRs. Complete and finite measurements for all three metrics retain 2,029 PRs:
1,005 agentic and 1,024 human-authored. Compact coverage and measurement tables
are versioned under `analysis/quantitative_analysis/results/structural/`.
The complete per-file retrieval record is retained in the source archive.
