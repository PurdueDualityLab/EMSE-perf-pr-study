# Performance Validation — Journal RQ3

This directory retains historical `rq2_*` names. It answers **RQ3** in the
journal article; optimization-pattern labeling in `rq1_optimization_patterns/`
answers journal RQ2. For the journal-wide offline workflow and temporal analysis,
see [../../REPRODUCING.md](../../REPRODUCING.md).

This directory classifies the performance-validation evidence reported in a
complete-observability subset of the balanced PR sample used by RQ1. GPT-5.6-sol,
Gemini 3.1 Pro Preview, and Qwen3.8-27B classify every PR independently.

## Method

The analysis records observable evidence in PR descriptions, comments, reviews, code
diffs, workflow runs, and check runs. It does not infer that validation happened
outside the captured artifacts. Passing CI or a workflow name containing
`benchmark` is not performance evidence by itself.

Preparation refuses PRs whose code, discussion, or CI endpoints were not
collected. Such rows are excluded through a predeclared sample artifact; missing
observability is never converted into a negative validation label. To preserve
the 1:1 design, each unavailable PR is paired with an exclusion from the other
arm in the same weekly stratum. The counterpart is selected deterministically as
the row with the greatest original `selection_hash`, without adding replacement
units after observing evidence.

The journal RQ3 sample contains 2,258 PRs: 1,129 agentic and 1,129 human-authored PRs.
The unavailable human PR `101194285:11618` and its deterministic agentic
counterpart `599431918:22509` in `2026-W20` are recorded in
`sample/evidence_exclusions.parquet`. The sample SHA-256 is
`01460f42a1f1b63ac40d4ed3302140cf3d125a4d8aafba22244d5bf64c688b59`.

The non-exclusive validation types are:

- `benchmark`: explicit quantitative runtime measurements.
- `profiling`: profiler-derived measurements or artifacts.
- `static-reasoning`: explicit analytical performance reasoning without runtime data.
- `anecdotal`: explicit local or manual performance testing without quantitative
  or analytical support.

Each positive label includes a primary type, evidence sources, short verbatim
quotes, and any explicitly reported metrics. Multi-label types avoid discarding
evidence when a PR combines, for example, benchmark results and static
reasoning. Primary type preserves a single-label comparison with prior analyses.

## Classification Workflow

```bash
.venv/bin/python analysis/rq2_validation/run_rq2.py prepare-sample \
  --source-sample data/data/sample/balanced_sample.parquet \
  --evidence-dir mining/sample_evidence/final \
  --output-dir analysis/rq2_validation/sample

.venv/bin/python analysis/rq2_validation/run_rq2.py prepare \
  --sample analysis/rq2_validation/sample/balanced_sample.parquet \
  --evidence-dir mining/sample_evidence/final \
  --output-dir analysis/rq2_validation/results_gpt

.venv/bin/python analysis/rq2_validation/run_rq2.py submit \
  --output-dir analysis/rq2_validation/results_gpt

.venv/bin/python analysis/rq2_validation/run_rq2.py status \
  --output-dir analysis/rq2_validation/results_gpt

.venv/bin/python analysis/rq2_validation/run_rq2.py collect \
  --output-dir analysis/rq2_validation/results_gpt
```

Use `run_rq2_gemini.py` with the same actions and
`analysis/rq2_validation/results_gemini`. Gemini partitions one logical run into
provider jobs internally. Its `retry` action submits unresolved rows only.
OpenAI exposes `prepare-retry` and `merge-retry`.

Both providers use medium reasoning, a 4,096-token output limit, the same prompt
and semantic schema, and deterministic input ordering. Gemini uses temperature
zero; GPT-5.6-sol does not accept a temperature parameter. Preparation records
hashes for the sample, every evidence table, schema, system instruction,
provider configuration, rendered prompt, and per-PR input.

## Pairwise Agreement Assessment

```bash
.venv/bin/python analysis/rq2_validation/analyze_rq2_agreement.py \
  --gpt analysis/rq2_validation/results_gpt/validation_labels.parquet \
  --gemini analysis/rq2_validation/results_gemini/validation_labels.parquet \
  --sample analysis/rq2_validation/sample/balanced_sample.parquet \
  --output-dir analysis/rq2_validation/agreement
```

Agreement is calculated globally and by sample arm for validation presence,
primary type, the complete type set, and each binary validation type. The script
requires complete, error-free results over the official sample with identical
study contracts and per-row inputs. Disagreements retain both models' labels,
quotes, and explanations as review candidates; neither model is used as an
automatic fallback. These intermediate candidates are not the final labeling
procedure. The journal uses the three-model consensus described below.

The completed run contains 2,258 labels from each model. Full agreement across
presence, primary type, and the complete type set is 63.99% (1,445/2,258).
Validation-presence agreement is 86.80% (Cohen's kappa 0.5841), primary-type
agreement is 80.60% (kappa 0.7086), and complete type-set agreement is 64.66%
(kappa 0.5559). Full agreement is 63.95% for agentic PRs and 64.04% for
human-candidate PRs. The 813 complete disagreements are exported to
`agreement/adjudication_template.csv` with empty decision fields for human
review.

The final provider outputs contain 2,258 classified rows each. Completion,
schema validity, and consistency of input identities are checked before
consensus construction.

## Consensus and Comparison

```bash
.venv/bin/python analysis/rq2_validation/build_rq2_consensus.py \
  --gpt analysis/rq2_validation/results_gpt/validation_labels_complete.parquet \
  --gemini analysis/rq2_validation/results_gemini/validation_labels.parquet \
  --qwen analysis/rq2_validation/results_qwen/validation_labels.parquet \
  --sample analysis/rq2_validation/sample/balanced_sample.parquet \
  --output-dir analysis/rq2_validation/consensus

.venv/bin/python analysis/rq2_validation/analyze_rq2_comparison.py \
  --consensus analysis/rq2_validation/consensus/rq2_consensus.parquet \
  --output-dir analysis/rq2_validation/comparison
```

Stage 1 takes the majority validation-presence vote. The primary Stage 2
analysis requires two positive models to agree on `primary_validation_type`.
It resolves 1,819 positive PRs and preserves 20 as unresolved. A separate
multi-label sensitivity requires agreement on the complete
`(primary_validation_type, validation_types)` value, resolving 1,707 positive
PRs and preserving 132 as unresolved. This secondary rule avoids synthesizing
a type set that no model emitted. Primary comparisons use chi-square tests and
Cramer's V, including Yates's correction for validation presence. Evidence-type
proportions are calculated within each author group among resolved positive PRs.

The machine-readable output additionally includes repository-cluster bootstrap
intervals, Fisher tests with Holm correction, and non-exclusive type summaries
over the stricter multi-label subset. These are retained as optional robustness
analyses distinct from the primary journal RQ3 comparison.

## Validation Reporting over Time

The [temporal analysis](temporal/README.md) evaluates validation presence and
benchmark-based primary evidence across the 65 sampled weekly strata. It reports
quarterly proportions and logistic-regression estimates with repository-clustered
standard errors. The temporal comparisons are exploratory and use a distinct
inferential procedure from the aggregate comparisons. Their interpretation
concerns reported evidence rather than independently measured agent capability.
