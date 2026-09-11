# RQ1 Optimization Patterns

This directory contains the optimization-pattern catalog, GPT-5.6-sol, Gemini
3.1 Pro Preview, and Qwen3.8-27B classifiers, consensus construction, and
supporting analyses.

## Contents

- `run_rq1.py`: prepares, submits, monitors, and collects the RQ1 batch.
- `run_rq1_gemini.py`: runs one logical Gemini classification over the same RQ1
  sample and handles provider job limits internally.
- `analyze_rq1_agreement.py`: calculates agreement and Cohen's kappa globally and
  by sample arm, then exports agreements and human-review candidates.
- `build_rq1_consensus.py`: builds strict atomic two-of-three consensus labels.
- `analyze_rq1_comparison.py`: compares study arms and exports statistical
  tables, a machine-readable summary, and a figure.
- `adjudicate_gemini_parent_labels.py`: applies an explicitly authorized,
  auditable correction when Gemini returns a catalog sub-pattern under the wrong
  high-level parent.
- `catalog/`: original and updated optimization-pattern taxonomies.
- `pattern_analysis.ipynb`: distribution and statistical analyses.
- `compare_pattern.py` and `label_analysis.ipynb`: agreement and manual-review
  utilities for multi-model labeling.
- `optimization_pattern_detection_*.ipynb` and
  `optimization_pattern_detection_qwen.py`: alternative classifier notebooks
  and scripts.

## Batch Workflow

```bash
.venv/bin/python analysis/rq1_optimization_patterns/run_rq1.py prepare \
  --sample data/data/sample/balanced_sample.parquet \
  --evidence-dir mining/sample_evidence/final \
  --catalog analysis/rq1_optimization_patterns/catalog/updated_optimization_catalog.csv \
  --output-dir analysis/rq1_optimization_patterns/results_gpt

.venv/bin/python analysis/rq1_optimization_patterns/run_rq1.py submit \
  --output-dir analysis/rq1_optimization_patterns/results_gpt

.venv/bin/python analysis/rq1_optimization_patterns/run_rq1.py status \
  --output-dir analysis/rq1_optimization_patterns/results_gpt

.venv/bin/python analysis/rq1_optimization_patterns/run_rq1.py collect \
  --output-dir analysis/rq1_optimization_patterns/results_gpt
```

`results_gpt/optimization_pattern_labels.parquet` is the collected
GPT PR-level label table. Batch payloads, raw API responses, and local state
are ignored by Git.

## Agreement

After both models have classified the same sample with the same prompt version:

```bash
.venv/bin/python analysis/rq1_optimization_patterns/analyze_rq1_agreement.py \
  --gpt analysis/rq1_optimization_patterns/results_gpt/optimization_pattern_labels.parquet \
  --gemini analysis/rq1_optimization_patterns/results_gemini/optimization_pattern_labels.parquet \
  --sample data/data/sample/balanced_sample.parquet \
  --output-dir analysis/rq1_optimization_patterns/agreement
```

The summary reports exact agreement and Cohen's kappa globally and separately
for agentic and human-candidate PRs. Disagreements are exported to
`disagreements.parquet`, and `adjudication_template.csv` provides empty decision
and notes columns for human review. The script does not select either model as a
fallback.

The completed run contains 2,260 labels from each model. Hierarchical agreement
is 70.0% (1,582/2,260; Cohen's kappa 0.6764), with 69.82% agreement for agentic
PRs and 70.18% for human-candidate PRs. The remaining 678 model disagreements
require human adjudication.

Gemini produced 20 persistent responses whose sub-pattern was in the catalog
but whose high-level parent was inconsistent with the catalog hierarchy. After
four model attempts, the study owner authorized preserving each returned
sub-pattern and replacing only its high-level parent with the sub-pattern's
unique catalog parent. The complete audit is stored in
`results_gemini/manual_parent_adjudications.csv`; the correction is reproducible
with `adjudicate_gemini_parent_labels.py`.

Both runners use medium reasoning, a 4,096-token output limit, and the same
structured schema. Gemini uses temperature zero. GPT-5.6-sol does not support
a temperature parameter, so no equivalent parameter is sent. This prevents
strict generation-level equivalence with the prior GPT-5.1 configuration and
is reported as a methodological limitation. Preparation snapshots the catalog and
records hashes for the sample, evidence, schema, system instruction, manifest,
and rendered prompt per PR. Collection refuses modified snapshots and records
invalid or missing responses as row-level errors.

Gemini's `retry` command submits only unresolved rows and folds successful
responses into the same logical result. OpenAI exposes `prepare-retry` and
`merge-retry`. Agreement requires the complete balanced sample and refuses to
calculate kappa while either model still has errors or the study contracts and
per-row input hashes differ.

## Consensus and Comparison

After all three model-specific label tables are complete:

```bash
.venv/bin/python analysis/rq1_optimization_patterns/build_rq1_consensus.py
.venv/bin/python analysis/rq1_optimization_patterns/analyze_rq1_comparison.py
```

Consensus is computed over the atomic `(high_level_pattern, sub_pattern)` pair.
Rows without an exact two-of-three majority remain unresolved. The primary
comparison preserves the previous paper's chi-square, Cramer's V, permutation,
and rarefaction analyses. The generated grouped-bar figure also preserves the
previous ordering, normalization, labels, and layout while reading the new
consensus table.

The machine-readable output additionally includes repository-cluster bootstrap
intervals and category-level Fisher tests with Holm correction. These are
retained as optional robustness instruments and are not part of the paper's
primary RQ1 analysis.
