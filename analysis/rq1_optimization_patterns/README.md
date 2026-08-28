# RQ1 Optimization Patterns

This directory contains the optimization-pattern catalog, GPT-5.6-sol and
Gemini 3.1 Pro Preview Batch API classifiers, and supporting analysis notebooks.

## Contents

- `run_rq1.py`: prepares, submits, monitors, and collects the RQ1 batch.
- `run_rq1_gemini.py`: runs one logical Gemini classification over the same RQ1
  sample and handles provider job limits internally.
- `analyze_agreement.py`: calculates agreement and Cohen's kappa globally and
  by sample arm, then exports agreements and human-review candidates.
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
  --evidence-dir mining/sample_evidence_v1/final \
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
.venv/bin/python analysis/rq1_optimization_patterns/analyze_agreement.py \
  --gpt analysis/rq1_optimization_patterns/results_gpt/optimization_pattern_labels.parquet \
  --gemini analysis/rq1_optimization_patterns/results_gemini/optimization_pattern_labels.parquet \
  --sample data/data/sample/balanced_sample.parquet \
  --output-dir analysis/rq1_optimization_patterns/agreement
```

The summary reports exact agreement and Cohen's kappa globally and separately
for agentic and human-candidate PRs. Disagreements are exported for human
adjudication; the script does not select either model as a fallback.

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
