# Historical RQ Analysis

This directory preserves the executable RQ1 and RQ2 analysis code from
`PurdueDualityLab/github_perf_patch_study` at commit
`76374c11c0027492b3f4a7983fa22b583bbe1f92`.

The source analyses were consolidated in December 2025 and last updated in
January 2026. Notebook outputs and execution counts were removed during the
copy. Historical datasets, LLM responses, manual labels, generated figures,
and result tables were intentionally excluded.

## Layout

- `rq1_optimization_patterns/`: optimization-pattern labeling, agreement, and
  analysis code, plus the original and revised taxonomies required by the
  classifiers.
- `rq2_validation/`: performance-validation labeling, label merging, analysis,
  and plotting notebooks.

## Data Migration Status

These files preserve the historical analysis logic but still reference the
old AIDev dataset layout and historical relative paths. Before execution, they
must be adapted to the private Hugging Face dataset pinned by the root `data/`
submodule. The primary target is the `balanced-sample` dataset configuration.

Do not commit API keys, generated model responses, checkpoints, manual-review
exports, or result datasets when adapting these analyses.
