# Current RQ3 Results

This directory publishes the derived tables, statistical results, and figures
for the current balanced sample. PR-level intermediate data under `data/` are
excluded from Git and remain available in the private Hugging Face dataset.
Compact public labels are available in `analysis/classification_labels/`.

The current analysis populations are:

| Layer | All | agentic | human_candidate |
| --- | ---: | ---: | ---: |
| RQ1 included and RQ2 presence available | 2,081 | 1,048 | 1,033 |
| Positive validation presence / metric layer | 1,699 | 858 | 841 |
| Resolved positive validation type | 1,581 | 807 | 774 |

- `results/` contains the complete analysis report, individual CSV tables, the
  60-test family with Benjamini-Hochberg corrections, and qualitative extremes.
- `figures/` contains the four vector PDF figures generated from the current
  analysis.
- `summary.json` records cohort counts and SHA-256 hashes of the inputs.
