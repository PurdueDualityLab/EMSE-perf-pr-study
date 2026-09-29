# Research Question Analyses

This directory contains the methods and inputs used to reproduce the four
research questions in the journal article. Analyses use fixed PR identities,
archived measurements, and recorded classification decisions.

## Manuscript Correspondence

Directory and file names retain the numbering used when the analysis modules
were introduced. Their correspondence with the journal article is:

| Journal question | Analysis directory | Principal inputs |
| --- | --- | --- |
| RQ1: adoption, patch size, and structure | `quantitative_analysis/`, `maintainability/` | `quantitative_analysis/results/pr_quantitative.csv`, `quantitative_analysis/results/structural/` |
| RQ2: optimization patterns | `rq1_optimization_patterns/` | `classification_labels/rq1_labels.csv` |
| RQ3: validation evidence and temporal trends | `rq2_validation/` | `classification_labels/rq2_labels.csv`, `rq2_validation/temporal/pr_created_at.csv` |
| RQ4: reported performance metrics | `rq3_pattern_and_validation/` | `classification_labels/rq3_labels.csv`, `artifact_inputs/rq4_metadata.csv` |

The stored arm identifiers are `agentic` and `human_candidate`; the latter is
displayed as human-authored in the manuscript. Metric identifiers `D0`–`D9` in
the data correspond one-to-one to `M0`–`M9` in the manuscript.

## Reproduction

From the repository root:

```bash
python analysis/reproduce_paper.py --check-only
python analysis/reproduce_paper.py --output-dir reproduction
```

The first command verifies input integrity and population consistency. The
second recomputes statistical results, supporting tables, figures, and audit
summaries. Environment requirements and output descriptions are provided in
[REPRODUCING.md](../REPRODUCING.md).

## Analysis Documentation

- [Adoption and patch characteristics](quantitative_analysis/README.md).
- [Paired-file structural measurements](maintainability/README.md).
- [Optimization-pattern classification](rq1_optimization_patterns/README.md).
- [Validation-evidence classification](rq2_validation/README.md).
- [Temporal analysis](rq2_validation/temporal/README.md).
- [Performance-metric analysis](rq3_pattern_and_validation/README.md).
- [Audit procedures and results](audits/README.md).
- [Compact label dictionary](classification_labels/README.md).

## Classification and Provenance

Optimization patterns and validation evidence are classified independently by
GPT, Gemini, and Qwen. Optimization labels require a majority on the complete
category–pattern pair. Validation labels use separate decisions for presence
and primary evidence type. Performance metrics are extracted deterministically
from the archived PR evidence.

The offline workflow uses recorded labels and measurements. Reconstructing
their collection requires the archived evidence and the additional resources
described in the module documentation. Input hashes and selected revisions are
recorded in the [artifact manifest](../artifact_manifest.json).
