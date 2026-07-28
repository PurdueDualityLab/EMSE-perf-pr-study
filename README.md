# EMSE Performance PR Study

This repository contains the mining and selection code for the human
performance-PR arm used in the EMSE study.

## Pipelines

| Pipeline | Entrypoint | Purpose |
| --- | --- | --- |
| Rebalancing | `mining/src/build_rebalanced_dataset.py` | Mine AIDev-Pop repositories through GitHub, checkpoint per repository, and build agentic and human comparison outputs. |
| Official selection | `mining/src/build_official_selection.py` | Apply the fixed paper window, heuristic, quality filters, pinned PerfAnnotator model, and author-arm assignment without sampling. |

The pipelines are independent. Experimental diff classification and historical
recovery scripts do not define the official population.

## Layout

```text
mining/
  analysis/       Output validation and reporting
  experimental/   Non-official diff-based experiments
  legacy/         Superseded scripts retained for auditability
  src/            Supported pipeline code
  tests/          Pytest regression tests
  tools/          Artifact maintenance and recovery utilities
  ARTIFACTS.md     Local artifact inventory and checksums
  config.example.yaml
```

Large datasets, model caches, outputs, local configuration, and tokens are
ignored by Git. Their expected locations are documented in
[`mining/ARTIFACTS.md`](mining/ARTIFACTS.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp mining/config.example.yaml mining/config.local.yaml
```

GitHub runs require `mining/github_tokens.txt`, with one token per line. The
file is local-only and must have restrictive permissions, for example
`chmod 600 mining/github_tokens.txt`.

## Commands

```bash
# Rebalancing pipeline
.venv/bin/python mining/src/build_rebalanced_dataset.py --config mining/config.local.yaml

# Official selection into a new, empty directory
.venv/bin/python mining/src/build_official_selection.py \
  --input mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --output-dir mining/official_selection_v2

# Validate existing official outputs
.venv/bin/python mining/analysis/official_selection_report.py

# Validate a newly built v2 directory
.venv/bin/python mining/analysis/official_selection_report.py \
  --output-dir mining/official_selection_v2

# Tests
.venv/bin/python -m pytest mining/tests -q
```

Do not rerun or resume the completed snapshot in `mining/official_selection/`.
It was produced by the archived v1 implementation and is preserved as a study
artifact. Current v2 runs require exact code and artifact integrity matches and
produce a sibling `.sha256` lock that must be preserved.

See [`mining/README.md`](mining/README.md) for configuration, resume semantics,
outputs, and auxiliary tools.
