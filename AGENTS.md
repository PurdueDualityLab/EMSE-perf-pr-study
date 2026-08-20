# AGENTS.md

## Purpose

This repository is a small Python research/mining project for rebuilding the human performance-PR arm used in the EMSE study. Most work will be PR evaluation and targeted fixes, not broad refactors.

## Repository layout

- `mining/src/`: main pipeline code.
- `mining/tests/`: pytest-based regression tests.
- `mining/analysis/`: reporting helpers for mined outputs.
- `mining/config.example.yaml`: tracked runtime configuration template.
- `mining/config.local.yaml`: ignored local runtime configuration.
- `mining/github_tokens.txt`: local GitHub token file used for mining runs.
- `requirements.txt`: Python dependencies.

## Important code paths

- `mining/src/collect_pull_requests.py`: GitHub mining entrypoint. Handles config loading, resume state, batching, mining, filtering, and raw outputs.
- `mining/src/github_client.py`: GitHub API access, token rotation, rate-limit handling, PR search/fetch logic.
- `mining/src/resume.py`: checkpoint and resume-state behavior.
- `mining/src/quality_filters.py`: row-removal rules for dataset quality filtering.
- `mining/src/author_filter.py`: agent-vs-human author filtering.
- `mining/src/classify_task_type.py`: task-type attachment/classification.
- `mining/src/classify_agentic_prs.py`: published-rule AIDev attribution across the raw snapshot.
- `mining/src/classify_performance_prs.py`: AIDev-compatible task classification through the Batch API.
- `mining/src/select_human_candidates.py`: strict observable-signal filtering for human candidates.
- `mining/src/build_balanced_sample.py`: deterministic weekly 1:1 sampling.
- `mining/src/schema.py`: shared column-name constants and output helpers.

## Working conventions

- Prefer minimal fixes. This codebase is compact and mostly function-oriented.
- Preserve the current style: plain functions, small helpers, pandas dataframes, explicit dict/list structures.
- Avoid introducing new abstractions unless a change clearly needs them.
- Be careful with column-name compatibility. Several modules intentionally support multiple possible input column names.
- Avoid changing output file names, report field names, or resume-state structure unless the task explicitly requires it.

## Testing

- Expected test command: `python -m pytest mining/tests -q`
- Install dev deps with `pip install -r requirements-dev.txt` inside a virtualenv before relying on test results.

## Run commands

- Mining: `python mining/src/collect_pull_requests.py --config mining/config.local.yaml`
- Publication validation: `python mining/tools/publish_artifacts.py --dry-run`

## PR review focus

When evaluating or fixing this PR, prioritize:

- Correctness of batching and per-repo checkpoint writes.
- Resume behavior and signature compatibility.
- GitHub token rotation and rate-limit failure handling.
- Dataset filtering correctness, especially author and quality filters.
- Stability of parquet/report outputs and summary counts.

## Notes for edits

- Tests currently modify `sys.path` to import from `mining/src`; keep that in mind before reorganizing imports.
- This repo may contain local config or secrets-related files; do not print or commit token values.
- Real GitHub mining now requires `github.token_file`; keep token-file validation and token-rotation behavior intact.
- If a change touches mining behavior, add or update a focused test under `mining/tests/` whenever feasible.
