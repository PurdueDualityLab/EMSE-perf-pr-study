# AGENTS.md

## Purpose

This repository is a small Python research/mining project for reproducing the five-stage performance-PR study pipeline: GitHub mining, AIDev attribution, performance classification, strict human-candidate filtering, and deterministic weekly sampling. Most work should be targeted fixes and artifact validation, not broad refactors.

## Repository layout

- `mining/src/`: main pipeline code.
- `mining/tests/`: pytest-based regression tests.
- `mining/tools/`: artifact validation and private Hugging Face publication tooling.
- `mining/analysis/`: ignored local analyses; do not add them unless explicitly requested.
- `report/`: Overleaf paper submodule; update its pinned commit separately from pipeline changes.
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
- `mining/src/aidev_attribution.py`: internal implementation of the published AIDev attribution rules.
- `mining/src/classify_agentic_prs.py`: published-rule AIDev attribution across the raw snapshot.
- `mining/src/aidev_task_classifier.py`: internal AIDev-compatible task-type classifier.
- `mining/src/classify_performance_prs.py`: AIDev-compatible task classification through the Batch API.
- `mining/src/select_human_candidates.py`: strict observable-signal filtering for human candidates.
- `mining/src/build_balanced_sample.py`: deterministic weekly 1:1 sampling.
- `mining/src/schema.py`: shared column-name constants and output helpers.
- `mining/tools/publish_artifacts.py`: allowlisted validation and publication of official artifacts to a private Hugging Face dataset.

## Working conventions

- Prefer minimal fixes. This codebase is compact and mostly function-oriented.
- Preserve the current style: plain functions, small helpers, pandas dataframes, explicit dict/list structures.
- Avoid introducing new abstractions unless a change clearly needs them.
- Be careful with column-name compatibility. Several modules intentionally support multiple possible input column names.
- Avoid changing output file names, report field names, or resume-state structure unless the task explicitly requires it.

## Testing

- Expected test command: `python -m pytest mining/tests -q`
- Install all dependencies with `pip install -r requirements.txt` inside a virtualenv before relying on test results.

## Run commands

- Mining: `python mining/src/collect_pull_requests.py --config mining/config.local.yaml`
- Publication validation: `python mining/tools/publish_artifacts.py --dry-run`
- Private Hugging Face publication: `python mining/tools/publish_artifacts.py --repo-id namespace/dataset --upload`

## PR review focus

When evaluating or fixing this PR, prioritize:

- Correctness of batching and per-repo checkpoint writes.
- Resume behavior and signature compatibility.
- GitHub token rotation and rate-limit failure handling.
- Dataset filtering correctness, especially author and quality filters.
- Stability of parquet/report outputs and summary counts.
- Artifact allowlists, checksums, path sanitization, and private-only Hugging Face uploads.

## Notes for edits

- Tests currently modify `sys.path` to import from `mining/src`; keep that in mind before reorganizing imports.
- This repo may contain local config or secrets-related files; do not print or commit token values.
- Real GitHub mining now requires `github.token_file`; keep token-file validation and token-rotation behavior intact.
- If a change touches mining behavior, add or update a focused test under `mining/tests/` whenever feasible.
- Do not commit generated Parquets, Batch payloads, checkpoints, local analyses, `.env`, or token files.
- Keep the Hugging Face dataset private. The Dataset Viewer is unavailable for private datasets without a PRO or Enterprise account, but authenticated downloads remain supported.
- Treat `report/` as a separate Git repository: commit and push paper edits inside the submodule first, then update and commit the parent repository's gitlink.
