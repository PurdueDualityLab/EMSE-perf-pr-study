# Mining rebalanced human performance PRs

This directory contains the real mining pipeline for rebuilding the human
performance-PR arm with the same inclusion criteria as the AIDev-pop agent arm:
repositories with at least 100 stars, the same AIDev-pop collection time window,
the same `perf` task-type criterion, shared author filtering, and the same
quality filters used in the workshop study.

The methodological goal is not to force equal sample sizes. The final count is
the natural yield after applying the matched inclusion criteria.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r mining/requirements.txt
```

## Secrets

Copy `mining/.env.example` to `mining/.env` and fill in your GitHub tokens as
separate variables:

```env
GITHUB_TOKEN_1=ghp_first_token
GITHUB_TOKEN_2=ghp_second_token
GITHUB_TOKEN_3=ghp_third_token
GITHUB_TOKEN_4=ghp_fourth_token
GITHUB_TOKEN_5=ghp_fifth_token
```

The pipeline loads `mining/.env` automatically. Keep `github.token_env:
GITHUB_TOKEN` in the YAML; that value is treated as the prefix, so the miner
reads `GITHUB_TOKEN_1`, `GITHUB_TOKEN_2`, and so on. If one token hits rate
limit, the GitHub client rotates to the next token and retries the same request.

## Real mining run

Copy `mining/config.example.yaml` to `mining/config.local.yaml` and run:

```bash
python mining/src/build_rebalanced_dataset.py --config mining/config.local.yaml
```

GitHub mining runs in parallel batches. The default is 20 repositories at a
time:

```yaml
github:
  batch_size: 20
```

Set `batch_size: 1` to run repo-by-repo. Even with parallel batches, the
pipeline writes the resume checkpoint after each completed repository.

If AIDev parquets are already downloaded locally, set `source.data_root` to the
directory containing `pull_request.parquet`, `repository.parquet`,
`pr_task_type.parquet`, `human_pull_request.parquet`, and
`human_pr_task_type.parquet`.

The pipeline resumes automatically from `mining/outputs/raw/github_human_prs.parquet`
and `mining/outputs/state/rebalancing_state.json`. If you want to start over from
scratch, set:

```yaml
resume:
  enabled: true
  reset: true
```

Run it once with `reset: true`, then put it back to `false` for normal resume.

## Outputs

```text
outputs/
  raw/github_human_prs.parquet
  intermediate/human_prs_task_typed.parquet
  intermediate/human_prs_filtered.parquet
  final/agent_perf_prs_matched_criteria.parquet
  final/human_perf_prs_matched_criteria.parquet
  final/rebalancing_summary.json
  final/rebalancing_summary.md
  final/rebalancing_detailed_report.json
  final/rebalancing_detailed_report.md
  state/rebalancing_state.json
```

`rebalancing_summary.md` reports the time window, repository count, AI perf PR
count, human perf PR counts before/after quality filters, task-type label
sources, quality-filter removals, and the required methodological note:
same inclusion criteria, no imposed target count.

`rebalancing_detailed_report.json` and `rebalancing_detailed_report.md` add the
row-level details: failed repositories, failed PR fetches, author-filter
removals, enrichment failures, and the removed rows for each quality filter.

## Inspect the mined data

Use the analysis helper to read the final parquets and print a detailed summary:

```bash
python mining/analysis/rebalancing_report.py
python mining/analysis/rebalancing_report.py --json
```

The script reports AI vs human counts, totals, unique repos/authors, time
range, task-type source breakdowns, and the quality-filter removals.
