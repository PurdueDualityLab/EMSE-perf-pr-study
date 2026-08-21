# Mining Pipeline

The pipeline exposes one entrypoint per study stage. Run commands from the
repository root.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp mining/config.example.yaml mining/config.local.yaml
cp mining/github_tokens.example.txt mining/github_tokens.txt
```

Keep GitHub tokens in `mining/github_tokens.txt` and the OpenAI key in
`OPENAI_API_KEY` or an ignored `.env` file. Never commit or publish these
files.

## Stage 1: Pull Request Mining

```bash
.venv/bin/python mining/src/collect_pull_requests.py \
  --config mining/config.local.yaml
```

The canonical raw snapshot is
`mining/outputs_2026_06_01/raw/github_human_prs.parquet` (1,603,213 rows).
Mining checkpoints are written per repository and may be resumed with the
same configuration signature.

## Stage 2: Agentic Attribution

The AIDev attribution stage applies the five published GitHub-search rules and
preserves historical AIDev positives. Preparation and execution are separate
so the pending partition can be reviewed before API use.

```bash
.venv/bin/python mining/src/classify_agentic_prs.py \
  --raw mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --existing mining/aidev_attribution/aidev_window.parquet \
  --existing-state mining/aidev_attribution/aidev_window.state.json \
  --output-dir mining/aidev_attribution_all_available_dates \
  --end-date 2026-06-01

.venv/bin/python mining/src/classify_agentic_prs.py \
  --raw mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --existing mining/aidev_attribution/aidev_window.parquet \
  --existing-state mining/aidev_attribution/aidev_window.state.json \
  --output-dir mining/aidev_attribution_all_available_dates \
  --end-date 2026-06-01 --execute
```

The final attribution has 78,696 `agentic`, 1,524,066 `human_candidate`, and
451 `unresolved` rows.

## Stage 3: Performance Classification

`classify_performance_prs.py` reproduces AIDev's Conventional Commit cascade
and sends unmatched rows to an OpenAI-compatible Batch endpoint. Use the same
arguments for each action and add `--resume` after the first submission.

```bash
.venv/bin/python mining/src/classify_performance_prs.py submit \
  --input mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --output-dir mining/runs/performance --model gpt-5.6-luna
.venv/bin/python mining/src/classify_performance_prs.py status \
  --input mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --output-dir mining/runs/performance --model gpt-5.6-luna --resume
.venv/bin/python mining/src/classify_performance_prs.py collect \
  --input mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --output-dir mining/runs/performance --model gpt-5.6-luna --resume
.venv/bin/python mining/src/classify_performance_prs.py finalize \
  --input mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --output-dir mining/runs/performance --model gpt-5.6-luna --resume
```

The consolidated study output contains 29,483 performance PRs. It incorporates
141 recovered requests and retains six explicit API errors.

## Stage 4: Human Candidates

```bash
.venv/bin/python mining/src/select_human_candidates.py \
  --task-type mining/experiments/aidev_luna_full_v1/task_type_decisions_with_retry.parquet \
  --attribution mining/aidev_attribution_all_available_dates/aidev_all_available_dates.parquet \
  --output-dir mining/aidev_human_candidates_v1
```

The strict filter excludes bot authors and observable coding-agent, AI
authorship, AI review, and generated-metadata signals. It retains 24,680 human
candidates and excludes 3,447 records.

## Stage 5: Weekly Balanced Sample

```bash
.venv/bin/python mining/src/build_balanced_sample.py \
  --task-type mining/experiments/aidev_luna_full_v1/task_type_decisions_with_retry.parquet \
  --attribution mining/aidev_attribution_all_available_dates/aidev_all_available_dates.parquet \
  --human-candidates mining/aidev_human_candidates_v1/human_candidates.parquet \
  --output-dir mining/aidev_weekly_balanced_sample_v1
```

All 1,356 agentic performance PRs are retained. Human candidates are sampled
without replacement to the same quota in each ISO week using seed
`emse-primary-human-sample-v1` and ordering hash
`SHA256(seed + NUL + repo_id + NUL + number)`.

## Validation and Publication

```bash
.venv/bin/python -m pytest mining/tests -q
.venv/bin/python mining/tools/publish_artifacts.py --dry-run
```

To upload, pass a private Hugging Face dataset identifier explicitly:

```bash
.venv/bin/python mining/tools/publish_artifacts.py \
  --repo-id namespace/dataset --upload
```

The publisher rejects public repositories, excludes operational state and API
payloads, rewrites local absolute paths in summaries, and creates a portable
manifest with SHA-256 checksums.
