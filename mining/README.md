# Mining Module

The module supports two dataset pipelines. Neither pipeline imposes a target
sample size; both retain the natural yield after their configured criteria.

## Configuration

Create `mining/config.local.yaml` from the tracked
`mining/config.example.yaml`. The local file remains ignored so output paths and
other machine-specific settings are not committed.

Important settings:

- `source.aidev_revision` pins the AIDev dataset revision.
- `github.token_file` identifies the local token file.
- `github.batch_size` controls concurrent repository workers.
- `github.per_repo_search_limit` is normally unset; setting it intentionally truncates each repository search.
- `criteria.start_date` and `criteria.end_date` override the AIDev-derived time window.
- `resume.reset: true` intentionally clears the configured rebalancing output tree.
- `resume.trust_legacy_checkpoint: true` performs a one-time adoption of an older unhashed checkpoint after manual verification.

The GitHub client paginates PR files and commits, subdivides searches that hit
GitHub's 1,000-result cap, rotates shared tokens on rate limits or HTTP 401, and
retries transient network and server failures.

## Rebalancing Pipeline

```bash
.venv/bin/python mining/src/build_rebalanced_dataset.py \
  --config mining/config.local.yaml
```

The pipeline checkpoints the raw dataframe and resume state after each
repository result. A repository is complete only after its search and all PR
fetches succeed; partial and failed repositories remain eligible for retry.
Resume signatures include the dataset revision, repository identities, time
window, search limit, and relevant criteria. A raw checkpoint without its
matching state is rejected rather than inferred as complete. Current
checkpoints use a journal and SHA-256 so an interruption between Parquet and
state replacement can be recovered. Older states that already mark repositories
complete require the explicit one-time trust setting above.

The configured `outputs.dir` contains:

```text
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

Known AIDev agentic URLs are removed from the human candidate arm before author
and quality filtering.

## Official Selection

Run the current implementation only with a new output directory:

```bash
.venv/bin/python mining/src/build_official_selection.py \
  --input mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --output-dir mining/official_selection_v2
```

The fixed inclusive UTC window is
`2024-12-24T00:23:09+00:00` through
`2025-07-30T19:36:13+00:00`. Processing order is date, project heuristic,
quality, pinned CPU PerfAnnotator inference, then author-arm assignment.
Sampling is not performed.

Version 2 records input and schema hashes, hashes all code dependencies, writes
Parquet files atomically, verifies completed stages before reuse, and records
final output hashes. `--resume` succeeds only for the exact input,
configuration, code, and stage artifacts recorded by that v2 run.

On completion it also writes a sibling checksum lock such as
`mining/official_selection_v2.sha256`. Preserve and version that small file with
the study metadata; the report requires it to authenticate the manifest and
validates all final and intermediate Parquets against it.

The completed `mining/official_selection/` snapshot is an immutable v1 artifact
and cannot be resumed with v2. See [`ARTIFACTS.md`](ARTIFACTS.md).

## PerfAnnotator Metadata

The official selector invokes the resumable metadata classifier internally. It
can also be run directly:

```bash
.venv/bin/python mining/src/classify_perfannotator_metadata.py \
  --input mining/query_cache/input.parquet \
  --output mining/query_cache/perfannotator_metadata.parquet \
  --model-revision 7d7ba362c257c3ea8c69d52b4a736ae4f182e68c \
  --device cpu --seed 20260720 --deterministic
```

Its checkpoint signature includes the input, schema, exact model bytes,
inference package versions, model settings, and code hashes. Every part records
its row count, schema hash, and SHA-256 digest.

The pinned upstream model exposes only generic `LABEL_0`/`LABEL_1` metadata.
This study therefore records an explicit methodological convention:
`positive_label_id = 1` means performance-improving. The validator rejects
descriptive model metadata that contradicts that convention but does not infer
semantics from the generic names.

## Reports

```bash
.venv/bin/python mining/analysis/rebalancing_report.py
.venv/bin/python mining/analysis/rebalancing_report.py --json
.venv/bin/python mining/analysis/official_selection_report.py
.venv/bin/python mining/analysis/official_selection_report.py --json
.venv/bin/python mining/analysis/official_selection_report.py \
  --output-dir mining/official_selection_v2
```

Both report commands return a nonzero exit status when required artifacts or
count invariants are invalid.

## Auxiliary Code

- `mining/tools/mine_missing_aidev_pop_prs.py` recovers known AIDev references absent from a raw checkpoint.
- `mining/experimental/` contains non-official diff enrichment and classification.
- `mining/legacy/` contains superseded code retained only for auditability.

## Tests

```bash
.venv/bin/python -m pytest mining/tests -q
```
