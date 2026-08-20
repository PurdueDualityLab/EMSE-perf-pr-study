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

## Historical PerfAnnotator Metadata

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

This metadata mode is preserved for v1 auditability only. It classifies one PR
row built from title, body, commit-message list, and filenames. It does not
implement PerfMiner's published per-commit `(commit_message, code_diff)` input
and must not be reported as a PerfMiner reproduction.

## PerfMiner Reproduction

The corrected pipeline writes to a new directory and never mutates the v1
snapshot. Start from the 55,114 quality-passed PRs:

```bash
.venv/bin/python mining/src/fetch_pr_commit_manifest.py \
  --input mining/official_selection/intermediate/quality_passed.parquet \
  --output-dir mining/perfminer_reproduction/manifest \
  --token-file mining/github_tokens.txt

.venv/bin/python mining/src/extract_perfminer_commit_evidence.py \
  --pull-requests mining/perfminer_reproduction/manifest/pull_requests.parquet \
  --commits mining/perfminer_reproduction/manifest/commits.parquet \
  --output-dir mining/perfminer_reproduction/evidence

.venv/bin/python mining/src/classify_perfminer_commits.py \
  --input mining/perfminer_reproduction/evidence/commit_evidence.parquet \
  --output-dir mining/perfminer_reproduction/predictions \
  --model-dir mining/query_cache/models/perfannotator-mini-ease-2026 \
  --device cpu --batch-size 1

.venv/bin/python mining/src/aggregate_perfminer_prs.py \
  --pull-requests mining/perfminer_reproduction/manifest/pull_requests.parquet \
  --commit-predictions mining/perfminer_reproduction/predictions/commit_predictions.parquet \
  --output-dir mining/perfminer_reproduction/prs
```

Use `--resume` for an interrupted stage. Manifest checkpoints contain at most
50 PRs. Evidence targets about 100 commit associations per part but never
splits one PR, so a single part can contain up to 250. Git objects are fetched
into the ignored
`mining/query_cache/perfminer_repositories/` cache with partial-clone filtering.

The EASE 2026 model is distributed in Figshare package
`10.6084/m9.figshare.32142181.v1`. Extract
`replication_package/perfannotator-mini/perfannotator-mini/` into the model path
shown above. Classification rejects the directory unless all eight files match
the published bundle. The expected model-weights SHA-256 is
`c8e71790c6dc286562df297b40405d5c7ee8ad8bb0ca1fb47490949f6b5a47dd`;
the complete extracted artifact hash is
`9a0107e3fde617214c7d58cffda8dc29043adfcdeeee63331c51f458b05b40a6`.

The operational contract is pinned to PerfMiner revision
`73d1c2cb23e3f1f1ce9d4909262a1763ccadae29`:

- one commit per inference;
- the upstream commit-message cleanup and message-length filters;
- exactly one changed non-binary Java, Python, C, or C++ file and one changed method;
- PyDriller's raw file diff without synthetic filename headers;
- tokenizer pair input with `truncation="only_second"` and 512 total tokens;
- class 1 at `P(class 1) >= 0.5`.

Each PR head is read before and after its ordered commit list. A changed head,
count, or base is retried and never emits partial commit rows. Every downstream
row carries a digest of the frozen ordered commit list; extraction and
aggregation reject mixed snapshots or model contracts. PRs over the REST
endpoint's 250-commit limit remain explicitly unresolved.

PR aggregation uses `any_positive_commit`. A positive commit is sufficient
positive evidence. A PR is negative only when its manifest and every eligible
commit classification are complete. PRs with no eligible operational-PerfMiner
commit are `out_of_scope_no_eligible_commit`, not negative.

## AIDev Attribution Enrichment

The standalone attribution enricher applies only the five published AIDev
searches. It does not use the local author heuristic and never modifies the
input Parquet:

```bash
.venv/bin/python mining/src/enrich_aidev_attribution.py \
  --input mining/official_selection/intermediate/date_filtered.parquet \
  --output mining/aidev_attribution/aidev_window.parquet \
  --end-date 2025-07-30 \
  --token-file mining/github_tokens.txt
```

The script searches each input repository for Codex, Devin, GitHub Copilot,
Cursor, and Claude Code using the published agent-specific start dates. It
resolves repositories by immutable GitHub ID, fetches PR details only for matches, appends
attribution evidence and head-ref fields, and checkpoints in groups of 25
repositories by default. Use `--resume` to retry failed searches or detail
fetches. A nonmatching row becomes `human_candidate` only when every search
applicable on its creation date completed successfully and the repository's
immutable GitHub ID was verified; otherwise it remains `unresolved`.

Deleted repositories are terminally unavailable and remain `unresolved`; they
do not block finalization. The one-time
`--adopt-repository-resolution-upgrade` option accepts only the exact
pre-upgrade checkpoint code hash and preserves already completed searches while
retrying renamed or unavailable repositories by ID.

The output has sibling `.state.json` and `.summary.json` files containing the
run signature, exact rules, progress, failures, counts, and artifact hashes.

### All Available Dates Attribution

`enrich_aidev_all_available_dates.py` extends attribution to every PR in the
fixed full raw-mining snapshot without re-enriching identities already present
in `aidev_window.parquet`. It preserves the official-window artifact and writes
an independent output. Its scope is all available dates in the raw snapshot,
not a census of GitHub.

Prepare the deterministic pending partition first. This command performs no
GitHub requests:

```bash
.venv/bin/python mining/src/enrich_aidev_all_available_dates.py \
  --raw mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --existing mining/aidev_attribution/aidev_window.parquet \
  --existing-state mining/aidev_attribution/aidev_window.state.json \
  --output-dir mining/aidev_attribution_all_available_dates \
  --end-date 2026-06-01
```

Review `plan.json` before starting the network stage. Only then run the same
command with `--execute` and the desired GitHub token file. The network stage
uses `pending.parquet` as its only enricher input, so PRs already attributed in
the official window are never submitted for a second PR-detail enrichment.

The final merge applies pinned AIDev historical-positive precedence only when a
live result is not already `agentic`. It records these changes in `summary.json`
and leaves the official-window output unchanged.

### Method A Binary Agentic Classification

`classify_agentic_method_a.py` applies the agreed binary Method A policy to the
all-available-dates attribution artifact. A PR is `agentic` when it matches an
AIDev agent mechanism, an official coding-agent task/session URL, an explicit
agent-generation statement, `Co-Authored-By: Claude`, or any GitHub author whose
type is `Bot`. All other PRs are `non_agentic`; AI review, generated summaries,
and generated descriptions alone do not produce a positive decision.

The output label is binary, while `method_a_rules` preserves the matching rule
IDs for auditability. `summary.json` reports the number of PRs classified only
because of the broad `generic_bot_author` rule, which includes conventional bots
such as Dependabot and Renovate.

```bash
.venv/bin/python mining/src/classify_agentic_method_a.py \
  --input mining/aidev_attribution_all_available_dates/aidev_all_available_dates.parquet \
  --output-dir mining/agentic_method_a
```

The current input has no GitHub App ID/slug. Bot-authored app PRs are covered by
the generic bot rule; detecting apps that submit through a human account would
require a separate GitHub metadata enrichment.

### Strict AIDev Human-Candidate Filter

`filter_aidev_human_candidates.py` starts from performance PRs classified as
`human_candidate` by the all-dates AIDev reproduction. It excludes any
observable bot author, coding-agent account or task URL, explicit AI authorship,
AI coauthor trailer, AI review marker, or AI-generated PR metadata. Under the
selected strict policy, delimited automation or agent tokens in a `User` login
are also exclusion signals.

```bash
.venv/bin/python mining/src/filter_aidev_human_candidates.py \
  --task-type mining/experiments/aidev_luna_full_v1/task_type_decisions_with_retry.parquet \
  --attribution mining/aidev_attribution_all_available_dates/aidev_all_available_dates.parquet \
  --output-dir mining/aidev_human_candidates_v1
```

The output directory contains all decisions, retained human candidates,
excluded rows, and a summary with rule counts and overlaps. The retained label
means no observable signal under this method; it does not confirm human
authorship.

### Weekly Balanced Performance Sample

`sample_aidev_performance_weekly.py` includes every AIDev agentic-performance
PR and samples an equal number of strict human candidates within each ISO week
of `created_at` in UTC. Human selection is without replacement and ordered by a
deterministic SHA-256 key derived from the fixed seed and immutable PR identity.

```bash
.venv/bin/python mining/src/sample_aidev_performance_weekly.py \
  --task-type mining/experiments/aidev_luna_full_v1/task_type_decisions_with_retry.parquet \
  --attribution mining/aidev_attribution_all_available_dates/aidev_all_available_dates.parquet \
  --human-candidates mining/aidev_human_candidates_v1/human_candidates.parquet \
  --output-dir mining/aidev_weekly_balanced_sample_v1 \
  --seed emse-primary-human-sample-v1
```

The output includes both sample arms, their combined sample, all weekly quotas,
and a manifest covering the full sampling frame with inclusion probabilities,
selection hashes, and selection status.

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

## Performance Experiments

The experiment scripts keep every PR-level decision separate from the immutable
study artifacts. Build the first experiment's all-dates candidates from the raw
snapshot and final AIDev attribution output:

```bash
.venv/bin/python mining/src/select_performance_experiment_candidates.py \
  --raw mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --attribution mining/aidev_attribution_all_available_dates/aidev_all_available_dates.parquet \
  --output-dir mining/experiments/project_heuristic_perfminer/selection \
  --heuristic project \
  --start-date 2024-12-24T00:00:05Z \
  --end-date 2026-06-01T23:58:49Z

```

`project` uses the project title/body heuristic. The selection directory contains
`decisions.parquet` for all input PRs and `candidates.parquet` for its matches.

Run the existing four-stage PerfMiner pipeline with the project candidates as
its input, using `mining/experiments/project_heuristic_perfminer/` as the new
root for its manifest, evidence, predictions, and PR aggregation outputs.

The AIDev-plus-Luna experiment reproduces the public AIDev cascade over every
PR. It labels Conventional Commit titles locally and sends only titles without a
match to the replacement LLM. Its public source is pinned to
`SAILResearch/AI_Teammates_in_SE3@be60e52962ef867b300aac433a05f93440cbb653`.
It requires an API key only in the process environment. Do not place it in a
command, config file, or artifact:

```bash
.venv/bin/python mining/src/classify_aidev_task_type_luna_batch.py submit \
  --input mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --output-dir mining/experiments/aidev_luna_task_type \
  --model gpt-5.6-luna
```

The Batch runner uses AIDev's twelve Conventional Commit labels, title-first
regex, GPT-4 tokenizer with a 10,000-token body limit, and the same Chat
Completions JSON schema. Luna requires its default temperature,
so the runner omits AIDev's legacy `temperature=0` parameter. It writes local JSONL request files,
uploads them to OpenAI Batch, and preserves every batch ID in `state.json`.
The default submission creates one 10,000-request batch; increase
`--max-batches` only after confirming the account's queued-token quota.

Poll and collect completed batches, then finalize the one-row-per-PR output:

```bash
.venv/bin/python mining/src/classify_aidev_task_type_luna_batch.py status \
  --input mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --output-dir mining/experiments/aidev_luna_task_type \
  --model gpt-5.6-luna --resume

.venv/bin/python mining/src/classify_aidev_task_type_luna_batch.py collect \
  --input mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --output-dir mining/experiments/aidev_luna_task_type \
  --model gpt-5.6-luna --resume

.venv/bin/python mining/src/classify_aidev_task_type_luna_batch.py finalize \
  --input mining/outputs_2026_06_01/raw/github_human_prs.parquet \
  --output-dir mining/experiments/aidev_luna_task_type \
  --model gpt-5.6-luna --resume
```

`finalize` writes `task_type_decisions.parquet` with every PR: title-matched
rows never call the LLM; unmatched rows receive either the LLM's task-type label
or an explicit Batch error record. Submit additional batches with `submit
--resume` after completed requests have been collected.

After PerfMiner and the LLM finish, produce the one-row-per-PR comparison:

```bash
.venv/bin/python mining/src/report_performance_experiments.py \
  --project-decisions mining/experiments/project_heuristic_perfminer/selection/decisions.parquet \
  --perfminer-prs mining/experiments/project_heuristic_perfminer/prs/pr_predictions.parquet \
  --aidev-task-type mining/experiments/aidev_luna_task_type/task_type_decisions.parquet \
  --output-dir mining/experiments/comparison
```

## Auxiliary Code

- `mining/tools/mine_missing_aidev_pop_prs.py` recovers known AIDev references absent from a raw checkpoint.
- `mining/experimental/` contains non-official diff enrichment and classification.
- `mining/legacy/` contains superseded code retained only for auditability.

## Tests

```bash
.venv/bin/python -m pytest mining/tests -q
```
