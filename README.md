# Performance Pull Request Study

This repository provides a reproducible Python pipeline for identifying and
sampling performance-improving pull requests authored by coding agents and
human candidates. It combines GitHub mining, the published AIDev attribution
rules, AIDev-compatible task classification, strict observable-signal
filtering, and deterministic temporal sampling.

## Pipeline

The workflow has five stages:

1. **Pull request mining:** collect and checkpoint GitHub pull requests for the
   selected repositories and time window.
2. **Agentic attribution:** apply the published AIDev rules and preserve
   historical AIDev positives.
3. **Performance classification:** reproduce AIDev's Conventional Commit
   cascade and classify unmatched pull requests through the Batch API.
4. **Human-candidate filtering:** exclude observable bot, coding-agent,
   AI-authorship, AI-review, and generated-metadata signals.
5. **Weekly balanced sampling:** retain every agentic performance pull request
   and sample human candidates to the same quota within each ISO week.

## Results

| Artifact | Pull requests |
| --- | ---: |
| Raw GitHub snapshot | 1,603,213 |
| Agentic attribution | 78,696 |
| Performance classification | 29,483 |
| Agentic performance cohort | 1,356 |
| Strict human-candidate cohort | 24,680 |
| Final balanced sample | 2,712 |

The final sample contains 1,356 agentic and 1,356 human-candidate performance
pull requests across 67 ISO-week strata. "Human candidate" means that no
selected observable agent signal was found; it does not establish confirmed
human authorship.

## Repository Structure

```text
mining/
  src/                 Five-stage pipeline and shared helpers
  tests/               Pytest regression tests
  tools/               Artifact validation and publication tooling
  ARTIFACTS.md          Official artifact inventory and checksums
  config.example.yaml  Mining configuration template
```

Generated datasets, checkpoints, API payloads, caches, local configuration,
and credentials are intentionally excluded from Git.

## Quick Start

Python 3.12 is the tested environment.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp mining/config.example.yaml mining/config.local.yaml
cp mining/github_tokens.example.txt mining/github_tokens.txt
```

Add one GitHub token per line to the ignored token file, then start the mining
stage from the repository root:

```bash
.venv/bin/python mining/src/collect_pull_requests.py \
  --config mining/config.local.yaml
```

The remaining stage commands, inputs, and resume behavior are documented in
[`mining/README.md`](mining/README.md).

## Validation

Run the regression suite and validate the complete artifact allowlist with:

```bash
.venv/bin/python -m pytest mining/tests -q
.venv/bin/python mining/tools/publish_artifacts.py --dry-run
```

The publication validator checks file presence, row counts, schemas, SHA-256
digests, and metadata portability before any upload is allowed.

## Data Access

The approved artifacts are stored in the private Hugging Face dataset
[`rcalvome/EMSE-perf-pr-study`](https://huggingface.co/datasets/rcalvome/EMSE-perf-pr-study).
Authenticated users can download individual Parquet files or load a declared
dataset subset. `balanced-sample` is the default subset.

The Hub Dataset Viewer is unavailable for private datasets without a PRO or
Enterprise account; this does not affect authenticated downloads. See
[`mining/ARTIFACTS.md`](mining/ARTIFACTS.md) for the exact artifact inventory
and checksums.

## License

The processing code is licensed under the [MIT License](LICENSE). Generated
records retain their original source metadata and may be subject to GitHub and
source-repository terms.
