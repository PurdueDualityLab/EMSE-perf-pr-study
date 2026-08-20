# Performance Pull Request Study

This repository contains the reproducible data pipeline for a study of
performance-improving pull requests authored by coding agents and humans. The
public workflow has five stages:

1. Mine pull requests from GitHub.
2. Attribute pull requests with the published AIDev rules.
3. Classify performance work with the AIDev task taxonomy and Luna Batch.
4. Apply a strict observable-signal filter to human candidates.
5. Build a deterministic 1:1 sample stratified by ISO week.

The final sample contains 1,356 agentic and 1,356 human-candidate pull
requests across 67 weekly strata. "Human candidate" means that no selected
agent signal was observed; it is not a claim of confirmed human authorship.

See [`mining/README.md`](mining/README.md) for commands and methodology, and
[`mining/ARTIFACTS.md`](mining/ARTIFACTS.md) for the artifact inventory.

Source code is licensed under the [MIT License](LICENSE). Generated records
retain their original source metadata and may be subject to upstream terms.
