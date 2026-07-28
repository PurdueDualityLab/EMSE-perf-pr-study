# Experimental Tools

These scripts support exploratory diff-based PerfAnnotator runs. They are not
part of either reproducible dataset pipeline and their outputs are not official
study artifacts.

- `enrich_prs_with_diffs.py` fetches commit messages and bounded patch text.
- `classify_perfannotator_diffs.py` classifies the enriched text.

Use explicit input and output paths under `mining/query_cache/`. Validate an
experimental output independently before using it in an analysis.
