# Executed Optimization Catalog (Journal RQ2)

`updated_optimization_catalog.csv` is the canonical 58-pattern catalog used by
the pattern-classification stage (historical RQ1, journal RQ2). The submitted
paper reports 59 patterns, but the material catalog and its
revision history support 58 rows: 17 patterns were added, `Loop Strip-mining`
was removed, and two spatial-locality patterns were consolidated into one. The
reported 59 is consistent with counting the additions and deletion without
accounting for that 2-to-1 consolidation.

The consolidation is:

- Original: `Increase Cache Efficiency via Locality`.
- Original: `Improve cache locality via data structure`.
- Revised: `Improve cache locality - spatial locality`.

Both original definitions concern grouping data accessed together in memory.
The revised definition incorporates that spatial-locality scope. Consequently,
the accounting is **43 + 17 - 1 - 1 = 58**, not 59. No named I/O pattern has been
identified as missing from the execution inputs.

## Source Provenance

- Pilot source repository: [PurdueDualityLab/github_perf_patch_study](https://github.com/PurdueDualityLab/github_perf_patch_study).
- The earliest revised CSV available in the pilot repository history is commit
  `9a7b3ca` (2025-12-09), under
  `RQ2/pattern_analysis/catalog/updated_optimization_catalog.csv`: 58 rows.
  Subsequent moves at `66dba76` and `0209a11` retain the same pattern set.
- Parent-repository commit `0f1101d` trims whitespace in two labels and records
  this consolidation; it does not add or delete a row.
- The GPT and Gemini preparation snapshots both record catalog SHA-256
  `8a4b2e59a83a3994ee228515ba47ce2712c43112df3070640ea0b3ef7fcf803c`,
  matching the current 58-row CSV. The nine category counts are recorded in
  `provenance.json`.
- Overleaf commit `a932f18` (2026-09-16) removes the consolidation explanation
  and changes the table's I/O count from 6 to 7 and total from 58 to 59, without
  changing the executed catalog. This documents the manuscript discrepancy;
  it does not establish the editorial reason for that change.

The submitted manuscript is retained as a historical reference. Reproduction
uses the executed snapshot and its actual row count. Coverage counts (44 agentic
and 43 human-authored patterns) and category-level tests are reproduced from the
frozen labels; no extra pattern is imputed.

Catalog revisions require corresponding updates to the row count, checksum,
classification-prompt identifier, and publication metadata. The reproduction
inputs retain the catalog used by the recorded classification procedure.
