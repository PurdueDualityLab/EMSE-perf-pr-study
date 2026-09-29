# Metric-Analysis Provenance

## Extraction and Classification Layers

The deterministic metric extraction and the primary validation-type labels have
separate provenance. The original extraction used validation consensus with
SHA-256 prefix `a29be9ab`. The subsequent primary-type consensus has prefix
`5e31d0c7`; complete hashes are recorded in
[`analysis/classification_labels/manifest.json`](../classification_labels/manifest.json).

Validation presence and the extracted metric indicators are unchanged between
these records. Updating the primary-type layer resolves additional positive
cases: 1,581 becomes 1,684, and the number with unresolved types falls from 118
to 15. Previously resolved primary types retain their classifications.

The submitted paper uses the 1,684-row resolved-type population. The
`in_type_layer` field in the compact label export identifies that population.
The 15 unresolved positive cases remain in the general 1,699-PR metric population.

## Metadata Interpretation

`summary.json` identifies the original extraction and therefore retains its
original population fields and input hashes. It is not the specification of
the current primary-type population. The compact-label manifest records the
original extraction source, the classification update, and the current export
checksum independently.

`refresh_type_layer.py` implements this classification update. It verifies
unchanged validation presence, updates primary-type fields, and rejects changes
to previously resolved types. It writes a separate output directory for
comparison with the recorded results.

## Supporting Analyses

The recorded results incorporate updated type-dependent metric comparisons.
Some historical supporting tables retain the original extraction metadata;
their scope is identified in `results/rq3_results.md`. In particular, the
previously recorded per-agent benchmark summaries preceded the primary-type
update.

The journal-wide offline reproduction uses the current compact labels and the
archived metadata supplement. It recomputes the full statistical family and
the per-agent descriptive summaries under the selected primary-type definition.
Its output manifest records the input and implementation checksums.

## Assessment References

Metric-extraction precision and runtime–memory classifications are assessed
separately from the metric-profile computations. The selected assessment records
and sensitivity analyses are documented in
[`analysis/rq3_llm_validation/PROVENANCE.md`](../rq3_llm_validation/PROVENANCE.md).
