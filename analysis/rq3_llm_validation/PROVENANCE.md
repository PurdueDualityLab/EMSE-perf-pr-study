# Assessment Provenance

This document identifies the archived executions associated with the
model-based assessments. Directory and module identifiers are retained to
support reproducibility. The main [README](README.md) describes the methods;
the [assessment documentation](../audits/README.md) describes the compact inputs
used to reconstruct reported estimates.

## Execution Identifiers

| Assessment | Archived directory | Implementation | Use in the artifact |
| --- | --- | --- | --- |
| Selected metric-extraction precision assessment | `generated_regex_v3/` | `regex_v3.py`, `run_regex_v3.py`, `watch_regex_v3.py` | Source of `regex_occurrence_votes.csv`; reproduces 77/11 |
| Selected binary runtime–memory assessment | `generated/` | `experiment.py`, provider interfaces, `build_consensus.py` | Source of `tradeoff_votes.csv`; reproduces 17/12 |
| Expanded evidence assessment | `generated_v2/` | `v2.py`, `v2_evidence.py`, `run_v2.py`, `v2_consensus.py` | Additional assessment with expanded occurrence-level evidence and four runtime–memory labels |
| Binary-decision sensitivity assessment | `generated_binary_v1/` | `binary.py`, `run_binary.py`, `binary_validation.py` | Sensitivity to requiring a binary relationship decision |

The selected metric-extraction assessment has the recorded identifier
`rq4-regex-positive-audit-v3`. The runtime–memory sensitivity assessment has the
recorded identifier `rq3-tradeoff-forced-binary-v1`. The `rq3` prefix in these
implementation identifiers corresponds to journal RQ4.

## Metric-Extraction Assessment

The selected assessment uses the same 88 PRs, 1,910 occurrence groups, and 2,384
activations as the expanded evidence assessment. Its copied prompts and
evidence are byte-identical to that preparation. Providers receive no prior
classification labels. OpenAI and Gemini use Batch APIs; Qwen processes the
same recorded inputs locally.

The selected and expanded-evidence assessments both classify 77 PRs as positive
and 11 as negative relative to the reference judgments. The earlier assessment
classifies 68 and 20, respectively. The selected result is aggregated from
occurrence-level majorities, with a PR considered positive when any occurrence
has at least two valid judgments.

The initial evidence presentation emphasized representative matches. The
expanded assessment supplies complete matched records, preserves all original
activations, and explicitly addresses automated reports, baseline values,
unchanged measurements, and regressions. These differences are relevant when
comparing the reference labels across assessments.

## Runtime–Memory Sensitivity

All three assessments below concern the same 29 selected PRs:

| Assessment | Trade-off | Joint improvement | Indeterminate | Unsupported |
| --- | ---: | ---: | ---: | ---: |
| Selected binary assessment reported in the manuscript | 17 | 12 | Not available | Not available |
| Expanded evidence assessment with four labels | 3 | 4 | 8 | 14 |
| Binary-decision sensitivity assessment | 16 | 13 | Not available | Not available |

The final sensitivity assessment records that 24 of 29 majority decisions
require inference. Its evidence is identical to the expanded-evidence input,
while the classification task requires one of two labels. The selected
assessment and sensitivity assessments therefore differ in their evidence or
decision requirements and should not be treated as interchangeable outputs.

An additional evidence review records five supported binary labels, twelve
unsupported claims, and twelve inconclusive cases. These are agent-assisted
review judgments with prior consensus exposure, not blinded human adjudication.
They do not replace the stored provider votes.

## Evidence Normalization and Review Records

Quotation normalization permits unambiguous HTML, Markdown, entity, or
whitespace differences while preserving substantive words, quantities, units,
and classification decisions. Original and normalized quotations remain
associated with their source records.

The binary-decision sensitivity archive contains one explicitly recorded
dimension correction for occurrence `o0138` of rspack PR #13490: a memory-mode
baseline was detected under artifact size because the benchmark name contained
`bundle`. The correction is documented in
`generated_binary_v1/validation_corrections.json`, with request, prompt, and
source-record identifiers. The supplied model inputs and model decisions remain
recorded separately from this validation adjustment.

Complete review packages preserve provider decisions, evidence records,
occurrence-level associations, and checksums. Intermediate incomplete packages
are retained in the archive for traceability; reported results use the complete
selected samples identified above.

## Reproduction Boundary

The compact artifact reproduces aggregation of recorded judgments. Reconstructing
the evidence preparation or reassessing individual claims requires authorized
access to the corresponding archive. A new model execution may yield different
judgments even when generation settings and inputs are held fixed.

Source-file hashes and selected input identities are recorded in
[`analysis/artifact_inputs/manifest.json`](../artifact_inputs/manifest.json).
The [root artifact specification](../../ARTIFACT.md) explains the relationship
between these execution records and the submitted manuscript.
