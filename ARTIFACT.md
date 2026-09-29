# Artifact Specification and Provenance

This replication package accompanies **How Do Coding Agents Optimize Software
and Report Performance Validation? A Large-Scale Empirical Study of Open-Source
Pull Requests**. The [reproduction guide](REPRODUCING.md) describes the supported
computational workflow.

## Version Identification

[`artifact_manifest.json`](artifact_manifest.json) specifies input schemas,
row counts, checksums, source-code digests, the reference Python environment,
and the selected data and manuscript revisions. Source digests identify the
implementation used for reproduction; `code_base_revision` records its repository
baseline. Reproduction outputs also record the executing checkout's revision
and package versions.

| Component | Identification |
| --- | --- |
| Full dataset | `4464afd020ece2a619649a185460fde9d02625fd` |
| Submitted manuscript reference | `40bc4fc`; full revision recorded in the manifest |
| Mining and classification inventory | [`mining/ARTIFACTS.md`](mining/ARTIFACTS.md) |
| Classification and metric-label provenance | [`analysis/classification_labels/manifest.json`](analysis/classification_labels/manifest.json) |
| Assessment inputs | [`analysis/artifact_inputs/manifest.json`](analysis/artifact_inputs/manifest.json) |
| Manuscript compilation environment | Image digest specified in [`Dockerfile.paper`](Dockerfile.paper) |

The dataset and manuscript submodule revisions identify the associated source
snapshots. The repository revision and source digests jointly identify the
analysis implementation.

## Manuscript–Artifact Correspondence

### Optimization Catalog

The execution snapshot contains 58 patterns, whereas the submitted manuscript
reports 59. A two-to-one consolidation of spatial-locality patterns accounts for
the difference. The recorded manuscript history changes the I/O category count
from six to seven without a corresponding change to the execution catalog.
Pattern definitions, accounting, checksums, and revision evidence are documented
in the [catalog provenance](analysis/rq1_optimization_patterns/catalog/README.md).

Reproduction uses the recorded 58-pattern snapshot and the classifications
generated from it. The submitted manuscript is retained as the reference
document for this correspondence.

### Research Question and Field Names

Stored `rq1`, `rq2`, and `rq3` label files correspond to journal RQ2, RQ3, and
RQ4, respectively. The stored label `human_candidate` corresponds to the
human-authored group described in the manuscript. Stored metric identifiers
`D0`–`D9` correspond to manuscript identifiers `M0`–`M9`.

### Assessment Procedures

The selected model assessments and additional sensitivity analyses are identified
in [Assessment Provenance](analysis/rq3_llm_validation/PROVENANCE.md). Aggregating
recorded votes reproduces classification counts; it does not independently
verify the associated performance measurements or interpretations.

The manual classifier assessment uses 100 PRs selected through disproportionate
stratification. The manuscript's precision, recall, and F1 values describe the
unweighted sample. Sampling weights are retained to distinguish sample estimates
from weighted estimates of the represented population.

## Data Access and Licensing

The compact inputs required for offline reproduction are included in the
repository. Access to the full Hugging Face dataset requires authentication and
compliance with its access conditions. The original untruncated evidence and
provider-response archives are maintained separately; access arrangements are
described in the [reproduction guide](REPRODUCING.md#reconstructing-measurements-and-model-labels).

The processing code is distributed under the MIT License. Third-party
source-repository content remains subject to its original terms. Dataset-specific
access and reuse conditions are documented in the dataset repository.

## Validation and Publication Interfaces

The reproduction workflow generates an independent output directory and records
the provenance of its results. The manually authored methodology diagrams can
be included from an authorized manuscript checkout.

The publication tool validates an explicit artifact allowlist. `--validate-only`
and `--check-inventory` perform integrity verification; `--dry-run` constructs
local publication metadata. Publication requires the explicit `--upload` option
and is restricted by the tool to private dataset repositories.
