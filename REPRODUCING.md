# Reproduction Guide

The offline reproduction workflow uses fixed PR-level labels, source-code
measurements, timestamps, and compact audit votes. It recomputes statistics and
renders the nine quantitative figures used by the manuscript. No provider API,
GitHub mining, Hugging Face download, or GPU is needed for this path.

## 1. Environment

The reference environment uses Python **3.12.11**. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r analysis/quantitative_analysis/requirements.txt
```

That requirements file includes the root dependencies, Lizard 1.24.0, and the
complete reference dependency constraints in `analysis/constraints-python312.txt`.
The reference environment was verified on Linux x86_64 with CPython 3.12.11.
Qwen inference has a separate GPU environment in `analysis/requirements-qwen.txt`.
Exporting the original annotation workbook additionally requires
`pip install -r analysis/requirements-export.txt`; it is not needed for replay.

## 2. Input Verification

```bash
python analysis/reproduce_paper.py --check-only
python mining/tools/publish_artifacts.py --check-inventory
```

These commands verify the research inputs without generating analysis outputs.
`artifact_manifest.json` records input hashes, row counts, schemas, and source
digests. PR identities and population intersections are also checked.

## 3. Statistical Analysis and Figure Generation

```bash
python analysis/reproduce_paper.py --output-dir reproduction
```

The output directory contains:

| Directory | Contents |
| --- | --- |
| `rq1/` | Adoption/patch summaries, seven-test BH family, structural measures and three-test BH family |
| `rq2/patterns/` | Category distributions, association/effect sizes, coverage diagnostics |
| `rq2/structure/` | NLOC/function-count contrasts within categories, with BH correction |
| `rq3/evidence/` | Validation presence and primary-type comparisons |
| `rq3/temporal/` | Quarterly estimates, Wilson intervals, temporal regressions, and diagnostic analyses |
| `rq4/metrics/` | Metric profiles and the complete 60-test family, using compact archived outcome metadata |
| `rq4/correspondence/` | Pattern–metric permutation results and runtime/memory co-reporting |
| `rq4/code-smells/` | Code-smell comparisons and category-by-author interactions |
| `audits/` | Replayed regex precision, selected trade-off votes, and manual-classifier audit |
| `figures/` | Nine quantitative PDFs with the names included by the manuscript |
| `tables/` | Five supporting LaTeX result tables; the manuscript may present their values in prose |
| `logs/` | Per-stage commands' output and diagnostics |

`reproduction_manifest.json` records population controls, package versions,
commands, input-manifest hash, and output hashes. PDF metadata is timestamped
deterministically. Numeric results can differ in final floating-point digits
across statistical-library versions. Reference-table comparisons use relative
tolerance `1e-6` and absolute tolerance `1e-7`.

The generated paper figure names are:

```text
agent_sample_distribution.pdf
merge_rate_and_time.pdf
structural_change_distributions.pdf
optimization_pattern_distribution.pdf
validation_evidence_types.pdf
validation_over_time.pdf
metric_count_distribution.pdf
metric_frequency.pdf
metric_profile_by_category.pdf
```

The two methodology diagrams (`perf_pr_flow.pdf`, `Methodology.pdf`) are manually
authored, not statistical outputs. If an authorized report checkout is present,
include exact copies with:

```bash
python analysis/reproduce_paper.py --output-dir reproduction --paper-dir report
```

The runner writes to its output directory rather than replacing the submitted
paper's figures. Historical script/output prefixes remain for compatibility;
the wrapper maps them to final manuscript names.

## Reference Populations

| Control | Expected value |
| --- | ---: |
| Balanced sample | 2,260; 1,130 per arm |
| RQ2 resolved category–pattern pairs | 2,083; 1,049 agentic, 1,034 human-authored |
| RQ3 available evidence | 2,258 |
| RQ3 positive validation presence | 1,839 |
| RQ3 resolved positive primary types | 1,819 |
| RQ4 joint analytic population | 2,081 |
| RQ4 positive metric population | 1,699 |
| RQ4 resolved primary types | 1,684 |
| RQ4 at least one detected metric | 878 |
| Regex-audit reference labels | 77 positive, 11 negative, out of 88 |
| Selected runtime–memory assessment | 17 trade-offs, 12 joint improvements, out of 29 |

The execution catalog contains **58 patterns**. Its correspondence with the
count reported in the manuscript is documented in the
[catalog provenance](analysis/rq1_optimization_patterns/catalog/README.md).
The last two rows aggregate model judgments rather than independent ground
truth. The assessment procedures are described in
[analysis/audits/README.md](analysis/audits/README.md), with execution identifiers
and sensitivity results in [Assessment Provenance](analysis/rq3_llm_validation/PROVENANCE.md).

## Reconstructing Measurements and Model Labels

Full reconstruction is documented in [mining/README.md](mining/README.md),
[analysis/maintainability/README.md](analysis/maintainability/README.md), and the
classification-module documentation. It requires authorized full-data access,
GitHub credentials for collection, and provider access or a compatible Qwen
installation for inference.

The original evidence snapshot was collected on 2026-08-27 under
`mining/sample_evidence/final/`. It contains mutable PR descriptions, comments,
diffs, and check/workflow evidence; it is not part of the core published dataset.
Access to that archived snapshot must be arranged with the study maintainers.
Fetching GitHub again creates a new snapshot and is not an exact replay of the
published labels. See the SHA-256 provenance recorded with each analysis.

The submodule is an authenticated Hugging Face repository. Sign in and satisfy
its access conditions before using `hf download` or materializing its files.
The offline commands above do not require either private submodule.

## Input Export and Manifest Maintenance

`analysis/build_classification_labels.py` exports compact model/metric tables
from the archived sources. `analysis/export_artifact_inputs.py` exports the
additional metadata and selected assessment votes. Adopting new input snapshots
requires reviewing their provenance and updating the associated manifests:

```bash
python analysis/validate_artifact.py --write-manifest
python mining/tools/publish_artifacts.py --write-inventory
```

For full-data verification without construction or uploads:

```bash
python mining/tools/publish_artifacts.py --validate-only --dataset-dir data
```

`publish_artifacts.py --dry-run` builds local publication metadata and requires
the original full source files. Actual uploads remain private-only and require
the explicit `--upload` option.

## Software Verification

```bash
python -m pytest mining/tests -q
```

The verification suite checks data contracts, consensus procedures, sampling
logic, and artifact integrity. The reproduction workflow additionally checks
the generated result tables against recorded reference outputs. Provider
interfaces are assessed using fixtures rather than live inference requests.

## Manuscript Compilation

An authorized `report/` checkout can be built independently of the analyses.
The root `Dockerfile.paper`, distributed with the artifact, pins the
multi-platform TeX Live image at
`texlive/texlive@sha256:16c556aeb4095b47245fcd05db47061529e5df8fc8c04d97fb0f049055244526`
instead of following the mutable `latest` tag. Paper compilation is optional;
the offline numerical reproduction does not require Docker or LaTeX.

```bash
docker build -f Dockerfile.paper -t emse-paper report
docker run --rm --network none --user "$(id -u):$(id -g)" \
  -v "$PWD/report:/workspace" emse-paper
```

Container construction requires network access to obtain the image and build
utilities. The build specification is independent of the private manuscript
submodule's original Dockerfile.
