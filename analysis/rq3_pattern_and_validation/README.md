# Reported Performance Metrics — Journal RQ4

This module characterizes the quantitative performance metrics reported in PR
artifacts and relates them to optimization patterns and validation evidence.
Optimization labels are supplied by journal RQ2, and validation labels by
journal RQ3. Directory and output names retain the original `rq3` prefix.

Stored metric identifiers `D0`–`D9` correspond one-to-one to the manuscript's
`M0`–`M9` identifiers.

## Reproduction

The complete offline workflow is documented in
[REPRODUCING.md](../../REPRODUCING.md). Individual analyses can also be executed
from the repository root:

```bash
python analysis/rq3_pattern_and_validation/rq3_statistics.py --current \
  --data analysis/classification_labels/rq3_labels.csv \
  --metadata analysis/artifact_inputs/rq4_metadata.csv \
  --output-dir reproduction/rq4/metrics

python analysis/rq3_pattern_and_validation/metric_alignment.py \
  --labels analysis/classification_labels/rq3_labels.csv \
  --output-dir reproduction/rq4/correspondence \
  --figure-dir reproduction/rq4/correspondence/figures

python analysis/rq3_pattern_and_validation/code_smells_by_author.py \
  --labels analysis/classification_labels/rq3_labels.csv \
  --output-dir reproduction/rq4/code-smells
```

These commands use recorded extraction results and archived metadata. They do
not retrieve PR content or perform model inference.

## Analytical Populations

| Population | Total | Agentic | Human-authored |
| --- | ---: | ---: | ---: |
| Resolved optimization label and available validation-presence label | 2,081 | 1,048 | 1,033 |
| Reported validation; general metric population | 1,699 | 858 | 841 |
| Resolved positive primary evidence type | 1,684 | 851 | 833 |

The 15 PRs with positive validation presence and unresolved evidence types
remain in general metric profiles. Comparisons conditioned on evidence type
exclude them. An unresolved type is not treated as non-benchmark evidence.
Metric indicators are available for every PR in the 2,081-row joint population,
including PRs without reported validation.

## Metric Extraction

`extract_metrics.py` and `metric_patterns.py` implement deterministic extraction
of nine named metric categories and a residual category for unspecified
quantified performance gains. Each detection associates a metric cue with a
quantitative expression in the PR description, issue/review comments, commit
messages, file changes, or workflow names.

The extractor applies the following rules:

- A metric cue and a compatible quantitative expression must occur within a
  12-token window. Quantitative expressions include numbers with appropriate
  units, rates, comparative statements, and benchmark-table values.
- Each expression is attributed to the nearest compatible cue.
- Cues near prospective or hypothetical language are excluded using an
  eight-token window. Sections describing expected or estimated improvements
  are also excluded.
- Text normalization handles markup, URLs, identifiers, unit synonyms, and
  supported Japanese and Chinese units and comparative expressions.
- Extraction is multi-label. `D0` identifies a quantified gain without a named
  metric; static complexity arguments are represented by the validation taxonomy.

PR-level outputs retain metric indicators, dimension counts, and source
provenance. The compact labels include a sensitivity variant that excludes
code diffs. Metric detection indicates a reported quantity rather than an
independently verified improvement.

## Statistical Analysis

### Metric Profiles and Evidence Types

`rq3_statistics.py` produces category-by-validation tables, metric incidence,
metric-count distributions, evidence-type comparisons, and supporting
descriptive analyses. Categorical comparisons use chi-square tests when expected
cell counts are adequate; otherwise they use Fisher's exact test or a seeded
Monte Carlo conditional test for larger contingency tables. Ordinal comparisons
use Mann–Whitney or Kruskal–Wallis tests with effect sizes.

Benjamini–Hochberg adjustment is applied to the 60-test family. The compact
metadata supplement permits reproduction of the outcome-related comparisons
that participate in that family. Per-agent summaries are descriptive.

### Metric–Optimization Correspondence

`catalog_expected_dims.py` derives the expected-metric mapping from the
optimization catalog. `metric_alignment.py` compares the observed proportion of
PRs reporting an expected metric with a permutation distribution. Expected-metric
sets are reassigned among optimization patterns while each PR's reported metrics
remain fixed. The analysis uses 20,000 permutations and a one-sided hypothesis.

For caching and buffering, the module also describes co-reporting of runtime-side
metrics (`D1`, `D2`, or `D5`) and memory (`D3`). Metric presence alone does not
establish the direction of the performance change. The selected model-based
relationship assessment is documented in [the audit module](../audits/README.md).

### Code-Smell Refactorings

`code_smells_by_author.py` compares the Code Smells and Structural Simplification
category with the remaining eight categories. Outcomes are validation presence,
metric presence among PRs with validation, and their combined occurrence.
Within-group contingency comparisons and logistic-regression interaction tests
evaluate whether these contrasts differ by author type. Benjamini–Hochberg
adjustment is applied separately to the interaction and simple-effect families.

## Figure Generation

```bash
python analysis/rq3_pattern_and_validation/make_figures.py --current \
  --data analysis/classification_labels/rq3_labels.csv \
  --paper-names --output-dir reproduction/figures
```

The paper figures are `metric_count_distribution.pdf`, `metric_frequency.pdf`,
and `metric_profile_by_category.pdf`. Additional descriptive figures and tables
are included in the analysis outputs.

## Reconstructing the Extraction

`run_current.py` reconstructs the extraction from the balanced sample, the
optimization and validation consensus tables, and the archived evidence:

```bash
python analysis/rq3_pattern_and_validation/run_current.py --help
```

The evidence snapshot is dated August 27, 2026. It contains 2,259 complete PR
collections and one unavailable PR among the 2,260 sampled identities. The
classification populations additionally account for the paired exclusion used
to preserve weekly balance. Access to the untruncated snapshot must be arranged
with the maintainers; it is not included in the core dataset bundle. A new
GitHub collection constitutes a different evidence snapshot.

## Outputs and Provenance

| Location | Contents |
| --- | --- |
| `results/rq3_results.md`, `results/rq3_tests.csv` | Metric-profile results and statistical comparisons |
| `results/rq3_step2_results.md`, `results/rq3_step2_tests.csv` | Correspondence and runtime/memory co-reporting |
| `results/rq3_code_smells_results.md`, `results/rq3_code_smells_tests.csv` | Code-smell contrasts and interactions |
| `results/tables/` | Individual result tables |
| `figures/` | Recorded descriptive figures |
| `summary.json` | Provenance of the original extraction |
| `data/` | Archived PR-level extraction and occurrence records, maintained separately |

The compact labels contain the primary-type classification used by the submitted
paper. Their provenance is distinct from the original extraction's metadata.
[PROVENANCE.md](PROVENANCE.md) documents that relationship and the interpretation
of previously generated supporting tables.

## Interpretation

Results are observational PR-level associations. Benchmark classifications and
metric detections share evidence, so their association does not independently
validate the extractor. The positive-only three-model assessment does not
estimate recall. False positives and missed evidence are both possible;
extracted frequencies are not verified performance effects or
guaranteed lower bounds on true reporting rates.
