# Assessment of Classification and Metric Extraction

This module reproduces the assessment results associated with the performance
classifier, quantitative-metric extractor, and runtime–memory classifications.
It uses the annotation records and model votes in `analysis/artifact_inputs/`.

## Reproduction

From the repository root:

```bash
python analysis/audits/replay.py --output-dir reproduction/audits
```

The output, `audit_summary.json`, reports the reconstructed counts and estimates.
Input identities, schemas, and source checksums are recorded in
[`analysis/artifact_inputs/manifest.json`](../artifact_inputs/manifest.json).
The computation aggregates existing annotations and does not require model
inference. The compact inputs exclude PR text, model rationales, and evidence
quotations.

## Metric-Extraction Assessment

`regex_occurrence_votes.csv` contains three Boolean judgments for each of 1,910
occurrence groups across 88 PRs, with 44 PRs per author group. An occurrence is
accepted when at least two models judge it valid. A PR is classified as a true
positive relative to this reference when at least one occurrence is accepted.

The selected assessment yields **77 true-positive and 11 false-positive
reference labels**, corresponding to **87.5% PR-level precision**. The reference
is three-model consensus; it is not independently adjudicated human ground
truth. This positive-only sample does not estimate recall or the precision of
each individual metric label.

## Runtime–Memory Classification Assessment

`tradeoff_votes.csv` records the three-model binary classifications for 29 PRs.
Majority aggregation reproduces **17 trade-offs and 12 joint improvements**, as
reported in the submitted manuscript. By author group, the counts are 10 and 7
for agentic PRs, and 7 and 5 for human-authored PRs.

The labels characterize reported evidence rather than independently measured
performance effects. Additional assessments produced different classifications
under alternative evidence and decision requirements. Their outcomes and the
identifiers of the selected execution records are documented in
[Assessment Provenance](../rq3_llm_validation/PROVENANCE.md).

Reproduction establishes the correspondence between the stored votes and the
reported counts. Co-reporting a runtime metric and memory does not by itself
establish an improvement, a regression, or a runtime–memory trade-off.

## Manual Performance-Classifier Assessment

`manual_classifier_audit.csv` contains 100 human annotations, model predictions,
PR identities, selection hashes, and sampling information. The sample uses
disproportionate stratification: 30 predicted positives from the final study
sample, 30 predicted positives outside it, and 40 predicted negatives. Selection
is balanced by author group, and predicted negatives are further stratified by
task label.

The unweighted sample confusion matrix contains 50 true positives, 10 false
positives, 9 false negatives, and 31 true negatives. These counts reproduce the
reported precision of **83.3%**, recall of **84.7%**, and F1 score of **84.0%**.

The output additionally provides inverse-probability-weighted estimates for the
auditable, model-classified population represented by the sampling strata.
These estimates have a different target from the unweighted sample statistics
and from the complete title-prefix-plus-model classification pipeline. The
limited negative sample introduces uncertainty in recall estimation; the
artifact does not provide design-based confidence intervals.

The selection procedure and blinded annotation-workbook generator are provided
in [`build_luna_manual_audit.py`](../../mining/manual_review/build_luna_manual_audit.py).

## Exporting Archived Inputs

The original annotation workbook and model-evidence records are maintained
separately from the compact reproduction inputs. Authorized copies can be
exported with:

```bash
pip install -r analysis/requirements-export.txt
python analysis/export_artifact_inputs.py \
  --manual-workbook /path/to/completed-workbook.xlsx \
  --metric-source /path/to/archived-pr-metrics.csv \
  --agent-sample /path/to/agentic_sample.parquet \
  --regex-dir /path/to/metric-assessment-records \
  --tradeoff-dir /path/to/tradeoff-assessment-records \
  --output-dir reproduction/exported-inputs
```

The expected archive layouts are specified in
[Assessment Provenance](../rq3_llm_validation/PROVENANCE.md). Changes to the
selection or annotation protocol constitute a new assessment and should be
recorded with separate provenance.
