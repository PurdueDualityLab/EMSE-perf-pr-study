# Model-Based Assessment of Performance Evidence

This module implements the model-based assessments associated with journal RQ4.
It evaluates quantitative-metric detections and classifies reported runtime–
memory relationships. The module name retains the original analysis numbering.

## Reported Assessments

| Assessment | Sample | Decision procedure | Reported outcome |
| --- | ---: | --- | --- |
| Metric-extraction precision | 88 PRs; 44 per author group | Majority judgment per occurrence, followed by aggregation to PR level | 77 true-positive and 11 false-positive reference labels |
| Runtime–memory classification | 29 PRs | Majority judgment on a binary PR-level label | 17 trade-offs and 12 joint improvements |

The models are GPT-5.6-sol, Gemini-3.1-Pro-Preview, and Qwen3.8-27B. Their
classifications provide an assessment of the archived evidence rather than
independent measurements of software performance.

## Reproducing the Reported Counts

The compact inputs and aggregation procedures are provided by the
[assessment module](../audits/README.md):

```bash
python analysis/audits/replay.py --output-dir reproduction/audits
```

This is the recommended entry point for reproducing the results. It operates
offline from recorded annotations and votes. Input checksums are included in
the root [artifact manifest](../../artifact_manifest.json).

## Evidence and Decision Rules

The PR is the reporting unit. Full assessments use descriptions, file changes,
commit messages, issue and review discussions, and workflow/check information.
Input preparation records PR identities, evidence completeness, prompts,
schemas, and cryptographic checksums.

For metric extraction, three providers assess the same detected occurrences.
An occurrence requires two valid judgments, and a PR is positive when at least
one occurrence satisfies that criterion. Independent positive judgments on
different occurrences do not constitute an occurrence-level majority.

For runtime–memory classification, the selected assessment assigns either
`tradeoff` or `joint_improvement`. Separate sensitivity analyses examine the
consequences of allowing indeterminate or unsupported relationships and of
requiring a binary decision when evidence is incomplete.

All selected execution identifiers, sensitivity outcomes, and interpretation
limits are documented in [PROVENANCE.md](PROVENANCE.md). Model agreement should
not be interpreted as independently verified ground truth.

## Reconstructing Model Assessments

Full reconstruction requires authorized copies of the archived evidence and
preparation records, provider credentials, and a compatible Qwen runtime.
Acquiring new PR evidence or generating new model responses creates a new
execution record; it does not reproduce the original responses exactly.

The implementation provides the following interfaces:

| Component | Interface |
| --- | --- |
| Sample selection and evidence preparation | `experiment.py` |
| OpenAI requests and result collection | `run_openai.py` |
| Gemini requests and result collection | `run_gemini.py` |
| Local Qwen inference | `run_qwen.py` |
| Binary PR-level consensus | `build_consensus.py` |
| Occurrence-level metric assessment | Modules identified in [PROVENANCE.md](PROVENANCE.md#execution-identifiers) |
| Evidence packages for detailed review | `export_review.py`, `export_binary_review.py` |

The provider interfaces expose preparation, execution, result collection, and
recovery operations. Their command-line help specifies the required paths and
configuration:

```bash
python -m analysis.rq3_llm_validation.experiment --help
python -m analysis.rq3_llm_validation.run_openai --help
python -m analysis.rq3_llm_validation.run_gemini --help
python -m analysis.rq3_llm_validation.run_qwen --help
python -m analysis.rq3_llm_validation.build_consensus --help
```

OpenAI and Gemini require `OPENAI_API_KEY` and `GEMINI_API_KEY`, respectively.
Qwen execution requirements are described in
[`analysis/QWEN_GAUTSCHI.md`](../QWEN_GAUTSCHI.md). Provider requests, responses,
execution checkpoints, and full evidence packages are maintained outside the
versioned compact artifact.

## Evidence Integrity

Consensus construction requires complete provider results with matching PR
identities and input, prompt, and study-contract hashes. Evidence quotations
are checked against the supplied records. Permitted corrections to quotation
formatting preserve the model's classifications and record the corresponding
source text.

PR-specific automated measurements, baselines, unchanged outcomes, and
regressions can constitute valid quantitative evidence. Configuration values,
generic integration telemetry, and prospective claims do not automatically
establish a measured performance result. These distinctions govern the
assessment independently of whether the change was authored by an agent or
a human.
