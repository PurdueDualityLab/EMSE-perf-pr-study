# RQ1 and RQ2 classification labels

These compact tables expose the model labels and derived consensus used in the
current study. PR identities are keyed by `(repo_id, number)`; `sample_arm` is
`agentic` or `human_candidate`. The latter does not establish human authorship.

| File | Rows | Contents |
| --- | ---: | --- |
| `rq1_labels.csv` | 2,260 | GPT, Gemini, and Qwen optimization-pattern labels and atomic-pair consensus |
| `rq2_labels.csv` | 2,258 | Each model's validation presence, primary type, and type set, plus two-stage consensus |
| `rq2_exclusions.csv` | 2 | Evidence-related exclusions from the balanced sample, including paired exclusion |
| `manifest.json` | | Source hashes, export hashes, columns, model identifiers, and summary counts |

## Reading the labels

Each CSV contains one row per PR, with separate `gpt_*`, `gemini_*`, and `qwen_*`
columns. Models are GPT-5.6-sol, Gemini-3.1-Pro-Preview, and Qwen3.8 27B.
Consensus is model agreement, not manually adjudicated ground truth.

**RQ1:** `consensus_high_level_pattern` and `consensus_sub_pattern` are voted as
an atomic pair. There are 1,238 unanimous pairs, 845 exact majorities, and 177
unresolved rows. `included_in_analysis` selects the 2,083 resolved rows. A blank
consensus label means unresolved; it must not be interpreted as a negative label.

**RQ2:** `consensus_validation_present` records the first-stage presence vote.
There are 1,839 positive and 419 negative consensuses. Of the positives, 1,707
have a resolved primary-type/type-set tuple and 132 have unresolved types.
`included_in_stage2_analysis` selects the 1,707 resolved positives. A positive
presence with an unresolved type is still positive validation evidence.
`*_validation_types` cells are JSON arrays; empty or missing type information
must be interpreted together with the stage status and inclusion fields.

These counts refer to each RQ independently. Joint RQ3 analyses have different
denominators after intersecting the RQ1 and RQ2 inclusion criteria.

## Provenance and full artifacts

The tables are column-selected exports of the local consensus artifacts produced
by [`build_rq1_consensus.py`](../rq1_optimization_patterns/build_rq1_consensus.py)
and [`build_rq2_consensus.py`](../rq2_validation/build_rq2_consensus.py). The manifest
records the exact input hashes. RQ2 exports retain each provider's model ID;
RQ1 provider identities follow the study's completed three-model workflow.

The complete dataset and full model artifacts remain in the private
[Hugging Face dataset](https://huggingface.co/datasets/rcalvome/EMSE-perf-pr-study).
These GitHub exports contain identities, labels, voting/status fields, and
inclusion decisions. PR bodies, source code, diffs, comments, evidence quotations,
prompts, provider responses, token usage, and operational checkpoints are not
included.

In Python, load CSVs with `pandas.read_csv` and decode type sets with `json.loads`.
Booleans are serialized as `True`/`False`; unresolved scalar labels are blank.
