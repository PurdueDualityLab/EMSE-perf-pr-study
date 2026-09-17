# RQ1, RQ2, and RQ3 labels

These compact tables expose model labels, derived consensus, and extracted
metric dimensions used in the current study. PR identities are keyed by
`(repo_id, number)`; `sample_arm` is
`agentic` or `human_candidate`. The latter does not establish human authorship.

| File | Rows | Contents |
| --- | ---: | --- |
| `rq1_labels.csv` | 2,260 | GPT, Gemini, and Qwen optimization-pattern labels and atomic-pair consensus |
| `rq2_labels.csv` | 2,258 | Each model's validation presence, primary type, and type set, plus two-stage consensus |
| `rq2_exclusions.csv` | 2 | Evidence-related exclusions from the balanced sample, including paired exclusion |
| `rq3_labels.csv` | 2,081 | Extracted D0-D9 metric dimensions, no-diff sensitivity flags, and joint-analysis inclusion |
| `manifest.json` | | Source hashes, export hashes, columns, model identifiers, and summary counts |

## Reading the labels

Each CSV contains one row per PR. RQ1/RQ2 have separate `gpt_*`, `gemini_*`, and
`qwen_*` columns. Models are GPT-5.6-sol, Gemini-3.1-Pro-Preview, and Qwen3.8 27B.
Consensus is model agreement, not manually adjudicated ground truth.

**RQ1:** `consensus_high_level_pattern` and `consensus_sub_pattern` are voted as
an atomic pair. There are 1,238 unanimous pairs, 845 exact majorities, and 177
unresolved rows. `included_in_analysis` selects the 2,083 resolved rows. A blank
consensus label means unresolved; it must not be interpreted as a negative label.

**RQ2:** `consensus_validation_present` records the first-stage presence vote.
There are 1,839 positive and 419 negative consensuses. Of the positives, 1,819
have a resolved primary type and 20 have an unresolved primary type.
`included_in_stage2_analysis` selects the 1,819 resolved positives. The stricter
secondary multi-label rule resolves 1,707 complete primary-type/type-set tuples
and leaves 132 unresolved; `included_in_multilabel_analysis` identifies that
subset. A positive presence with an unresolved primary type is still positive
validation evidence.
`*_validation_types` cells are JSON arrays; empty or missing type information
must be interpreted together with the stage status and inclusion fields.

These counts refer to each RQ independently. Joint RQ3 analyses have different
denominators after intersecting the RQ1 and RQ2 inclusion criteria.

**RQ3:** labels are deterministic text-extraction outputs, not additional LLM
votes. The export contains 2,081 analytic PRs (1,048 agentic, 1,033 human-candidate).
`pattern`, `sub_pattern`, and validation labels come from the RQ1/RQ2 consensuses.
`in_metric_layer` includes all 1,699 positive validation consensuses (858 agentic,
841 human-candidate); `in_type_layer` includes the 1,581 resolved positive types
(807 agentic, 774 human-candidate). The 118 unresolved positive types remain in
general metric profiles, but not in type-dependent comparisons. PRs outside
these 2,081 analytic identities have not been assigned RQ3 labels in this export.

The boolean dimension columns mean:

| Column | Reported dimension |
| --- | --- |
| `D0` | Unspecified quantified performance gain |
| `D1` | Latency / execution time |
| `D2` | Throughput |
| `D3` | Memory |
| `D4` | CPU / compute work |
| `D5` | I/O and network |
| `D6` | Artifact size |
| `D7` | Build and CI time |
| `D8` | Energy and cost |
| `D9` | Scalability / concurrency |

`n_dims` counts D0-D9; `n_dims_specific` counts D1-D9. `dims` is pipe-separated;
a blank value denotes no detected dimension. `*_nodiff` and `n_dims_nodiff`
exclude code diffs, while `n_dims_description_only` uses only the description.
Flags are extracted for all analytic PRs, including those without positive RQ2
validation. A detected dimension is not an independently verified improvement.

## Provenance and full artifacts

RQ1/RQ2 tables are column-selected exports of the local consensus artifacts produced
by [`build_rq1_consensus.py`](../rq1_optimization_patterns/build_rq1_consensus.py)
and [`build_rq2_consensus.py`](../rq2_validation/build_rq2_consensus.py). The manifest
records the exact input hashes. RQ2 exports retain each provider's model ID;
RQ1 provider identities follow the study's completed three-model workflow.

The RQ3 source CSV was checked against the SHA-256 in the published Hugging Face
bundle at revision `4464afd020ece2a619649a185460fde9d02625fd`. Its remote path,
revision, and source/export hashes are recorded in the manifest. The full RQ3
results and evidence snippets remain in that private bundle.

The complete dataset and full model artifacts remain in the private
[Hugging Face dataset](https://huggingface.co/datasets/rcalvome/EMSE-perf-pr-study).
These GitHub exports contain identities, labels, voting/status fields, and
inclusion decisions. PR bodies, source code, diffs, comments, evidence quotations,
prompts, provider responses, token usage, and operational checkpoints are not
included.

In Python, load CSVs with `pandas.read_csv` and decode type sets with `json.loads`.
Booleans are serialized as `True`/`False`; unresolved scalar labels are blank.
