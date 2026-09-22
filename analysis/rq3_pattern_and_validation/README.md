# RQ3 — Optimization, Validation, and Reported Metrics

RQ1 characterizes *what* perf PRs change (optimization pattern), RQ2 *whether* the
change is accompanied by validation evidence. RQ3 adds the third element — the
performance metric dimensions reported as that evidence — and relates the three.

> Naming: this directory uses `D0`–`D9` for the metric dimensions, in column
> names and tables alike. The paper renames them `M0`–`M9` and says "metric"
> rather than "dimension"; the codes correspond one-to-one.

## Analytic sample

Run the consensus adapter from the repository root, in an environment installed
from `requirements.txt`:

```bash
python analysis/rq3_pattern_and_validation/run_current.py
```

The runner accepts `--sample`, `--rq1`, `--rq2`, `--evidence-dir`, and
`--output-dir`. Defaults read the official balanced sample under `data/data/sample/`,
the local RQ1/RQ2 `consensus/*.parquet` files, and the original untruncated evidence
snapshot under `mining/sample_evidence/final/`. The latter is an ignored local
artifact, not part of the published dataset. The snapshot used for the existing
RQ1/RQ2 labels was recovered from Gautschi; it is dated 2026-08-27 and records
2,259 complete collections and one unavailable PR out of 2,260 sample identities.
Reuse this snapshot rather than fetching newer versions of the discussions.

The adapter validates immutable `(repo_id, number)` identities, sample arms,
selection hashes, snapshot consistency, and evidence completeness. Equal-distance
quantitative-claim ties use sorted claim kinds instead of set iteration, making
extraction independent of Python's randomized hash seed. Sources use current PR
file patches (not repeated per-commit patches), commit messages, issue/review
comments, description, and workflow names. It does not inject provider-specific
RQ2 pipeline names into the corpus.

Denominators:

| Layer | All | agentic | human_candidate |
| --- | ---: | ---: | ---: |
| RQ1 included × available RQ2 presence consensus | 2,081 | 1,048 | 1,033 |
| Positive validation presence / general metric layer | 1,699 | 858 | 841 |
| Resolved positive type layer | 1,684 | 851 | 833 |
| Measured evidence (benchmark or profiling primary type) | 906 | 413 | 493 |

The 15 positive PRs without resolved validation types are included in the
general metric profile, dimensionality by sample arm/category, merge-status
comparisons, and no-diff sensitivity. They appear as `unresolved` in descriptive
type distributions. Only comparisons involving validation type exclude them,
including benchmark-versus-other tests: an unresolved type is not known to be
non-benchmark. `in_metric_layer` marks all positive presence consensuses, while
`in_type_layer` marks the resolved positive type subset. Dimension flags are
still extracted for every analytic PR.
Thus, neither 2,083 (RQ1 alone) nor 1,707 (RQ2 alone) is the joint RQ3 denominator.

Outputs are written alongside the code: `data/` (ignored by Git), `results/`,
five PDF figures under `figures/`, and `summary.json` with input SHA-256 hashes
and cohort counts. `data/sample_inclusion.csv` records inclusion for all 2,260
sample identities. Compact public labels live in `analysis/classification_labels/`.

## Pipeline (run from the repo root)

`run_current.py` drives Step 1 end to end — extraction, statistics, and figures.
The remaining entry points are run individually:

```bash
python analysis/rq3_pattern_and_validation/run_current.py          # Step 1, full
python analysis/rq3_pattern_and_validation/refresh_type_layer.py   # type layer only
python analysis/rq3_pattern_and_validation/catalog_expected_dims.py
python analysis/rq3_pattern_and_validation/metric_alignment.py     # Step 2
python analysis/rq3_pattern_and_validation/code_smells_by_author.py # post-hoc
```

### Step 1 — `extract_metrics.py` + `metric_patterns.py`

Deterministic regular-expression extraction of the metric dimensions
**D1–D9** (latency/exec time, throughput, memory, CPU work, I/O & network,
artifact size, build/CI time, energy & cost, scalability/concurrency) plus
**D0 unspecified performance** — a quantified gain whose dimension is not named
("3-5x overall performance improvement"), credited only when no D1–D9 cue is in
the claim's window — from each PR's corpus: description, issue and review
comments, commit messages, CI workflow names, and the code diff (each source
tracked separately). `n_dims` counts D0; `n_dims_specific` counts D1–D9 only.

A dimension is *reported* when a dimension cue and a **quantitative claim about
it** co-occur within a bounded window (`WINDOW_TOKENS = 12`):

* unit-carrying numerals appropriate to the dimension — time (`120ms`), bytes
  (`4MB`), rates (`1200 req/s`), energy/currency — or
* comparative statements — `2.5x faster`, `from 125ns to 83ns`, `-40%`,
  `before: 3.2s | after: 1.1s`, benchmark-table cells (`| 44.31 ns |`).

Each claim is credited to its **nearest** cue (`NEAREST_CUE_WINS`), so
"thread pool | 30-50% latency reduction" is D1, not D9. Cues within
`EXCLUSION_TOKENS = 8` of a hypothetical/prospective construction (`should
measure`, `would reduce`, `TODO: benchmark`, `estimated`, …) or inside a block
headed "Expected/Estimated improvements" are discarded. Static properties
(asymptotic complexity) are not dimensions — RQ2 handles them as static
reasoning. Text is normalized first (lower-casing; URL/e-mail/version/SHA/bot
boilerplate stripping; unit synonym folding, including Japanese/Chinese units
such as 秒 and 倍; CJK cues and comparatives such as 実行時間, メモリ, 削減, 耗时
are mapped onto the same dimensions). Extraction is multi-label.

`metric_list_coverage.py` re-runs the extractor under alternative
dimension lists (storage, GPU, errors/timeouts, UI rendering, work volume, LLM
tokens, cache effectiveness, bundle-size tables) and reports each list's
coverage → `results/metric_list_coverage.md`. On this corpus the extra
resource dimensions fire in 0–3 PRs each; D0 and the CJK handling were the only
additions with a measurable effect and were adopted.

`data/rq3_metric_matches.csv` holds one audit row per (PR, dimension,
source) with the cue, the claim and a snippet — use it to spot-check precision.
`*_nodiff` columns in `data/rq3_pr_level.csv` give the same flags with
the code diff excluded (sensitivity analysis, §5 of the results).

### Step 1 statistics — `rq3_statistics.py`

1. **Category × validation** (n = 2,081): presence and type, pooled and by author;
   agent-vs-human validation rate within each category.
2. **Metric profile** (n = 1,699): category × D0–D9 incidence by author; the
   *dimensionality* (distinct dimensions per PR) across authors, categories, and
   validation types; benchmark-based vs other evidence (n = 1,684, resolved types
   only). Table 2.5 breaks metric reporting down by author type and by individual
   agent under three denominators → `T2_5_metric_reporting_by_agent.csv`.
3. **Merge status** vs dimensionality.
4. **Extremes** exported as CSVs for qualitative reading, including PRs with a
   quantitative metric claim but no RQ2 validation label (RQ2 recheck candidates).
5. **Sensitivity** with the code diff excluded from the corpus.

Tests: chi-square when Cochran's rule holds, otherwise Fisher's exact (2×2) or a
Monte-Carlo Fisher–Freeman–Halton test (fixed margins, B = 20,000, seeded);
Cramér's V (OR for 2×2); Mann–Whitney U with Cliff's δ; Kruskal–Wallis with ε².
Benjamini–Hochberg is applied across the whole 60-test RQ3 Step 1 family
(`results/rq3_tests.csv`); Step 2 is a separate, smaller family
(`results/rq3_step2_tests.csv`). The conditional Monte Carlo test uses
Pearson's statistic with fixed margins. BH adjustment uses SciPy's implementation; `statsmodels` is used only by the
post-hoc Code-smells script for its regression-based interaction tests.

### Type-layer refresh — `refresh_type_layer.py`

RQ2 was rerun on the primary evidence type after the Step 1 run was committed.
Validation *presence* is identical under both consensuses, so the D0–D9
extraction and every presence-based result stand unchanged; only
`validation_type` / `in_type_layer` and the statistics conditioning on them
moved (resolved type layer 1,581 → 1,684; unresolved 118 → 15). The refresh is
purely additive — no already-resolved type changed, and the script aborts if
that is not the case.

The script applies `run_current.build_base`'s type rules to the already-published
extraction output and re-drives `rq3_statistics`, so its numbers come from the
same statistical code as the original run. **Its output has been folded into
`results/`**, which is now the single authoritative Step 1 result set.
Re-running the script writes a fresh staging directory
(`--output-dir`, default `results_type_refresh/`) for comparison; fold any
accepted changes back into `results/` rather than keeping both.

Conclusions were unaffected: all 60 tests match one-to-one, 25 significant after
BH before and after, no test gained or lost, largest adjusted-p shift ≈ 0.03.
Sections 2.5 (per-agent) and 3 (merge) are not regenerated by the refresh —
they need the `agent` and `is_merged` columns, which are absent from the
published compact labels. Section 3 does not condition on validation type, so
its committed values remain correct; section 2.5's `benchmark` column does, and
is flagged inline in `results/rq3_results.md` as reflecting the
pre-refresh layer.

### Step 2 — `catalog_expected_dims.py` + `metric_alignment.py`

Run from the refreshed compact labels in `analysis/classification_labels/`:

1. **Alignment** of reported dimensions with the pattern's expected dimensions,
   taken mechanically from the catalog (`catalog_expected_dims.py` →
   `catalog_expected_dims.csv`) and tested against a null that permutes expected
   sets across sub-patterns, each PR keeping its reported dimensions.
2. **The memory-for-time check** within Caching and Buffering: how often the gain
   is reported, how often memory, how often both. Descriptive, with no
   cross-pattern comparison, since other patterns are not expected to move memory.

An earlier three-class trade-off classification and composite adequacy outcome
were removed as unvalidatable.

### Post-hoc — `code_smells_by_author.py`

Contrasts the *Code Smells and Structural Simplification* category (structural
cleanups) with the eight remaining categories (actual performance changes) on
validation presence (category layer, n = 2,081), any dimension reported and
number of dimensions (metric layer, n = 1,699), and the combined outcome
*validated and ≥1 dimension* on the category layer — each within the agentic
and human arms, agent vs human within each group, and a category × author
interaction test (logistic-regression LRT for binary outcomes; permutation on
the difference of Cliff's δ with a negative-binomial LRT cross-check for
#dims). A final section tests each D0–D9 unconditionally and conditional on
≥1 dimension, to separate *fewer* metrics from *different* metrics.

Finding: the Code-smells gap exists only in the human arm. Humans validate and
quantify actual performance changes at a high rate and structural cleanups at
a markedly lower one (validated-and-quantified 25% vs 46%), while agents apply
the same verification behaviour regardless of category (38% vs 42%); the
interaction is significant (ratio of ORs 2.08 [1.22, 3.56], p = 0.007,
q = 0.028); for validation presence and any dimension it is significant before
correction and borderline after (p = 0.043 and 0.034; q = 0.11 and 0.10). Code
smells PRs that do quantify report the same metric mix as other PRs. Reads the
compact labels only. BH is applied across the 20 pre-specified tests (within-arm,
within-group, interaction), a family separate from Step 1 and Step 2; the
per-dimension, pairwise, and negative-binomial follow-ups are descriptive and
carry raw p only (`results/rq3_code_smells_tests.csv`, column `prespecified`).

## Outputs

| path | content |
| --- | --- |
| `data/rq3_pr_level.csv` | per-PR: RQ1 pattern, RQ2 labels, outcome metadata, D0–D9 flags, `n_dims`, `n_dims_specific`, sources (Git-ignored) |
| `data/rq3_metric_matches.csv` | audit snippets per (PR, dimension, source) (Git-ignored) |
| `data/sample_inclusion.csv` | inclusion decision for all 2,260 sample identities (Git-ignored) |
| `results/rq3_results.md` | every Step 1 table and test |
| `results/rq3_tests.csv` | Step 1 test family (60 tests) with BH-adjusted p |
| `results/tables/T0–T5*.csv` | the Step 1 tables individually |
| `results/rq3_step2_results.md` | Step 2 narrative: alignment and memory-for-time |
| `results/rq3_step2_tests.csv` | Step 2 test family with BH-adjusted p |
| `results/rq3_code_smells_results.md` | Post-hoc: structural cleanups vs performance changes, by author |
| `results/rq3_code_smells_tests.csv` | Post-hoc test family with BH-adjusted p |
| `results/tables/T7_*.csv` | the post-hoc tables individually |
| `results/tables/T6_*.csv` | the Step 2 tables individually |
| `results/extremes_*.csv` | PR lists at the distribution extremes |
| `summary.json` | cohort counts and input SHA-256 hashes of the extraction run |
| `figures/rq3_dimensionality.pdf` | dimensions per validated PR, by author and by evidence type **(in paper)** |
| `figures/rq3_metric_frequency.pdf` | % of validated PRs reporting each dimension D0–D9, agent vs human **(in paper)** |
| `figures/rq3_metric_profile_heatmap.pdf` | category × D0–D9 incidence (plus a no-metric column) among validated PRs **(in paper)** |
| `figures/rq3_validation_by_category.pdf` | validation type share per category, agent vs human |
| `figures/rq3_memory_for_time.pdf` | gain vs memory reporting within caching and buffering |

`data/` is excluded from Git; the PR-level intermediates remain available in the
private Hugging Face dataset, and the compact public labels in
`analysis/classification_labels/`. `summary.json` still records
`type_layer_rows: 1581` and the pre-refresh RQ2 consensus hash `a29be9ab…`: it
documents the provenance of the extraction run that produced it, not the current
layer sizes. The authoritative denominators are the table at the top of this file.

To refresh paper figures without rerunning extraction:

```bash
python analysis/rq3_pattern_and_validation/make_figures.py --current \
  --data analysis/classification_labels/rq3_labels.csv \
  --output-dir report/figures
python analysis/make_methodology_figure.py
python analysis/make_paper_comparison_figures.py
```

The paper layout stacks category/type and heatmap panels for readability and
hatches unresolved validation types. Figure captions define the separate
positive-presence and resolved-type denominators. The comparison-figure script
reads published compact RQ1/RQ2 labels and the local complete-case structural
deltas, rendering vector PDFs without changing the underlying statistics.

## Interpretation

Interpret results as exploratory PR-level associations. In particular,
benchmark labels and quantitative metric detection share evidence, so their
association does not independently validate the extractor. Extraction rules
have not been evaluated against a new manually annotated precision/recall set,
and the regular expressions capture explicit surface mentions only, so coverage
rates are lower bounds.
