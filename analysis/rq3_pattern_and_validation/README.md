# RQ3 — Optimization, Validation, and Reported Metrics

RQ1 characterizes *what* perf PRs change (optimization pattern), RQ2 *whether* the
change is accompanied by validation evidence. RQ3 adds the third element — the
performance metric dimensions reported as that evidence — and relates the three.

## Analytic sample

### Current balanced sample

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
selection hashes, snapshot consistency, and evidence completeness. It preserves
the existing D0–D9 extraction rules and the historical exploratory test family.
Equal-distance quantitative-claim ties now use sorted claim kinds instead of
set iteration, making extraction independent of Python's randomized hash seed.
Sources use current PR file patches (not repeated per-commit patches), commit
messages, issue/review comments, description, and workflow names. It does not
inject provider-specific RQ2 pipeline names into the corpus. These source
differences should be considered when comparing against the historical results.

Current denominators:

| Layer | All | agentic | human_candidate |
| --- | ---: | ---: | ---: |
| RQ1 included × available RQ2 presence consensus | 2,081 | 1,048 | 1,033 |
| Positive validation presence / general metric layer | 1,699 | 858 | 841 |
| Resolved positive type layer | 1,581 | 807 | 774 |

The 118 positive PRs without resolved validation types are included in the
general metric profile, dimensionality by sample arm/category, merge-status
comparisons, and no-diff sensitivity. They appear as `unresolved` in descriptive
type distributions. Only comparisons involving validation type exclude them,
including benchmark-versus-other tests: an unresolved type is not known to be
non-benchmark. `in_metric_layer` marks all positive presence consensuses, while
`in_type_layer` marks the resolved positive type subset. Dimension flags are
still extracted for every analytic PR.
Thus, neither 2,083 (RQ1 alone) nor 1,707 (RQ2 alone) is the joint RQ3 denominator.

Outputs are written to ignored `current/`: `data/`, `results/`, four PDF figures
under `figures/`, and `summary.json` with input SHA-256 hashes and cohort counts.
`data/sample_inclusion.csv` records inclusion for all 2,260 sample identities.
The existing committed historical outputs remain the reference for the old run.

To refresh paper figures without rerunning extraction:

```bash
python analysis/rq3_pattern_and_validation/make_figures.py --current \
  --data analysis/rq3_pattern_and_validation/current/data/rq3_pr_level.csv \
  --output-dir report/figures
python analysis/make_methodology_figure.py
python analysis/make_paper_comparison_figures.py
```

The paper layout stacks category/type and heatmap panels for readability and
hatches unresolved validation types. Figure captions define the separate
positive-presence and resolved-type denominators.
The comparison-figure script reads published compact RQ1/RQ2 labels and the
local complete-case structural deltas, rendering vector PDFs without changing
the underlying statistics.

Interpret results as exploratory PR-level associations. In particular,
benchmark labels and quantitative metric detection share evidence, so their
association does not independently validate the extractor. Extraction rules
have not been evaluated against a new manually annotated precision/recall set.
The conditional Monte Carlo test uses Pearson's statistic with fixed margins;
its label now describes that implementation accurately. BH adjustment uses
SciPy's equivalent implementation, avoiding an undeclared statsmodels dependency.

### Historical sample

| layer | PRs | agent | human | source |
| --- | --- | --- | --- | --- |
| category × validation | 357 | 280 | 77 | 407 valid perf PRs − 50 `No Meaningful Change or Not Performance PR` (RQ1) |
| metric layer | 177 | 128 | 49 | the 357 with `validation_present == True` (RQ2) |

PRs without validation evidence form the "no metric reported" stratum.

## Pipeline (run from the repo root)

```bash
python analysis/rq3_pattern_and_validation/extract_metrics.py    # Requires historical loader inputs
python analysis/rq3_pattern_and_validation/rq3_statistics.py
python analysis/rq3_pattern_and_validation/make_figures.py
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

`experiments/metric_list_coverage.py` re-runs the extractor under alternative
dimension lists (storage, GPU, errors/timeouts, UI rendering, work volume, LLM
tokens, cache effectiveness, bundle-size tables) and reports each list's
coverage → `results/metric_list_coverage.md`. On this corpus the extra
resource dimensions fire in 0–3 PRs each; D0 and the CJK handling were the only
additions with a measurable effect and were adopted.

`data/rq3_metric_matches.csv` holds one audit row per (PR, dimension, source)
with the cue, the claim and a snippet — use it to spot-check precision.
`*_nodiff` columns in `data/rq3_pr_level.csv` give the same flags with the code
diff excluded (sensitivity analysis, §5 of the results).

### Step 2 — `rq3_statistics.py`

1. **Category × validation** (n = 357): presence and type, pooled and by author;
   agent-vs-human validation rate within each category.
2. **Metric profile** (n = 177): category × D0–D9 incidence by author; the
   *dimensionality* (distinct dimensions per PR) across authors, categories, and
   validation types; benchmark-based vs other evidence. Table 2.5 breaks metric
   reporting down by author type and by individual agent under three
   denominators (all PRs, validated PRs, benchmark PRs) → `T2_5_metric_reporting_by_agent.csv`.
3. **Merge status** vs dimensionality.
4. **Extremes** exported as CSVs for qualitative reading, including PRs with a
   quantitative metric claim but no RQ2 validation label (RQ2 recheck candidates).

Tests: chi-square when Cochran's rule holds, otherwise Fisher's exact (2×2) or a
Monte-Carlo Fisher–Freeman–Halton test (fixed margins, B = 20,000, seeded);
Cramér's V (OR for 2×2); Mann–Whitney U with Cliff's δ; Kruskal–Wallis with ε².
Benjamini–Hochberg is applied across the whole RQ3 family
(`results/rq3_tests.csv`). Categories with n < 10 on the 357 PRs are pooled as
"Other" for inferential tests only.

## Outputs

| path | content |
| --- | --- |
| `data/rq3_pr_level.csv` | 357 PRs: RQ1 pattern, RQ2 labels, outcome metadata, D0–D9 flags, `n_dims`, `n_dims_specific`, sources |
| `data/rq3_metric_matches.csv` | audit snippets per (PR, dimension, source) |
| `results/rq3_results.md` | every table and test |
| `results/tables/*.csv` | the tables individually |
| `results/rq3_tests.csv` | RQ3 test family with BH-adjusted p |
| `STATISTICS.md` | statistical analysis summary: methods, which tests are significant after BH and which are not, power caveats |
| `results/extremes_*.csv` | PR lists at the distribution extremes |
| `figures/rq3_validation_by_category.pdf` | validation type share per category, agent vs human |
| `figures/rq3_metric_frequency.pdf` | % of validated PRs reporting each dimension D0–D9, agent vs human |
| `figures/rq3_metric_profile_heatmap.pdf` | category × D0–D9 incidence (plus a no-metric column) among validated PRs |
| `figures/rq3_dimensionality.pdf` | dimensions per validated PR, by author and by evidence type |
