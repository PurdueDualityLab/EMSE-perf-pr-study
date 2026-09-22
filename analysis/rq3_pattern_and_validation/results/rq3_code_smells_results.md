# RQ3 post-hoc — structural cleanups vs actual performance changes, by author
Source: `analysis/classification_labels/rq3_labels.csv`. *Code smells* = "Code Smells and Structural Simplification" (structural cleanups; n = 290 on the category layer, 226 validated); *Other* = the eight remaining RQ1 categories (actual performance changes; n = 1791, 1473 validated). Validation presence and the combined outcome use the category layer (n = 2081); any dimension and #dims use the metric layer (validated PRs, n = 1699). Tests: chi-square / Fisher's exact with OR (Haldane–Anscombe) for 2×2; Mann–Whitney U with Cliff's δ for counts; interaction by logistic-regression LRT (binary) and a permutation test on the difference of Cliff's δ (#dims, arm shuffled within category group, B = 20,000, seed 20250911) with a negative-binomial LRT as cross-check. Benjamini–Hochberg across the 20 pre-specified tests (§2–§4); the per-dimension (§5), pairwise (§6) and negative-binomial tests are descriptive follow-ups reported with raw p only. All tests: `rq3_code_smells_tests.csv`.

## 1. Descriptives
`quantified` = validated **and** ≥1 dimension reported, as a share of the category layer.
| arm | group | n_category_layer | validated_n | validated_pct | n_metric_layer | any_dim_n | any_dim_pct | mean_dims | median_dims | q1_dims | q3_dims | quantified_n | quantified_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Agent | Code smells | 135 | 112 | 83.0 | 112 | 51 | 45.5 | 0.759 | 0.0 | 0.0 | 1.0 | 51 | 37.8 |
| Agent | Other | 913 | 746 | 81.7 | 746 | 386 | 51.7 | 0.94 | 1.0 | 0.0 | 1.0 | 386 | 42.3 |
| Human | Code smells | 155 | 114 | 73.5 | 114 | 39 | 34.2 | 0.535 | 0.0 | 0.0 | 1.0 | 39 | 25.2 |
| Human | Other | 878 | 727 | 82.8 | 727 | 402 | 55.3 | 0.878 | 1.0 | 0.0 | 1.0 | 402 | 45.8 |

## 2. Code smells vs Other, within each arm
OR < 1 and Cliff's δ < 0: Code smells PRs have the outcome less often / report fewer dimensions.
| arm | outcome | code_smells | other | effect | test | p_raw | p_bh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Agent | validation present | 83.0% | 81.7% | OR 1.09 [0.68, 1.76] | chi-square | 0.724 | 0.762 |
| Agent | any dimension (validated) | 45.5% | 51.7% | OR 0.78 [0.52, 1.16] | chi-square | 0.220 | 0.276 |
| Agent | #dims (validated) | mean 0.76, md 0 | mean 0.94, md 1 | Cliff's δ -0.073; Δmean -0.18 | Mann-Whitney U | 0.178 | 0.238 |
| Agent | validated and ≥1 dim (category layer) | 37.8% | 42.3% | OR 0.83 [0.57, 1.20] | chi-square | 0.322 | 0.379 |
| Human | validation present | 73.5% | 82.8% | OR 0.58 [0.39, 0.86] | chi-square | 0.006 | 0.028 |
| Human | any dimension (validated) | 34.2% | 55.3% | OR 0.42 [0.28, 0.64] | chi-square | <0.001 | <0.001 |
| Human | #dims (validated) | mean 0.54, md 0 | mean 0.88, md 1 | Cliff's δ -0.211; Δmean -0.34 | Mann-Whitney U | <0.001 | <0.001 |
| Human | validated and ≥1 dim (category layer) | 25.2% | 45.8% | OR 0.40 [0.27, 0.59] | chi-square | <0.001 | <0.001 |

## 3. Agent vs Human, within each category group
OR > 1 and Cliff's δ > 0: agents have the outcome more often / report more dimensions than humans.
| group | outcome | agent | human | effect | test | p_raw | p_bh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Code smells | validation present | 83.0% | 73.5% | OR 1.75 [0.99, 3.11] | chi-square | 0.054 | 0.120 |
| Code smells | any dimension (validated) | 45.5% | 34.2% | OR 1.61 [0.94, 2.75] | chi-square | 0.082 | 0.137 |
| Code smells | #dims (validated) | mean 0.76, md 0 | mean 0.54, md 0 | Cliff's δ +0.122; Δmean +0.22 | Mann-Whitney U | 0.071 | 0.129 |
| Code smells | validated and ≥1 dim (category layer) | 37.8% | 25.2% | OR 1.81 [1.09, 2.99] | chi-square | 0.021 | 0.068 |
| Other | validation present | 81.7% | 82.8% | OR 0.93 [0.73, 1.18] | chi-square | 0.545 | 0.606 |
| Other | any dimension (validated) | 51.7% | 55.3% | OR 0.87 [0.71, 1.06] | chi-square | 0.172 | 0.238 |
| Other | #dims (validated) | mean 0.94, md 1 | mean 0.88, md 1 | Cliff's δ -0.008; Δmean +0.06 | Mann-Whitney U | 0.763 | 0.763 |
| Other | validated and ≥1 dim (category layer) | 42.3% | 45.8% | OR 0.87 [0.72, 1.05] | chi-square | 0.135 | 0.208 |

## 4. Interaction: does the Code-smells gap differ between agents and humans?
Ratio of ORs = OR(Agent) / OR(Human); a ratio > 1 means the Code-smells penalty is smaller for agents.
| outcome | OR_agent | OR_human | ratio_of_ORs | ratio_lo | ratio_hi | wald_z | p_wald | LRT_chi2 | p_LRT | p_bh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation present | 1.09 | 0.58 | 1.89 | 1.01 | 3.52 | 2.0 | 0.045 | 4.09 | 0.043 | 0.108 |
| any dimension (validated) | 0.78 | 0.42 | 1.85 | 1.04 | 3.29 | 2.11 | 0.035 | 4.47 | 0.034 | 0.098 |
| validated and ≥1 dim (category layer) | 0.83 | 0.4 | 2.08 | 1.22 | 3.56 | 2.68 | 0.007 | 7.26 | 0.007 | 0.028 |

**#dims (validated PRs, n = 1699)** — permutation test BH q = 0.125; the NB cross-check is descriptive (raw p only).
| quantity | value |
| --- | --- |
| Cliff's δ (Code smells vs Other), Agent | -0.073 |
| Cliff's δ (Code smells vs Other), Human | -0.211 |
| δ(Agent) − δ(Human) [bootstrap 95% CI] | +0.138 [-0.004, +0.282] |
| permutation p (arm shuffled within group, B=20,000) | 0.063 |
| mean #dims difference-in-differences (Agent gap − Human gap) | +0.162 |
| NB regression: RR(Code smells vs Other) Agent / Human; ratio [95% CI] | 0.81 / 0.61; 1.32 [0.89, 1.96] |
| NB regression: LRT χ²(1) on cs:agent, p | 1.98, 0.159 |

## 5. Fewer metrics or different metrics? (pooled metric layer; descriptive, raw p)
Unconditional incidence among validated PRs, then the *mix* among PRs that report at least one dimension.
| dimension | denominator | code_smells | other | OR | test | p_raw | p_bh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D1 latency/exec time | validated PRs | 31.9% (72/226) | 37.7% (556/1473) | 0.77 [0.57, 1.04] | chi-square | 0.088 | n/a |
| D2 throughput | validated PRs | 1.8% (4/226) | 3.9% (58/1473) | 0.44 [0.16, 1.22] | chi-square | 0.106 | n/a |
| D3 memory | validated PRs | 11.9% (27/226) | 18.9% (278/1473) | 0.58 [0.38, 0.89] | chi-square | 0.012 | n/a |
| D4 CPU work | validated PRs | 1.8% (4/226) | 3.2% (47/1473) | 0.55 [0.20, 1.53] | chi-square | 0.244 | n/a |
| D5 I/O & network | validated PRs | 1.3% (3/226) | 4.9% (72/1473) | 0.26 [0.08, 0.84] | chi-square | 0.015 | n/a |
| D6 artifact size | validated PRs | 8.0% (18/226) | 9.2% (135/1473) | 0.86 [0.51, 1.43] | chi-square | 0.557 | n/a |
| D7 build/CI time | validated PRs | 0.0% (0/226) | 1.9% (28/1473) | 0.11 [0.01, 1.84] | Fisher's exact | 0.043 | n/a |
| D8 energy & cost | validated PRs | 0.4% (1/226) | 0.5% (7/1473) | 0.93 [0.11, 7.60] | Fisher's exact | 1.000 | n/a |
| D9 scalability/concurrency | validated PRs | 2.7% (6/226) | 5.6% (83/1473) | 0.46 [0.20, 1.06] | chi-square | 0.061 | n/a |
| D0 unspecified perf | validated PRs | 4.9% (11/226) | 5.1% (75/1473) | 0.95 [0.50, 1.82] | chi-square | 0.886 | n/a |
| D1 latency/exec time | PRs reporting ≥1 dim | 80.0% (72/90) | 70.6% (556/788) | 1.67 [0.97, 2.86] | chi-square | 0.060 | n/a |
| D2 throughput | PRs reporting ≥1 dim | 4.4% (4/90) | 7.4% (58/788) | 0.59 [0.21, 1.65] | chi-square | 0.306 | n/a |
| D3 memory | PRs reporting ≥1 dim | 30.0% (27/90) | 35.3% (278/788) | 0.79 [0.49, 1.26] | chi-square | 0.319 | n/a |
| D4 CPU work | PRs reporting ≥1 dim | 4.4% (4/90) | 6.0% (47/788) | 0.73 [0.26, 2.09] | chi-square | 0.559 | n/a |
| D5 I/O & network | PRs reporting ≥1 dim | 3.3% (3/90) | 9.1% (72/788) | 0.34 [0.11, 1.11] | chi-square | 0.062 | n/a |
| D6 artifact size | PRs reporting ≥1 dim | 20.0% (18/90) | 17.1% (135/788) | 1.21 [0.70, 2.09] | chi-square | 0.497 | n/a |
| D7 build/CI time | PRs reporting ≥1 dim | 0.0% (0/90) | 3.6% (28/788) | 0.15 [0.01, 2.44] | Fisher's exact | 0.104 | n/a |
| D8 energy & cost | PRs reporting ≥1 dim | 1.1% (1/90) | 0.9% (7/788) | 1.25 [0.15, 10.31] | Fisher's exact | 0.581 | n/a |
| D9 scalability/concurrency | PRs reporting ≥1 dim | 6.7% (6/90) | 10.5% (83/788) | 0.61 [0.26, 1.43] | chi-square | 0.250 | n/a |
| D0 unspecified perf | PRs reporting ≥1 dim | 12.2% (11/90) | 9.5% (75/788) | 1.32 [0.67, 2.60] | chi-square | 0.414 | n/a |

## 6. Code smells vs each other category on #dims (validated PRs, pooled; descriptive, raw p)
Code smells: n = 226, mean = 0.65, median = 0, any dimension = 39.8%. Cliff's δ < 0: Code smells reports fewer dimensions than that category.
| category | n | mean_dims | median_dims | any_dim_pct | cliffs_delta | p_raw | p_bh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Algorithm-Level Optimizations | 367 | 0.92 | 1.0 | 52.6 | -0.138 | 0.002 | n/a |
| Build & Compilation & Infrastructure Optimization | 128 | 0.95 | 1.0 | 59.4 | -0.185 | 0.001 | n/a |
| Control-Flow and Branching Optimizations | 35 | 0.51 | 0.0 | 37.1 | 0.048 | 0.605 | n/a |
| Data Structure Selection and Adaptation | 47 | 1.06 | 1.0 | 59.6 | -0.209 | 0.012 | n/a |
| I/O and Synchronization | 161 | 0.99 | 1.0 | 59.6 | -0.195 | <0.001 | n/a |
| Loop Transformations | 9 | 1.0 | 1.0 | 55.6 | -0.19 | 0.274 | n/a |
| Memory and Data Locality Optimizations | 590 | 0.96 | 1.0 | 54.1 | -0.154 | <0.001 | n/a |
| Network, Database, and Data Access Optimization | 136 | 0.59 | 0.0 | 42.6 | 0.0 | 1.000 | n/a |

## 7. Interpretation
Humans calibrate verification to the nature of the change: they validate and quantify actual performance changes at a high rate and structural cleanups at a markedly lower one. Agents apply the same verification behaviour regardless of category. The category × author interaction is therefore human selectivity, not agent over-reporting on cleanups: within the *Other* categories agents and humans are indistinguishable on every outcome, and the gap opens only where humans pull back. Code smells PRs that do quantify report the same metric mix as everyone else (§5); they are simply less likely to quantify at all. The two binary interactions (validation presence, any dimension) are significant before correction and borderline after it; the combined outcome is significant under any family. This family is separate from the RQ3 Step 1 and Step 2 families.
