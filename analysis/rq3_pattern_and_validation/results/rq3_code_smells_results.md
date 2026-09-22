# RQ3 post-hoc — code-smell refactorings vs actual performance changes, by author
Source: `analysis/classification_labels/rq3_labels.csv`. *Code smells* = "Code Smells and Structural Simplification" (code-smell refactorings; n = 290 on the category layer, 226 validated); *Other* = the eight remaining RQ1 categories (actual performance changes; n = 1791, 1473 validated). Validation presence and the combined outcome use the category layer (n = 2081); any dimension uses the metric layer (validated PRs, n = 1699). Tests: chi-square / Fisher's exact with OR (Haldane–Anscombe) for 2×2; interaction by logistic-regression LRT. Benjamini–Hochberg in two families: the 3 interaction tests (§4) and the 12 simple-effect contrasts (§2–§3); the per-dimension follow-ups (§5) are descriptive and carry raw p only. All tests: `rq3_code_smells_tests.csv`.

## 1. Descriptives
`quantified` = validated **and** ≥1 dimension reported, as a share of the category layer.
| arm | group | n_category_layer | validated_n | validated_pct | n_metric_layer | any_dim_n | any_dim_pct | quantified_n | quantified_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Agent | Code smells | 135 | 112 | 83.0 | 112 | 51 | 45.5 | 51 | 37.8 |
| Agent | Other | 913 | 746 | 81.7 | 746 | 386 | 51.7 | 386 | 42.3 |
| Human | Code smells | 155 | 114 | 73.5 | 114 | 39 | 34.2 | 39 | 25.2 |
| Human | Other | 878 | 727 | 82.8 | 727 | 402 | 55.3 | 402 | 45.8 |

## 2. Code smells vs Other, within each arm
OR < 1: Code smells PRs have the outcome less often.
| arm | outcome | code_smells | other | effect | test | p_raw | p_bh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Agent | validation present | 83.0% | 81.7% | OR 1.09 [0.68, 1.76] | chi-square | 0.724 | 0.724 |
| Agent | any dimension (validated) | 45.5% | 51.7% | OR 0.78 [0.52, 1.16] | chi-square | 0.220 | 0.294 |
| Agent | validated and >=1 dim (category layer) | 37.8% | 42.3% | OR 0.83 [0.57, 1.20] | chi-square | 0.322 | 0.387 |
| Human | validation present | 73.5% | 82.8% | OR 0.58 [0.39, 0.86] | chi-square | 0.006 | 0.025 |
| Human | any dimension (validated) | 34.2% | 55.3% | OR 0.42 [0.28, 0.64] | chi-square | <0.001 | <0.001 |
| Human | validated and >=1 dim (category layer) | 25.2% | 45.8% | OR 0.40 [0.27, 0.59] | chi-square | <0.001 | <0.001 |

## 3. Agent vs Human, within each category group
OR > 1: agents have the outcome more often than humans.
| group | outcome | agent | human | effect | test | p_raw | p_bh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Code smells | validation present | 83.0% | 73.5% | OR 1.75 [0.99, 3.11] | chi-square | 0.054 | 0.129 |
| Code smells | any dimension (validated) | 45.5% | 34.2% | OR 1.61 [0.94, 2.75] | chi-square | 0.082 | 0.164 |
| Code smells | validated and >=1 dim (category layer) | 37.8% | 25.2% | OR 1.81 [1.09, 2.99] | chi-square | 0.021 | 0.062 |
| Other | validation present | 81.7% | 82.8% | OR 0.93 [0.73, 1.18] | chi-square | 0.545 | 0.595 |
| Other | any dimension (validated) | 51.7% | 55.3% | OR 0.87 [0.71, 1.06] | chi-square | 0.172 | 0.257 |
| Other | validated and >=1 dim (category layer) | 42.3% | 45.8% | OR 0.87 [0.72, 1.05] | chi-square | 0.135 | 0.231 |

## 4. Interaction: does the Code-smells gap differ between agents and humans?
Ratio of ORs = OR(Agent) / OR(Human); a ratio > 1 means the Code-smells penalty is smaller for agents.
| outcome | OR_agent | OR_human | ratio_of_ORs | ratio_lo | ratio_hi | wald_z | p_wald | LRT_chi2 | p_LRT | p_bh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation present | 1.09 | 0.58 | 1.89 | 1.01 | 3.52 | 2.0 | 0.045 | 4.09 | 0.043 | 0.043 |
| any dimension (validated) | 0.78 | 0.42 | 1.85 | 1.04 | 3.29 | 2.11 | 0.035 | 4.47 | 0.034 | 0.043 |
| validated and >=1 dim (category layer) | 0.83 | 0.4 | 2.08 | 1.22 | 3.56 | 2.68 | 0.007 | 7.26 | 0.007 | 0.021 |

## 5. Fewer metrics or different metrics? (pooled metric layer; descriptive, raw p)
Unconditional incidence among validated PRs, then the *mix* among PRs that report at least one dimension.
| dimension | denominator | code_smells | other | OR | test | p_raw |
| --- | --- | --- | --- | --- | --- | --- |
| D1 latency/exec time | validated PRs | 31.9% (72/226) | 37.7% (556/1473) | 0.77 [0.57, 1.04] | chi-square | 0.088 |
| D2 throughput | validated PRs | 1.8% (4/226) | 3.9% (58/1473) | 0.44 [0.16, 1.22] | chi-square | 0.106 |
| D3 memory | validated PRs | 11.9% (27/226) | 18.9% (278/1473) | 0.58 [0.38, 0.89] | chi-square | 0.012 |
| D4 CPU work | validated PRs | 1.8% (4/226) | 3.2% (47/1473) | 0.55 [0.20, 1.53] | chi-square | 0.244 |
| D5 I/O & network | validated PRs | 1.3% (3/226) | 4.9% (72/1473) | 0.26 [0.08, 0.84] | chi-square | 0.015 |
| D6 artifact size | validated PRs | 8.0% (18/226) | 9.2% (135/1473) | 0.86 [0.51, 1.43] | chi-square | 0.557 |
| D7 build/CI time | validated PRs | 0.0% (0/226) | 1.9% (28/1473) | 0.11 [0.01, 1.84] | Fisher's exact | 0.043 |
| D8 energy & cost | validated PRs | 0.4% (1/226) | 0.5% (7/1473) | 0.93 [0.11, 7.60] | Fisher's exact | 1.000 |
| D9 scalability/concurrency | validated PRs | 2.7% (6/226) | 5.6% (83/1473) | 0.46 [0.20, 1.06] | chi-square | 0.061 |
| D0 unspecified perf | validated PRs | 4.9% (11/226) | 5.1% (75/1473) | 0.95 [0.50, 1.82] | chi-square | 0.886 |
| D1 latency/exec time | PRs reporting >=1 dim | 80.0% (72/90) | 70.6% (556/788) | 1.67 [0.97, 2.86] | chi-square | 0.060 |
| D2 throughput | PRs reporting >=1 dim | 4.4% (4/90) | 7.4% (58/788) | 0.59 [0.21, 1.65] | chi-square | 0.306 |
| D3 memory | PRs reporting >=1 dim | 30.0% (27/90) | 35.3% (278/788) | 0.79 [0.49, 1.26] | chi-square | 0.319 |
| D4 CPU work | PRs reporting >=1 dim | 4.4% (4/90) | 6.0% (47/788) | 0.73 [0.26, 2.09] | chi-square | 0.559 |
| D5 I/O & network | PRs reporting >=1 dim | 3.3% (3/90) | 9.1% (72/788) | 0.34 [0.11, 1.11] | chi-square | 0.062 |
| D6 artifact size | PRs reporting >=1 dim | 20.0% (18/90) | 17.1% (135/788) | 1.21 [0.70, 2.09] | chi-square | 0.497 |
| D7 build/CI time | PRs reporting >=1 dim | 0.0% (0/90) | 3.6% (28/788) | 0.15 [0.01, 2.44] | Fisher's exact | 0.104 |
| D8 energy & cost | PRs reporting >=1 dim | 1.1% (1/90) | 0.9% (7/788) | 1.25 [0.15, 10.31] | Fisher's exact | 0.581 |
| D9 scalability/concurrency | PRs reporting >=1 dim | 6.7% (6/90) | 10.5% (83/788) | 0.61 [0.26, 1.43] | chi-square | 0.250 |
| D0 unspecified perf | PRs reporting >=1 dim | 12.2% (11/90) | 9.5% (75/788) | 1.32 [0.67, 2.60] | chi-square | 0.414 |

## 6. Interpretation
Humans calibrate verification to the nature of the change: they validate and quantify actual performance changes at a high rate and code-smell refactorings at a markedly lower one. Agents apply the same verification behaviour regardless of category. The category × author interaction is therefore human selectivity, not agent over-reporting on code-smell refactorings: within the *Other* categories agents and humans are indistinguishable on every outcome, and the gap opens only where humans pull back. Code smells PRs that do quantify report the same metric mix as everyone else (§5); they are simply less likely to quantify at all. The two component interactions (validation presence, any dimension) are significant before correction and borderline after it; the combined outcome is significant after correction. These families are separate from the RQ3 Step 1 and Step 2 families.
