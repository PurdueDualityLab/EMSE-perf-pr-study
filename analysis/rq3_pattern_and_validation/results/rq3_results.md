Metric profiles use all positive validation consensuses, including unresolved types. Only comparisons involving validation type require a resolved positive type; unresolved types are never treated as non-benchmark evidence. Tests are PR-level exploratory associations. The resolved-type layer was refreshed against the current RQ2 consensus (1,581 -> 1,684); extraction and validation presence are unchanged, so presence-based results are identical under both.

# RQ3 — Optimization, validation, and reported metrics
Source: `analysis/classification_labels/rq3_labels.csv` + `analysis/classification_labels/rq2_labels.csv` (n = 2081 PRs; 1699 in the positive metric layer; 1684 in the resolved type layer). Rare categories pooled as 'Other' for tests: none.

## 0. Analytic sample

| author_type | PRs (category layer) | with validation evidence | metric layer (all positive) | resolved type layer |
| --- | --- | --- | --- | --- |
| agentic | 1048 | 858 | 858 | 851 |
| human_candidate | 1033 | 841 | 841 | 833 |
| All | 2081 | 1699 | 1699 | 1684 |

Extractor settings: window = 12 tokens; exclusion window = 8 tokens; nearest-cue attribution = True. Categories with n < 10 on the 2081 PRs are pooled as 'Other' for inferential tests only.

Validation type (RQ2) by author type, % of the author's PRs (2081 PRs):
| validation_type | agentic | human_candidate | All |
| --- | --- | --- | --- |
| benchmark | 37.8 | 45.6 | 41.7 |
| profiling | 1.6 | 2.1 | 1.9 |
| static-reasoning | 41.6 | 32.0 | 36.9 |
| anecdotal | 0.2 | 0.9 | 0.5 |
| none | 18.1 | 18.6 | 18.4 |
| unresolved | 0.7 | 0.8 | 0.7 |
| n | 1048 | 1033 | 2081 |

## 1. Optimization category × validation evidence (n = 2081)

### 1.1 Validation presence by category (% of the category's PRs for that author)
| Category | n | agentic n | agentic validated | human_candidate n | human_candidate validated | All validated |
| --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 691 | 347 | 85.6% | 344 | 85.2% | 85.4% |
| Algorithm-Level Optimizations | 437 | 222 | 85.1% | 215 | 82.8% | 84.0% |
| Code Smells and Structural Simplification | 290 | 135 | 83.0% | 155 | 73.5% | 77.9% |
| I/O and Synchronization | 215 | 92 | 63.0% | 123 | 83.7% | 74.9% |
| Build & Compilation & Infrastructure Optimization | 175 | 99 | 75.8% | 76 | 69.7% | 73.1% |
| Network, Database, and Data Access Optimization | 166 | 92 | 80.4% | 74 | 83.8% | 81.9% |
| Data Structure Selection and Adaptation | 55 | 35 | 80.0% | 20 | 95.0% | 85.5% |
| Control-Flow and Branching Optimizations | 39 | 21 | 95.2% | 18 | 83.3% | 89.7% |
| Loop Transformations | 13 | 5 | 100.0% | 8 | 50.0% | 69.2% |

- **Category × validation present** [pooled], n=2081: chi-square, stat=29.29, p=<0.001, Cramér's V=0.119 (min expected=2.39)

- **Category × validation present** [agentic], n=1048: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=33.25, p=<0.001, Cramér's V=0.178 (min expected=0.91)

- **Category × validation present** [human_candidate], n=1033: chi-square, stat=25.08, p=0.002, Cramér's V=0.156 (min expected=1.49)

Validation rate by sample arm within each category (adaptive contingency test):

- **Author × validation present — Algorithm** [within category], n=437: chi-square, stat=0.45, p=0.504, Cramér's V=0.032 (min expected=34.44; OR=1.19 [0.71, 1.99])

- **Author × validation present — Build/Infra** [within category], n=175: chi-square, stat=0.79, p=0.373, Cramér's V=0.067 (min expected=20.41; OR=1.36 [0.69, 2.65])

- **Author × validation present — Code smells** [within category], n=290: chi-square, stat=3.72, p=0.054, Cramér's V=0.113 (min expected=29.79; OR=1.75 [0.99, 3.11])

- **Author × validation present — Control-flow** [within category], n=39: Fisher's exact, stat=1.49, p=0.318, Cramér's V=0.196 (min expected=1.85; OR=4.00 [0.38, 42.37])

- **Author × validation present — Data structure** [within category], n=55: Fisher's exact, stat=2.30, p=0.234, Cramér's V=0.205 (min expected=2.91; OR=0.21 [0.02, 1.85])

- **Author × validation present — I/O & sync** [within category], n=215: chi-square, stat=11.99, p=<0.001, Cramér's V=0.236 (min expected=23.11; OR=0.33 [0.17, 0.63])

- **Author × validation present — Loop** [within category], n=13: Fisher's exact, stat=3.61, p=0.105, Cramér's V=0.527 (min expected=1.54; OR=11.00 [0.46, 263.54])

- **Author × validation present — Memory/locality** [within category], n=691: chi-square, stat=0.02, p=0.877, Cramér's V=0.006 (min expected=50.28; OR=1.03 [0.68, 1.58])

- **Author × validation present — Network/DB** [within category], n=166: chi-square, stat=0.31, p=0.577, Cramér's V=0.043 (min expected=13.37; OR=0.80 [0.36, 1.78])

| Category | agentic validated | human_candidate validated | test | p (raw) | note |
| --- | --- | --- | --- | --- | --- |
| Algorithm | 85.1% | 82.8% | chi-square | 0.504 | min expected=34.44; OR=1.19 [0.71, 1.99] |
| Build/Infra | 75.8% | 69.7% | chi-square | 0.373 | min expected=20.41; OR=1.36 [0.69, 2.65] |
| Code smells | 83.0% | 73.5% | chi-square | 0.054 | min expected=29.79; OR=1.75 [0.99, 3.11] |
| Control-flow | 95.2% | 83.3% | Fisher's exact | 0.318 | min expected=1.85; OR=4.00 [0.38, 42.37] |
| Data structure | 80.0% | 95.0% | Fisher's exact | 0.234 | min expected=2.91; OR=0.21 [0.02, 1.85] |
| I/O & sync | 63.0% | 83.7% | chi-square | <0.001 | min expected=23.11; OR=0.33 [0.17, 0.63] |
| Loop | 100.0% | 50.0% | Fisher's exact | 0.105 | min expected=1.54; OR=11.00 [0.46, 263.54] |
| Memory/locality | 85.6% | 85.2% | chi-square | 0.877 | min expected=50.28; OR=1.03 [0.68, 1.58] |
| Network/DB | 80.4% | 83.8% | chi-square | 0.577 | min expected=13.37; OR=0.80 [0.36, 1.78] |

### 1.2 Validation type by category (% of the category's PRs; 'none' = no validation evidence)

**agentic**
| pattern | benchmark | profiling | static-reasoning | anecdotal | none | unresolved | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 42.7 | 2.0 | 39.8 | 0.3 | 14.4 | 0.9 | 347 |
| Algorithm-Level Optimizations | 35.6 | 1.8 | 47.3 | 0.5 | 14.9 | 0.0 | 222 |
| Code Smells and Structural Simplification | 42.2 | 1.5 | 37.8 | 0.0 | 17.0 | 1.5 | 135 |
| Build & Compilation & Infrastructure Optimization | 44.4 | 1.0 | 30.3 | 0.0 | 24.2 | 0.0 | 99 |
| I/O and Synchronization | 17.4 | 0.0 | 44.6 | 0.0 | 37.0 | 1.1 | 92 |
| Network, Database, and Data Access Optimization | 27.2 | 3.3 | 48.9 | 0.0 | 19.6 | 1.1 | 92 |
| Data Structure Selection and Adaptation | 48.6 | 0.0 | 31.4 | 0.0 | 20.0 | 0.0 | 35 |
| Control-Flow and Branching Optimizations | 33.3 | 0.0 | 61.9 | 0.0 | 4.8 | 0.0 | 21 |
| Loop Transformations | 60.0 | 0.0 | 40.0 | 0.0 | 0.0 | 0.0 | 5 |

**human_candidate**
| pattern | benchmark | profiling | static-reasoning | anecdotal | none | unresolved | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 46.5 | 2.9 | 34.0 | 0.9 | 14.8 | 0.9 | 344 |
| Algorithm-Level Optimizations | 53.0 | 1.9 | 26.5 | 0.9 | 17.2 | 0.5 | 215 |
| Code Smells and Structural Simplification | 34.2 | 1.9 | 36.1 | 1.3 | 26.5 | 0.0 | 155 |
| I/O and Synchronization | 36.6 | 0.8 | 43.9 | 0.8 | 16.3 | 1.6 | 123 |
| Build & Compilation & Infrastructure Optimization | 56.6 | 1.3 | 11.8 | 0.0 | 30.3 | 0.0 | 76 |
| Network, Database, and Data Access Optimization | 39.2 | 2.7 | 39.2 | 1.4 | 16.2 | 1.4 | 74 |
| Data Structure Selection and Adaptation | 75.0 | 5.0 | 10.0 | 0.0 | 5.0 | 5.0 | 20 |
| Control-Flow and Branching Optimizations | 44.4 | 0.0 | 38.9 | 0.0 | 16.7 | 0.0 | 18 |
| Loop Transformations | 50.0 | 0.0 | 0.0 | 0.0 | 50.0 | 0.0 | 8 |

**All**
| pattern | benchmark | profiling | static-reasoning | anecdotal | none | unresolved | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 44.6 | 2.5 | 36.9 | 0.6 | 14.6 | 0.9 | 691 |
| Algorithm-Level Optimizations | 44.2 | 1.8 | 37.1 | 0.7 | 16.0 | 0.2 | 437 |
| Code Smells and Structural Simplification | 37.9 | 1.7 | 36.9 | 0.7 | 22.1 | 0.7 | 290 |
| I/O and Synchronization | 28.4 | 0.5 | 44.2 | 0.5 | 25.1 | 1.4 | 215 |
| Build & Compilation & Infrastructure Optimization | 49.7 | 1.1 | 22.3 | 0.0 | 26.9 | 0.0 | 175 |
| Network, Database, and Data Access Optimization | 32.5 | 3.0 | 44.6 | 0.6 | 18.1 | 1.2 | 166 |
| Data Structure Selection and Adaptation | 58.2 | 1.8 | 23.6 | 0.0 | 14.5 | 1.8 | 55 |
| Control-Flow and Branching Optimizations | 38.5 | 0.0 | 51.3 | 0.0 | 10.3 | 0.0 | 39 |
| Loop Transformations | 53.8 | 0.0 | 15.4 | 0.0 | 30.8 | 0.0 | 13 |

- **Category × validation type (validated PRs)** [pooled], n=1684: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=49.90, p=0.009, Cramér's V=0.099 (min expected=0.06)

- **Category × validation type (validated PRs)** [agentic], n=851: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=31.38, p=0.164, Cramér's V=0.111 (min expected=0.01)

- **Category × validation type (validated PRs)** [human_candidate], n=833: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=44.81, p=0.033, Cramér's V=0.134 (min expected=0.04)

- **Category × benchmark-based vs other evidence (validated PRs)** [pooled], n=1684: chi-square, stat=41.78, p=<0.001, Cramér's V=0.158 (min expected=4.37)

## 2. Metric profile: category × reported dimensions (validated PRs, n = 1699)

Dimensions: **D1** Latency / execution time; **D2** Throughput; **D3** Memory; **D4** CPU / compute work; **D5** I/O and network; **D6** Artifact size; **D7** Build and CI time; **D8** Energy and cost; **D9** Scalability / concurrency; **D0** Unspecified performance. D0 records a quantified gain whose dimension is not named and is credited only when no D1–D9 cue is in the claim's window.

**agentic (n = 858)** — % of the category's validated PRs reporting each dimension
| Category | n validated | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D0 | ≥1 dim | mean #dims |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 297 | 40% | 4% | 22% | 4% | 5% | 18% | 2% | 1% | 2% | 6% | 53.9% | 1.02 |
| Algorithm-Level Optimizations | 189 | 32% | 6% | 16% | 4% | 4% | 8% | 2% | 1% | 6% | 8% | 45.5% | 0.87 |
| Code Smells and Structural Simplification | 112 | 36% | 1% | 12% | 3% | 1% | 14% | 0% | 0% | 3% | 7% | 45.5% | 0.76 |
| Build & Compilation & Infrastructure Optimization | 75 | 35% | 4% | 12% | 4% | 5% | 17% | 8% | 1% | 5% | 4% | 58.7% | 0.96 |
| Network, Database, and Data Access Optimization | 74 | 20% | 0% | 4% | 1% | 16% | 7% | 1% | 0% | 7% | 0% | 41.9% | 0.57 |
| I/O and Synchronization | 58 | 34% | 10% | 24% | 7% | 10% | 7% | 2% | 0% | 19% | 0% | 67.2% | 1.14 |
| Data Structure Selection and Adaptation | 28 | 46% | 0% | 36% | 0% | 7% | 36% | 4% | 0% | 0% | 4% | 57.1% | 1.32 |
| Control-Flow and Branching Optimizations | 20 | 30% | 5% | 5% | 5% | 0% | 0% | 0% | 0% | 10% | 5% | 40.0% | 0.60 |
| Loop Transformations | 5 | 40% | 0% | 0% | 20% | 0% | 0% | 0% | 0% | 0% | 0% | 40.0% | 0.60 |
| All | 858 | 35% | 4% | 17% | 4% | 5% | 13% | 2% | 1% | 5% | 5% | 50.9% | 0.92 |

**human_candidate (n = 841)** — % of the category's validated PRs reporting each dimension
| Category | n validated | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D0 | ≥1 dim | mean #dims |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 293 | 36% | 4% | 25% | 3% | 4% | 4% | 1% | 0% | 4% | 6% | 54.3% | 0.89 |
| Algorithm-Level Optimizations | 178 | 52% | 3% | 20% | 3% | 4% | 4% | 1% | 0% | 5% | 4% | 60.1% | 0.97 |
| Code Smells and Structural Simplification | 114 | 28% | 3% | 12% | 1% | 2% | 2% | 0% | 1% | 3% | 3% | 34.2% | 0.54 |
| I/O and Synchronization | 103 | 40% | 5% | 16% | 1% | 3% | 5% | 2% | 0% | 17% | 4% | 55.3% | 0.91 |
| Network, Database, and Data Access Optimization | 62 | 31% | 0% | 5% | 2% | 3% | 6% | 2% | 2% | 6% | 5% | 43.5% | 0.61 |
| Build & Compilation & Infrastructure Optimization | 53 | 40% | 2% | 25% | 2% | 4% | 9% | 2% | 0% | 2% | 8% | 60.4% | 0.92 |
| Data Structure Selection and Adaptation | 19 | 53% | 0% | 11% | 0% | 0% | 0% | 0% | 0% | 0% | 5% | 63.2% | 0.68 |
| Control-Flow and Branching Optimizations | 15 | 27% | 0% | 7% | 0% | 0% | 7% | 0% | 0% | 0% | 0% | 33.3% | 0.40 |
| Loop Transformations | 4 | 75% | 25% | 25% | 0% | 0% | 25% | 0% | 0% | 0% | 0% | 75.0% | 1.50 |
| All | 841 | 39% | 3% | 19% | 2% | 3% | 5% | 1% | 0% | 5% | 5% | 52.4% | 0.83 |

**All (n = 1699)** — % of the category's validated PRs reporting each dimension
| Category | n validated | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D0 | ≥1 dim | mean #dims |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 590 | 38% | 4% | 24% | 4% | 5% | 11% | 2% | 1% | 3% | 6% | 54.1% | 0.96 |
| Algorithm-Level Optimizations | 367 | 41% | 5% | 18% | 4% | 4% | 6% | 1% | 1% | 6% | 7% | 52.6% | 0.92 |
| Code Smells and Structural Simplification | 226 | 32% | 2% | 12% | 2% | 1% | 8% | 0% | 0% | 3% | 5% | 39.8% | 0.65 |
| I/O and Synchronization | 161 | 38% | 7% | 19% | 3% | 6% | 6% | 2% | 0% | 17% | 2% | 59.6% | 0.99 |
| Network, Database, and Data Access Optimization | 136 | 25% | 0% | 4% | 1% | 10% | 7% | 1% | 1% | 7% | 2% | 42.6% | 0.59 |
| Build & Compilation & Infrastructure Optimization | 128 | 37% | 3% | 17% | 3% | 5% | 14% | 5% | 1% | 4% | 5% | 59.4% | 0.95 |
| Data Structure Selection and Adaptation | 47 | 49% | 0% | 26% | 0% | 4% | 21% | 2% | 0% | 0% | 4% | 59.6% | 1.06 |
| Control-Flow and Branching Optimizations | 35 | 29% | 3% | 6% | 3% | 0% | 3% | 0% | 0% | 6% | 3% | 37.1% | 0.51 |
| Loop Transformations | 9 | 56% | 11% | 11% | 11% | 0% | 11% | 0% | 0% | 0% | 0% | 55.6% | 1.00 |
| All | 1699 | 37% | 4% | 18% | 3% | 4% | 9% | 2% | 0% | 5% | 5% | 51.7% | 0.87 |

### 2.2 Tests on the metric profile

- **Category × D1 reported** [pooled (validated)], n=1699: chi-square, stat=19.61, p=0.012, Cramér's V=0.107 (min expected=3.33)

- **Category × D2 reported** [pooled (validated)], n=1699: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=17.18, p=0.042, Cramér's V=0.101 (min expected=0.33)

- **Category × D3 reported** [pooled (validated)], n=1699: chi-square, stat=40.84, p=<0.001, Cramér's V=0.155 (min expected=1.62)

- **Category × D4 reported** [pooled (validated)], n=1699: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=6.77, p=0.536, Cramér's V=0.063 (min expected=0.27)

- **Category × D5 reported** [pooled (validated)], n=1699: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=19.18, p=0.022, Cramér's V=0.106 (min expected=0.40)

- **Category × D6 reported** [pooled (validated)], n=1699: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=24.81, p=0.003, Cramér's V=0.121 (min expected=0.81)

- **Category × D7 reported** [pooled (validated)], n=1699: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=16.38, p=0.054, Cramér's V=0.098 (min expected=0.15)

- **Category × D8 reported** [pooled (validated)], n=1699: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=1.72, p=0.976, Cramér's V=0.032 (min expected=0.04)

- **Category × D9 reported** [pooled (validated)], n=1699: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=60.89, p=<0.001, Cramér's V=0.189 (min expected=0.47)

- **Category × D0 reported** [pooled (validated)], n=1699: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=7.76, p=0.441, Cramér's V=0.068 (min expected=0.46)

- **Author × D1 reported** [validated], n=1699: chi-square, stat=2.97, p=0.085, Cramér's V=0.042 (min expected=310.86; OR=0.84 [0.69, 1.02])

- **Author × D2 reported** [validated], n=1699: chi-square, stat=0.48, p=0.486, Cramér's V=0.017 (min expected=30.69; OR=1.20 [0.72, 1.99])

- **Author × D3 reported** [validated], n=1699: chi-square, stat=1.30, p=0.254, Cramér's V=0.028 (min expected=150.97; OR=0.87 [0.68, 1.11])

- **Author × D4 reported** [validated], n=1699: chi-square, stat=2.22, p=0.136, Cramér's V=0.036 (min expected=25.24; OR=1.54 [0.87, 2.72])

- **Author × D5 reported** [validated], n=1699: chi-square, stat=3.68, p=0.055, Cramér's V=0.047 (min expected=37.12; OR=1.59 [0.99, 2.55])

- **Author × D6 reported** [validated], n=1699: chi-square, stat=40.91, p=<0.001, Cramér's V=0.155 (min expected=75.73; OR=3.27 [2.24, 4.78])

- **Author × D7 reported** [validated], n=1699: chi-square, stat=3.43, p=0.064, Cramér's V=0.045 (min expected=13.86; OR=2.09 [0.94, 4.65])

- **Author × D8 reported** [validated], n=1699: Fisher's exact, stat=1.93, p=0.288, Cramér's V=0.034 (min expected=3.96; OR=2.95 [0.59, 14.68])

- **Author × D9 reported** [validated], n=1699: chi-square, stat=0.04, p=0.837, Cramér's V=0.005 (min expected=44.05; OR=0.96 [0.62, 1.47])

- **Author × D0 reported** [validated], n=1699: chi-square, stat=0.32, p=0.569, Cramér's V=0.014 (min expected=42.57; OR=1.13 [0.73, 1.75])

- **Author × any dimension reported** [validated], n=1699: chi-square, stat=0.39, p=0.535, Cramér's V=0.015 (min expected=406.39; OR=1.06 [0.88, 1.28])

- **Category × any dimension reported** [pooled (validated)], n=1699: chi-square, stat=29.93, p=<0.001, Cramér's V=0.133 (min expected=4.35)

### 2.3 Dimensionality of the evidence (number of distinct dimensions per validated PR)

% of the author's validated PRs (counting D0):
| n_dims | agentic | human_candidate | All |
| --- | --- | --- | --- |
| 0 | 49.1 | 47.6 | 48.3 |
| 1 | 27.9 | 31.2 | 29.5 |
| 2 | 10.3 | 14.4 | 12.3 |
| 3 | 9.1 | 5.0 | 7.1 |
| 4 | 2.9 | 1.4 | 2.2 |
| 5 | 0.6 | 0.4 | 0.5 |
| 6 | 0.2 | 0.1 | 0.2 |
| n | 858 | 841 | 1699 |

PRs whose only reported 'dimension' is **D0** (a quantified gain with no named dimension): agentic: 0.9%, human_candidate: 1.2%; all: 1.1%. Excluding D0, the share of validated PRs with ≥1 named dimension is agentic: 50.0%, human_candidate: 51.2%; all: 50.6%.

- **#dims: agent vs human** [validated], n=1699: Mann–Whitney U, stat=364049.50, p=0.728, Cliff's delta=0.009 (median agent=1.00 (mean 0.92); median human=1.00 (mean 0.83))

- **#dims across categories** [pooled (validated)], n=1699: Kruskal–Wallis, stat=32.12, p=<0.001, epsilon²=0.014 (Algorithm md=1.0; Build/Infra md=1.0; Code smells md=0.0; Control-flow md=0.0; Data structure md=1.0; I/O & sync md=1.0; Loop md=1.0; Memory/locality md=1.0; Network/DB md=0.0)

- **#dims across categories** [agentic], n=858: Kruskal–Wallis, stat=17.09, p=0.029, epsilon²=0.011 (Algorithm md=0.0; Build/Infra md=1.0; Code smells md=0.0; Control-flow md=0.0; Data structure md=1.0; I/O & sync md=1.0; Loop md=0.0; Memory/locality md=1.0; Network/DB md=0.0)

- **#dims across categories** [human_candidate], n=841: Kruskal–Wallis, stat=27.31, p=<0.001, epsilon²=0.023 (Algorithm md=1.0; Build/Infra md=1.0; Code smells md=0.0; Control-flow md=0.0; Data structure md=1.0; I/O & sync md=1.0; Loop md=1.5; Memory/locality md=1.0; Network/DB md=0.0)

Per category and author:
| pattern | author_type | count | mean | median | max |
| --- | --- | --- | --- | --- | --- |
| Algorithm-Level Optimizations | agentic | 189 | 0.87 | 0.00 | 5 |
| Algorithm-Level Optimizations | human_candidate | 178 | 0.97 | 1.00 | 4 |
| Build & Compilation & Infrastructure Optimization | agentic | 75 | 0.96 | 1.00 | 4 |
| Build & Compilation & Infrastructure Optimization | human_candidate | 53 | 0.92 | 1.00 | 4 |
| Code Smells and Structural Simplification | agentic | 112 | 0.76 | 0.00 | 3 |
| Code Smells and Structural Simplification | human_candidate | 114 | 0.54 | 0.00 | 4 |
| Control-Flow and Branching Optimizations | agentic | 20 | 0.60 | 0.00 | 3 |
| Control-Flow and Branching Optimizations | human_candidate | 15 | 0.40 | 0.00 | 2 |
| Data Structure Selection and Adaptation | agentic | 28 | 1.32 | 1.00 | 4 |
| Data Structure Selection and Adaptation | human_candidate | 19 | 0.68 | 1.00 | 2 |
| I/O and Synchronization | agentic | 58 | 1.14 | 1.00 | 6 |
| I/O and Synchronization | human_candidate | 103 | 0.91 | 1.00 | 4 |
| Loop Transformations | agentic | 5 | 0.60 | 0.00 | 2 |
| Loop Transformations | human_candidate | 4 | 1.50 | 1.50 | 3 |
| Memory and Data Locality Optimizations | agentic | 297 | 1.02 | 1.00 | 5 |
| Memory and Data Locality Optimizations | human_candidate | 293 | 0.89 | 1.00 | 6 |
| Network, Database, and Data Access Optimization | agentic | 74 | 0.57 | 0.00 | 3 |
| Network, Database, and Data Access Optimization | human_candidate | 62 | 0.61 | 0.00 | 5 |

### 2.4 Metric profile × validation type (resolved positive types, n = 1684)

% of PRs with that validation type reporting each dimension:
| Validation type | n | ≥1 dim | mean #dims | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark | 867 | 79.8% | 1.43 | 62% | 5% | 29% | 5% | 7% | 16% | 3% | 1% | 7% | 8% |
| profiling | 39 | 30.8% | 0.36 | 15% | 0% | 10% | 0% | 0% | 3% | 0% | 0% | 3% | 5% |
| static-reasoning | 767 | 21.6% | 0.29 | 10% | 2% | 6% | 1% | 2% | 2% | 1% | 0% | 4% | 2% |
| anecdotal | 11 | 18.2% | 0.18 | 18% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

- **#dims: benchmark vs other evidence** [resolved types], n=1684: Mann–Whitney U, stat=576710.50, p=<0.001, Cliff's delta=0.628 (median benchmark=1.00 (mean 1.43); median other=0.00 (mean 0.29))

- **Benchmark evidence × any dimension reported** [resolved types], n=1684: chi-square, stat=562.50, p=<0.001, Cramér's V=0.578 (min expected=393.95; OR=13.99 [11.07, 17.69])

- **Benchmark evidence × D1 reported** [resolved types], n=1684: chi-square, stat=492.94, p=<0.001, Cramér's V=0.541 (min expected=301.77; OR=14.80 [11.34, 19.32])

- **Benchmark evidence × D2 reported** [resolved types], n=1684: chi-square, stat=14.50, p=<0.001, Cramér's V=0.093 (min expected=29.59; OR=3.00 [1.66, 5.41])

- **Benchmark evidence × D3 reported** [resolved types], n=1684: chi-square, stat=144.58, p=<0.001, Cramér's V=0.293 (min expected=147.97; OR=5.91 [4.31, 8.09])

- **Benchmark evidence × D4 reported** [resolved types], n=1684: chi-square, stat=28.44, p=<0.001, Cramér's V=0.130 (min expected=24.74; OR=7.40 [3.14, 17.44])

- **Benchmark evidence × D5 reported** [resolved types], n=1684: chi-square, stat=18.89, p=<0.001, Cramér's V=0.106 (min expected=36.39; OR=3.12 [1.82, 5.35])

- **Benchmark evidence × D6 reported** [resolved types], n=1684: chi-square, stat=94.26, p=<0.001, Cramér's V=0.237 (min expected=74.23; OR=8.76 [5.24, 14.64])

- **Benchmark evidence × D7 reported** [resolved types], n=1684: chi-square, stat=13.36, p=<0.001, Cramér's V=0.089 (min expected=13.58; OR=5.79 [2.00, 16.75])

- **Benchmark evidence × D8 reported** [resolved types], n=1684: Fisher's exact, stat=4.17, p=0.071, Cramér's V=0.050 (min expected=3.88; OR=6.64 [0.82, 54.10])

- **Benchmark evidence × D9 reported** [resolved types], n=1684: chi-square, stat=10.94, p=<0.001, Cramér's V=0.081 (min expected=43.18; OR=2.13 [1.35, 3.37])

- **Benchmark evidence × D0 reported** [resolved types], n=1684: chi-square, stat=35.03, p=<0.001, Cramér's V=0.144 (min expected=41.72; OR=4.77 [2.71, 8.40])

> Sections 2.5, 3, 4 and 5 are carried over from the pre-refresh run. Sections 3, 4 and 5 do not condition on validation type, so their values are unchanged. Section 2.5's `benchmark` column does condition on it and reflects the 1,581-PR type layer; it was not regenerated because the per-agent breakdown needs the `agent` column, which is absent from the published compact labels.

### 2.5 Metric reporting by author type and by agent

Rates are % of the row's PRs. *all PRs* = the category layer; the metric layer contains all positive validation consensuses, including unresolved types; *benchmark* = PRs whose evidence is a benchmark; *no diff* = dimension flags recomputed with the code diff excluded. The *all PRs* quantification columns count only PRs that are validated **and** report a dimension, i.e. they are the product of the validation rate and the conditional quantification rate; PRs with a quantitative claim but no RQ2 validation label (`extremes_metric_without_validation_label.csv`) are not counted. Per-agent rows are descriptive and are not part of the RQ3 test family.

| author | n (all PRs) | validation present | metric layer | n resolved type layer | benchmark evidence | metric layer & ≥1 dim, all PRs | metric layer & ≥1 dim, all PRs (no diff) | n metric layer | ≥1 dim, metric layer | mean #dims, metric layer | n benchmark | ≥1 dim, benchmark |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| human_candidate | 1033 | 81.4% | 81.4% | 774 | 42.1% | 42.7% | 41.0% | 841 | 52.4% | 0.83 | 435 | 79.3% |
| agentic (all) | 1048 | 81.9% | 81.9% | 807 | 34.9% | 41.7% | 39.7% | 858 | 50.9% | 0.92 | 366 | 82.0% |
|   github_copilot | 421 | 92.2% | 92.2% | 362 | 29.0% | 41.3% | 40.4% | 388 | 44.8% | 0.77 | 122 | 86.1% |
|   openai_codex | 366 | 71.9% | 71.9% | 251 | 43.4% | 40.2% | 39.6% | 263 | 55.9% | 1.09 | 159 | 78.6% |
|   cursor | 146 | 71.2% | 71.2% | 96 | 21.2% | 37.7% | 28.8% | 104 | 52.9% | 1.00 | 31 | 90.3% |
|   devin | 62 | 85.5% | 85.5% | 50 | 30.6% | 48.4% | 46.8% | 53 | 56.6% | 0.81 | 19 | 84.2% |
|   claude_code | 53 | 94.3% | 94.3% | 48 | 66.0% | 58.5% | 56.6% | 50 | 62.0% | 1.10 | 35 | 74.3% |

## 3. Merge status

- **#dims: merged vs not merged** [pooled (validated)], n=1699: Mann–Whitney U, stat=319983.50, p=0.023, Cliff's delta=-0.061 (median merged=1.00 (mean 0.82); median not merged=1.00 (mean 0.96))

- **Merged × any dimension reported** [pooled (validated)], n=1699: chi-square, stat=2.43, p=0.119, Cramér's V=0.038 (min expected=313.61; OR=0.86 [0.70, 1.04])

- **#dims: merged vs not merged** [agentic (validated)], n=858: Mann–Whitney U, stat=84622.50, p=0.033, Cliff's delta=-0.078 (median merged=0.00 (mean 0.84); median not merged=1.00 (mean 1.00))

- **Merged × any dimension reported** [agentic (validated)], n=858: chi-square, stat=3.02, p=0.082, Cramér's V=0.059 (min expected=199.71; OR=0.79 [0.60, 1.03])

- **#dims: merged vs not merged** [human_candidate (validated)], n=841: Mann–Whitney U, stat=69775.50, p=0.360, Cliff's delta=-0.037 (median merged=1.00 (mean 0.80); median not merged=1.00 (mean 0.90))

- **Merged × any dimension reported** [human_candidate (validated)], n=841: chi-square, stat=0.39, p=0.532, Cramér's V=0.022 (min expected=115.10; OR=0.91 [0.67, 1.23])

| Author | Merged | n | ≥1 dim | mean #dims | median #dims | median time-to-merge (days) |
| --- | --- | --- | --- | --- | --- | --- |
| agentic | False | 407 | 54.1% | 1.00 | 1 |  |
| agentic | True | 451 | 48.1% | 0.84 | 0 | 0.28 |
| human_candidate | False | 242 | 54.1% | 0.90 | 1 |  |
| human_candidate | True | 599 | 51.8% | 0.80 | 1 | 0.84 |

## 4. PRs at the extremes (exported for qualitative reading)

Highest dimensionality (top 15) → `results/extremes_top_dimensionality.csv`
| html_url | author_type | pattern | validation_type | n_dims | dims |
| --- | --- | --- | --- | --- | --- |
| https://github.com/CapSoftware/Cap/pull/1598 | agentic | I/O and Synchronization | static-reasoning | 6 | D1|D2|D3|D4|D5|D9 |
| https://github.com/lambdaclass/ethrex/pull/6639 | human_candidate | Memory and Data Locality Optimizations | benchmark | 6 | D0|D1|D2|D3|D4|D9 |
| https://github.com/rustfs/rustfs/pull/461 | agentic | I/O and Synchronization | benchmark | 6 | D1|D2|D3|D4|D5|D9 |
| https://github.com/chrxh/alien/pull/409 | agentic | Algorithm-Level Optimizations | static-reasoning | 5 | D0|D1|D3|D6|D9 |
| https://github.com/rustfs/rustfs/pull/449 | agentic | Memory and Data Locality Optimizations | benchmark | 5 | D1|D3|D5|D6|D9 |
| https://github.com/NethermindEth/nethermind/pull/11241 | agentic | Algorithm-Level Optimizations | benchmark | 5 | D0|D1|D4|D5|D8 |
| https://github.com/dotnet/fsharp/pull/19072 | agentic | Memory and Data Locality Optimizations | benchmark | 5 | D0|D1|D3|D5|D7 |
| https://github.com/modular/modular/pull/6633 | agentic | Memory and Data Locality Optimizations | benchmark | 5 | D0|D1|D2|D3|D5 |
| https://github.com/datahub-project/datahub/pull/16938 | human_candidate | Network, Database, and Data Access Optimization | benchmark | 5 | D0|D1|D3|D4|D9 |
| https://github.com/Azure/azure-sdk-for-java/pull/48617 | human_candidate | Memory and Data Locality Optimizations | benchmark | 5 | D1|D2|D3|D4|D9 |
| https://github.com/Azure/azure-sdk-for-net/pull/56872 | human_candidate | Memory and Data Locality Optimizations | benchmark | 5 | D2|D3|D5|D6|D9 |
| https://github.com/web-infra-dev/rspack/pull/13876 | agentic | Memory and Data Locality Optimizations | benchmark | 4 | D1|D3|D5|D6 |
| https://github.com/dotnet/runtime/pull/126063 | agentic | Algorithm-Level Optimizations | static-reasoning | 4 | D0|D1|D2|D3 |
| https://github.com/dotnet/runtime/pull/125799 | human_candidate | Algorithm-Level Optimizations | benchmark | 4 | D1|D3|D5|D9 |
| https://github.com/web-infra-dev/rspack/pull/13564 | agentic | Data Structure Selection and Adaptation | benchmark | 4 | D0|D1|D3|D6 |

Validated PRs reporting **no** metric dimension (48.3% overall), as % of each validation type → `results/extremes_validated_no_metric.csv`
| validation_type | agentic | human_candidate | n (validated PRs) |
| --- | --- | --- | --- |
| anecdotal | 100.0 | 66.7 | 8 |
| benchmark | 18.0 | 20.7 | 801 |
| profiling | 86.7 | 50.0 | 31 |
| static-reasoning | 76.9 | 83.0 | 741 |
| unresolved | 27.5 | 52.2 | 118 |

PRs *without* an RQ2 validation label in which the extractor still found a quantitative metric claim (n = 42; candidates for an RQ2 label recheck) → `results/extremes_metric_without_validation_label.csv`
| html_url | author_type | pattern | n_dims | dims |
| --- | --- | --- | --- | --- |
| https://github.com/hmislk/hmis/pull/17201 | human_candidate | Code Smells and Structural Simplification | 3 | D1|D5|D8 |
| https://github.com/triton-lang/triton/pull/7849 | human_candidate | Build & Compilation & Infrastructure Optimization | 1 | D1 |
| https://github.com/ant-design/ant-design/pull/56516 | human_candidate | Build & Compilation & Infrastructure Optimization | 1 | D6 |
| https://github.com/dotnet/macios/pull/24775 | human_candidate | Memory and Data Locality Optimizations | 1 | D2 |
| https://github.com/ray-project/ray/pull/61555 | human_candidate | Control-Flow and Branching Optimizations | 2 | D1|D9 |
| https://github.com/commaai/openpilot/pull/37904 | human_candidate | Code Smells and Structural Simplification | 1 | D1 |
| https://github.com/swc-project/swc/pull/11310 | human_candidate | Loop Transformations | 1 | D1 |
| https://github.com/fatedier/fft/pull/22 | agentic | I/O and Synchronization | 2 | D1|D2 |
| https://github.com/mlflow/mlflow/pull/19078 | agentic | Code Smells and Structural Simplification | 1 | D0 |
| https://github.com/Azure/azure-sdk-tools/pull/11379 | human_candidate | Code Smells and Structural Simplification | 3 | D1|D3|D5 |
| https://github.com/Azure/azure-sdk-tools/pull/11382 | human_candidate | Build & Compilation & Infrastructure Optimization | 2 | D3|D5 |
| https://github.com/dotnet/runtime/pull/123776 | agentic | Memory and Data Locality Optimizations | 1 | D1 |
| https://github.com/kentcdodds/kentcdodds.com/pull/694 | agentic | I/O and Synchronization | 1 | D1 |
| https://github.com/medplum/medplum/pull/6650 | human_candidate | Control-Flow and Branching Optimizations | 1 | D1 |
| https://github.com/c3rb3ru5d3d53c/binlex/pull/171 | agentic | Build & Compilation & Infrastructure Optimization | 2 | D1|D2 |
| https://github.com/freenet/freenet-core/pull/2234 | human_candidate | I/O and Synchronization | 5 | D0|D1|D2|D5|D9 |
| https://github.com/open-metadata/OpenMetadata/pull/21376 | human_candidate | Memory and Data Locality Optimizations | 1 | D1 |
| https://github.com/OneKeyHQ/app-monorepo/pull/11798 | agentic | Memory and Data Locality Optimizations | 2 | D1|D4 |
| https://github.com/surrealdb/surrealdb/pull/6824 | human_candidate | Memory and Data Locality Optimizations | 1 | D3 |
| https://github.com/microsoft/onnxscript/pull/2441 | human_candidate | Code Smells and Structural Simplification | 1 | D1 |
| https://github.com/embeddings-benchmark/mteb/pull/3131 | human_candidate | I/O and Synchronization | 1 | D1 |
| https://github.com/actualbudget/actual/pull/5899 | agentic | Network, Database, and Data Access Optimization | 1 | D6 |
| https://github.com/actualbudget/actual/pull/5921 | agentic | Network, Database, and Data Access Optimization | 1 | D6 |
| https://github.com/renegade-fi/renegade/pull/911 | human_candidate | I/O and Synchronization | 1 | D1 |
| https://github.com/HotCakeX/Harden-Windows-Security/pull/975 | human_candidate | Code Smells and Structural Simplification | 1 | D1 |
| https://github.com/activepieces/activepieces/pull/9982 | agentic | Network, Database, and Data Access Optimization | 1 | D1 |
| https://github.com/epicweb-dev/epicshop/pull/422 | agentic | Algorithm-Level Optimizations | 1 | D1 |
| https://github.com/epicweb-dev/epicshop/pull/553 | agentic | Network, Database, and Data Access Optimization | 1 | D1 |
| https://github.com/Stirling-Tools/Stirling-PDF/pull/3992 | agentic | I/O and Synchronization | 1 | D1 |
| https://github.com/Shelf-nu/shelf.nu/pull/2088 | agentic | I/O and Synchronization | 1 | D2 |
| https://github.com/bayesianbandits/bayesianbandits/pull/220 | human_candidate | Algorithm-Level Optimizations | 1 | D1 |
| https://github.com/jirihofman/portfolio/pull/309 | agentic | Memory and Data Locality Optimizations | 1 | D1 |
| https://github.com/langfuse/langfuse/pull/12572 | agentic | I/O and Synchronization | 1 | D4 |
| https://github.com/langfuse/langfuse/pull/13479 | agentic | I/O and Synchronization | 1 | D9 |
| https://github.com/langfuse/langfuse-docs/pull/2143 | agentic | Memory and Data Locality Optimizations | 1 | D1 |
| https://github.com/TypedDevs/bashunit/pull/439 | human_candidate | Memory and Data Locality Optimizations | 1 | D1 |
| https://github.com/CapSoftware/Cap/pull/1595 | agentic | Algorithm-Level Optimizations | 2 | D1|D2 |
| https://github.com/CapSoftware/Cap/pull/1596 | agentic | I/O and Synchronization | 2 | D1|D2 |
| https://github.com/rustfs/rustfs/pull/916 | agentic | Memory and Data Locality Optimizations | 6 | D1|D2|D3|D5|D6|D9 |
| https://github.com/bolna-ai/bolna/pull/236 | agentic | Memory and Data Locality Optimizations | 1 | D1 |
| https://github.com/Effect-TS/effect-smol/pull/794 | human_candidate | Algorithm-Level Optimizations | 1 | D6 |
| https://github.com/facet-rs/facet/pull/741 | agentic | Build & Compilation & Infrastructure Optimization | 1 | D9 |

## 5. Sensitivity: excluding the code diff from the corpus

% of the author's validated PRs:
| Author | n | Dimension | with diff | without diff |
| --- | --- | --- | --- | --- |
| agentic | 858 | D1 | 35.0% | 33.4% |
| agentic | 858 | D2 | 4.0% | 3.5% |
| agentic | 858 | D3 | 16.9% | 14.7% |
| agentic | 858 | D4 | 3.6% | 3.3% |
| agentic | 858 | D5 | 5.4% | 4.4% |
| agentic | 858 | D6 | 13.4% | 12.5% |
| agentic | 858 | D7 | 2.2% | 2.1% |
| agentic | 858 | D8 | 0.7% | 0.6% |
| agentic | 858 | D9 | 5.1% | 4.7% |
| agentic | 858 | D0 | 5.4% | 4.7% |
| agentic | 858 | ≥1 dim | 50.9% | 48.5% |
| human_candidate | 841 | D1 | 39.0% | 37.6% |
| human_candidate | 841 | D2 | 3.3% | 3.2% |
| human_candidate | 841 | D3 | 19.0% | 17.5% |
| human_candidate | 841 | D4 | 2.4% | 1.9% |
| human_candidate | 841 | D5 | 3.4% | 3.0% |
| human_candidate | 841 | D6 | 4.5% | 4.0% |
| human_candidate | 841 | D7 | 1.1% | 1.0% |
| human_candidate | 841 | D8 | 0.2% | 0.1% |
| human_candidate | 841 | D9 | 5.4% | 4.5% |
| human_candidate | 841 | D0 | 4.8% | 4.6% |
| human_candidate | 841 | ≥1 dim | 52.4% | 50.4% |
| All | 1699 | D1 | 37.0% | 35.5% |
| All | 1699 | D2 | 3.6% | 3.4% |
| All | 1699 | D3 | 18.0% | 16.1% |
| All | 1699 | D4 | 3.0% | 2.6% |
| All | 1699 | D5 | 4.4% | 3.7% |
| All | 1699 | D6 | 9.0% | 8.3% |
| All | 1699 | D7 | 1.6% | 1.5% |
| All | 1699 | D8 | 0.5% | 0.4% |
| All | 1699 | D9 | 5.2% | 4.6% |
| All | 1699 | D0 | 5.1% | 4.6% |
| All | 1699 | ≥1 dim | 51.7% | 49.4% |

PRs whose dimension count changes when the diff is excluded: 79 of 1699.

Dimensions found in the description alone: 503 PRs have ≥1; bot-authored issue comments contribute a dimension in 294 PRs.

## 6. RQ3 test family (Benjamini–Hochberg across all 60 tests)

| family | test_label | stratum | n | shape | test | statistic | dof | p_raw | p_bh | effect_name | effect | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A. category×validation | Category × validation present | pooled | 2081 | 9x2 | chi-square | 29.29 | 8.00 | <0.001 | <0.001 | Cramér's V | 0.12 | min expected=2.39 |
| A. category×validation | Category × validation present | agentic | 1048 | 9x2 | Conditional Pearson chi-square (Monte Carlo, B=20,000) | 33.25 | 8.00 | <0.001 | <0.001 | Cramér's V | 0.18 | min expected=0.91 |
| A. category×validation | Category × validation present | human_candidate | 1033 | 9x2 | chi-square | 25.08 | 8.00 | 0.002 | 0.004 | Cramér's V | 0.16 | min expected=1.49 |
| A. category×validation | Author × validation present — Algorithm | within category | 437 | 2x2 | chi-square | 0.45 | 1.00 | 0.504 | 0.593 | Cramér's V | 0.03 | min expected=34.44; OR=1.19 [0.71, 1.99] |
| A. category×validation | Author × validation present — Build/Infra | within category | 175 | 2x2 | chi-square | 0.79 | 1.00 | 0.373 | 0.466 | Cramér's V | 0.07 | min expected=20.41; OR=1.36 [0.69, 2.65] |
| A. category×validation | Author × validation present — Code smells | within category | 290 | 2x2 | chi-square | 3.72 | 1.00 | 0.054 | 0.097 | Cramér's V | 0.11 | min expected=29.79; OR=1.75 [0.99, 3.11] |
| A. category×validation | Author × validation present — Control-flow | within category | 39 | 2x2 | Fisher's exact | 1.49 | 1.00 | 0.318 | 0.415 | Cramér's V | 0.20 | min expected=1.85; OR=4.00 [0.38, 42.37] |
| A. category×validation | Author × validation present — Data structure | within category | 55 | 2x2 | Fisher's exact | 2.30 | 1.00 | 0.234 | 0.327 | Cramér's V | 0.20 | min expected=2.91; OR=0.21 [0.02, 1.85] |
| A. category×validation | Author × validation present — I/O & sync | within category | 215 | 2x2 | chi-square | 11.99 | 1.00 | <0.001 | 0.002 | Cramér's V | 0.24 | min expected=23.11; OR=0.33 [0.17, 0.63] |
| A. category×validation | Author × validation present — Loop | within category | 13 | 2x2 | Fisher's exact | 3.61 | 1.00 | 0.105 | 0.161 | Cramér's V | 0.53 | min expected=1.54; OR=11.00 [0.46, 263.54] |
| A. category×validation | Author × validation present — Memory/locality | within category | 691 | 2x2 | chi-square | 0.02 | 1.00 | 0.877 | 0.892 | Cramér's V | 0.01 | min expected=50.28; OR=1.03 [0.68, 1.58] |
| A. category×validation | Author × validation present — Network/DB | within category | 166 | 2x2 | chi-square | 0.31 | 1.00 | 0.577 | 0.619 | Cramér's V | 0.04 | min expected=13.37; OR=0.80 [0.36, 1.78] |
| A. category×validation | Category × validation type (validated PRs) | pooled | 1684 | 9x4 | Conditional Pearson chi-square (Monte Carlo, B=20,000) | 49.90 | 24.00 | 0.009 | 0.022 | Cramér's V | 0.10 | min expected=0.06 |
| A. category×validation | Category × validation type (validated PRs) | agentic | 851 | 9x4 | Conditional Pearson chi-square (Monte Carlo, B=20,000) | 31.38 | 24.00 | 0.164 | 0.235 | Cramér's V | 0.11 | min expected=0.01 |
| A. category×validation | Category × validation type (validated PRs) | human_candidate | 833 | 9x4 | Conditional Pearson chi-square (Monte Carlo, B=20,000) | 44.81 | 24.00 | 0.033 | 0.066 | Cramér's V | 0.13 | min expected=0.04 |
| A. category×validation | Category × benchmark-based vs other evidence (validated PRs) | pooled | 1684 | 9x2 | chi-square | 41.78 | 8.00 | <0.001 | <0.001 | Cramér's V | 0.16 | min expected=4.37 |
| B. metric profile | Category × D1 reported | pooled (validated) | 1699 | 9x2 | chi-square | 19.61 | 8.00 | 0.012 | 0.029 | Cramér's V | 0.11 | min expected=3.33 |
| B. metric profile | Category × D2 reported | pooled (validated) | 1699 | 9x2 | Conditional Pearson chi-square (Monte Carlo, B=20,000) | 17.18 | 8.00 | 0.042 | 0.081 | Cramér's V | 0.10 | min expected=0.33 |
| B. metric profile | Category × D3 reported | pooled (validated) | 1699 | 9x2 | chi-square | 40.84 | 8.00 | <0.001 | <0.001 | Cramér's V | 0.15 | min expected=1.62 |
| B. metric profile | Category × D4 reported | pooled (validated) | 1699 | 9x2 | Conditional Pearson chi-square (Monte Carlo, B=20,000) | 6.77 | 8.00 | 0.536 | 0.596 | Cramér's V | 0.06 | min expected=0.27 |
| B. metric profile | Category × D5 reported | pooled (validated) | 1699 | 9x2 | Conditional Pearson chi-square (Monte Carlo, B=20,000) | 19.18 | 8.00 | 0.022 | 0.051 | Cramér's V | 0.11 | min expected=0.40 |
| B. metric profile | Category × D6 reported | pooled (validated) | 1699 | 9x2 | Conditional Pearson chi-square (Monte Carlo, B=20,000) | 24.81 | 8.00 | 0.003 | 0.008 | Cramér's V | 0.12 | min expected=0.81 |
| B. metric profile | Category × D7 reported | pooled (validated) | 1699 | 9x2 | Conditional Pearson chi-square (Monte Carlo, B=20,000) | 16.38 | 8.00 | 0.054 | 0.097 | Cramér's V | 0.10 | min expected=0.15 |
| B. metric profile | Category × D8 reported | pooled (validated) | 1699 | 9x2 | Conditional Pearson chi-square (Monte Carlo, B=20,000) | 1.72 | 8.00 | 0.976 | 0.976 | Cramér's V | 0.03 | min expected=0.04 |
| B. metric profile | Category × D9 reported | pooled (validated) | 1699 | 9x2 | Conditional Pearson chi-square (Monte Carlo, B=20,000) | 60.89 | 8.00 | <0.001 | <0.001 | Cramér's V | 0.19 | min expected=0.47 |
| B. metric profile | Category × D0 reported | pooled (validated) | 1699 | 9x2 | Conditional Pearson chi-square (Monte Carlo, B=20,000) | 7.76 | 8.00 | 0.441 | 0.540 | Cramér's V | 0.07 | min expected=0.46 |
| B. metric profile | Author × D1 reported | validated | 1699 | 2x2 | chi-square | 2.97 | 1.00 | 0.085 | 0.134 | Cramér's V | 0.04 | min expected=310.86; OR=0.84 [0.69, 1.02] |
| B. metric profile | Author × D2 reported | validated | 1699 | 2x2 | chi-square | 0.48 | 1.00 | 0.486 | 0.584 | Cramér's V | 0.02 | min expected=30.69; OR=1.20 [0.72, 1.99] |
| B. metric profile | Author × D3 reported | validated | 1699 | 2x2 | chi-square | 1.30 | 1.00 | 0.254 | 0.346 | Cramér's V | 0.03 | min expected=150.97; OR=0.87 [0.68, 1.11] |
| B. metric profile | Author × D4 reported | validated | 1699 | 2x2 | chi-square | 2.22 | 1.00 | 0.136 | 0.199 | Cramér's V | 0.04 | min expected=25.24; OR=1.54 [0.87, 2.72] |
| B. metric profile | Author × D5 reported | validated | 1699 | 2x2 | chi-square | 3.68 | 1.00 | 0.055 | 0.097 | Cramér's V | 0.05 | min expected=37.12; OR=1.59 [0.99, 2.55] |
| B. metric profile | Author × D6 reported | validated | 1699 | 2x2 | chi-square | 40.91 | 1.00 | <0.001 | <0.001 | Cramér's V | 0.15 | min expected=75.73; OR=3.27 [2.24, 4.78] |
| B. metric profile | Author × D7 reported | validated | 1699 | 2x2 | chi-square | 3.43 | 1.00 | 0.064 | 0.110 | Cramér's V | 0.04 | min expected=13.86; OR=2.09 [0.94, 4.65] |
| B. metric profile | Author × D8 reported | validated | 1699 | 2x2 | Fisher's exact | 1.93 | 1.00 | 0.288 | 0.384 | Cramér's V | 0.03 | min expected=3.96; OR=2.95 [0.59, 14.68] |
| B. metric profile | Author × D9 reported | validated | 1699 | 2x2 | chi-square | 0.04 | 1.00 | 0.837 | 0.866 | Cramér's V | 0.01 | min expected=44.05; OR=0.96 [0.62, 1.47] |
| B. metric profile | Author × D0 reported | validated | 1699 | 2x2 | chi-square | 0.32 | 1.00 | 0.569 | 0.619 | Cramér's V | 0.01 | min expected=42.57; OR=1.13 [0.73, 1.75] |
| B. metric profile | Author × any dimension reported | validated | 1699 | 2x2 | chi-square | 0.39 | 1.00 | 0.535 | 0.596 | Cramér's V | 0.01 | min expected=406.39; OR=1.06 [0.88, 1.28] |
| B. metric profile | Category × any dimension reported | pooled (validated) | 1699 | 9x2 | chi-square | 29.93 | 8.00 | <0.001 | <0.001 | Cramér's V | 0.13 | min expected=4.35 |
| B. metric profile | #dims: agent vs human | validated | 1699 | agent n=858, human n=841 | Mann–Whitney U | 364049.50 |  | 0.728 | 0.766 | Cliff's delta | 0.01 | median agent=1.00 (mean 0.92); median human=1.00 (mean 0.83) |
| B. metric profile | #dims across categories | pooled (validated) | 1699 | k=9 | Kruskal–Wallis | 32.12 | 8.00 | <0.001 | <0.001 | epsilon² | 0.01 | Algorithm md=1.0; Build/Infra md=1.0; Code smells md=0.0; Control-flow md=0.0; Data structure md=1.0; I/O & sync md=1.0; Loop md=1.0; Memory/locality md=1.0; Network/DB md=0.0 |
| B. metric profile | #dims across categories | agentic | 858 | k=9 | Kruskal–Wallis | 17.09 | 8.00 | 0.029 | 0.063 | epsilon² | 0.01 | Algorithm md=0.0; Build/Infra md=1.0; Code smells md=0.0; Control-flow md=0.0; Data structure md=1.0; I/O & sync md=1.0; Loop md=0.0; Memory/locality md=1.0; Network/DB md=0.0 |
| B. metric profile | #dims across categories | human_candidate | 841 | k=9 | Kruskal–Wallis | 27.31 | 8.00 | <0.001 | 0.002 | epsilon² | 0.02 | Algorithm md=1.0; Build/Infra md=1.0; Code smells md=0.0; Control-flow md=0.0; Data structure md=1.0; I/O & sync md=1.0; Loop md=1.5; Memory/locality md=1.0; Network/DB md=0.0 |
| B. metric profile | #dims: benchmark vs other evidence | resolved types | 1684 | benchmark n=867, other n=817 | Mann–Whitney U | 576710.50 |  | <0.001 | <0.001 | Cliff's delta | 0.63 | median benchmark=1.00 (mean 1.43); median other=0.00 (mean 0.29) |
| B. metric profile | Benchmark evidence × any dimension reported | resolved types | 1684 | 2x2 | chi-square | 562.50 | 1.00 | <0.001 | <0.001 | Cramér's V | 0.58 | min expected=393.95; OR=13.99 [11.07, 17.69] |
| B. metric profile | Benchmark evidence × D1 reported | resolved types | 1684 | 2x2 | chi-square | 492.94 | 1.00 | <0.001 | <0.001 | Cramér's V | 0.54 | min expected=301.77; OR=14.80 [11.34, 19.32] |
| B. metric profile | Benchmark evidence × D2 reported | resolved types | 1684 | 2x2 | chi-square | 14.50 | 1.00 | <0.001 | <0.001 | Cramér's V | 0.09 | min expected=29.59; OR=3.00 [1.66, 5.41] |
| B. metric profile | Benchmark evidence × D3 reported | resolved types | 1684 | 2x2 | chi-square | 144.58 | 1.00 | <0.001 | <0.001 | Cramér's V | 0.29 | min expected=147.97; OR=5.91 [4.31, 8.09] |
| B. metric profile | Benchmark evidence × D4 reported | resolved types | 1684 | 2x2 | chi-square | 28.44 | 1.00 | <0.001 | <0.001 | Cramér's V | 0.13 | min expected=24.74; OR=7.40 [3.14, 17.44] |
| B. metric profile | Benchmark evidence × D5 reported | resolved types | 1684 | 2x2 | chi-square | 18.89 | 1.00 | <0.001 | <0.001 | Cramér's V | 0.11 | min expected=36.39; OR=3.12 [1.82, 5.35] |
| B. metric profile | Benchmark evidence × D6 reported | resolved types | 1684 | 2x2 | chi-square | 94.26 | 1.00 | <0.001 | <0.001 | Cramér's V | 0.24 | min expected=74.23; OR=8.76 [5.24, 14.64] |
| B. metric profile | Benchmark evidence × D7 reported | resolved types | 1684 | 2x2 | chi-square | 13.36 | 1.00 | <0.001 | <0.001 | Cramér's V | 0.09 | min expected=13.58; OR=5.79 [2.00, 16.75] |
| B. metric profile | Benchmark evidence × D8 reported | resolved types | 1684 | 2x2 | Fisher's exact | 4.17 | 1.00 | 0.071 | 0.118 | Cramér's V | 0.05 | min expected=3.88; OR=6.64 [0.82, 54.10] |
| B. metric profile | Benchmark evidence × D9 reported | resolved types | 1684 | 2x2 | chi-square | 10.94 | 1.00 | <0.001 | 0.003 | Cramér's V | 0.08 | min expected=43.18; OR=2.13 [1.35, 3.37] |
| B. metric profile | Benchmark evidence × D0 reported | resolved types | 1684 | 2x2 | chi-square | 35.03 | 1.00 | <0.001 | <0.001 | Cramér's V | 0.14 | min expected=41.72; OR=4.77 [2.71, 8.40] |
| C. merge | #dims: merged vs not merged | pooled (validated) | 1699 | merged n=1050, not merged n=649 | Mann–Whitney U | 319983.50 |  | 0.023 | 0.051 | Cliff's delta | -0.06 | median merged=1.00 (mean 0.82); median not merged=1.00 (mean 0.96) |
| C. merge | Merged × any dimension reported | pooled (validated) | 1699 | 2x2 | chi-square | 2.43 | 1.00 | 0.119 | 0.178 | Cramér's V | 0.04 | min expected=313.61; OR=0.86 [0.70, 1.04] |
| C. merge | #dims: merged vs not merged | agentic (validated) | 858 | merged n=451, not merged n=407 | Mann–Whitney U | 84622.50 |  | 0.033 | 0.066 | Cliff's delta | -0.08 | median merged=0.00 (mean 0.84); median not merged=1.00 (mean 1.00) |
| C. merge | Merged × any dimension reported | agentic (validated) | 858 | 2x2 | chi-square | 3.02 | 1.00 | 0.082 | 0.133 | Cramér's V | 0.06 | min expected=199.71; OR=0.79 [0.60, 1.03] |
| C. merge | #dims: merged vs not merged | human_candidate (validated) | 841 | merged n=599, not merged n=242 | Mann–Whitney U | 69775.50 |  | 0.360 | 0.460 | Cliff's delta | -0.04 | median merged=1.00 (mean 0.80); median not merged=1.00 (mean 0.90) |
| C. merge | Merged × any dimension reported | human_candidate (validated) | 841 | 2x2 | chi-square | 0.39 | 1.00 | 0.532 | 0.596 | Cramér's V | 0.02 | min expected=115.10; OR=0.91 [0.67, 1.23] |

**Significant after BH (q < 0.05):**

- **Category × validation present** [pooled], n=2081: chi-square, stat=29.29, p=<0.001, Cramér's V=0.119 (min expected=2.39) → q=<0.001

- **Category × validation present** [agentic], n=1048: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=33.25, p=<0.001, Cramér's V=0.178 (min expected=0.91) → q=<0.001

- **Category × validation present** [human_candidate], n=1033: chi-square, stat=25.08, p=0.002, Cramér's V=0.156 (min expected=1.49) → q=0.004

- **Author × validation present — I/O & sync** [within category], n=215: chi-square, stat=11.99, p=<0.001, Cramér's V=0.236 (min expected=23.11; OR=0.33 [0.17, 0.63]) → q=0.002

- **Category × validation type (validated PRs)** [pooled], n=1684: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=49.90, p=0.009, Cramér's V=0.099 (min expected=0.06) → q=0.022

- **Category × benchmark-based vs other evidence (validated PRs)** [pooled], n=1684: chi-square, stat=41.78, p=<0.001, Cramér's V=0.158 (min expected=4.37) → q=<0.001

- **Category × D1 reported** [pooled (validated)], n=1699: chi-square, stat=19.61, p=0.012, Cramér's V=0.107 (min expected=3.33) → q=0.029

- **Category × D3 reported** [pooled (validated)], n=1699: chi-square, stat=40.84, p=<0.001, Cramér's V=0.155 (min expected=1.62) → q=<0.001

- **Category × D6 reported** [pooled (validated)], n=1699: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=24.81, p=0.003, Cramér's V=0.121 (min expected=0.81) → q=0.008

- **Category × D9 reported** [pooled (validated)], n=1699: Conditional Pearson chi-square (Monte Carlo, B=20,000), stat=60.89, p=<0.001, Cramér's V=0.189 (min expected=0.47) → q=<0.001

- **Author × D6 reported** [validated], n=1699: chi-square, stat=40.91, p=<0.001, Cramér's V=0.155 (min expected=75.73; OR=3.27 [2.24, 4.78]) → q=<0.001

- **Category × any dimension reported** [pooled (validated)], n=1699: chi-square, stat=29.93, p=<0.001, Cramér's V=0.133 (min expected=4.35) → q=<0.001

- **#dims across categories** [pooled (validated)], n=1699: Kruskal–Wallis, stat=32.12, p=<0.001, epsilon²=0.014 (Algorithm md=1.0; Build/Infra md=1.0; Code smells md=0.0; Control-flow md=0.0; Data structure md=1.0; I/O & sync md=1.0; Loop md=1.0; Memory/locality md=1.0; Network/DB md=0.0) → q=<0.001

- **#dims across categories** [human_candidate], n=841: Kruskal–Wallis, stat=27.31, p=<0.001, epsilon²=0.023 (Algorithm md=1.0; Build/Infra md=1.0; Code smells md=0.0; Control-flow md=0.0; Data structure md=1.0; I/O & sync md=1.0; Loop md=1.5; Memory/locality md=1.0; Network/DB md=0.0) → q=0.002

- **#dims: benchmark vs other evidence** [resolved types], n=1684: Mann–Whitney U, stat=576710.50, p=<0.001, Cliff's delta=0.628 (median benchmark=1.00 (mean 1.43); median other=0.00 (mean 0.29)) → q=<0.001

- **Benchmark evidence × any dimension reported** [resolved types], n=1684: chi-square, stat=562.50, p=<0.001, Cramér's V=0.578 (min expected=393.95; OR=13.99 [11.07, 17.69]) → q=<0.001

- **Benchmark evidence × D1 reported** [resolved types], n=1684: chi-square, stat=492.94, p=<0.001, Cramér's V=0.541 (min expected=301.77; OR=14.80 [11.34, 19.32]) → q=<0.001

- **Benchmark evidence × D2 reported** [resolved types], n=1684: chi-square, stat=14.50, p=<0.001, Cramér's V=0.093 (min expected=29.59; OR=3.00 [1.66, 5.41]) → q=<0.001

- **Benchmark evidence × D3 reported** [resolved types], n=1684: chi-square, stat=144.58, p=<0.001, Cramér's V=0.293 (min expected=147.97; OR=5.91 [4.31, 8.09]) → q=<0.001

- **Benchmark evidence × D4 reported** [resolved types], n=1684: chi-square, stat=28.44, p=<0.001, Cramér's V=0.130 (min expected=24.74; OR=7.40 [3.14, 17.44]) → q=<0.001

- **Benchmark evidence × D5 reported** [resolved types], n=1684: chi-square, stat=18.89, p=<0.001, Cramér's V=0.106 (min expected=36.39; OR=3.12 [1.82, 5.35]) → q=<0.001

- **Benchmark evidence × D6 reported** [resolved types], n=1684: chi-square, stat=94.26, p=<0.001, Cramér's V=0.237 (min expected=74.23; OR=8.76 [5.24, 14.64]) → q=<0.001

- **Benchmark evidence × D7 reported** [resolved types], n=1684: chi-square, stat=13.36, p=<0.001, Cramér's V=0.089 (min expected=13.58; OR=5.79 [2.00, 16.75]) → q=<0.001

- **Benchmark evidence × D9 reported** [resolved types], n=1684: chi-square, stat=10.94, p=<0.001, Cramér's V=0.081 (min expected=43.18; OR=2.13 [1.35, 3.37]) → q=0.003

- **Benchmark evidence × D0 reported** [resolved types], n=1684: chi-square, stat=35.03, p=<0.001, Cramér's V=0.144 (min expected=41.72; OR=4.77 [2.71, 8.40]) → q=<0.001

