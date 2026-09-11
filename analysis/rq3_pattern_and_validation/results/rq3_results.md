# RQ3 — Optimization, validation, and reported metrics
Source: `RQ3_metric_targeting/data/rq3_pr_level.csv` (n = 357 PRs; 177 with validation evidence). Rare categories pooled as 'Other' for tests: Loop Transformations.

## 0. Analytic sample

| author_type | PRs (category layer) | with validation evidence (metric layer) |
| --- | --- | --- |
| AI Agent | 280 | 128 |
| Human | 77 | 49 |
| All | 357 | 177 |

Extractor settings: window = 12 tokens; exclusion window = 8 tokens; nearest-cue attribution = True. Categories with n < 10 on the 357 PRs are pooled as 'Other' for inferential tests only.

Validation type (RQ2) by author type, % of the author's PRs (357 PRs):
| validation_type | AI Agent | Human | All |
| --- | --- | --- | --- |
| benchmark | 11.4 | 31.2 | 15.7 |
| profiling | 2.1 | 0.0 | 1.7 |
| static-analysis | 30.7 | 28.6 | 30.3 |
| anecdotal | 1.4 | 3.9 | 2.0 |
| none | 54.3 | 36.4 | 50.4 |
| n | 280 | 77 | 357 |

## 1. Optimization category × validation evidence (n = 357)

### 1.1 Validation presence by category (% of the category's PRs for that author)
| Category | n | AI Agent n | AI Agent validated | Human n | Human validated | All validated |
| --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 114 | 92 | 52.2% | 22 | 86.4% | 58.8% |
| Algorithm-Level Optimizations | 78 | 63 | 36.5% | 15 | 66.7% | 42.3% |
| Code Smells and Structural Simplification | 43 | 33 | 27.3% | 10 | 20.0% | 25.6% |
| Build & Compilation & Infrastructure Optimization | 34 | 28 | 57.1% | 6 | 66.7% | 58.8% |
| I/O and Synchronization | 34 | 22 | 45.5% | 12 | 58.3% | 50.0% |
| Network, Database, and Data Access Optimization | 21 | 15 | 93.3% | 6 | 50.0% | 81.0% |
| Control-Flow and Branching Optimizations | 19 | 15 | 13.3% | 4 | 75.0% | 26.3% |
| Data Structure Selection and Adaptation | 12 | 10 | 60.0% | 2 | 50.0% | 58.3% |
| Loop Transformations | 2 | 2 | 0.0% | 0 | – | 0.0% |

- **Category × validation present** [pooled], n=357: Fisher–Freeman–Halton (Monte Carlo, B=20,000), stat=31.29, p=<0.001, Cramér's V=0.296 (min expected=0.99)

- **Category × validation present** [AI Agent], n=280: Fisher–Freeman–Halton (Monte Carlo, B=20,000), stat=32.25, p=<0.001, Cramér's V=0.339 (min expected=0.91)

- **Category × validation present** [Human], n=77: Fisher–Freeman–Halton (Monte Carlo, B=20,000), stat=14.23, p=0.037, Cramér's V=0.430 (min expected=0.73)

Agent vs human validation rate within each category (Fisher's exact, 2×2):

- **Author × validation present — Algorithm** [within category], n=78: chi-square, stat=4.51, p=0.034, Cramér's V=0.241 (min expected=6.35; OR=0.29 [0.09, 0.94])

- **Author × validation present — Build/Infra** [within category], n=34: Fisher's exact, stat=0.19, p=1.000, Cramér's V=0.074 (min expected=2.47; OR=0.67 [0.10, 4.26])

- **Author × validation present — Code smells** [within category], n=43: Fisher's exact, stat=0.21, p=1.000, Cramér's V=0.070 (min expected=2.56; OR=1.50 [0.27, 8.45])

- **Author × validation present — Control-flow** [within category], n=19: Fisher's exact, stat=6.19, p=0.037, Cramér's V=0.571 (min expected=1.05; OR=0.05 [0.00, 0.77])

- **Author × validation present — Data structure** [within category], n=12: Fisher's exact, stat=0.07, p=1.000, Cramér's V=0.076 (min expected=0.83; OR=1.50 [0.07, 31.58])

- **Author × validation present — I/O & sync** [within category], n=34: chi-square, stat=0.52, p=0.473, Cramér's V=0.123 (min expected=6.00; OR=0.60 [0.14, 2.47])

- **Author × validation present — Memory/locality** [within category], n=114: chi-square, stat=8.57, p=0.003, Cramér's V=0.274 (min expected=9.07; OR=0.17 [0.05, 0.62])

- **Author × validation present — Network/DB** [within category], n=21: Fisher's exact, stat=5.22, p=0.053, Cramér's V=0.499 (min expected=1.14; OR=14.00 [1.06, 185.50])

| Category | Agent validated | Human validated | test | p (raw) | note |
| --- | --- | --- | --- | --- | --- |
| Algorithm | 36.5% | 66.7% | chi-square | 0.034 | min expected=6.35; OR=0.29 [0.09, 0.94] |
| Build/Infra | 57.1% | 66.7% | Fisher's exact | 1.000 | min expected=2.47; OR=0.67 [0.10, 4.26] |
| Code smells | 27.3% | 20.0% | Fisher's exact | 1.000 | min expected=2.56; OR=1.50 [0.27, 8.45] |
| Control-flow | 13.3% | 75.0% | Fisher's exact | 0.037 | min expected=1.05; OR=0.05 [0.00, 0.77] |
| Data structure | 60.0% | 50.0% | Fisher's exact | 1.000 | min expected=0.83; OR=1.50 [0.07, 31.58] |
| I/O & sync | 45.5% | 58.3% | chi-square | 0.473 | min expected=6.00; OR=0.60 [0.14, 2.47] |
| Memory/locality | 52.2% | 86.4% | chi-square | 0.003 | min expected=9.07; OR=0.17 [0.05, 0.62] |
| Network/DB | 93.3% | 50.0% | Fisher's exact | 0.053 | min expected=1.14; OR=14.00 [1.06, 185.50] |
| Other | 0.0% | – | n/a (degenerate table) |  |  |

### 1.2 Validation type by category (% of the category's PRs; 'none' = no validation evidence)

**AI Agent**
| pattern | benchmark | profiling | static-analysis | anecdotal | none | n |
| --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 7.6 | 5.4 | 38.0 | 1.1 | 47.8 | 92 |
| Algorithm-Level Optimizations | 15.9 | 1.6 | 17.5 | 1.6 | 63.5 | 63 |
| Code Smells and Structural Simplification | 6.1 | 0.0 | 21.2 | 0.0 | 72.7 | 33 |
| Build & Compilation & Infrastructure Optimization | 14.3 | 0.0 | 39.3 | 3.6 | 42.9 | 28 |
| I/O and Synchronization | 18.2 | 0.0 | 27.3 | 0.0 | 54.5 | 22 |
| Control-Flow and Branching Optimizations | 6.7 | 0.0 | 6.7 | 0.0 | 86.7 | 15 |
| Network, Database, and Data Access Optimization | 6.7 | 0.0 | 80.0 | 6.7 | 6.7 | 15 |
| Data Structure Selection and Adaptation | 30.0 | 0.0 | 30.0 | 0.0 | 40.0 | 10 |
| Loop Transformations | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 2 |

**Human**
| pattern | benchmark | profiling | static-analysis | anecdotal | none | n |
| --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 36.4 | 0.0 | 40.9 | 9.1 | 13.6 | 22 |
| Algorithm-Level Optimizations | 46.7 | 0.0 | 20.0 | 0.0 | 33.3 | 15 |
| I/O and Synchronization | 16.7 | 0.0 | 41.7 | 0.0 | 41.7 | 12 |
| Code Smells and Structural Simplification | 10.0 | 0.0 | 10.0 | 0.0 | 80.0 | 10 |
| Build & Compilation & Infrastructure Optimization | 33.3 | 0.0 | 33.3 | 0.0 | 33.3 | 6 |
| Network, Database, and Data Access Optimization | 16.7 | 0.0 | 16.7 | 16.7 | 50.0 | 6 |
| Control-Flow and Branching Optimizations | 50.0 | 0.0 | 25.0 | 0.0 | 25.0 | 4 |
| Data Structure Selection and Adaptation | 50.0 | 0.0 | 0.0 | 0.0 | 50.0 | 2 |

**All**
| pattern | benchmark | profiling | static-analysis | anecdotal | none | n |
| --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 13.2 | 4.4 | 38.6 | 2.6 | 41.2 | 114 |
| Algorithm-Level Optimizations | 21.8 | 1.3 | 17.9 | 1.3 | 57.7 | 78 |
| Code Smells and Structural Simplification | 7.0 | 0.0 | 18.6 | 0.0 | 74.4 | 43 |
| Build & Compilation & Infrastructure Optimization | 17.6 | 0.0 | 38.2 | 2.9 | 41.2 | 34 |
| I/O and Synchronization | 17.6 | 0.0 | 32.4 | 0.0 | 50.0 | 34 |
| Network, Database, and Data Access Optimization | 9.5 | 0.0 | 61.9 | 9.5 | 19.0 | 21 |
| Control-Flow and Branching Optimizations | 15.8 | 0.0 | 10.5 | 0.0 | 73.7 | 19 |
| Data Structure Selection and Adaptation | 33.3 | 0.0 | 25.0 | 0.0 | 41.7 | 12 |
| Loop Transformations | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 2 |

- **Category × validation type (validated PRs)** [pooled], n=177: Fisher–Freeman–Halton (Monte Carlo, B=20,000), stat=25.05, p=0.242, Cramér's V=0.217 (min expected=0.17)

- **Category × validation type (validated PRs)** [AI Agent], n=128: Fisher–Freeman–Halton (Monte Carlo, B=20,000), stat=21.14, p=0.425, Cramér's V=0.235 (min expected=0.06)

- **Category × validation type (validated PRs)** [Human], n=49: Fisher–Freeman–Halton (Monte Carlo, B=20,000), stat=10.72, p=0.717, Cramér's V=0.331 (min expected=0.06)

- **Category × benchmark-based vs other evidence (validated PRs)** [pooled], n=177: Fisher–Freeman–Halton (Monte Carlo, B=20,000), stat=15.97, p=0.024, Cramér's V=0.300 (min expected=1.58)

## 2. Metric profile: category × reported dimensions (validated PRs, n = 177)

Dimensions: **D1** Latency / execution time; **D2** Throughput; **D3** Memory; **D4** CPU / compute work; **D5** I/O and network; **D6** Artifact size; **D7** Build and CI time; **D8** Energy and cost; **D9** Scalability / concurrency; **D0** Unspecified performance. D0 records a quantified gain whose dimension is not named and is credited only when no D1–D9 cue is in the claim's window.

**AI Agent (n = 128)** — % of the category's validated PRs reporting each dimension
| Category | n validated | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D0 | ≥1 dim | mean #dims |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 48 | 29% | 0% | 4% | 10% | 2% | 0% | 0% | 0% | 0% | 2% | 33.3% | 0.48 |
| Algorithm-Level Optimizations | 23 | 52% | 4% | 17% | 0% | 0% | 0% | 0% | 0% | 4% | 0% | 56.5% | 0.78 |
| Build & Compilation & Infrastructure Optimization | 16 | 25% | 12% | 6% | 0% | 0% | 19% | 12% | 0% | 6% | 0% | 43.8% | 0.81 |
| Network, Database, and Data Access Optimization | 14 | 7% | 0% | 7% | 0% | 21% | 0% | 0% | 0% | 0% | 0% | 28.6% | 0.36 |
| I/O and Synchronization | 10 | 30% | 10% | 20% | 0% | 0% | 0% | 10% | 10% | 10% | 20% | 50.0% | 1.10 |
| Code Smells and Structural Simplification | 9 | 44% | 0% | 0% | 0% | 0% | 0% | 0% | 11% | 0% | 11% | 44.4% | 0.67 |
| Data Structure Selection and Adaptation | 6 | 17% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 16.7% | 0.17 |
| Control-Flow and Branching Optimizations | 2 | 50% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 50.0% | 0.50 |
| All | 128 | 31% | 3% | 8% | 4% | 3% | 2% | 2% | 2% | 2% | 3% | 39.8% | 0.61 |

**Human (n = 49)** — % of the category's validated PRs reporting each dimension
| Category | n validated | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D0 | ≥1 dim | mean #dims |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 19 | 37% | 5% | 21% | 0% | 5% | 0% | 0% | 0% | 0% | 0% | 52.6% | 0.68 |
| Algorithm-Level Optimizations | 10 | 70% | 0% | 10% | 0% | 10% | 0% | 0% | 0% | 0% | 20% | 70.0% | 1.10 |
| I/O and Synchronization | 7 | 43% | 0% | 0% | 0% | 0% | 14% | 0% | 0% | 0% | 0% | 42.9% | 0.57 |
| Build & Compilation & Infrastructure Optimization | 4 | 25% | 0% | 0% | 0% | 0% | 0% | 25% | 0% | 0% | 25% | 50.0% | 0.75 |
| Control-Flow and Branching Optimizations | 3 | 33% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 33.3% | 0.33 |
| Network, Database, and Data Access Optimization | 3 | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 33% | 33% | 33.3% | 0.67 |
| Code Smells and Structural Simplification | 2 | 50% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 50.0% | 0.50 |
| Data Structure Selection and Adaptation | 1 | 100% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 100.0% | 1.00 |
| All | 49 | 43% | 2% | 10% | 0% | 4% | 2% | 2% | 0% | 2% | 8% | 53.1% | 0.73 |

**All (n = 177)** — % of the category's validated PRs reporting each dimension
| Category | n validated | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D0 | ≥1 dim | mean #dims |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Memory and Data Locality Optimizations | 67 | 31% | 1% | 9% | 7% | 3% | 0% | 0% | 0% | 0% | 1% | 38.8% | 0.54 |
| Algorithm-Level Optimizations | 33 | 58% | 3% | 15% | 0% | 3% | 0% | 0% | 0% | 3% | 6% | 60.6% | 0.88 |
| Build & Compilation & Infrastructure Optimization | 20 | 25% | 10% | 5% | 0% | 0% | 15% | 15% | 0% | 5% | 5% | 45.0% | 0.80 |
| I/O and Synchronization | 17 | 35% | 6% | 12% | 0% | 0% | 6% | 6% | 6% | 6% | 12% | 47.1% | 0.88 |
| Network, Database, and Data Access Optimization | 17 | 6% | 0% | 6% | 0% | 18% | 0% | 0% | 0% | 6% | 6% | 29.4% | 0.41 |
| Code Smells and Structural Simplification | 11 | 45% | 0% | 0% | 0% | 0% | 0% | 0% | 9% | 0% | 9% | 45.5% | 0.64 |
| Data Structure Selection and Adaptation | 7 | 29% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 28.6% | 0.29 |
| Control-Flow and Branching Optimizations | 5 | 40% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 40.0% | 0.40 |
| All | 177 | 34% | 3% | 8% | 3% | 3% | 2% | 2% | 1% | 2% | 5% | 43.5% | 0.64 |

### 2.2 Tests on the metric profile

- **Category × D1 reported** [pooled (validated)], n=177: Fisher–Freeman–Halton (Monte Carlo, B=20,000), stat=15.80, p=0.025, Cramér's V=0.299 (min expected=1.72)

- **Category × D3 reported** [pooled (validated)], n=177: Fisher–Freeman–Halton (Monte Carlo, B=20,000), stat=4.74, p=0.698, Cramér's V=0.164 (min expected=0.42)

- **Category × D0 reported** [pooled (validated)], n=177: Fisher–Freeman–Halton (Monte Carlo, B=20,000), stat=4.86, p=0.654, Cramér's V=0.166 (min expected=0.23)

- **Author × D1 reported** [validated], n=177: chi-square, stat=2.11, p=0.146, Cramér's V=0.109 (min expected=16.89; OR=0.61 [0.31, 1.19])

- **Author × D2 reported** [validated], n=177: Fisher's exact, stat=0.15, p=1.000, Cramér's V=0.029 (min expected=1.38; OR=1.55 [0.17, 14.21])

- **Author × D3 reported** [validated], n=177: Fisher's exact, stat=0.26, p=0.563, Cramér's V=0.038 (min expected=4.15; OR=0.75 [0.24, 2.30])

- **Author × D4 reported** [validated], n=177: Fisher's exact, stat=1.97, p=0.324, Cramér's V=0.105 (min expected=1.38; OR=4.41 [0.24, 81.24])

- **Author × D5 reported** [validated], n=177: Fisher's exact, stat=0.10, p=0.669, Cramér's V=0.024 (min expected=1.66; OR=0.76 [0.13, 4.28])

- **Author × D0 reported** [validated], n=177: Fisher's exact, stat=2.08, p=0.219, Cramér's V=0.109 (min expected=2.21; OR=0.36 [0.09, 1.51])

- **Author × any dimension reported** [validated], n=177: chi-square, stat=2.52, p=0.112, Cramér's V=0.119 (min expected=21.32; OR=1.71 [0.88, 3.31])

- **Category × any dimension reported** [pooled (validated)], n=177: Fisher–Freeman–Halton (Monte Carlo, B=20,000), stat=6.69, p=0.476, Cramér's V=0.194 (min expected=2.18)

### 2.3 Dimensionality of the evidence (number of distinct dimensions per validated PR)

% of the author's validated PRs (counting D0):
| n_dims | AI Agent | Human | All |
| --- | --- | --- | --- |
| 0 | 60.2 | 46.9 | 56.5 |
| 1 | 23.4 | 32.7 | 26.0 |
| 2 | 14.1 | 20.4 | 15.8 |
| 3 | 0.8 | 0.0 | 0.6 |
| 4 | 0.8 | 0.0 | 0.6 |
| 5 | 0.8 | 0.0 | 0.6 |
| n | 128 | 49 | 177 |

PRs whose only reported 'dimension' is **D0** (a quantified gain with no named dimension): AI Agent: 0.8%, Human: 0.0%; all: 0.6%. Excluding D0, the share of validated PRs with ≥1 named dimension is AI Agent: 39.1%, Human: 53.1%; all: 42.9%.

- **#dims: agent vs human** [validated], n=177: Mann–Whitney U, stat=2754.50, p=0.162, Cliff's delta=-0.122 (median agent=0.00 (mean 0.61); median human=1.00 (mean 0.73))

- **#dims across categories** [pooled (validated)], n=177: Kruskal–Wallis, stat=7.53, p=0.376, epsilon²=0.003 (Algorithm md=1.0; Build/Infra md=0.0; Code smells md=0.0; Control-flow md=0.0; Data structure md=0.0; I/O & sync md=0.0; Memory/locality md=0.0; Network/DB md=0.0)

- **#dims across categories** [AI Agent], n=128: Kruskal–Wallis, stat=6.53, p=0.479, epsilon²=-0.004 (Algorithm md=1.0; Build/Infra md=0.0; Code smells md=0.0; Control-flow md=0.5; Data structure md=0.0; I/O & sync md=0.5; Memory/locality md=0.0; Network/DB md=0.0)

- **#dims across categories** [Human], n=48: Kruskal–Wallis, stat=3.19, p=0.785, epsilon²=-0.069 (Algorithm md=1.0; Build/Infra md=0.5; Code smells md=0.5; Control-flow md=0.0; I/O & sync md=0.0; Memory/locality md=1.0; Network/DB md=0.0)

Per category and author:
| pattern | author_type | count | mean | median | max |
| --- | --- | --- | --- | --- | --- |
| Algorithm-Level Optimizations | AI Agent | 23 | 0.78 | 1.00 | 2 |
| Algorithm-Level Optimizations | Human | 10 | 1.10 | 1.00 | 2 |
| Build & Compilation & Infrastructure Optimization | AI Agent | 16 | 0.81 | 0.00 | 4 |
| Build & Compilation & Infrastructure Optimization | Human | 4 | 0.75 | 0.50 | 2 |
| Code Smells and Structural Simplification | AI Agent | 9 | 0.67 | 0.00 | 2 |
| Code Smells and Structural Simplification | Human | 2 | 0.50 | 0.50 | 1 |
| Control-Flow and Branching Optimizations | AI Agent | 2 | 0.50 | 0.50 | 1 |
| Control-Flow and Branching Optimizations | Human | 3 | 0.33 | 0.00 | 1 |
| Data Structure Selection and Adaptation | AI Agent | 6 | 0.17 | 0.00 | 1 |
| Data Structure Selection and Adaptation | Human | 1 | 1.00 | 1.00 | 1 |
| I/O and Synchronization | AI Agent | 10 | 1.10 | 0.50 | 5 |
| I/O and Synchronization | Human | 7 | 0.57 | 0.00 | 2 |
| Memory and Data Locality Optimizations | AI Agent | 48 | 0.48 | 0.00 | 2 |
| Memory and Data Locality Optimizations | Human | 19 | 0.68 | 1.00 | 2 |
| Network, Database, and Data Access Optimization | AI Agent | 14 | 0.36 | 0.00 | 2 |
| Network, Database, and Data Access Optimization | Human | 3 | 0.67 | 0.00 | 2 |

### 2.4 Metric profile × validation type (validated PRs)

% of PRs with that validation type reporting each dimension:
| Validation type | n | ≥1 dim | mean #dims | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| benchmark | 56 | 83.9% | 1.29 | 73% | 7% | 14% | 0% | 4% | 5% | 4% | 4% | 5% | 12% |
| profiling | 6 | 83.3% | 1.67 | 83% | 0% | 0% | 83% | 0% | 0% | 0% | 0% | 0% | 0% |
| static-analysis | 108 | 21.3% | 0.28 | 13% | 1% | 6% | 0% | 4% | 1% | 1% | 0% | 1% | 1% |
| anecdotal | 7 | 28.6% | 0.29 | 14% | 0% | 0% | 0% | 0% | 0% | 14% | 0% | 0% | 0% |

- **#dims: benchmark vs other evidence** [validated], n=177: Mann–Whitney U, stat=5471.00, p=<0.001, Cliff's delta=0.615 (median benchmark=1.00 (mean 1.29); median other=0.00 (mean 0.35))

- **Benchmark evidence × any dimension reported** [validated], n=177: chi-square, stat=54.47, p=<0.001, Cramér's V=0.555 (min expected=24.36; OR=15.84 [6.95, 36.11])

- **Benchmark evidence × D1 reported** [validated], n=177: chi-square, stat=54.46, p=<0.001, Cramér's V=0.555 (min expected=19.30; OR=13.80 [6.45, 29.56])

- **Benchmark evidence × D2 reported** [validated], n=177: Fisher's exact, stat=5.56, p=0.035, Cramér's V=0.177 (min expected=1.58; OR=9.23 [1.01, 84.60])

- **Benchmark evidence × D3 reported** [validated], n=177: Fisher's exact, stat=3.57, p=0.080, Cramér's V=0.142 (min expected=4.75; OR=2.71 [0.93, 7.91])

- **Benchmark evidence × D4 reported** [validated], n=177: Fisher's exact, stat=2.38, p=0.181, Cramér's V=0.116 (min expected=1.58; OR=0.19 [0.01, 3.45])

- **Benchmark evidence × D5 reported** [validated], n=177: Fisher's exact, stat=0.01, p=1.000, Cramér's V=0.007 (min expected=1.90; OR=1.08 [0.19, 6.10])

- **Benchmark evidence × D0 reported** [validated], n=177: Fisher's exact, stat=12.09, p=0.001, Cramér's V=0.261 (min expected=2.53; OR=17.14 [2.05, 143.04])

### 2.5 Metric reporting by author type and by agent

Rates are % of the row's PRs. *all PRs* = the category layer; *validated* = PRs with RQ2 validation evidence; *benchmark* = PRs whose evidence is a benchmark; *no diff* = dimension flags recomputed with the code diff excluded. The *all PRs* quantification columns count only PRs that are validated **and** report a dimension, i.e. they are the product of the validation rate and the conditional quantification rate; PRs with a quantitative claim but no RQ2 validation label (`extremes_metric_without_validation_label.csv`) are not counted. Per-agent rows are descriptive and are not part of the RQ3 test family.

| author | n (all PRs) | validated | benchmark evidence | validated & ≥1 dim, all PRs | validated & ≥1 dim, all PRs (no diff) | n validated | ≥1 dim, validated | mean #dims, validated | n benchmark | ≥1 dim, benchmark |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Human | 77 | 63.6% | 31.2% | 33.8% | 32.5% | 49 | 53.1% | 0.73 | 24 | 95.8% |
| AI Agent (all) | 280 | 45.7% | 11.4% | 18.2% | 14.3% | 128 | 39.8% | 0.61 | 32 | 75.0% |
|   OpenAI_Codex | 164 | 18.9% | 6.7% | 4.9% | 1.8% | 31 | 25.8% | 0.42 | 11 | 54.5% |
|   Devin | 57 | 80.7% | 14.0% | 36.8% | 35.1% | 46 | 45.7% | 0.70 | 8 | 87.5% |
|   Copilot | 36 | 91.7% | 25.0% | 27.8% | 25.0% | 33 | 30.3% | 0.36 | 9 | 77.8% |
|   Cursor | 20 | 75.0% | 10.0% | 50.0% | 30.0% | 15 | 66.7% | 1.27 | 2 | 100.0% |
|   Claude_Code | 3 | 100.0% | 66.7% | 66.7% | 66.7% | 3 | 66.7% | 0.67 | 2 | 100.0% |

## 3. Merge status

- **#dims: merged vs not merged** [pooled (validated)], n=177: Mann–Whitney U, stat=3093.50, p=0.015, Cliff's delta=-0.191 (median merged=0.00 (mean 0.43); median not merged=0.00 (mean 0.80))

- **Merged × any dimension reported** [pooled (validated)], n=177: chi-square, stat=2.98, p=0.084, Cramér's V=0.130 (min expected=32.63; OR=0.59 [0.32, 1.08])

- **#dims: merged vs not merged** [AI Agent (validated)], n=128: Mann–Whitney U, stat=1502.00, p=0.024, Cliff's delta=-0.211 (median merged=0.00 (mean 0.34); median not merged=0.00 (mean 0.77))

- **Merged × any dimension reported** [AI Agent (validated)], n=128: chi-square, stat=3.13, p=0.077, Cramér's V=0.156 (min expected=18.73; OR=0.50 [0.24, 1.08])

- **#dims: merged vs not merged** [Human (validated)], n=49: Mann–Whitney U, stat=222.50, p=0.121, Cliff's delta=-0.243 (median merged=0.00 (mean 0.57); median not merged=1.00 (mean 0.95))

- **Merged × any dimension reported** [Human (validated)], n=49: chi-square, stat=1.15, p=0.283, Cramér's V=0.153 (min expected=9.86; OR=0.53 [0.17, 1.69])

| Author | Merged | n | ≥1 dim | mean #dims | median #dims | median time-to-merge (days) |
| --- | --- | --- | --- | --- | --- | --- |
| AI Agent | False | 81 | 45.7% | 0.77 | 0 |  |
| AI Agent | True | 47 | 29.8% | 0.34 | 0 | 0.09 |
| Human | False | 21 | 61.9% | 0.95 | 1 |  |
| Human | True | 28 | 46.4% | 0.57 | 0 | 0.28 |

## 4. PRs at the extremes (exported for qualitative reading)

Highest dimensionality (top 15) → `results/extremes_top_dimensionality.csv`
| html_url | author_type | pattern | validation_type | n_dims | dims |
| --- | --- | --- | --- | --- | --- |
| https://github.com/nnstreamer/nntrainer/pull/3312 | AI Agent | I/O and Synchronization | benchmark | 5 | D1|D2|D3|D8|D0 |
| https://github.com/gmathi/NovelLibrary/pull/246 | AI Agent | Build & Compilation & Infrastructure Optimization | static-analysis | 4 | D1|D2|D6|D9 |
| https://github.com/test-zeus-ai/testzeus-hercules/pull/61 | AI Agent | I/O and Synchronization | benchmark | 3 | D1|D9|D0 |
| https://github.com/yamadashy/repomix/pull/309 | Human | I/O and Synchronization | benchmark | 2 | D1|D6 |
| https://github.com/saturday06/VRM-Addon-for-Blender/pull/797 | AI Agent | Memory and Data Locality Optimizations | profiling | 2 | D1|D4 |
| https://github.com/oven-sh/bun/pull/18585 | Human | Memory and Data Locality Optimizations | benchmark | 2 | D1|D2 |
| https://github.com/saturday06/VRM-Addon-for-Blender/pull/964 | AI Agent | Code Smells and Structural Simplification | benchmark | 2 | D1|D0 |
| https://github.com/sourcebot-dev/sourcebot/pull/357 | AI Agent | Network, Database, and Data Access Optimization | static-analysis | 2 | D1|D5 |
| https://github.com/buger/probe/pull/56 | AI Agent | Memory and Data Locality Optimizations | static-analysis | 2 | D1|D3 |
| https://github.com/saturday06/VRM-Addon-for-Blender/pull/800 | AI Agent | Memory and Data Locality Optimizations | profiling | 2 | D1|D4 |
| https://github.com/mochilang/mochi/pull/13059 | AI Agent | Algorithm-Level Optimizations | benchmark | 2 | D1|D3 |
| https://github.com/saturday06/VRM-Addon-for-Blender/pull/798 | AI Agent | Memory and Data Locality Optimizations | profiling | 2 | D1|D4 |
| https://github.com/saturday06/VRM-Addon-for-Blender/pull/796 | AI Agent | Memory and Data Locality Optimizations | profiling | 2 | D1|D4 |
| https://github.com/onnx/onnx/pull/7057 | AI Agent | Algorithm-Level Optimizations | benchmark | 2 | D1|D2 |
| https://github.com/saturday06/VRM-Addon-for-Blender/pull/795 | AI Agent | Memory and Data Locality Optimizations | profiling | 2 | D1|D4 |

Validated PRs reporting **no** metric dimension (56.5% overall), as % of each validation type → `results/extremes_validated_no_metric.csv`
| validation_type | AI Agent | Human | n (validated PRs) |
| --- | --- | --- | --- |
| anecdotal | 50.0 | 100.0 | 7 |
| benchmark | 25.0 | 4.2 | 56 |
| profiling | 16.7 |  | 6 |
| static-analysis | 76.7 | 86.4 | 108 |

PRs *without* an RQ2 validation label in which the extractor still found a quantitative metric claim (n = 9; candidates for an RQ2 label recheck) → `results/extremes_metric_without_validation_label.csv`
| html_url | author_type | pattern | n_dims | dims |
| --- | --- | --- | --- | --- |
| https://github.com/microsoft/HydraLab/pull/695 | AI Agent | I/O and Synchronization | 1 | D1 |
| https://github.com/mochilang/mochi/pull/3943 | AI Agent | Algorithm-Level Optimizations | 1 | D1 |
| https://github.com/Stirling-Tools/Stirling-PDF/pull/3992 | AI Agent | I/O and Synchronization | 1 | D1 |
| https://github.com/mochilang/mochi/pull/6522 | AI Agent | Code Smells and Structural Simplification | 1 | D8 |
| https://github.com/MontrealAI/AGI-Alpha-Agent-v0/pull/3666 | AI Agent | I/O and Synchronization | 1 | D1 |
| https://github.com/lunasaw/gb28181-proxy/pull/38 | AI Agent | Memory and Data Locality Optimizations | 3 | D1|D2|D3 |
| https://github.com/onflow/flow-go/pull/7598 | AI Agent | Algorithm-Level Optimizations | 1 | D1 |
| https://github.com/getsentry/sentry/pull/87963 | Human | Data Structure Selection and Adaptation | 1 | D1 |
| https://github.com/gofiber/fiber/pull/3532 | Human | Code Smells and Structural Simplification | 1 | D1 |

## 5. Sensitivity: excluding the code diff from the corpus

% of the author's validated PRs:
| Author | n | Dimension | with diff | without diff |
| --- | --- | --- | --- | --- |
| AI Agent | 128 | D1 | 31.2% | 25.0% |
| AI Agent | 128 | D2 | 3.1% | 2.3% |
| AI Agent | 128 | D3 | 7.8% | 3.9% |
| AI Agent | 128 | D4 | 3.9% | 3.9% |
| AI Agent | 128 | D5 | 3.1% | 1.6% |
| AI Agent | 128 | D6 | 2.3% | 1.6% |
| AI Agent | 128 | D7 | 2.3% | 2.3% |
| AI Agent | 128 | D8 | 1.6% | 0.8% |
| AI Agent | 128 | D9 | 2.3% | 1.6% |
| AI Agent | 128 | D0 | 3.1% | 3.1% |
| AI Agent | 128 | ≥1 dim | 39.8% | 31.2% |
| Human | 49 | D1 | 42.9% | 42.9% |
| Human | 49 | D2 | 2.0% | 2.0% |
| Human | 49 | D3 | 10.2% | 10.2% |
| Human | 49 | D4 | 0.0% | 0.0% |
| Human | 49 | D5 | 4.1% | 2.0% |
| Human | 49 | D6 | 2.0% | 0.0% |
| Human | 49 | D7 | 2.0% | 2.0% |
| Human | 49 | D8 | 0.0% | 0.0% |
| Human | 49 | D9 | 2.0% | 2.0% |
| Human | 49 | D0 | 8.2% | 8.2% |
| Human | 49 | ≥1 dim | 53.1% | 51.0% |
| All | 177 | D1 | 34.5% | 29.9% |
| All | 177 | D2 | 2.8% | 2.3% |
| All | 177 | D3 | 8.5% | 5.6% |
| All | 177 | D4 | 2.8% | 2.8% |
| All | 177 | D5 | 3.4% | 1.7% |
| All | 177 | D6 | 2.3% | 1.1% |
| All | 177 | D7 | 2.3% | 2.3% |
| All | 177 | D8 | 1.1% | 0.6% |
| All | 177 | D9 | 2.3% | 1.7% |
| All | 177 | D0 | 4.5% | 4.5% |
| All | 177 | ≥1 dim | 43.5% | 36.7% |

PRs whose dimension count changes when the diff is excluded: 14 of 177.

Dimensions found in the description alone: 56 PRs have ≥1; bot-authored issue comments contribute a dimension in 6 PRs.

## 6. RQ3 test family (Benjamini–Hochberg across all 44 tests)

| family | test_label | stratum | n | shape | test | statistic | dof | p_raw | p_bh | effect_name | effect | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A. category×validation | Category × validation present | pooled | 357 | 9x2 | Fisher–Freeman–Halton (Monte Carlo, B=20,000) | 31.29 | 8.00 | <0.001 | <0.001 | Cramér's V | 0.30 | min expected=0.99 |
| A. category×validation | Category × validation present | AI Agent | 280 | 9x2 | Fisher–Freeman–Halton (Monte Carlo, B=20,000) | 32.25 | 8.00 | <0.001 | <0.001 | Cramér's V | 0.34 | min expected=0.91 |
| A. category×validation | Category × validation present | Human | 77 | 8x2 | Fisher–Freeman–Halton (Monte Carlo, B=20,000) | 14.23 | 7.00 | 0.037 | 0.110 | Cramér's V | 0.43 | min expected=0.73 |
| A. category×validation | Author × validation present — Algorithm | within category | 78 | 2x2 | chi-square | 4.51 | 1.00 | 0.034 | 0.110 | Cramér's V | 0.24 | min expected=6.35; OR=0.29 [0.09, 0.94] |
| A. category×validation | Author × validation present — Build/Infra | within category | 34 | 2x2 | Fisher's exact | 0.19 | 1.00 | 1.000 | 1.000 | Cramér's V | 0.07 | min expected=2.47; OR=0.67 [0.10, 4.26] |
| A. category×validation | Author × validation present — Code smells | within category | 43 | 2x2 | Fisher's exact | 0.21 | 1.00 | 1.000 | 1.000 | Cramér's V | 0.07 | min expected=2.56; OR=1.50 [0.27, 8.45] |
| A. category×validation | Author × validation present — Control-flow | within category | 19 | 2x2 | Fisher's exact | 6.19 | 1.00 | 0.037 | 0.110 | Cramér's V | 0.57 | min expected=1.05; OR=0.05 [0.00, 0.77] |
| A. category×validation | Author × validation present — Data structure | within category | 12 | 2x2 | Fisher's exact | 0.07 | 1.00 | 1.000 | 1.000 | Cramér's V | 0.08 | min expected=0.83; OR=1.50 [0.07, 31.58] |
| A. category×validation | Author × validation present — I/O & sync | within category | 34 | 2x2 | chi-square | 0.52 | 1.00 | 0.473 | 0.639 | Cramér's V | 0.12 | min expected=6.00; OR=0.60 [0.14, 2.47] |
| A. category×validation | Author × validation present — Memory/locality | within category | 114 | 2x2 | chi-square | 8.57 | 1.00 | 0.003 | 0.022 | Cramér's V | 0.27 | min expected=9.07; OR=0.17 [0.05, 0.62] |
| A. category×validation | Author × validation present — Network/DB | within category | 21 | 2x2 | Fisher's exact | 5.22 | 1.00 | 0.053 | 0.145 | Cramér's V | 0.50 | min expected=1.14; OR=14.00 [1.06, 185.50] |
| A. category×validation | Author × validation present — Other | within category | 2 | 1x1 | n/a (degenerate table) |  |  |  |  | Cramér's V |  |  |
| A. category×validation | Category × validation type (validated PRs) | pooled | 177 | 8x4 | Fisher–Freeman–Halton (Monte Carlo, B=20,000) | 25.05 | 21.00 | 0.242 | 0.409 | Cramér's V | 0.22 | min expected=0.17 |
| A. category×validation | Category × validation type (validated PRs) | AI Agent | 128 | 8x4 | Fisher–Freeman–Halton (Monte Carlo, B=20,000) | 21.14 | 21.00 | 0.425 | 0.623 | Cramér's V | 0.23 | min expected=0.06 |
| A. category×validation | Category × validation type (validated PRs) | Human | 49 | 8x3 | Fisher–Freeman–Halton (Monte Carlo, B=20,000) | 10.72 | 14.00 | 0.717 | 0.830 | Cramér's V | 0.33 | min expected=0.06 |
| A. category×validation | Category × benchmark-based vs other evidence (validated PRs) | pooled | 177 | 8x2 | Fisher–Freeman–Halton (Monte Carlo, B=20,000) | 15.97 | 7.00 | 0.024 | 0.100 | Cramér's V | 0.30 | min expected=1.58 |
| B. metric profile | Category × D1 reported | pooled (validated) | 177 | 8x2 | Fisher–Freeman–Halton (Monte Carlo, B=20,000) | 15.80 | 7.00 | 0.025 | 0.100 | Cramér's V | 0.30 | min expected=1.72 |
| B. metric profile | Category × D3 reported | pooled (validated) | 177 | 8x2 | Fisher–Freeman–Halton (Monte Carlo, B=20,000) | 4.74 | 7.00 | 0.698 | 0.830 | Cramér's V | 0.16 | min expected=0.42 |
| B. metric profile | Category × D0 reported | pooled (validated) | 177 | 8x2 | Fisher–Freeman–Halton (Monte Carlo, B=20,000) | 4.86 | 7.00 | 0.654 | 0.818 | Cramér's V | 0.17 | min expected=0.23 |
| B. metric profile | Author × D1 reported | validated | 177 | 2x2 | chi-square | 2.11 | 1.00 | 0.146 | 0.292 | Cramér's V | 0.11 | min expected=16.89; OR=0.61 [0.31, 1.19] |
| B. metric profile | Author × D2 reported | validated | 177 | 2x2 | Fisher's exact | 0.15 | 1.00 | 1.000 | 1.000 | Cramér's V | 0.03 | min expected=1.38; OR=1.55 [0.17, 14.21] |
| B. metric profile | Author × D3 reported | validated | 177 | 2x2 | Fisher's exact | 0.26 | 1.00 | 0.563 | 0.728 | Cramér's V | 0.04 | min expected=4.15; OR=0.75 [0.24, 2.30] |
| B. metric profile | Author × D4 reported | validated | 177 | 2x2 | Fisher's exact | 1.97 | 1.00 | 0.324 | 0.509 | Cramér's V | 0.10 | min expected=1.38; OR=4.41 [0.24, 81.24] |
| B. metric profile | Author × D5 reported | validated | 177 | 2x2 | Fisher's exact | 0.10 | 1.00 | 0.669 | 0.818 | Cramér's V | 0.02 | min expected=1.66; OR=0.76 [0.13, 4.28] |
| B. metric profile | Author × D0 reported | validated | 177 | 2x2 | Fisher's exact | 2.08 | 1.00 | 0.219 | 0.386 | Cramér's V | 0.11 | min expected=2.21; OR=0.36 [0.09, 1.51] |
| B. metric profile | Author × any dimension reported | validated | 177 | 2x2 | chi-square | 2.52 | 1.00 | 0.112 | 0.247 | Cramér's V | 0.12 | min expected=21.32; OR=1.71 [0.88, 3.31] |
| B. metric profile | Category × any dimension reported | pooled (validated) | 177 | 8x2 | Fisher–Freeman–Halton (Monte Carlo, B=20,000) | 6.69 | 7.00 | 0.476 | 0.639 | Cramér's V | 0.19 | min expected=2.18 |
| B. metric profile | #dims: agent vs human | validated | 177 | agent n=128, human n=49 | Mann–Whitney U | 2754.50 |  | 0.162 | 0.310 | Cliff's delta | -0.12 | median agent=0.00 (mean 0.61); median human=1.00 (mean 0.73) |
| B. metric profile | #dims across categories | pooled (validated) | 177 | k=8 | Kruskal–Wallis | 7.53 | 7.00 | 0.376 | 0.571 | epsilon² | 0.00 | Algorithm md=1.0; Build/Infra md=0.0; Code smells md=0.0; Control-flow md=0.0; Data structure md=0.0; I/O & sync md=0.0; Memory/locality md=0.0; Network/DB md=0.0 |
| B. metric profile | #dims across categories | AI Agent | 128 | k=8 | Kruskal–Wallis | 6.53 | 7.00 | 0.479 | 0.639 | epsilon² | -0.00 | Algorithm md=1.0; Build/Infra md=0.0; Code smells md=0.0; Control-flow md=0.5; Data structure md=0.0; I/O & sync md=0.5; Memory/locality md=0.0; Network/DB md=0.0 |
| B. metric profile | #dims across categories | Human | 48 | k=7 | Kruskal–Wallis | 3.19 | 6.00 | 0.785 | 0.885 | epsilon² | -0.07 | Algorithm md=1.0; Build/Infra md=0.5; Code smells md=0.5; Control-flow md=0.0; I/O & sync md=0.0; Memory/locality md=1.0; Network/DB md=0.0 |
| B. metric profile | #dims: benchmark vs other evidence | validated | 177 | benchmark n=56, other n=121 | Mann–Whitney U | 5471.00 |  | <0.001 | <0.001 | Cliff's delta | 0.61 | median benchmark=1.00 (mean 1.29); median other=0.00 (mean 0.35) |
| B. metric profile | Benchmark evidence × any dimension reported | validated | 177 | 2x2 | chi-square | 54.47 | 1.00 | <0.001 | <0.001 | Cramér's V | 0.56 | min expected=24.36; OR=15.84 [6.95, 36.11] |
| B. metric profile | Benchmark evidence × D1 reported | validated | 177 | 2x2 | chi-square | 54.46 | 1.00 | <0.001 | <0.001 | Cramér's V | 0.56 | min expected=19.30; OR=13.80 [6.45, 29.56] |
| B. metric profile | Benchmark evidence × D2 reported | validated | 177 | 2x2 | Fisher's exact | 5.56 | 1.00 | 0.035 | 0.110 | Cramér's V | 0.18 | min expected=1.58; OR=9.23 [1.01, 84.60] |
| B. metric profile | Benchmark evidence × D3 reported | validated | 177 | 2x2 | Fisher's exact | 3.57 | 1.00 | 0.080 | 0.195 | Cramér's V | 0.14 | min expected=4.75; OR=2.71 [0.93, 7.91] |
| B. metric profile | Benchmark evidence × D4 reported | validated | 177 | 2x2 | Fisher's exact | 2.38 | 1.00 | 0.181 | 0.331 | Cramér's V | 0.12 | min expected=1.58; OR=0.19 [0.01, 3.45] |
| B. metric profile | Benchmark evidence × D5 reported | validated | 177 | 2x2 | Fisher's exact | 0.01 | 1.00 | 1.000 | 1.000 | Cramér's V | 0.01 | min expected=1.90; OR=1.08 [0.19, 6.10] |
| B. metric profile | Benchmark evidence × D0 reported | validated | 177 | 2x2 | Fisher's exact | 12.09 | 1.00 | 0.001 | 0.011 | Cramér's V | 0.26 | min expected=2.53; OR=17.14 [2.05, 143.04] |
| C. merge | #dims: merged vs not merged | pooled (validated) | 177 | merged n=75, not merged n=102 | Mann–Whitney U | 3093.50 |  | 0.015 | 0.083 | Cliff's delta | -0.19 | median merged=0.00 (mean 0.43); median not merged=0.00 (mean 0.80) |
| C. merge | Merged × any dimension reported | pooled (validated) | 177 | 2x2 | chi-square | 2.98 | 1.00 | 0.084 | 0.195 | Cramér's V | 0.13 | min expected=32.63; OR=0.59 [0.32, 1.08] |
| C. merge | #dims: merged vs not merged | AI Agent (validated) | 128 | merged n=47, not merged n=81 | Mann–Whitney U | 1502.00 |  | 0.024 | 0.100 | Cliff's delta | -0.21 | median merged=0.00 (mean 0.34); median not merged=0.00 (mean 0.77) |
| C. merge | Merged × any dimension reported | AI Agent (validated) | 128 | 2x2 | chi-square | 3.13 | 1.00 | 0.077 | 0.195 | Cramér's V | 0.16 | min expected=18.73; OR=0.50 [0.24, 1.08] |
| C. merge | #dims: merged vs not merged | Human (validated) | 49 | merged n=28, not merged n=21 | Mann–Whitney U | 222.50 |  | 0.121 | 0.253 | Cliff's delta | -0.24 | median merged=0.00 (mean 0.57); median not merged=1.00 (mean 0.95) |
| C. merge | Merged × any dimension reported | Human (validated) | 49 | 2x2 | chi-square | 1.15 | 1.00 | 0.283 | 0.461 | Cramér's V | 0.15 | min expected=9.86; OR=0.53 [0.17, 1.69] |

**Significant after BH (q < 0.05):**

- **Category × validation present** [pooled], n=357: Fisher–Freeman–Halton (Monte Carlo, B=20,000), stat=31.29, p=<0.001, Cramér's V=0.296 (min expected=0.99) → q=<0.001

- **Category × validation present** [AI Agent], n=280: Fisher–Freeman–Halton (Monte Carlo, B=20,000), stat=32.25, p=<0.001, Cramér's V=0.339 (min expected=0.91) → q=<0.001

- **Author × validation present — Memory/locality** [within category], n=114: chi-square, stat=8.57, p=0.003, Cramér's V=0.274 (min expected=9.07; OR=0.17 [0.05, 0.62]) → q=0.022

- **#dims: benchmark vs other evidence** [validated], n=177: Mann–Whitney U, stat=5471.00, p=<0.001, Cliff's delta=0.615 (median benchmark=1.00 (mean 1.29); median other=0.00 (mean 0.35)) → q=<0.001

- **Benchmark evidence × any dimension reported** [validated], n=177: chi-square, stat=54.47, p=<0.001, Cramér's V=0.555 (min expected=24.36; OR=15.84 [6.95, 36.11]) → q=<0.001

- **Benchmark evidence × D1 reported** [validated], n=177: chi-square, stat=54.46, p=<0.001, Cramér's V=0.555 (min expected=19.30; OR=13.80 [6.45, 29.56]) → q=<0.001

- **Benchmark evidence × D0 reported** [validated], n=177: Fisher's exact, stat=12.09, p=0.001, Cramér's V=0.261 (min expected=2.53; OR=17.14 [2.05, 143.04]) → q=0.011
