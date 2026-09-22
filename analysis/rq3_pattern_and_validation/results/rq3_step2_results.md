# RQ3 Step 2 — do the reported metrics fit the optimization applied?
Source: `analysis/classification_labels/rq3_labels.csv`. Population: the RQ3 metric layer (n = 1699 of 2081 analytic PRs; 1684 with a resolved primary evidence type, of which 906 measured = benchmark or profiling).
Expected dimensions per sub-pattern: `catalog_expected_dims.csv`, derived mechanically from the catalog's Optimized Metrics column (`catalog_expected_dims.py`).

## 1. Alignment: do reported dimensions track the pattern?
Share of validated PRs reporting >= 1 dimension their pattern is expected to improve, against a null that permutes the expected sets across sub-patterns (each sub-pattern receives another sub-pattern's expected set; every PR keeps its reported dimensions). One-sided p: observed >= null.
| population | n | observed | null mean | null 95% | p (one-sided) |
| --- | --- | --- | --- | --- | --- |
| All validated | 1699 | 34.5% | 23.7% | 14.5%-32.1% | 0.0017 |
| agentic | 858 | 32.3% | 23.5% | 14.1%-31.7% | 0.0148 |
| human_candidate | 841 | 36.7% | 24.6% | 15.2%-33.4% | 0.0016 |
| measured evidence | 906 | 54.2% | 40.5% | 25.6%-53.5% | 0.0168 |
| static reasoning | 767 | 11.6% | 7.2% | 4.4%-9.9% | 0.0009 |

## 2. Memory-for-time patterns: is the memory cost reported alongside the gain?
Memory-for-time = Caching, Buffering (spend memory to save time or I/O). A gain is any of D1/D2/D5. Descriptive: rates within these PRs, by author type; no comparison group, since other patterns are not expected to move memory.
### 2.1 All validated memory-for-time PRs
| author_type | n | reports a gain (D1/D2/D5) | reports memory (D3) | gain and memory | gain without memory | of gain reporters, memory too | any dimension |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agentic | 141 | 64 (45.4%) | 25 (17.7% [12.3%-24.9%]) | 18 (12.8%) | 46 | 28.1% | 57.4% |
| human_candidate | 108 | 46 (42.6%) | 15 (13.9% [8.6%-21.7%]) | 12 (11.1%) | 34 | 26.1% | 51.9% |
| All | 249 | 110 (44.2%) | 40 (16.1% [12.0%-21.1%]) | 30 (12.0%) | 80 | 27.3% | 55.0% |

### 2.2 Measured evidence only
| author_type | n | reports a gain (D1/D2/D5) | reports memory (D3) | gain and memory | gain without memory | of gain reporters, memory too | any dimension |
| --- | --- | --- | --- | --- | --- | --- | --- |
| agentic | 74 | 55 (74.3%) | 18 (24.3% [16.0%-35.2%]) | 17 (23.0%) | 38 | 30.9% | 85.1% |
| human_candidate | 56 | 41 (73.2%) | 14 (25.0% [15.5%-37.7%]) | 12 (21.4%) | 29 | 29.3% | 89.3% |
| All | 130 | 96 (73.8%) | 32 (24.6% [18.0%-32.7%]) | 29 (22.3%) | 67 | 30.2% | 86.9% |

### 2.3 By pattern (all validated)
| sub_pattern | n | reports a gain | reports memory | gain and memory | n measured | reports memory (measured) |
| --- | --- | --- | --- | --- | --- | --- |
| Caching | 235 | 43.4% | 15.7% | 11.5% | 122 | 23.8% |
| Buffering | 14 | 57.1% | 21.4% | 21.4% | 8 | 37.5% |

### 2.4 Sensitivity: code diff excluded from the corpus
Memory-for-time PRs reporting memory: 16.1% -> 14.5% without the diff.

## 3. Extremes
80 memory-for-time PRs report a gain and no memory figure (`extremes_memory_for_time_gain_no_memory.csv`); 30 report both. Sample of the latter:

| html_url | author_type | sub_pattern | validation_type | dims | n_dims |
| --- | --- | --- | --- | --- | --- |
| https://github.com/bitcoin/bitcoin/pull/34636 | human_candidate | Caching | benchmark | D1|D3 | 2 |
| https://github.com/bitcoin/bitcoin/pull/35195 | human_candidate | Caching | benchmark | D1|D3|D9 | 3 |
| https://github.com/Azure/azure-sdk-for-net/pull/56872 | human_candidate | Buffering | benchmark | D2|D3|D5|D6|D9 | 5 |
| https://github.com/Azure/azure-sdk-for-java/pull/48617 | human_candidate | Caching | benchmark | D1|D2|D3|D4|D9 | 5 |
| https://github.com/dotnet/fsharp/pull/19072 | agentic | Caching | benchmark | D0|D1|D3|D5|D7 | 5 |
| https://github.com/MetaMask/metamask-extension/pull/42862 | agentic | Caching | benchmark | D1|D3|D6 | 3 |
| https://github.com/datahub-project/datahub/pull/16241 | human_candidate | Caching | benchmark | D1|D3|D6 | 3 |
| https://github.com/maplibre/martin/pull/2688 | agentic | Caching | benchmark | D1|D3|D4 | 3 |
| https://github.com/esphome/esphome/pull/12628 | human_candidate | Buffering | benchmark | D1|D3|D5 | 3 |
| https://github.com/saturday06/VRM-Addon-for-Blender/pull/795 | agentic | Caching | benchmark | D1|D3|D4 | 3 |

## 4. Step 2 test family (Benjamini-Hochberg)
| test_label | stratum | n | test | statistic | p_raw | p_bh | effect_name | effect | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reported dim in pattern's expected set vs permutation null | All validated | 1699 | permutation (B=20000) | 0.108 | 0.002 | 0.003 | observed - null | 0.108 | observed=34.5%; null=23.7% |
| Reported dim in pattern's expected set vs permutation null | agentic | 858 | permutation (B=20000) | 0.088 | 0.015 | 0.017 | observed - null | 0.088 | observed=32.3%; null=23.5% |
| Reported dim in pattern's expected set vs permutation null | human_candidate | 841 | permutation (B=20000) | 0.121 | 0.002 | 0.003 | observed - null | 0.121 | observed=36.7%; null=24.6% |
| Reported dim in pattern's expected set vs permutation null | measured evidence | 906 | permutation (B=20000) | 0.137 | 0.017 | 0.017 | observed - null | 0.137 | observed=54.2%; null=40.5% |
| Reported dim in pattern's expected set vs permutation null | static reasoning | 767 | permutation (B=20000) | 0.044 | 0.001 | 0.003 | observed - null | 0.044 | observed=11.6%; null=7.2% |

Significant after BH (q < 0.05): 5 of 5.
- Reported dim in pattern's expected set vs permutation null [All validated]: q=0.0029, observed=34.5%; null=23.7%
- Reported dim in pattern's expected set vs permutation null [agentic]: q=0.0168, observed=32.3%; null=23.5%
- Reported dim in pattern's expected set vs permutation null [human_candidate]: q=0.0029, observed=36.7%; null=24.6%
- Reported dim in pattern's expected set vs permutation null [measured evidence]: q=0.0168, observed=54.2%; null=40.5%
- Reported dim in pattern's expected set vs permutation null [static reasoning]: q=0.0029, observed=11.6%; null=7.2%
