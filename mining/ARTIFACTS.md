# Official Artifact Inventory

Generated data remains outside Git because of its size. The private dataset
publication contains only the files below; checkpoints, Batch payloads, retry
directories, caches, secrets, and historical experiments are excluded.

| Stage | Artifact | Rows | SHA-256 |
| --- | --- | ---: | --- |
| Mining | `raw/github_pull_requests.parquet` | 1,603,213 | `a3d737ce2cc32e15b6e14b91578217873fcd9739bebc664fdc04f31209256bef` |
| Agentic attribution | `agentic/aidev_attribution.parquet` | 1,603,213 | `305238366a898fc95a963a96f86d026166d3d4f68a968711b7aee3fe16d8b6a0` |
| Performance classification | `performance/task_type_decisions.parquet` | 1,603,213 | `16a9a281029321541c7370c3445a69c896c05dd7fce2a0dbcb4c7ce70b2afbf4` |
| Human filtering | `human/decisions.parquet` | 28,127 | `72922a804a87063867927f12dc39e8eabfb3a1230905153aa9ff31ba13b5105b` |
| Human filtering | `human/human_candidates.parquet` | 24,680 | `b08525d2c6e634c6ed667000bf24d22cc956b85ca5a844a6acbbeb24ff79637f` |
| Human filtering | `human/excluded.parquet` | 3,447 | `5c6813a81251e963bac43ec03f5a5249d24d4ab5125b2734544d399a1b9a6e19` |
| Sampling | `sample/agentic_sample.parquet` | 1,130 | `66643a58480668ef62295e1d3cf3c162cb89a79a81d6741fac7bec2ddbb1820e` |
| Sampling | `sample/human_sample.parquet` | 1,130 | `66970968eba6666b752f1306fe204d215f424e8aab086d71bfc78084e06ea835` |
| Sampling | `sample/balanced_sample.parquet` | 2,260 | `10f46fae9c880325f6574ca308f1857a4798c2ac1548bd0511502cb8687b357f` |
| Sampling | `sample/sampling_manifest.parquet` | 25,365 | `ad11ad502cc689608278993225bb59e1568e51377d710fed3da072f649b3a1b5` |
| Sampling | `sample/weekly_strata.parquet` | 65 | `670eabaaf28672a10966de3112067000ebfc70f6e0aa0016249191da5e878027` |
| Sampling | `sample/quality_exclusions.parquet` | 671 | `09a67ecf51e4cb4945b5f00ab65939708cdf912fed437600a0c82cf05c6b055b` |
| Curated labels | `curated/pull_request_labels.parquet` | 1,603,213 | `f058825dfbc1c56f621a03ccef9480d06dcedab13e138e7e4cd97fbd1a60117a` |
| RQ1 catalog | `catalog/original_optimization_catalog.csv` | 43 | `a951a9acf91ebe9a775009e95d187a3ea59f5a4ae4b2875d5ebc9788c1bc6d1e` |
| RQ1 catalog | `catalog/updated_optimization_catalog.csv` | 58 | `8a4b2e59a83a3994ee228515ba47ce2712c43112df3070640ea0b3ef7fcf803c` |

The publisher additionally generates sanitized stage summaries,
`publication_manifest.json`, `checksums.sha256`, and the dataset card. It
verifies table row counts and hashes before an upload can start.

The final cohort is 1:1 balanced: 1,130 agentic and 1,130 human-candidate
performance PRs. Human candidates are negative under the selected observable
signals, not verified human authors.

The curated-label table has one compact row per mined PR. It combines AIDev
attribution and task-type labels with the performance indicator, strict human
filter decision, and sampling fields. Human-filter and sampling values are null
outside the populations evaluated by those stages.
