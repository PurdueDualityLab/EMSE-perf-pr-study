# Artifact Inventory

This inventory identifies the local study artifacts that must be preserved.
Generated data remains outside Git because of its size. Checksums are lowercase
SHA-256 values.

Verify the locked study artifacts from the repository root with:

```bash
sha256sum -c mining/artifacts.sha256
```

## Official Selection V1

Location: `mining/official_selection/`

Status: completed on 2026-07-20, immutable, no sampling. Do not resume this
directory with `build_official_selection.py` v2.

The exact generating script is archived as
`mining/legacy/official_selection_v1/build_official_selection.py`, together
with its local source dependencies; its hash matches the v1 run manifest.

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| `candidate_decisions.parquet` | 64,770 | `b65c84a5ca8ec0eea2c7f067e4caa9213415049adcc18e7e47f01739ef80d836` |
| `official_population.parquet` | 14,969 | `f124f2d44f3a5494d51024464d5005761077f7c36886423f31538087195af841` |
| `official_agentic_prs.parquet` | 624 | `179d3099ee8acaf4017b68020800851430c363b8dbe92daf620303cdd507f8c5` |
| `official_human_non_agentic_candidates.parquet` | 14,345 | `859b7c8114c64adb77b429391a8eeb133fe209541f1b79ec02dcda3a0217e2a3` |
| `intermediate/date_filtered.parquet` | 633,350 | `213071630a467376d447fcaeb1a2ce3203c0080379d9faf9353a83dbc17cb884` |
| `intermediate/heuristic_candidates.parquet` | 64,770 | `cf873823d1cdddeb6f0a6e5ea46b97cab094db98ab3b49242b645b94fca76e4b` |
| `intermediate/quality_passed.parquet` | 55,114 | `9b4347e4b96194a0ab003478a8d81b5de84c4ad8c39fab0f26df2a8f446b9e49` |
| `intermediate/model_predictions.parquet` | 55,114 | `cb6fa4f7d3f699305a4eac1343cf5742e22b185d8d62575f7f08f2fad649c5f5` |
| `selection_summary.json` | n/a | `0c46ee36558d11f371948167a7fad117efb2201ef7f9f2edb0ca36522349cb1f` |
| `run_manifest.json` | n/a | `4ffc6822716b2d230df040038002c74bb599de9f2680728dca9cccbc1acafddc` |

The v1 manifest records these reproducibility pins:

- Raw input: `mining/outputs_2026_06_01/raw/github_human_prs.parquet`
- Raw rows: 1,603,213
- Raw SHA-256: `a3d737ce2cc32e15b6e14b91578217873fcd9739bebc664fdc04f31209256bef`
- PerfAnnotator revision: `7d7ba362c257c3ea8c69d52b4a736ae4f182e68c`
- Extracted PerfAnnotator tree SHA-256: `e2ed2bd495eb7ccea90da33037d2d444be344183cc8a747949a660248541531f`
- AIDev revision: `b6d1b8af952053f20fd9f9aa03e66bddd12228fe`
- Seed: `20260720`
- Inference device: CPU
- Positive-label convention: class ID `1` is performance-improving; the upstream config contains only generic label names.

## AIDev Attribution

Location: `mining/aidev_attribution/`

Status: completed on 2026-08-03 for all 2,774 repositories in the official
window input. Preserve this directory as an immutable generated output.

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| `aidev_window.parquet` | 633,350 | `8eb98287cb57f6c0d308ce80a6448fc840c618cbb40a255211bc88d62a3a800d` |
| `aidev_window.state.json` | n/a | `cee7913348f91b946eaf98fad3a0b0785b2f9a4d81bb149274e05522301dcd3f` |
| `aidev_window.summary.json` | n/a | `aef154afa5bbbcd7b8b0d86e6798f822145fbc4dc10a368a533cf074d1b901ca` |

The output contains 33,632 `agentic`, 599,569 `human_candidate`, and 149
`unresolved` rows. All unresolved rows belong to the deleted
`agentmark-ai/agentmark` repository; its immutable repository ID was verified,
but the five searches could not be run. The generating enricher SHA-256 is
`cc5785bf5e3c9c8d60b91c2606b9363d4773bed7ee44160e168b384f3fcc4081`.
The historical coverage and live-search drift analysis is recorded in
`mining/analysis/aidev_attribution_findings.md`.

## Preserved Local Data

| Location | Purpose | Policy |
| --- | --- | --- |
| `mining/outputs_2026_06_01/` | Recovered full raw mining input | Preserve in place. |
| `mining/query_cache/` | Exploratory datasets, model cache, and resumable parts | Preserve in place; contents are not official outputs. |
| `mining/query_cache/models/perfannotator-mini-ease-2026/` | Exact EASE 2026 Figshare model, tokenizer, config, and operating point | Preserve in place; all eight files are verified before inference. |
| `mining/aidev_attribution/` | Completed published-rule attribution for the official date-window input | Immutable generated output; preserve state and summary with the Parquet. |
| `mining/perfminer_reproduction/` | Corrected commit-level manifest, evidence, predictions, and PR aggregation | Generated v2 research outputs; preserve completed state and summaries. |
| `mining/official_selection/` | Completed official v1 population | Immutable. |
| `mining/github_tokens.txt` | Local GitHub credentials | Never copy into artifacts, logs, Git, or backups. |

Exploratory query-cache files are intentionally not checksum-locked because the
directory contains mutable checkpoints. Completed study outputs are locked
above.

The EASE model bundle has weights SHA-256
`c8e71790c6dc286562df297b40405d5c7ee8ad8bb0ca1fb47490949f6b5a47dd`
and extracted tree SHA-256
`9a0107e3fde617214c7d58cffda8dc29043adfcdeeee63331c51f458b05b40a6`.
It differs from the older Hugging Face artifact recorded by official-selection
v1; the two checkpoints are not interchangeable.

## External Backup

The pre-cleanup backup is outside the repository at:

`/home/rcalvome/Documents/repositories/EMSE-perf-pr-study-backup/pre_cleanup_2026_07_27/`

It contains source, query-cache, and official-selection snapshots. The token
file was deliberately excluded.
