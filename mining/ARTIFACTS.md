# Artifact Inventory

This inventory identifies the local study artifacts that must be preserved.
Generated data remains outside Git because of its size. Checksums are lowercase
SHA-256 values.

Verify the locked raw and official files from the repository root with:

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

## Preserved Local Data

| Location | Purpose | Policy |
| --- | --- | --- |
| `mining/outputs_2026_06_01/` | Recovered full raw mining input | Preserve in place. |
| `mining/query_cache/` | Exploratory datasets, model cache, and resumable parts | Preserve in place; contents are not official outputs. |
| `mining/official_selection/` | Completed official v1 population | Immutable. |
| `mining/github_tokens.txt` | Local GitHub credentials | Never copy into artifacts, logs, Git, or backups. |

Exploratory query-cache files are intentionally not checksum-locked because the
directory contains mutable checkpoints. Completed study outputs are locked
above.

## External Backup

The pre-cleanup backup is outside the repository at:

`/home/rcalvome/Documents/repositories/EMSE-perf-pr-study-backup/pre_cleanup_2026_07_27/`

It contains source, query-cache, and official-selection snapshots. The token
file was deliberately excluded.
