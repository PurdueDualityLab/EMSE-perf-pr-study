# Official Core Artifact Inventory

Generated from the allowlist in `mining/tools/publish_artifacts.py`.
Paths below are relative to the Hugging Face dataset repository root.
This is the core mining/catalog/model-label bundle. The historical RQ4
extraction bundle has separate metadata; offline paper inputs and audits
are listed in the root `artifact_manifest.json`.

| Stage | Artifact | Rows | SHA-256 |
| --- | --- | ---: | --- |
| mining | `data/raw/github_pull_requests.parquet` | 1,603,213 | `a3d737ce2cc32e15b6e14b91578217873fcd9739bebc664fdc04f31209256bef` |
| agentic_attribution | `data/agentic/aidev_attribution.parquet` | 1,603,213 | `305238366a898fc95a963a96f86d026166d3d4f68a968711b7aee3fe16d8b6a0` |
| performance_classification | `data/performance/task_type_decisions.parquet` | 1,603,213 | `16a9a281029321541c7370c3445a69c896c05dd7fce2a0dbcb4c7ce70b2afbf4` |
| human_filtering | `data/human/decisions.parquet` | 28,127 | `72922a804a87063867927f12dc39e8eabfb3a1230905153aa9ff31ba13b5105b` |
| human_filtering | `data/human/human_candidates.parquet` | 24,680 | `b08525d2c6e634c6ed667000bf24d22cc956b85ca5a844a6acbbeb24ff79637f` |
| human_filtering | `data/human/excluded.parquet` | 3,447 | `5c6813a81251e963bac43ec03f5a5249d24d4ab5125b2734544d399a1b9a6e19` |
| weekly_sampling | `data/sample/agentic_sample.parquet` | 1,130 | `66643a58480668ef62295e1d3cf3c162cb89a79a81d6741fac7bec2ddbb1820e` |
| weekly_sampling | `data/sample/human_sample.parquet` | 1,130 | `66970968eba6666b752f1306fe204d215f424e8aab086d71bfc78084e06ea835` |
| weekly_sampling | `data/sample/balanced_sample.parquet` | 2,260 | `10f46fae9c880325f6574ca308f1857a4798c2ac1548bd0511502cb8687b357f` |
| weekly_sampling | `data/sample/sampling_manifest.parquet` | 25,365 | `ad11ad502cc689608278993225bb59e1568e51377d710fed3da072f649b3a1b5` |
| weekly_sampling | `data/sample/weekly_strata.parquet` | 65 | `670eabaaf28672a10966de3112067000ebfc70f6e0aa0016249191da5e878027` |
| weekly_sampling | `data/sample/quality_exclusions.parquet` | 671 | `09a67ecf51e4cb4945b5f00ab65939708cdf912fed437600a0c82cf05c6b055b` |
| curated_labels | `data/curated/pull_request_labels.parquet` | 1,603,213 | `f058825dfbc1c56f621a03ccef9480d06dcedab13e138e7e4cd97fbd1a60117a` |
| rq1_catalog | `data/catalog/original_optimization_catalog.csv` | 43 | `a951a9acf91ebe9a775009e95d187a3ea59f5a4ae4b2875d5ebc9788c1bc6d1e` |
| rq1_catalog | `data/catalog/updated_optimization_catalog.csv` | 58 | `8a4b2e59a83a3994ee228515ba47ce2712c43112df3070640ea0b3ef7fcf803c` |
| rq1_model_labels | `data/rq1/optimization_pattern_labels_gpt.parquet` | 2,260 | `6a34304ec4de221cd28108265b6d5787c5d684d2f444240253ff8cc898ce632b` |
| rq1_model_labels | `data/rq1/optimization_pattern_labels_gemini.parquet` | 2,260 | `fcb5e01a0da93f7718e6f2fa1dd4417c8e467b8c5b1c9da7df70205d4ec4e96a` |
| rq1_model_labels | `data/rq1/optimization_pattern_labels_qwen.parquet` | 2,260 | `69df410be22b738fbaef95385e04f8df755564e6a12578a85870ca9e833f719f` |
| rq2_model_labels | `data/rq2/performance_validation_labels_gpt.parquet` | 2,258 | `88fe2f918cc4d436a459f0c13ea5718a6ef37f896884088d552d8a876801f2a6` |
| rq2_model_labels | `data/rq2/performance_validation_labels_gemini.parquet` | 2,258 | `0a8e964ac6d9623e916e3478a9f1c688ec4c8667af5c74fbc79fe97bc7e675aa` |
| rq2_model_labels | `data/rq2/performance_validation_labels_qwen.parquet` | 2,258 | `7e455ad4028bcecdfd42ec3a74443195fdf3667b20c9b5f995415a06211eeb33` |

The executed optimization catalog has **58 patterns**. The submitted paper
reports 59; the documented consolidation and manuscript history are in
`analysis/rq1_optimization_patterns/catalog/README.md`.

Historical `rq1` and `rq2` dataset paths refer to journal RQ2 and RQ3.
`human_candidate` is the stored label for the manuscript's human-authored
group; it does not establish independently verified human authorship.

Full-data access requires authentication and the dataset's access conditions.
Publication tooling retains its private-only upload guard.

Validate installed dataset files without writes:

```bash
python mining/tools/publish_artifacts.py --validate-only --dataset-dir data
```

`--dry-run` validates original local sources and **builds** sanitized metadata
under `--bundle-dir`; it is not a read-only validation command.

Maintainers can check or regenerate this inventory and `mining/artifacts.sha256`:

```bash
python mining/tools/publish_artifacts.py --check-inventory
python mining/tools/publish_artifacts.py --write-inventory
```
