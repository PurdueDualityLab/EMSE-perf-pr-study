# How Do Coding Agents Optimize Software and Report Performance Validation?

**A Large-Scale Empirical Study of Open-Source Pull Requests**

Huiyun Peng, Ricardo Calvo, Kelechi Kalu, and James C. Davis · Purdue University

This repository provides the replication package for the study. It contains
the data-processing pipeline, analysis code, compact research datasets, and
instructions for reproducing the reported results.

The study examines 2,260 performance-oriented pull requests (PRs), comprising
1,130 agentic and 1,130 human-authored contributions to open-source projects,
covering submissions through June 1, 2026. It extends the authors' MSR 2026
Mining Challenge study of 407 PRs.

## Research Questions

| Question | Focus | Analysis directory |
| --- | --- | --- |
| RQ1 | Adoption outcomes, patch size, and structural indicators of maintainability | [`quantitative_analysis`](analysis/quantitative_analysis/README.md), [`maintainability`](analysis/maintainability/README.md) |
| RQ2 | Optimization patterns and their prevalence | [`rq1_optimization_patterns`](analysis/rq1_optimization_patterns/README.md) |
| RQ3 | Reported validation evidence and temporal trends | [`rq2_validation`](analysis/rq2_validation/README.md) |
| RQ4 | Quantitative performance metrics and their correspondence with optimizations | [`rq3_pattern_and_validation`](analysis/rq3_pattern_and_validation/README.md) |

Directory and data-field names retain the original numbering for compatibility.
The [analysis documentation](analysis/README.md) provides the correspondence
between manuscript terminology and stored identifiers.

## Reproduction

The analyses can be reproduced from the versioned labels and measurements
included in this repository. This workflow operates offline and requires
Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r analysis/quantitative_analysis/requirements.txt
python analysis/reproduce_paper.py --output-dir reproduction
```

The command generates statistical results, supporting tables, nine quantitative
figures, and audit summaries. Output filenames correspond to the manuscript's
figure references. Input and source checksums are validated before execution;
selected result tables are compared with the recorded reference outputs.

See [REPRODUCING.md](REPRODUCING.md) for environment requirements, expected
populations, output descriptions, and instructions for reconstructing the
original measurements and classifications.

## Repository Contents

| Location | Contents |
| --- | --- |
| [`mining/`](mining/README.md) | PR collection, authorship attribution, performance classification, filtering, and sampling |
| [`analysis/`](analysis/README.md) | Statistical analyses and figure generation for RQ1–RQ4 |
| [`analysis/classification_labels/`](analysis/classification_labels/README.md) | Model labels, consensus decisions, and extracted metric indicators |
| [`analysis/artifact_inputs/`](analysis/artifact_inputs/) | Compact metadata, annotation records, and model votes required for reproduction |
| [`analysis/audits/`](analysis/audits/README.md) | Assessment procedures and reproduction of audit summaries |
| [`data/`](data/) | Submodule identifying the full dataset revision |
| [`report/`](report/) | Submodule containing the manuscript, figures, and bibliography |
| [`artifact_manifest.json`](artifact_manifest.json) | Input schemas, checksums, source digests, and revision identifiers |

## Data Availability

Compact inputs for offline reproduction are included in this repository. The
full dataset is hosted on [Hugging Face](https://huggingface.co/datasets/rcalvome/EMSE-perf-pr-study).
Access to the full files requires authentication and acceptance of the dataset's
access conditions. The dataset revision is identified by the `data/` submodule.

The [artifact specification](ARTIFACT.md) describes provenance, access to
archived evidence, and documented differences between the manuscript and
execution records. The [core inventory](mining/ARTIFACTS.md) lists published
mining and classification artifacts with their checksums.

## Interpretation

The stored label `human_candidate` corresponds to the manuscript's
human-authored group. It indicates the absence of selected observable automation
signals rather than independently verified authorship. The analyses characterize
reported validation and performance claims; they do not independently reproduce
the performance effects asserted by PR authors.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). The preceding
conference study is:

Peng, H., Qiu, A. Z., Calvo Méndez, R. A., Kalu, K. G., and Davis, J. C. (2026).
*How Do Agents Perform Code Optimization? An Empirical Study.* Proceedings of
the 23rd International Conference on Mining Software Repositories, 732–736.
https://doi.org/10.1145/3793302.3793564

## License

The processing and analysis code is distributed under the [MIT License](LICENSE).
Third-party repository content remains subject to its original terms. Dataset
access and reuse conditions are documented in the dataset repository.
