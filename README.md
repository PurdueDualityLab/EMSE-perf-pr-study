# Performance Pull Request Study

This repository groups the source code, generated-data reference, and LaTeX
report for the performance pull request study. Detailed pipeline commands and
artifact metadata live in the corresponding directories rather than in this
root overview.

## Repository Structure

```text
.
├── mining/              Data collection, classification, filtering, sampling,
│   ├── src/             and publication pipeline
│   ├── tests/           Pytest regression tests
│   ├── tools/           Artifact validation and publication utilities
│   ├── README.md        Pipeline commands and methodology
│   └── ARTIFACTS.md     Published artifact inventory and checksums
├── data/                Private Hugging Face dataset submodule
├── report/              Overleaf LaTeX report submodule
├── AGENTS.md            Repository guidance for coding agents
├── requirements.txt     Pinned Python dependencies
├── LICENSE              Source-code license
└── README.md            Repository structure overview
```

## Mining

[`mining/`](mining/) contains the reproducible Python pipeline. Its
[`README.md`](mining/README.md) documents setup, stage inputs, execution,
resume behavior, validation, and publication. Generated outputs, checkpoints,
caches, local configuration, and credentials are excluded from Git.

## Data

[`data/`](data/) is a Git submodule pointing to the private Hugging Face
dataset:

https://huggingface.co/datasets/rcalvome/EMSE-perf-pr-study

Access requires authorization for the private dataset. The submodule pins the
exact dataset revision used by this repository. Its local checkout may use
sparse checkout to avoid materializing the large Parquet files.

## Report

[`report/`](report/) is a Git submodule connected to the private Overleaf
project containing the LaTeX report, figures, references, and build files:

https://www.overleaf.com/project/6a08c1aaa379b791cc6da4c2

Access requires authorization for the private Overleaf project. Changes to the
report are committed and pushed inside the submodule first; the parent
repository then records the updated report commit.

## Supporting Files

- [`requirements.txt`](requirements.txt) contains the pinned Python runtime and
  test dependencies.
- [`AGENTS.md`](AGENTS.md) records repository-specific maintenance guidance.
- [`LICENSE`](LICENSE) contains the MIT license for the processing code.
- [`.gitmodules`](.gitmodules) defines the private data and report submodules.
