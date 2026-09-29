# Qwen classification on Gautschi

`run_qwen.py` classifies the official RQ1 and RQ2 samples with the same input
construction, prompts, semantic schemas, and study contracts used by the OpenAI
runs. It uses Ollama structured outputs and writes provider-specific manifests,
per-PR JSON checkpoints, final Parquet labels, and summaries.

The Gautschi job requests one H100 on the `ai` partition for the maximum
14-day wall time. All mutable and large artifacts remain under
`/scratch/gautschi/rcalvome/`: the Apptainer image, caches, Python environment,
Ollama model store, checkpoints, results, and logs.

The default image source is `docker://ollama/ollama:latest`. For a pinned image,
export `OLLAMA_IMAGE_REF` with an immutable tag or export both
`OLLAMA_IMAGE_REF` and `OLLAMA_IMAGE_DIGEST`. The digest takes precedence over
any tag in the image reference. The job rebuilds `ollama.sif` only
when it is absent, fails inspection, or the configured source changes. It pulls
`qwen3.8:27b` only when `ollama show` cannot resolve it from the Scratch model
store.

Submit only after the repository and ignored evidence artifacts have been
synchronized:

```bash
sbatch analysis/run_qwen_gautschi.sbatch
```

Slurm sends `SIGUSR1` ten minutes before the time limit. The runner finishes the
active request, writes its atomic checkpoint, exits with the requeue status, and
the batch script requeues the same job. A later manual submission also resumes
from the existing per-PR checkpoints.

For output-length failures, `RETRY_ERRORS_NUM_PREDICT` selects only error
checkpoints and records the retry provider configuration separately. RQ2 also
supports a bounded retry schema that caps the two set-like arrays at their enum
cardinalities without changing the prompt or semantic label validator:

```bash
sbatch --export=ALL,RETRY_ERRORS_NUM_PREDICT=4096,BOUND_RQ2_RETRY_LISTS=1 \
  analysis/run_qwen_gautschi.sbatch
```

## Outputs and Integrity

Collected labels are stored in the ignored `results_qwen/` directories for each
study. Raw responses and checkpoints remain on Scratch. Provider manifests
record configuration, prompt lengths, and input hashes; normalization decisions
are recorded separately in `label_normalizations.json`.

The archived Qwen manifests omit per-field truncation flags. Inspecting which
fields were truncated requires reconstructing the inputs from their hashed
sources. When comparing archived RQ1 prompts across providers, account for the
recorded prompt-version names and trailing whitespace rather than assuming
byte-identical prompt hashes.
