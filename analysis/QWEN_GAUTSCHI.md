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

## Completed run

The completed `qwen3.8:27b` run produced 2,260/2,260 classified RQ1 rows and
2,258/2,258 classified RQ2 rows. Every final response ended with `stop`; no
final response reached its output-token limit. Final local artifacts are kept in
the ignored `results_qwen/` directories for each study, while raw responses and
checkpoints remain on Scratch.

RQ1 required 61 deterministic unique-parent taxonomy adjudications and one
retry of an empty response. Qwen-GPT hierarchical agreement is 67.83%
(1,533/2,260; Cohen's kappa 0.6513), and Qwen-Gemini agreement is 63.89%
(1,444/2,260; kappa 0.6089). All three models agree on 54.78% of rows
(1,238/2,260). Where GPT and Gemini agree, Qwen matches their label on 78.26%
of rows. Qwen's RQ1 input-row hashes match both prior providers for every row.
Its rendered prompts also match all prior prompts after removing trailing
whitespace, but their exact hashes differ because the prior artifacts retain the
`rq1-optimization-pattern-legacy-v2` prompt-version name and trailing whitespace.

RQ2 initially produced 74 output-length failures. An 8,192-token retry recovered
13 through short regenerated responses, but the remaining 61 repeated
`"code_diff"` hundreds of times in `evidence_sources` until exhausting the new
limit. A third retry capped `validation_types` at four items and
`evidence_sources` at five, preserving both earlier attempts, and classified all
61 with the standard 4,096-token budget. Duplicate set-like values were removed
in first-seen order under the approved normalization rule; 169 such corrections
are recorded in `label_normalizations.json`. Qwen-GPT full RQ2 agreement is
74.62% (1,685/2,258), and Qwen-Gemini agreement is 67.23% (1,518/2,258). All
three models fully agree on 55.85% of rows (1,261/2,258). Where GPT and Gemini
fully agree, Qwen matches them on 87.27% of rows. RQ2 prompt and input-row hashes
match both prior providers for all 2,258 rows.

The Qwen manifests record prompt lengths and input hashes but omit per-field
truncation flags. Reconstructing inputs from the hashed sources found 692 RQ1
code diffs over the 15,000-character limit. For RQ2, the corresponding counts
are 9 descriptions, 284 comment groups, 23 review groups, 712 code diffs, and
226 CI groups. The prompts apply the documented truncation marker in every such
case; this is an artifact-audit limitation rather than a classification error.
