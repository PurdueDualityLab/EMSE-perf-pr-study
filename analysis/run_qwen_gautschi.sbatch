#!/bin/bash
#SBATCH --job-name=emse-qwen
#SBATCH --account=davisjam
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:h100:1
#SBATCH --time=14-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=/scratch/gautschi/rcalvome/EMSE-perf-pr-study/logs/qwen-%j.out
#SBATCH --error=/scratch/gautschi/rcalvome/EMSE-perf-pr-study/logs/qwen-%j.err

set -euo pipefail

printf 'Starting Gautschi Qwen job %s on %s\n' "${SLURM_JOB_ID}" "$(hostname)"

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/gautschi/${USER}}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_ROOT}/EMSE-perf-pr-study}"
CONTAINER_DIR="${SCRATCH_ROOT}/containers"
OLLAMA_ROOT="${SCRATCH_ROOT}/ollama"
OLLAMA_MODELS="${OLLAMA_ROOT}/models"
OLLAMA_HOME="${OLLAMA_ROOT}/home"
APPTAINER_ROOT="${SCRATCH_ROOT}/apptainer"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/qwen3.8-27b}"
VENV="${SCRATCH_ROOT}/venvs/emse-qwen"
SIF="${CONTAINER_DIR}/ollama.sif"
SOURCE_STAMP="${SIF}.source"
OLLAMA_IMAGE_REF="${OLLAMA_IMAGE_REF:-docker://ollama/ollama:latest}"
OLLAMA_IMAGE_DIGEST="${OLLAMA_IMAGE_DIGEST:-}"
MODEL="${MODEL:-qwen3.8:27b}"
OLLAMA_HOST="127.0.0.1:11434"
RUNNER_PID=""
OLLAMA_PID=""
REQUEUE_REQUESTED=0
REQUEUE_ON_SIGNAL="${REQUEUE_ON_SIGNAL:-1}"
LIMIT_PER_ARM="${LIMIT_PER_ARM:-}"
RETRY_ERRORS_NUM_PREDICT="${RETRY_ERRORS_NUM_PREDICT:-}"
BOUND_RQ2_RETRY_LISTS="${BOUND_RQ2_RETRY_LISTS:-0}"

mkdir -p \
    "${CONTAINER_DIR}" \
    "${OLLAMA_MODELS}" \
    "${OLLAMA_HOME}" \
    "${APPTAINER_ROOT}/cache" \
    "${APPTAINER_ROOT}/tmp/${SLURM_JOB_ID}" \
    "${RESULT_ROOT}" \
    "${PROJECT_ROOT}/logs" \
    "${SCRATCH_ROOT}/venvs"

export APPTAINER_CACHEDIR="${APPTAINER_ROOT}/cache"
export APPTAINER_TMPDIR="${APPTAINER_ROOT}/tmp/${SLURM_JOB_ID}"
export APPTAINERENV_OLLAMA_MODELS="/ollama/models"
export APPTAINERENV_OLLAMA_HOST="${OLLAMA_HOST}"
export APPTAINERENV_OLLAMA_FLASH_ATTENTION="1"
export APPTAINERENV_OLLAMA_NUM_PARALLEL="1"
export APPTAINERENV_OLLAMA_MAX_LOADED_MODELS="1"

cleanup() {
    if [[ -n "${OLLAMA_PID}" ]] && kill -0 "${OLLAMA_PID}" 2>/dev/null; then
        kill "${OLLAMA_PID}" 2>/dev/null || true
        wait "${OLLAMA_PID}" 2>/dev/null || true
    fi
    rm -rf "${APPTAINER_TMPDIR}"
}

request_checkpoint() {
    REQUEUE_REQUESTED=1
    if [[ -n "${RUNNER_PID}" ]] && kill -0 "${RUNNER_PID}" 2>/dev/null; then
        kill -USR1 "${RUNNER_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap request_checkpoint USR1 TERM

PYTHON="/apps/spack/gautschi-cpu/apps/python/3.11.9-gcc-14.1.0-hbmvmni/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    printf 'Required Python executable is unavailable: %s\n' "${PYTHON}" >&2
    exit 1
fi
printf 'Using Python: %s\n' "${PYTHON}"

if [[ -n "${OLLAMA_IMAGE_DIGEST}" && "${OLLAMA_IMAGE_REF}" != *@sha256:* ]]; then
    IMAGE_REPOSITORY="${OLLAMA_IMAGE_REF}"
    if [[ "${OLLAMA_IMAGE_REF}" =~ ^(docker://.+):[^/:]+$ ]]; then
        IMAGE_REPOSITORY="${BASH_REMATCH[1]}"
    fi
    IMAGE_SOURCE="${IMAGE_REPOSITORY}@${OLLAMA_IMAGE_DIGEST}"
else
    IMAGE_SOURCE="${OLLAMA_IMAGE_REF}"
fi

CURRENT_SOURCE=""
if [[ -f "${SOURCE_STAMP}" ]]; then
    IFS= read -r CURRENT_SOURCE < "${SOURCE_STAMP}" || true
fi
if [[ ! -f "${SIF}" ]] || ! apptainer inspect "${SIF}" >/dev/null 2>&1 || [[ "${CURRENT_SOURCE}" != "${IMAGE_SOURCE}" ]]; then
    TEMP_SIF="${SIF}.tmp-${SLURM_JOB_ID}"
    rm -f "${TEMP_SIF}"
    apptainer pull --disable-cache "${TEMP_SIF}" "${IMAGE_SOURCE}"
    apptainer inspect "${TEMP_SIF}" >/dev/null
    mv "${TEMP_SIF}" "${SIF}"
    printf '%s\n' "${IMAGE_SOURCE}" > "${SOURCE_STAMP}"
fi

REQUIREMENTS="${PROJECT_ROOT}/analysis/requirements-qwen.txt"
REQUIREMENTS_HASH="$(sha256sum "${REQUIREMENTS}" | cut -d' ' -f1)"
INSTALLED_HASH=""
if [[ -f "${VENV}/requirements.sha256" ]]; then
    IFS= read -r INSTALLED_HASH < "${VENV}/requirements.sha256" || true
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
    "${PYTHON}" -m venv "${VENV}"
fi
if [[ "${INSTALLED_HASH}" != "${REQUIREMENTS_HASH}" ]]; then
    "${VENV}/bin/python" -m pip install --upgrade pip
    "${VENV}/bin/python" -m pip install --requirement "${REQUIREMENTS}"
    printf '%s\n' "${REQUIREMENTS_HASH}" > "${VENV}/requirements.sha256"
fi
"${VENV}/bin/python" -c 'import pandas, pyarrow, pydantic, openai'

BIND_PATHS="${OLLAMA_MODELS}:/ollama/models,${OLLAMA_HOME}:/ollama/home,${PROJECT_ROOT}:/workspace"
APPTAINER_EXEC=(apptainer exec --nv --home "${OLLAMA_HOME}" --bind "${BIND_PATHS}")
"${APPTAINER_EXEC[@]}" "${SIF}" ollama serve >> "${RESULT_ROOT}/ollama.log" 2>&1 &
OLLAMA_PID=$!

for _ in $(seq 1 60); do
    if "${APPTAINER_EXEC[@]}" "${SIF}" ollama list >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${OLLAMA_PID}" 2>/dev/null; then
        printf 'Ollama server exited during startup.\n' >&2
        exit 1
    fi
    sleep 2
done
"${APPTAINER_EXEC[@]}" "${SIF}" ollama list >/dev/null

if ! "${APPTAINER_EXEC[@]}" "${SIF}" ollama show "${MODEL}" >/dev/null 2>&1; then
    "${APPTAINER_EXEC[@]}" "${SIF}" ollama pull "${MODEL}"
fi
"${APPTAINER_EXEC[@]}" "${SIF}" ollama show "${MODEL}" >/dev/null
"${APPTAINER_EXEC[@]}" "${SIF}" ollama list > "${RESULT_ROOT}/model-inventory.txt"
du -sh "${OLLAMA_MODELS}" > "${RESULT_ROOT}/model-storage.txt"
sha256sum "${SIF}" > "${RESULT_ROOT}/container.sha256"
printf '%s\n' "${IMAGE_SOURCE}" > "${RESULT_ROOT}/container-source.txt"
nvidia-smi > "${RESULT_ROOT}/gpu-inventory.txt"

RUNNER_ARGS=(
    --study rq1 rq2
    --model "${MODEL}"
    --ollama-url "http://${OLLAMA_HOST}"
    --output-root "${RESULT_ROOT}"
    --evidence-dir "${PROJECT_ROOT}/mining/sample_evidence/final"
    --rq1-sample "${PROJECT_ROOT}/data/data/sample/balanced_sample.parquet"
    --rq1-catalog "${PROJECT_ROOT}/analysis/rq1_optimization_patterns/catalog/updated_optimization_catalog.csv"
    --rq2-sample "${PROJECT_ROOT}/analysis/rq2_validation/sample/balanced_sample.parquet"
    --adjudicate-unique-parents
    --normalize-duplicate-labels
    --defer-length-errors
    --max-consecutive-errors 10000
)
if [[ -n "${LIMIT_PER_ARM}" ]]; then
    RUNNER_ARGS+=(--limit-per-arm "${LIMIT_PER_ARM}")
fi
if [[ -n "${RETRY_ERRORS_NUM_PREDICT}" ]]; then
    RUNNER_ARGS+=(--retry-errors-num-predict "${RETRY_ERRORS_NUM_PREDICT}")
fi
if [[ "${BOUND_RQ2_RETRY_LISTS}" -eq 1 ]]; then
    RUNNER_ARGS+=(--bound-rq2-retry-lists)
fi
"${VENV}/bin/python" "${PROJECT_ROOT}/analysis/run_qwen.py" "${RUNNER_ARGS[@]}" &
RUNNER_PID=$!

set +e
wait "${RUNNER_PID}"
STATUS=$?
if kill -0 "${RUNNER_PID}" 2>/dev/null; then
    wait "${RUNNER_PID}"
    STATUS=$?
fi
set -e

if [[ "${REQUEUE_REQUESTED}" -eq 1 || "${STATUS}" -eq 75 ]]; then
    if [[ "${REQUEUE_ON_SIGNAL}" -eq 1 ]]; then
        scontrol requeue "${SLURM_JOB_ID}"
        exit 0
    fi
    exit 75
fi
exit "${STATUS}"
