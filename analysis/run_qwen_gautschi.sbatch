#!/bin/bash
#SBATCH --job-name=emse-qwen
#SBATCH --account=davisjam
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=120G
#SBATCH --time=14-00:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=/scratch/gautschi/rcalvome/EMSE-perf-pr-study/logs/qwen-%j.out
#SBATCH --error=/scratch/gautschi/rcalvome/EMSE-perf-pr-study/logs/qwen-%j.err

set -euo pipefail

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
export APPTAINERENV_HOME="/ollama/home"
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

source /etc/profile >/dev/null 2>&1
module load python/3.11.9

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
    python -m venv "${VENV}"
fi
if [[ "${INSTALLED_HASH}" != "${REQUIREMENTS_HASH}" ]]; then
    "${VENV}/bin/python" -m pip install --upgrade pip
    "${VENV}/bin/python" -m pip install --requirement "${REQUIREMENTS}"
    printf '%s\n' "${REQUIREMENTS_HASH}" > "${VENV}/requirements.sha256"
fi
"${VENV}/bin/python" -c 'import pandas, pyarrow, pydantic, openai'

BIND_PATHS="${OLLAMA_MODELS}:/ollama/models,${OLLAMA_HOME}:/ollama/home,${PROJECT_ROOT}:/workspace"
apptainer exec --nv --bind "${BIND_PATHS}" "${SIF}" ollama serve >> "${RESULT_ROOT}/ollama.log" 2>&1 &
OLLAMA_PID=$!

for _ in $(seq 1 60); do
    if apptainer exec --nv --bind "${BIND_PATHS}" "${SIF}" ollama list >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${OLLAMA_PID}" 2>/dev/null; then
        printf 'Ollama server exited during startup.\n' >&2
        exit 1
    fi
    sleep 2
done
apptainer exec --nv --bind "${BIND_PATHS}" "${SIF}" ollama list >/dev/null

if ! apptainer exec --nv --bind "${BIND_PATHS}" "${SIF}" ollama show "${MODEL}" >/dev/null 2>&1; then
    apptainer exec --nv --bind "${BIND_PATHS}" "${SIF}" ollama pull "${MODEL}"
fi
apptainer exec --nv --bind "${BIND_PATHS}" "${SIF}" ollama show "${MODEL}" >/dev/null
apptainer exec --nv --bind "${BIND_PATHS}" "${SIF}" ollama list > "${RESULT_ROOT}/model-inventory.txt"
du -sh "${OLLAMA_MODELS}" > "${RESULT_ROOT}/model-storage.txt"
sha256sum "${SIF}" > "${RESULT_ROOT}/container.sha256"
printf '%s\n' "${IMAGE_SOURCE}" > "${RESULT_ROOT}/container-source.txt"
nvidia-smi > "${RESULT_ROOT}/gpu-inventory.txt"

"${VENV}/bin/python" "${PROJECT_ROOT}/analysis/run_qwen.py" \
    --study rq1 rq2 \
    --model "${MODEL}" \
    --ollama-url "http://${OLLAMA_HOST}" \
    --output-root "${RESULT_ROOT}" \
    --evidence-dir "${PROJECT_ROOT}/mining/sample_evidence/final" \
    --rq1-sample "${PROJECT_ROOT}/data/data/sample/balanced_sample.parquet" \
    --rq1-catalog "${PROJECT_ROOT}/analysis/rq1_optimization_patterns/catalog/updated_optimization_catalog.csv" \
    --rq2-sample "${PROJECT_ROOT}/analysis/rq2_validation/sample/balanced_sample.parquet" &
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
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi
exit "${STATUS}"
