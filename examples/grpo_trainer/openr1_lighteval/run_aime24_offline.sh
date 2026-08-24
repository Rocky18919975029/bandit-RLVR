#!/usr/bin/env bash
# Run the pinned Open-R1/LightEval AIME24 protocol without network access.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

LIGHTEVAL_COMMIT=d3da6b9bbf38104c8b5e1acc86f83541f9a502d1
TASK_NAME='lighteval|aime24|0|0'
EVAL_TEMPERATURE=0.6
EVAL_TOP_P=0.95
EVAL_N=64
EXPECTED_PROBLEMS=30

REQUESTED_MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a base model or exported VERL checkpoint}
AIME24_FILE=${AIME24_FILE:?Set AIME24_FILE to the local repeated or unique AIME24 parquet}
EVAL_ENV_PATH=${EVAL_ENV_PATH:?Set EVAL_ENV_PATH to the isolated LightEval environment}
EVAL_SEED=${EVAL_SEED:-42}
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-8}
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}

if [ "${CONDA_PREFIX:-}" != "${EVAL_ENV_PATH}" ]; then
    echo "Wrong environment: expected CONDA_PREFIX=${EVAL_ENV_PATH}, got ${CONDA_PREFIX:-<unset>}" >&2
    echo "Use submit_openr1_aime24_lighteval_h100.slurm; do not activate the training environment." >&2
    exit 1
fi
if [ ! -f "${AIME24_FILE}" ]; then
    echo "AIME24 parquet not found: ${AIME24_FILE}" >&2
    exit 1
fi
if (( MAX_NEW_TOKENS >= MAX_MODEL_LENGTH )); then
    echo "MAX_NEW_TOKENS must be smaller than MAX_MODEL_LENGTH" >&2
    exit 1
fi

MODEL_PATH=""
for candidate in \
    "${REQUESTED_MODEL_PATH}" \
    "${REQUESTED_MODEL_PATH}/actor/huggingface" \
    "${REQUESTED_MODEL_PATH}/huggingface"; do
    if [ -f "${candidate}/config.json" ]; then
        MODEL_PATH=$(cd -- "${candidate}" && pwd)
        break
    fi
done
if [ -z "${MODEL_PATH}" ]; then
    echo "No Hugging Face config.json found under ${REQUESTED_MODEL_PATH}" >&2
    exit 1
fi

MODEL_TAG=$(basename -- "${REQUESTED_MODEL_PATH%/}")
RUN_NAME=${RUN_NAME:-openr1_aime24_${MODEL_TAG}_t0p6_p0p95_n64_seed${EVAL_SEED}_$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/validation_data/openr1_lighteval_aime24}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}
DATASET_DIR=${DATASET_DIR:-${OUTPUT_DIR}/dataset}

if [ -e "${OUTPUT_DIR}" ]; then
    echo "Output directory already exists: ${OUTPUT_DIR}" >&2
    echo "Set a new RUN_NAME or OUTPUT_DIR to keep runs isolated." >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH

python "${SCRIPT_DIR}/prepare_aime24_dataset.py" \
    --input "${AIME24_FILE}" \
    --output-dir "${DATASET_DIR}" \
    --expected-problems "${EXPECTED_PROBLEMS}"

export OPENR1_AIME24_DATASET_DIR=${DATASET_DIR}

python - "${OUTPUT_DIR}/protocol_manifest.json" <<PY
import json
import os
import sys
from pathlib import Path

manifest = {
    "protocol": "Open-R1 LightEval AIME24 sampling",
    "lighteval_commit": "${LIGHTEVAL_COMMIT}",
    "task": "${TASK_NAME}",
    "problem_count": ${EXPECTED_PROBLEMS},
    "samples_per_problem": ${EVAL_N},
    "total_completions": ${EXPECTED_PROBLEMS} * ${EVAL_N},
    "temperature": ${EVAL_TEMPERATURE},
    "top_p": ${EVAL_TOP_P},
    "seed": ${EVAL_SEED},
    "data_parallel_size": ${DATA_PARALLEL_SIZE},
    "tensor_parallel_size": 1,
    "enforce_eager": True,
    "vllm_compile_cache_disabled": os.environ.get("VLLM_DISABLE_COMPILE_CACHE") == "1",
    "vllm_cache_root": os.environ.get("VLLM_CACHE_ROOT"),
    "max_model_length": ${MAX_MODEL_LENGTH},
    "max_new_tokens": ${MAX_NEW_TOKENS},
    "model_path": "${MODEL_PATH}",
    "source_aime24_file": str(Path("${AIME24_FILE}").resolve()),
    "offline_flags": {
        key: os.environ[key]
        for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    },
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "Open-R1 LightEval AIME24 protocol audit"
echo "  LightEval commit: ${LIGHTEVAL_COMMIT}"
echo "  model: ${MODEL_PATH}"
echo "  problems: ${EXPECTED_PROBLEMS}"
echo "  samples/problem: ${EVAL_N} (total=$((EXPECTED_PROBLEMS * EVAL_N)))"
echo "  temperature/top_p: ${EVAL_TEMPERATURE}/${EVAL_TOP_P}"
echo "  seed: ${EVAL_SEED}"
echo "  topology: ${DATA_PARALLEL_SIZE} independent vLLM replicas x TP=1"
echo "  vLLM execution: enforce_eager=True, compile cache disabled"
echo "  max model/new tokens: ${MAX_MODEL_LENGTH}/${MAX_NEW_TOKENS}"
echo "  output: ${OUTPUT_DIR}"

MODEL_ARGS="model_name=${MODEL_PATH},dtype=bfloat16,tensor_parallel_size=1,data_parallel_size=${DATA_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_length=${MAX_MODEL_LENGTH},max_num_seqs=${MAX_NUM_SEQS},max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS},seed=${EVAL_SEED},trust_remote_code=True,use_chat_template=True,generation_parameters={temperature:${EVAL_TEMPERATURE},top_p:${EVAL_TOP_P},seed:${EVAL_SEED},max_new_tokens:${MAX_NEW_TOKENS}}"

PYTHONPATH="${SCRIPT_DIR}" python "${SCRIPT_DIR}/run_lighteval_vllm.py" \
    --model-args "${MODEL_ARGS}" \
    --tasks "${TASK_NAME}" \
    --custom-tasks openr1_aime24_task \
    --output-dir "${OUTPUT_DIR}/lighteval"

echo "LightEval completed. Result files:"
find "${OUTPUT_DIR}/lighteval" -type f \( -name 'results_*.json' -o -name 'details_*.parquet' \) -print | sort
