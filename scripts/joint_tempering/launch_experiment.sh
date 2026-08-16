#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

: "${MODEL_PATH:?Set MODEL_PATH to an HPC-local model or checkpoint directory}"

PYTHON_BIN=${PYTHON_BIN:-python3}
DATA_PATH=${DATA_PATH:-/data/user/zhongal/data/reschedule/DAPO-Math-17k.filtered.seed42.sample1536.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/outputs/joint_tempering}
NUM_PROBLEMS=${NUM_PROBLEMS:-128}
PROBLEM_SEED=${PROBLEM_SEED:-42}
POOL_CANDIDATES=${POOL_CANDIDATES:-16}
MAX_RESPONSE_TOKENS=${MAX_RESPONSE_TOKENS:-4096}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
DTYPE=${DTYPE:-auto}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
GENERATION_SEED=${GENERATION_SEED:-1234}
POOL_PROMPT_BATCH_SIZE=${POOL_PROMPT_BATCH_SIZE:-2}
MYOPIC_REQUEST_BATCH_SIZE=${MYOPIC_REQUEST_BATCH_SIZE:-16}
MYOPIC_REPEATS=${MYOPIC_REPEATS:-1}
BLOCK_LENGTHS=${BLOCK_LENGTHS:-"16 32 64 128"}
ALPHAS=${ALPHAS:-"1.25 1.5 2.0"}
CANDIDATE_COUNTS=${CANDIDATE_COUNTS:-"4 8 16"}
RUN_POOL=${RUN_POOL:-1}
RUN_SWEEP=${RUN_SWEEP:-1}
RUN_MYOPIC=${RUN_MYOPIC:-1}
RUN_ANALYSIS=${RUN_ANALYSIS:-1}

read -r -a BLOCK_LENGTH_ARGS <<< "${BLOCK_LENGTHS}"
read -r -a ALPHA_ARGS <<< "${ALPHAS}"
read -r -a CANDIDATE_COUNT_ARGS <<< "${CANDIDATE_COUNTS}"
MODEL_LENGTH_ARGS=()
if [[ -n "${MAX_MODEL_LEN}" ]]; then
    MODEL_LENGTH_ARGS=(--max-model-len "${MAX_MODEL_LEN}")
fi

POOL_PATH=${OUTPUT_DIR}/pool.jsonl
JOINT_PATH=${OUTPUT_DIR}/joint_sweep.jsonl
MYOPIC_PATH=${OUTPUT_DIR}/myopic.jsonl
SUMMARY_PREFIX=${OUTPUT_DIR}/summary

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" -c 'import pandas, pyarrow, torch, transformers, vllm; print("offline dependencies: ok")'

if [[ "${RUN_POOL}" == "1" ]]; then
    "${PYTHON_BIN}" examples/joint_tempering/generate_pool.py \
        --model "${MODEL_PATH}" \
        --data "${DATA_PATH}" \
        --output "${POOL_PATH}" \
        --num-problems "${NUM_PROBLEMS}" \
        --problem-seed "${PROBLEM_SEED}" \
        --num-candidates "${POOL_CANDIDATES}" \
        --max-response-tokens "${MAX_RESPONSE_TOKENS}" \
        --prompt-batch-size "${POOL_PROMPT_BATCH_SIZE}" \
        --generation-seed "${GENERATION_SEED}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --dtype "${DTYPE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        "${MODEL_LENGTH_ARGS[@]}"
fi

if [[ "${RUN_SWEEP}" == "1" ]]; then
    "${PYTHON_BIN}" examples/joint_tempering/sweep_joint.py \
        --pool "${POOL_PATH}" \
        --output "${JOINT_PATH}" \
        --block-lengths "${BLOCK_LENGTH_ARGS[@]}" \
        --alphas "${ALPHA_ARGS[@]}" \
        --candidate-counts "${CANDIDATE_COUNT_ARGS[@]}"
fi

if [[ "${RUN_MYOPIC}" == "1" ]]; then
    "${PYTHON_BIN}" examples/joint_tempering/generate_myopic.py \
        --model "${MODEL_PATH}" \
        --pool "${POOL_PATH}" \
        --output "${MYOPIC_PATH}" \
        --block-lengths "${BLOCK_LENGTH_ARGS[@]}" \
        --alphas "${ALPHA_ARGS[@]}" \
        --num-repeats "${MYOPIC_REPEATS}" \
        --max-response-tokens "${MAX_RESPONSE_TOKENS}" \
        --request-batch-size "${MYOPIC_REQUEST_BATCH_SIZE}" \
        --generation-seed "${GENERATION_SEED}" \
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
        --dtype "${DTYPE}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        "${MODEL_LENGTH_ARGS[@]}"
fi

if [[ "${RUN_ANALYSIS}" == "1" ]]; then
    "${PYTHON_BIN}" examples/joint_tempering/analyze_results.py \
        --joint "${JOINT_PATH}" \
        --myopic "${MYOPIC_PATH}" \
        --output-prefix "${SUMMARY_PREFIX}"
fi
