#!/usr/bin/env bash
# Evaluate a base model or exported VERL checkpoint using LightEval's sampling unit:
# one vLLM request per problem with n completions inside that request.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

REQUESTED_MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a base model or VERL checkpoint directory}
DATA_DIR=${DATA_DIR:-/data/user/zhongal/data/reschedule}
AIME24_FILE=${AIME24_FILE:-${DATA_DIR}/aime24.parquet}

EVAL_N=${EVAL_N:-64}
PASS_KS=${PASS_KS:-1,2,4,8,16,32,64}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.6}
EVAL_TOP_P=${EVAL_TOP_P:-0.95}
EVAL_SEED=${EVAL_SEED:-42}
# LightEval passes the same configured engine seed to every data-parallel
# vLLM replica. Keep this local to post-hoc evaluation; training retains VERL's
# historical seed + replica_rank behavior unless explicitly overridden.
ROLLOUT_REPLICA_SEED_MODE=${ROLLOUT_REPLICA_SEED_MODE:-shared}
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-${NGPUS_PER_NODE:-1}}
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-4096}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
EXPECTED_PROBLEMS=${EXPECTED_PROBLEMS:-30}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}

if (( EVAL_N < 1 )); then
    echo "EVAL_N must be positive; got ${EVAL_N}" >&2
    exit 1
fi
if (( EXPECTED_PROBLEMS < 1 )); then
    echo "EXPECTED_PROBLEMS must be positive; got ${EXPECTED_PROBLEMS}" >&2
    exit 1
fi
if [ ! -f "${AIME24_FILE}" ]; then
    echo "AIME24 parquet not found: ${AIME24_FILE}" >&2
    exit 1
fi
if [ "${ROLLOUT_REPLICA_SEED_MODE}" != "shared" ]; then
    echo "LightEval-compatible posthoc requires ROLLOUT_REPLICA_SEED_MODE=shared" >&2
    exit 1
fi

MODEL_PATH=""
for candidate in \
    "${REQUESTED_MODEL_PATH}" \
    "${REQUESTED_MODEL_PATH}/actor/huggingface" \
    "${REQUESTED_MODEL_PATH}/huggingface"; do
    if [ -f "${candidate}/config.json" ]; then
        MODEL_PATH=${candidate}
        break
    fi
done
if [ -z "${MODEL_PATH}" ]; then
    echo "No Hugging Face config.json found under ${REQUESTED_MODEL_PATH}." >&2
    echo "Pass a base-model directory, actor/huggingface directory, or global_step_* directory." >&2
    exit 1
fi

MODEL_TAG=$(basename -- "${REQUESTED_MODEL_PATH%/}")
RUN_NAME=${RUN_NAME:-aime24_${MODEL_TAG}_n${EVAL_N}_seed${EVAL_SEED}}
PROJECT_NAME=${PROJECT_NAME:-aime24_posthoc_validation}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/validation_data/${PROJECT_NAME}/${RUN_NAME}}
RAW_DIR=${RAW_DIR:-${OUTPUT_DIR}/raw}
UNIQUE_AIME24_FILE=${UNIQUE_AIME24_FILE:-${OUTPUT_DIR}/aime24.unique.parquet}
mkdir -p "${RAW_DIR}"
if compgen -G "${RAW_DIR}/*.jsonl" >/dev/null; then
    echo "Raw validation directory already contains JSONL files: ${RAW_DIR}" >&2
    echo "Choose a new RUN_NAME or OUTPUT_DIR to avoid mixing evaluation runs." >&2
    exit 1
fi

# Some AIME parquet files physically repeat each of the 30 prompts 32 times.
# VERL applies rollout.val_kwargs.n to every input row, so feeding those 960
# rows directly with n=32 would accidentally request 30,720 completions.
python3 "${SCRIPT_DIR}/prepare_aime24_validation.py" \
    --input "${AIME24_FILE}" \
    --output "${UNIQUE_AIME24_FILE}" \
    --expected-problems "${EXPECTED_PROBLEMS}"

EXPECTED_ROLLOUTS=$((EXPECTED_PROBLEMS * EVAL_N))
echo "AIME24 rollout preflight"
echo "  unique input rows: ${EXPECTED_PROBLEMS}"
echo "  rollouts/problem: ${EVAL_N}"
echo "  expected total rollouts: ${EXPECTED_ROLLOUTS}"

echo "AIME24 post-hoc validation"
echo "  requested model: ${REQUESTED_MODEL_PATH}"
echo "  resolved HF model: ${MODEL_PATH}"
echo "  source dataset: ${AIME24_FILE}"
echo "  validation dataset: ${UNIQUE_AIME24_FILE}"
echo "  samples/problem: ${EVAL_N}"
echo "  temperature/top_p: ${EVAL_TEMPERATURE}/${EVAL_TOP_P}"
echo "  seed: ${EVAL_SEED} (replica mode: ${ROLLOUT_REPLICA_SEED_MODE})"
echo "  sampling unit: one problem/request with n=${EVAL_N}"
echo "  data-parallel replicas: ${DATA_PARALLEL_SIZE} x TP=1"
echo "  max response tokens: ${MAX_RESPONSE_LENGTH}"
echo "  output: ${OUTPUT_DIR}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0}"
if (( DATA_PARALLEL_SIZE > ${#GPU_IDS[@]} )); then
    echo "Requested ${DATA_PARALLEL_SIZE} replicas but CUDA_VISIBLE_DEVICES has ${#GPU_IDS[@]} GPUs" >&2
    exit 1
fi

pids=()
for ((replica = 0; replica < DATA_PARALLEL_SIZE; replica++)); do
    replica_cache="${OUTPUT_DIR}/vllm-cache/replica-${replica}"
    mkdir -p "${replica_cache}"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[replica]}" \
    VLLM_CACHE_ROOT="${replica_cache}" \
    TORCHINDUCTOR_CACHE_DIR="${replica_cache}/torchinductor" \
    python3 "${SCRIPT_DIR}/generate_aime24_vllm_replica.py" \
        --model "${MODEL_PATH}" \
        --data "${UNIQUE_AIME24_FILE}" \
        --output "${RAW_DIR}/replica-${replica}.jsonl" \
        --replica-index "${replica}" \
        --num-replicas "${DATA_PARALLEL_SIZE}" \
        --samples-per-problem "${EVAL_N}" \
        --temperature "${EVAL_TEMPERATURE}" \
        --top-p "${EVAL_TOP_P}" \
        --seed "${EVAL_SEED}" \
        --max-model-len "${MAX_MODEL_LENGTH}" \
        --max-response-tokens "${MAX_RESPONSE_LENGTH}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
if (( status != 0 )); then
    echo "At least one vLLM replica failed" >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/analyze_aime24_validation.py" \
    --input "${RAW_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --pass-k "${PASS_KS}" \
    --expected-problems "${EXPECTED_PROBLEMS}" \
    --expected-samples-per-problem "${EVAL_N}" \
    --model-path "${MODEL_PATH}" \
    --requested-model-path "${REQUESTED_MODEL_PATH}" \
    --temperature "${EVAL_TEMPERATURE}" \
    --top-p "${EVAL_TOP_P}" \
    --seed "${EVAL_SEED}" \
    --replica-seed-mode "${ROLLOUT_REPLICA_SEED_MODE}" \
    --sampling-unit problem-request-n \
    --strict-boxed-verifier
