#!/usr/bin/env bash
# Reproduce the historical AIME24 protocol used for the original step-1
# exact-replay GRPO control: VERL validation expansion, 8 DP replicas with
# rank-offset seeds, n=32, temperature=1, and top_p=1.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

REQUESTED_MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a base model or VERL checkpoint directory}
DATA_DIR=${DATA_DIR:-/data/user/zhongal/data/reschedule}
AIME24_FILE=${AIME24_FILE:-${DATA_DIR}/aime24.parquet}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/DAPO-Math-17k.filtered.seed42.sample1536.parquet}

# These values define the historical control protocol and are intentionally
# not configurable. Use run_aime24_posthoc_validation_fsdp.sh for new protocols.
EVAL_N=32
PASS_KS=1,2,4,8,16,32
EVAL_TEMPERATURE=1.0
EVAL_TOP_P=1.0
EVAL_SEED=42
ROLLOUT_REPLICA_SEED_MODE=rank_offset
EXPECTED_PROBLEMS=30
VALIDATION_PROBLEM_BATCH_SIZE=8
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=3072

if [ ! -f "${AIME24_FILE}" ]; then
    echo "AIME24 parquet not found: ${AIME24_FILE}" >&2
    exit 1
fi
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Training parquet required to initialize VERL was not found: ${TRAIN_FILE}" >&2
    exit 1
fi
if [ "${NGPUS_PER_NODE:-0}" != "8" ] || [ "${ROLLOUT_TP:-0}" != "1" ]; then
    echo "Historical step-1 control evaluation requires 8 GPUs as 8 DP replicas with TP=1" >&2
    echo "Got NGPUS_PER_NODE=${NGPUS_PER_NODE:-unset}, ROLLOUT_TP=${ROLLOUT_TP:-unset}" >&2
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
RUN_NAME=${RUN_NAME:-aime24_step1_control_protocol_${MODEL_TAG}_n32_seed42}
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

python3 "${SCRIPT_DIR}/prepare_aime24_validation.py" \
    --input "${AIME24_FILE}" \
    --output "${UNIQUE_AIME24_FILE}" \
    --expected-problems "${EXPECTED_PROBLEMS}"

echo "Historical step-1 control AIME24 protocol audit"
echo "  requested model: ${REQUESTED_MODEL_PATH}"
echo "  resolved HF model: ${MODEL_PATH}"
echo "  problems: ${EXPECTED_PROBLEMS}"
echo "  samples/problem: ${EVAL_N} (total=$((EXPECTED_PROBLEMS * EVAL_N)))"
echo "  temperature/top_p: ${EVAL_TEMPERATURE}/${EVAL_TOP_P}"
echo "  seed: ${EVAL_SEED} (replica mode: ${ROLLOUT_REPLICA_SEED_MODE})"
echo "  sampling unit: expanded-requests"
echo "  topology: 8 DP replicas x TP=1"
echo "  max response tokens: ${MAX_RESPONSE_LENGTH}"
echo "  strict boxed verifier: true"
echo "  output: ${OUTPUT_DIR}"

MODE=eval \
MODEL_PATH="${MODEL_PATH}" \
DATA_DIR="${DATA_DIR}" \
TRAIN_FILE="${TRAIN_FILE}" \
VAL_FILES="['${UNIQUE_AIME24_FILE}']" \
PROJECT_NAME="${PROJECT_NAME}" \
RUN_NAME="${RUN_NAME}" \
VALIDATION_DATA_DIR="${RAW_DIR}" \
TRAINER_LOGGER='["console"]' \
WANDB_MODE=disabled \
ROLLOUT_N=2 \
ROLLOUT_VAL_N="${EVAL_N}" \
ROLLOUT_VAL_TEMPERATURE="${EVAL_TEMPERATURE}" \
ROLLOUT_VAL_TOP_P="${EVAL_TOP_P}" \
ROLLOUT_SEED="${EVAL_SEED}" \
ROLLOUT_REPLICA_SEED_MODE="${ROLLOUT_REPLICA_SEED_MODE}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
REWARD_STRICT_BOX_VERIFY=True \
SIR_ENABLE=False \
SIR_INITIAL_REPLAY_PATH="" \
TEMPERED_GRPO_ENABLE=False \
bash "${SCRIPT_DIR}/run_qwen2_5_math_7b_grpo_reschedule_baseline.sh" \
    data.val_batch_size="${VALIDATION_PROBLEM_BATCH_SIZE}" \
    "$@"

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
    --sampling-unit expanded-requests \
    --strict-boxed-verifier
