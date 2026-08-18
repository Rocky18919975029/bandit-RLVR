#!/usr/bin/env bash
# Evaluate a base model or an exported VERL checkpoint on AIME 2024 with FSDP.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

REQUESTED_MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a base model or VERL checkpoint directory}
DATA_DIR=${DATA_DIR:-/data/user/zhongal/data/reschedule}
AIME24_FILE=${AIME24_FILE:-${DATA_DIR}/aime24.parquet}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/DAPO-Math-17k.filtered.seed42.sample1536.parquet}

EVAL_N=${EVAL_N:-32}
PASS_KS=${PASS_KS:-1,2,4,8,16,32}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-1.0}
EVAL_TOP_P=${EVAL_TOP_P:-1.0}
EVAL_SEED=${EVAL_SEED:-42}
EXPECTED_PROBLEMS=${EXPECTED_PROBLEMS:-30}
VALIDATION_PROBLEM_BATCH_SIZE=${VALIDATION_PROBLEM_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
WANDB_MODE=${WANDB_MODE:-disabled}

if (( EVAL_N < 1 )); then
    echo "EVAL_N must be positive; got ${EVAL_N}" >&2
    exit 1
fi
if (( VALIDATION_PROBLEM_BATCH_SIZE < 1 )); then
    echo "VALIDATION_PROBLEM_BATCH_SIZE must be positive; got ${VALIDATION_PROBLEM_BATCH_SIZE}" >&2
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
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Training parquet required to initialize VERL was not found: ${TRAIN_FILE}" >&2
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
echo "  problems/generation batch: ${VALIDATION_PROBLEM_BATCH_SIZE}"
echo "  temperature/top_p: ${EVAL_TEMPERATURE}/${EVAL_TOP_P}"
echo "  max response tokens: ${MAX_RESPONSE_LENGTH}"
echo "  output: ${OUTPUT_DIR}"

MODE=eval \
MODEL_PATH="${MODEL_PATH}" \
DATA_DIR="${DATA_DIR}" \
TRAIN_FILE="${TRAIN_FILE}" \
VAL_FILES="['${UNIQUE_AIME24_FILE}']" \
PROJECT_NAME="${PROJECT_NAME}" \
RUN_NAME="${RUN_NAME}" \
VALIDATION_DATA_DIR="${RAW_DIR}" \
TRAINER_LOGGER="${TRAINER_LOGGER}" \
WANDB_MODE="${WANDB_MODE}" \
ROLLOUT_N=2 \
ROLLOUT_VAL_N="${EVAL_N}" \
ROLLOUT_VAL_TEMPERATURE="${EVAL_TEMPERATURE}" \
ROLLOUT_VAL_TOP_P="${EVAL_TOP_P}" \
ROLLOUT_SEED="${EVAL_SEED}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
REWARD_STRICT_BOX_VERIFY=True \
SIR_ENABLE=False \
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
    --strict-boxed-verifier
