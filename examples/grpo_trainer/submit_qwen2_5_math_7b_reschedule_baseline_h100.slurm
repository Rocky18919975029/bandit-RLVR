#!/bin/bash
# Slurm wrapper for the copied Re-Schedule GRPO baseline launcher.

#SBATCH --job-name=verl-resched-grpo
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -xeuo pipefail

VERL_WORKDIR=${VERL_WORKDIR:-${SLURM_SUBMIT_DIR}}
export VERL_WORKDIR

LD_LIBRARY_PATH=":${LD_LIBRARY_PATH:-}:"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH//:\/data\/apps\/glibc\/2.32\/lib:/:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH//:\/data\/apps\/glibc\/2.32\/lib64:/:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH#:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH%:}
export LD_LIBRARY_PATH

module purge
module load gcc/13.3
module load cuda/12.8
module list

CONDA_SH=${CONDA_SH:-/share/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/user/zhongal/.conda/envs/verl}
if [ ! -f "${CONDA_SH}" ]; then
    echo "Conda init script not found: ${CONDA_SH}" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_SH}"
conda deactivate || true
conda activate "${CONDA_ENV_PATH}"
hash -r

cd "${VERL_WORKDIR}"

python - <<'PY'
import os
import sys
from pathlib import Path

import torch
import vllm
import verl
from packaging import version

print("python:", sys.version)
print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
print("verl:", verl.__file__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device visible inside the Slurm allocation.")
print("gpu:", torch.cuda.get_device_name(0))

workdir = Path(os.environ["VERL_WORKDIR"]).resolve()
expected_verl = workdir / "verl" / "__init__.py"
actual_verl = Path(verl.__file__).resolve()
if actual_verl != expected_verl:
    raise SystemExit(f"Wrong VERL imported: expected {expected_verl}, got {actual_verl}")
if version.parse(vllm.__version__) < version.parse("0.18.0"):
    raise SystemExit(
        f"{workdir} requires vLLM >= 0.18.0; found {vllm.__version__}. "
        "Install the offline vLLM bundle before submitting GRPO."
    )
print("bandit-RLVR runtime compatibility: OK")
PY

MODEL_PATH=${MODEL_PATH:-/data/user/zhongal/.cache/qwen2.5-math-7b-local}
DATA_DIR=${DATA_DIR:-/data/user/zhongal/data/reschedule}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/DAPO-Math-17k.filtered.seed42.sample1536.parquet}
MODE=${MODE:-train}

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "Local model config not found: ${MODEL_PATH}/config.json" >&2
    exit 1
fi
if [ ! -f "${TRAIN_FILE}" ]; then
    echo "Training parquet not found: ${TRAIN_FILE}" >&2
    exit 1
fi

if [ "${MODE}" = "eval" ]; then
    REQUIRED_VAL_FILES=(
        aime24.parquet
        aime25.parquet
        amc23.parquet
        math500.parquet
        minerva_math.parquet
        olympiadbench.parquet
    )
else
    REQUIRED_VAL_FILES=(aime24.parquet)
fi
for filename in "${REQUIRED_VAL_FILES[@]}"; do
    if [ ! -f "${DATA_DIR}/${filename}" ]; then
        echo "Validation parquet not found: ${DATA_DIR}/${filename}" >&2
        exit 1
    fi
done

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export RAY_TMP_DIR=${RAY_TMP_DIR:-/tmp/ray_${USER:-u}_${SLURM_JOB_ID}}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-${VERL_WORKDIR}/wandb}
export WANDB_SILENT=${WANDB_SILENT:-true}
export WANDB_START_METHOD=${WANDB_START_METHOD:-thread}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CC=${CC:-$(which gcc)}
export CXX=${CXX:-$(which g++)}
export CUDAHOSTCXX=${CUDAHOSTCXX:-$(which g++)}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}

mkdir -p "${WANDB_DIR}" "${RAY_TMP_DIR}"

PROJECT_NAME=${PROJECT_NAME:-grpo_dapo_math17k_reschedule_baseline}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_math_7b_grpo_baseline_${SLURM_JOB_ID}}
ROLLOUT_N=${ROLLOUT_N:-32}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
ROLLOUT_TP=${ROLLOUT_TP:-4}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
TEST_FREQ=${TEST_FREQ:-1}
SAVE_CONTENTS=${SAVE_CONTENTS:-"['model','optimizer','extra','hf_model']"}
SAVE_FREQ=${SAVE_FREQ:-1}
RESUME_MODE=${RESUME_MODE:-disable}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
ROLLOUT_LOGPROB_REUSE=${ROLLOUT_LOGPROB_REUSE:-True}

if [[ ${SLURM_GPUS_ON_NODE:-} =~ ([0-9]+)$ ]]; then
    NGPUS_PER_NODE=${BASH_REMATCH[1]}
else
    IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    NGPUS_PER_NODE=${#VISIBLE_GPU_IDS[@]}
fi

SMOKE_TEST=${SMOKE_TEST:-0}
if [ "${SMOKE_TEST}" = "1" ]; then
    MODE=train
    EXPERIMENT_NAME=qwen2_5_math_7b_grpo_smoke_${SLURM_JOB_ID}
    TRAIN_BATCH_SIZE=8
    ROLLOUT_N=2
    PPO_MINI_BATCH_SIZE=8
    PPO_MICRO_BATCH_SIZE_PER_GPU=1
    LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
    MAX_RESPONSE_LENGTH=512
    ROLLOUT_TP=1
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.35
    TOTAL_TRAINING_STEPS=1
    TEST_FREQ=-1
    SAVE_FREQ=-1
    RESUME_MODE=disable
    TRAINER_LOGGER='["console"]'
    WANDB_MODE=disabled
fi

echo "Job ID: ${SLURM_JOB_ID}"
echo "Working directory: ${VERL_WORKDIR}"
echo "Model: ${MODEL_PATH}"
echo "Training file: ${TRAIN_FILE}"
echo "Smoke test: ${SMOKE_TEST}"
echo "Allocated GPUs: ${NGPUS_PER_NODE}"
echo "Rollout tensor parallelism: ${ROLLOUT_TP}"
echo "Training prompts per step: ${TRAIN_BATCH_SIZE}"
echo "Rollouts per prompt: ${ROLLOUT_N}"

MODEL_PATH="${MODEL_PATH}" \
DATA_DIR="${DATA_DIR}" \
TRAIN_FILE="${TRAIN_FILE}" \
PROJECT_NAME="${PROJECT_NAME}" \
RUN_NAME="${EXPERIMENT_NAME}" \
MODE="${MODE}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
ROLLOUT_N="${ROLLOUT_N}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
ROLLOUT_TP="${ROLLOUT_TP}" \
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
TEST_FREQ="${TEST_FREQ}" \
SAVE_CONTENTS="${SAVE_CONTENTS}" \
SAVE_FREQ="${SAVE_FREQ}" \
RESUME_MODE="${RESUME_MODE}" \
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD}" \
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD}" \
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD}" \
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
ROLLOUT_LOGPROB_REUSE="${ROLLOUT_LOGPROB_REUSE}" \
NNODES=1 \
NGPUS_PER_NODE="${NGPUS_PER_NODE}" \
bash examples/grpo_trainer/run_qwen2_5_math_7b_grpo_reschedule_baseline.sh "$@"
