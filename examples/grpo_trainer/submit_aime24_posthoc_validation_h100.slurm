#!/bin/bash

#SBATCH --job-name=aime24-eval
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --exclude=ACD1-16
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

VERL_WORKDIR=${VERL_WORKDIR:-${SLURM_SUBMIT_DIR}}
CONDA_SH=${CONDA_SH:-/share/anaconda3/etc/profile.d/conda.sh}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/user/zhongal/.conda/envs/verl}
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

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export CC=${CC:-$(which gcc)}
export CXX=${CXX:-$(which g++)}
export CUDAHOSTCXX=${CUDAHOSTCXX:-$(which g++)}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export RAY_TMP_DIR=${RAY_TMP_DIR:-/tmp/ray_${USER:-u}_${SLURM_JOB_ID}}
mkdir -p "${RAY_TMP_DIR}"

if [[ ${SLURM_GPUS_ON_NODE:-} =~ ([0-9]+)$ ]]; then
    NGPUS_PER_NODE=${BASH_REMATCH[1]}
else
    IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    NGPUS_PER_NODE=${#VISIBLE_GPU_IDS[@]}
fi
ROLLOUT_TP=${ROLLOUT_TP:-${NGPUS_PER_NODE}}
if (( ROLLOUT_TP > NGPUS_PER_NODE )); then
    echo "ROLLOUT_TP=${ROLLOUT_TP} exceeds allocated GPUs=${NGPUS_PER_NODE}" >&2
    exit 1
fi

python - <<'PY'
import os
from pathlib import Path

import torch
import verl
import vllm

workdir = Path(os.environ["VERL_WORKDIR"]).resolve()
actual_verl = Path(verl.__file__).resolve()
expected_verl = workdir / "verl" / "__init__.py"
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("verl:", actual_verl)
print("visible GPUs:", torch.cuda.device_count())
if actual_verl != expected_verl:
    raise SystemExit(f"Wrong VERL imported: expected {expected_verl}, got {actual_verl}")
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device visible inside the Slurm allocation")
PY

export NGPUS_PER_NODE
export ROLLOUT_TP
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
export WANDB_MODE=${WANDB_MODE:-disabled}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}

bash examples/grpo_trainer/run_aime24_posthoc_validation_fsdp.sh "$@"
