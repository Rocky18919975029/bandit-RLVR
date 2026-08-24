#!/bin/bash

#SBATCH --job-name=openr1-aime24
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --exclude=ACD1-16
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR}}
CONDA_SH=${CONDA_SH:-/share/anaconda3/etc/profile.d/conda.sh}
EVAL_ENV_PATH=${EVAL_ENV_PATH:-/data/user/zhongal/.conda/envs/openr1-lighteval-bandit}

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
if [ ! -f "${EVAL_ENV_PATH}/.openr1_lighteval_environment.json" ]; then
    echo "Isolated LightEval environment is not prepared: ${EVAL_ENV_PATH}" >&2
    echo "Run examples/grpo_trainer/openr1_lighteval/create_offline_env.sh first." >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_SH}"
conda deactivate || true
conda activate "${EVAL_ENV_PATH}"
hash -r

cd "${REPO_ROOT}"

unset PYTHONPATH
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset RAY_ADDRESS
unset RAY_TMPDIR
unset VLLM_ENABLE_V1_MULTIPROCESSING
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export VLLM_DISABLE_COMPILE_CACHE=1
# Match VERL's CUDA rollout runtime. vLLM/NCCL communicator initialization can
# crash on clusters without CUDA virtual-memory IPC support when this is on.
export NCCL_CUMEM_ENABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export VLLM_CACHE_ROOT=/tmp/vllm_${USER}_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=${VLLM_CACHE_ROOT}/torchinductor
export TRITON_CACHE_DIR=${VLLM_CACHE_ROOT}/triton
mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
export CC=${CC:-$(which gcc)}
export CXX=${CXX:-$(which g++)}
export CUDAHOSTCXX=${CUDAHOSTCXX:-$(which g++)}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}

if [[ ${SLURM_GPUS_ON_NODE:-} =~ ([0-9]+)$ ]]; then
    ALLOCATED_GPUS=${BASH_REMATCH[1]}
else
    IFS=',' read -r -a VISIBLE_GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    ALLOCATED_GPUS=${#VISIBLE_GPU_IDS[@]}
fi
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-8}
if (( DATA_PARALLEL_SIZE != ALLOCATED_GPUS )); then
    echo "This launcher requires one vLLM replica per GPU: DP=${DATA_PARALLEL_SIZE}, allocated=${ALLOCATED_GPUS}" >&2
    exit 1
fi
export DATA_PARALLEL_SIZE

python - "${EVAL_ENV_PATH}" <<'PY'
import json
import os
import sys
from pathlib import Path

import datasets
import lighteval
import torch
import transformers
import vllm
from packaging.version import Version

expected_prefix = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != expected_prefix:
    raise SystemExit(f"Wrong Python prefix: expected {expected_prefix}, got {sys.prefix}")
marker = json.loads((expected_prefix / ".openr1_lighteval_environment.json").read_text(encoding="utf-8"))
expected_commit = "865335e44fd84e0bae4a8b1ffcb65075e5080f31"
if Path(marker.get("environment_prefix", "")).resolve() != expected_prefix:
    raise SystemExit(f"Environment marker belongs to another prefix: {marker.get('environment_prefix')}")
if marker.get("lighteval_commit") != expected_commit:
    raise SystemExit(
        f"Wrong LightEval revision: expected {expected_commit}, got {marker.get('lighteval_commit')}"
    )
for name, package in (("lighteval", lighteval), ("torch", torch), ("vllm", vllm)):
    package_path = Path(package.__file__).resolve()
    if expected_prefix not in package_path.parents:
        raise SystemExit(f"{name} imported outside isolated environment: {package_path}")
if lighteval.__version__ != "0.10.1.dev0":
    raise SystemExit(f"Wrong LightEval version: {lighteval.__version__}")
if vllm.__version__ != "0.11.0":
    raise SystemExit(f"Wrong vLLM version: {vllm.__version__}")
if not (Version("4.55.2") <= Version(transformers.__version__) < Version("5")):
    raise SystemExit(f"Transformers is incompatible with vLLM 0.11: {transformers.__version__}")
if datasets.__version__ != "3.6.0":
    raise SystemExit(f"Wrong datasets version: {datasets.__version__}")
print("Isolated evaluation environment: PASS")
print("python:", sys.executable)
print("lighteval:", lighteval.__file__)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("NCCL_CUMEM_ENABLE:", os.environ.get("NCCL_CUMEM_ENABLE"))
print("visible GPUs:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device visible inside the Slurm allocation")
PY

bash examples/grpo_trainer/openr1_lighteval/run_aime24_offline.sh "$@"
