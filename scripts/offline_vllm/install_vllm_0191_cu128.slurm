#!/bin/bash
# Build vLLM 0.19.1 against torch 2.10 + CUDA 12.8 in a new offline conda
# environment, preserving the existing /data/user/zhongal/.conda/envs/verl.

#SBATCH --job-name=build-vllm-cu128
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -xeuo pipefail

VERL_WORKDIR=${VERL_WORKDIR:-${SLURM_SUBMIT_DIR}}
BUNDLE_DIR=${BUNDLE_DIR:-${VERL_WORKDIR}/offline/vllm0191-cu128}
SOURCE_ENV=${SOURCE_ENV:-/data/user/zhongal/.conda/envs/verl}
TARGET_ENV=${TARGET_ENV:-/data/user/zhongal/.conda/envs/bandit-verl}
CONDA_SH=${CONDA_SH:-/share/anaconda3/etc/profile.d/conda.sh}
VLLM_VERSION=0.19.1

for required in \
    "${CONDA_SH}" \
    "${BUNDLE_DIR}/cu128-constraints.txt" \
    "${BUNDLE_DIR}/source/vllm-${VLLM_VERSION}/setup.py"
do
    if [ ! -e "${required}" ]; then
        echo "Required offline file not found: ${required}" >&2
        exit 1
    fi
done
if [ -e "${TARGET_ENV}" ]; then
    echo "Target environment already exists: ${TARGET_ENV}" >&2
    echo "Choose a new TARGET_ENV or remove it manually after inspecting it." >&2
    exit 1
fi

LD_LIBRARY_PATH=":${LD_LIBRARY_PATH:-}:"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH//:\/data\/apps\/glibc\/2.32\/lib:/:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH//:\/data\/apps\/glibc\/2.32\/lib64:/:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH#:}
LD_LIBRARY_PATH=${LD_LIBRARY_PATH%:}
export LD_LIBRARY_PATH

module purge
module load gcc/13.3
module load cuda/12.8

# shellcheck source=/dev/null
source "${CONDA_SH}"
conda create --yes --prefix "${TARGET_ENV}" --clone "${SOURCE_ENV}"
conda activate "${TARGET_ENV}"
hash -r

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export MAX_JOBS=${MAX_JOBS:-12}
export NVCC_THREADS=${NVCC_THREADS:-2}
export VLLM_TARGET_DEVICE=cuda
export TMPDIR=${TMPDIR:-${VERL_WORKDIR}/.tmp/vllm-build-${SLURM_JOB_ID}}
mkdir -p "${TMPDIR}"

PIP_OFFLINE=(--no-index --find-links "${BUNDLE_DIR}/wheels")

python -m pip install "${PIP_OFFLINE[@]}" --upgrade \
    'torch==2.10.0+cu128' \
    'torchvision==0.25.0+cu128' \
    'torchaudio==2.10.0+cu128'

python -m pip install "${PIP_OFFLINE[@]}" --upgrade \
    --requirement "${BUNDLE_DIR}/source/vllm-${VLLM_VERSION}/requirements/cuda.txt" \
    --requirement "${BUNDLE_DIR}/source/vllm-${VLLM_VERSION}/requirements/build.txt" \
    'transformers>=5.5.3,<5.11'

cd "${BUNDLE_DIR}/source/vllm-${VLLM_VERSION}"
python use_existing_torch.py --prefix
python -m pip install "${PIP_OFFLINE[@]}" --no-build-isolation --no-deps .

cd "${VERL_WORKDIR}"
python -m pip install --no-deps --editable .

python - <<'PY'
import torch
import transformers
import vllm
from vllm.entrypoints.openai.parser.harmony_utils import get_encoding

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
print("gpu:", torch.cuda.get_device_name(0))
assert torch.__version__.startswith("2.10.0")
assert torch.version.cuda == "12.8"
assert vllm.__version__ == "0.19.1"
assert callable(get_encoding)
print("bandit-RLVR offline runtime: OK")
PY

python -m pip check

echo "Use this environment for GRPO: CONDA_ENV_PATH=${TARGET_ENV}"
