#!/usr/bin/env bash
# Run on an internet-connected machine. Downloads Linux x86_64 / Python 3.12
# wheels plus the vLLM 0.19.1 source distribution for an offline CUDA 12.8
# build on the HPC.

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/opt/anaconda3/bin/python3.12}
OUTPUT_DIR=${1:-}
VLLM_VERSION=0.19.1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONSTRAINTS=${SCRIPT_DIR}/cu128-constraints.txt

if [ -z "${OUTPUT_DIR}" ]; then
    echo "Usage: $0 /absolute/path/to/output-bundle" >&2
    exit 2
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python 3.12 not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [ ! -f "${CONSTRAINTS}" ]; then
    echo "Constraints file not found: ${CONSTRAINTS}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/wheels" "${OUTPUT_DIR}/source"

"${PYTHON_BIN}" -m pip download \
    --dest "${OUTPUT_DIR}/source" \
    --no-deps \
    --no-binary=:all: \
    "vllm==${VLLM_VERSION}"

SOURCE_ARCHIVE=$(find "${OUTPUT_DIR}/source" -maxdepth 1 -name "vllm-${VLLM_VERSION}.tar.gz" -print -quit)
if [ -z "${SOURCE_ARCHIVE}" ]; then
    echo "vLLM source archive was not downloaded" >&2
    exit 1
fi

SOURCE_TREE=${OUTPUT_DIR}/source/vllm-${VLLM_VERSION}
if [ ! -d "${SOURCE_TREE}" ]; then
    tar -xzf "${SOURCE_ARCHIVE}" -C "${OUTPUT_DIR}/source"
fi

PIP_TARGET=(
    --dest "${OUTPUT_DIR}/wheels"
    --platform manylinux_2_28_x86_64
    --python-version 312
    --implementation cp
    --abi cp312
    --only-binary=:all:
    --index-url https://pypi.org/simple
    --extra-index-url https://download.pytorch.org/whl/cu128
    --extra-index-url https://flashinfer.ai/whl/
    --constraint "${CONSTRAINTS}"
)

# Runtime and CUDA dependencies declared by vLLM. The file itself adds the
# FlashInfer wheel index required for flashinfer-cubin.
"${PYTHON_BIN}" -m pip download \
    "${PIP_TARGET[@]}" \
    --requirement "${SOURCE_TREE}/requirements/cuda.txt"

# Native build toolchain dependencies, downloaded as Linux wheels.
"${PYTHON_BIN}" -m pip download \
    "${PIP_TARGET[@]}" \
    --requirement "${SOURCE_TREE}/requirements/build.txt"

# The current bandit-RLVR checkout requires Transformers 5.5.3 or newer.
"${PYTHON_BIN}" -m pip download \
    "${PIP_TARGET[@]}" \
    'transformers>=5.5.3,<5.11'

cp "${CONSTRAINTS}" "${OUTPUT_DIR}/cu128-constraints.txt"

echo "Offline bundle prepared at: ${OUTPUT_DIR}"
echo "Next: rsync this directory to ~/bandit-RLVR/offline/vllm0191-cu128 on HPC."
