#!/usr/bin/env bash
# Build on an internet-connected machine; copy the resulting directory to HPC.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
LIGHTEVAL_COMMIT=d3da6b9bbf38104c8b5e1acc86f83541f9a502d1
BUNDLE_DIR=${BUNDLE_DIR:-${REPO_ROOT}/offline_bundles/openr1-lighteval-${LIGHTEVAL_COMMIT}}
PYTHON_BIN=${PYTHON_BIN:-/opt/anaconda3/bin/python3.12}

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "Python 3.12 not found at ${PYTHON_BIN}; set PYTHON_BIN explicitly." >&2
    exit 1
fi
if [ -e "${BUNDLE_DIR}" ]; then
    echo "Bundle path already exists: ${BUNDLE_DIR}" >&2
    echo "Choose a new BUNDLE_DIR; this script never overwrites a bundle." >&2
    exit 1
fi

WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/openr1-lighteval-bundle.XXXXXX")
trap 'rm -rf -- "${WORK_DIR}"' EXIT
WHEELHOUSE=${BUNDLE_DIR}/wheelhouse
mkdir -p "${WHEELHOUSE}"

git clone https://github.com/huggingface/lighteval.git "${WORK_DIR}/lighteval"
git -C "${WORK_DIR}/lighteval" checkout "${LIGHTEVAL_COMMIT}"
"${PYTHON_BIN}" -m pip wheel \
    --no-deps \
    --wheel-dir "${WHEELHOUSE}" \
    "${WORK_DIR}/lighteval"

# The target environment is an offline clone of the stable VERL environment,
# which already supplies torch, vLLM, Ray, Transformers, datasets, NumPy,
# pandas and pyarrow. Download the extra LightEval runtime dependencies as
# Linux/Python-3.12 wheels without pulling a second CUDA/PyTorch stack.
"${PYTHON_BIN}" -m pip download \
    --dest "${WHEELHOUSE}" \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 3.12 \
    --abi cp312 \
    --only-binary=:all: \
    termcolor==2.3.0 \
    pytablewriter \
    typer \
    rich \
    colorlog \
    aenum==3.1.15 \
    nltk==3.9.1 \
    sacrebleu \
    sentencepiece \
    protobuf \
    pycountry \
    fsspec \
    httpx==0.27.2 \
    GitPython \
    scikit-learn \
    numpy==1.26.4 \
    absl-py \
    hf-xet \
    more-itertools \
    latex2sympy2_extended==1.0.6

# rouge-score 0.1.2 is source-only on PyPI but builds a platform-independent
# wheel. Build it locally and reject the bundle if the resulting tag is not
# portable.
"${PYTHON_BIN}" -m pip wheel \
    --no-deps \
    --wheel-dir "${WHEELHOUSE}" \
    rouge-score==0.1.2

if find "${WHEELHOUSE}" -maxdepth 1 -name 'rouge_score-*.whl' ! -name '*-none-any.whl' | grep -q .; then
    echo "rouge-score produced a platform-specific wheel; refusing a nonportable bundle" >&2
    exit 1
fi

"${PYTHON_BIN}" - "${BUNDLE_DIR}/bundle_manifest.json" <<PY
import json
import sys
from pathlib import Path

wheelhouse = Path("${WHEELHOUSE}")
manifest = {
    "lighteval_commit": "${LIGHTEVAL_COMMIT}",
    "target": "CPython 3.12 / manylinux2014_x86_64",
    "wheels": sorted(path.name for path in wheelhouse.glob("*.whl")),
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "Offline bundle ready: ${BUNDLE_DIR}"
echo "Copy this directory to the same path relative to the repository on HPC."
