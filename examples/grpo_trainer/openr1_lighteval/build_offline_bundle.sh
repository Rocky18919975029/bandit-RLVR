#!/usr/bin/env bash
# Build on an internet-connected machine; copy the resulting directory to HPC.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
LIGHTEVAL_COMMIT=865335e44fd84e0bae4a8b1ffcb65075e5080f31
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

"${PYTHON_BIN}" - "${WHEELHOUSE}" <<'PY'
import zipfile
import sys
from pathlib import Path

wheelhouse = Path(sys.argv[1])
wheels = list(wheelhouse.glob("lighteval-0.10.1.dev0-*.whl"))
if len(wheels) != 1:
    raise SystemExit(f"Expected exactly one LightEval 0.10.1 wheel, found: {wheels}")

with zipfile.ZipFile(wheels[0]) as archive:
    vllm_model = archive.read("lighteval/models/vllm/vllm_model.py").decode("utf-8")
    model_loader = archive.read("lighteval/models/model_loader.py").decode("utf-8")
    metrics = archive.read("lighteval/metrics/metrics.py").decode("utf-8")
    metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
    metadata = archive.read(metadata_name).decode("utf-8")

required = {
    "official AsyncVLLMModel": "class AsyncVLLMModel(VLLMModel)" in vllm_model,
    "native data parallel argument": '"data_parallel_size": config.data_parallel_size' in vllm_model,
    "async model selection": "if config.is_async:" in model_loader,
    "AIME n=64 metric": "math_pass_at_1_64n" in metrics,
    "Transformers compatible with vLLM 0.11": "Requires-Dist: transformers>=4.54.0" in metadata,
    "NumPy 1.x compatible": "Requires-Dist: numpy<2" in metadata,
}
missing = [name for name, present in required.items() if not present]
if missing:
    raise SystemExit(f"LightEval wheel compatibility audit failed: {missing}")
print("Official LightEval async/backend metric audit: PASS")
PY

# The target environment is an offline clone of the stable VERL environment,
# which supplies torch, vLLM, Ray, pandas and pyarrow. Include the exact
# Transformers and datasets versions satisfying both LightEval and vLLM 0.11,
# without pulling a second CUDA/PyTorch stack.
"${PYTHON_BIN}" -m pip download \
    --dest "${WHEELHOUSE}" \
    --platform manylinux2014_x86_64 \
    --implementation cp \
    --python-version 3.12 \
    --abi cp312 \
    --only-binary=:all: \
    transformers==4.55.2 \
    datasets==3.6.0 \
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
