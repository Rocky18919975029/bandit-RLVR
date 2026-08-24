#!/usr/bin/env bash
# Create a separate LightEval environment without contacting package servers.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
LIGHTEVAL_COMMIT=24895519caecec2abeea53fa790021325ce7e59e
CONDA_SH=${CONDA_SH:-/share/anaconda3/etc/profile.d/conda.sh}
BASE_ENV_PATH=${BASE_ENV_PATH:-/data/user/zhongal/.conda/envs/verl}
EVAL_ENV_PATH=${EVAL_ENV_PATH:-/data/user/zhongal/.conda/envs/openr1-lighteval-bandit}
BUNDLE_DIR=${BUNDLE_DIR:-${REPO_ROOT}/offline_bundles/openr1-lighteval-${LIGHTEVAL_COMMIT}}

if [ "${BASE_ENV_PATH}" = "${EVAL_ENV_PATH}" ]; then
    echo "EVAL_ENV_PATH must differ from the training BASE_ENV_PATH" >&2
    exit 1
fi
if [ ! -f "${CONDA_SH}" ]; then
    echo "Conda init script not found: ${CONDA_SH}" >&2
    exit 1
fi
if [ ! -x "${BASE_ENV_PATH}/bin/python" ]; then
    echo "Stable source environment not found: ${BASE_ENV_PATH}" >&2
    exit 1
fi
if [ ! -f "${BUNDLE_DIR}/bundle_manifest.json" ]; then
    echo "Offline bundle not found: ${BUNDLE_DIR}" >&2
    exit 1
fi

"${BASE_ENV_PATH}/bin/python" - "${BUNDLE_DIR}/bundle_manifest.json" "${LIGHTEVAL_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("lighteval_commit") != sys.argv[2]:
    raise SystemExit(
        f"Wrong LightEval bundle commit: expected {sys.argv[2]}, got {manifest.get('lighteval_commit')}"
    )
if not any(name.startswith("lighteval-0.10.1.dev0-") for name in manifest.get("wheels", [])):
    raise SystemExit("Bundle manifest does not contain the pinned LightEval wheel")
print("Offline bundle audit: PASS")
PY

# shellcheck source=/dev/null
source "${CONDA_SH}"
conda deactivate || true

if [ ! -d "${EVAL_ENV_PATH}" ]; then
    echo "Cloning the stable training environment into an isolated prefix..."
    conda create --offline --yes --prefix "${EVAL_ENV_PATH}" --clone "${BASE_ENV_PATH}"
fi

"${EVAL_ENV_PATH}/bin/python" -m pip install \
    --no-index \
    --find-links "${BUNDLE_DIR}/wheelhouse" \
    "lighteval[math]==0.10.1.dev0" \
    more-itertools

PYTHONNOUSERSITE=1 "${EVAL_ENV_PATH}/bin/python" - "${EVAL_ENV_PATH}" "${LIGHTEVAL_COMMIT}" <<'PY'
import json
import sys
from pathlib import Path

import datasets
import inspect
import latex2sympy2_extended
import lighteval
import torch
import vllm
from lighteval.models.vllm.vllm_model import AsyncVLLMModel

prefix = Path(sys.argv[1]).resolve()
for package_name, package in (("lighteval", lighteval), ("vllm", vllm), ("torch", torch)):
    package_path = Path(package.__file__).resolve()
    if prefix not in package_path.parents:
        raise SystemExit(f"{package_name} escaped isolated prefix: {package_path}")

backend_path = Path(inspect.getsourcefile(AsyncVLLMModel)).resolve()
if prefix not in backend_path.parents:
    raise SystemExit(f"AsyncVLLMModel escaped isolated prefix: {backend_path}")
if lighteval.__version__ != "0.10.1.dev0":
    raise SystemExit(f"Wrong LightEval version: {lighteval.__version__}")
if vllm.__version__ != "0.11.0":
    raise SystemExit(f"Wrong vLLM version: {vllm.__version__}")

marker = {
    "environment_prefix": str(prefix),
    "lighteval_commit": sys.argv[2],
    "lighteval_path": str(Path(lighteval.__file__).resolve()),
    "torch_version": torch.__version__,
    "vllm_version": vllm.__version__,
    "datasets_version": datasets.__version__,
    "model_backend": "official AsyncVLLMModel",
    "model_backend_source": str(backend_path),
}
(prefix / ".openr1_lighteval_environment.json").write_text(
    json.dumps(marker, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(marker, indent=2, sort_keys=True))
PY

echo "Isolated offline LightEval environment is ready: ${EVAL_ENV_PATH}"
echo "The source training environment was not modified."
