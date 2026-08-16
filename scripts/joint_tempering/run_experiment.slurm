#!/usr/bin/env bash

#SBATCH --job-name=joint-tempering
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=joint-tempering-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}}
LAUNCH_SCRIPT=${REPO_ROOT}/scripts/joint_tempering/launch_experiment.sh

if [[ ! -f "${LAUNCH_SCRIPT}" ]]; then
    echo "Joint-tempering launcher not found: ${LAUNCH_SCRIPT}" >&2
    echo "Submit this job from the repository root or export REPO_ROOT explicitly." >&2
    exit 1
fi

# Activate the HPC-local VERL/vLLM environment before this line if it is not
# already active in submitted jobs. MODEL_PATH must point to a local checkpoint.
bash "${LAUNCH_SCRIPT}"
