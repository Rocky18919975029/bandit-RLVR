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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Activate the HPC-local VERL/vLLM environment before this line if it is not
# already active in submitted jobs. MODEL_PATH must point to a local checkpoint.
bash "${SCRIPT_DIR}/launch_experiment.sh"
