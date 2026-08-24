# Offline Open-R1 LightEval AIME24

This wrapper evaluates a local Hugging Face model or an exported VERL actor with the AIME24 sampling protocol used by Open-R1:

- LightEval commit: `d3da6b9bbf38104c8b5e1acc86f83541f9a502d1`
- official LightEval AIME24 prompt and math scorer
- 30 unique AIME24 problems
- temperature `0.6`
- top-p `0.95`
- 64 completions per problem (`1,920` completions total)
- one independent vLLM replica per GPU; tensor parallelism is fixed to one

The repository task keeps the official `lighteval|aime24` identity and differs from the pinned LightEval task only in its dataset path. The custom registration intentionally overrides the built-in task with a local `train.parquet`, so evaluation cannot silently download or substitute a Hub dataset.

The task also contains a narrow dataset-loader compatibility shim because the pinned LightEval commit still passes the removed `trust_remote_code` argument while the isolated environment uses `datasets>=5`. The shim changes only local dataset loading; it does not change the Open-R1 prompt, math scorer, generation parameters, or metric aggregation.

## Isolation model

The setup script makes an offline Conda clone at a separate prefix and installs LightEval only into that clone. The Slurm launcher:

- never activates `/data/user/zhongal/.conda/envs/verl`;
- disables user site packages and clears `PYTHONPATH`;
- requires `lighteval`, `torch`, and `vllm` to resolve inside the isolated prefix;
- enables the Hugging Face, datasets, and Transformers offline switches.
- runs vLLM in eager mode with its compile cache disabled and job-local cache roots, preventing independent data-parallel replicas from racing on shared TorchInductor artifacts.

The stable training environment is only the source of the one-time clone. It is never changed.

## One-time preparation

On the internet-connected Mac, build the Linux/Python 3.12 wheel bundle:

```bash
cd "/Users/zeshenghong/Documents/ChatGPT/bandit RLVR"

export PYTHON_BIN=/opt/anaconda3/bin/python3.12
export BUNDLE_DIR="$PWD/offline_bundles/openr1-lighteval-d3da6b9bbf38104c8b5e1acc86f83541f9a502d1"

bash examples/grpo_trainer/openr1_lighteval/build_offline_bundle.sh
```

Sync both the repository and the ignored `offline_bundles` directory to HPC. Then, in a fresh HPC terminal, create the isolated environment:

```bash
cd ~/bandit-RLVR

export CONDA_SH=/share/anaconda3/etc/profile.d/conda.sh
export BASE_ENV_PATH=/data/user/zhongal/.conda/envs/verl
export EVAL_ENV_PATH=/data/user/zhongal/.conda/envs/openr1-lighteval-bandit
export BUNDLE_DIR="$PWD/offline_bundles/openr1-lighteval-d3da6b9bbf38104c8b5e1acc86f83541f9a502d1"

bash examples/grpo_trainer/openr1_lighteval/create_offline_env.sh
```

This setup performs no network access. If the bundle is incomplete, installation fails instead of falling back to PyPI.

## Submit an evaluation

Every submission must provide a model and local AIME24 parquet. `MODEL_PATH` may be a base model directory, `actor/huggingface`, or a `global_step_*` directory.

```bash
cd ~/bandit-RLVR

export REPO_ROOT="$PWD"
export CONDA_SH=/share/anaconda3/etc/profile.d/conda.sh
export EVAL_ENV_PATH=/data/user/zhongal/.conda/envs/openr1-lighteval-bandit
export MODEL_PATH=/data/user/zhongal/.cache/qwen2.5-math-7b-local
export AIME24_FILE=/data/user/zhongal/data/reschedule/aime24.parquet
export EVAL_SEED=42
export DATA_PARALLEL_SIZE=8
export MAX_MODEL_LENGTH=4096
export MAX_NEW_TOKENS=3072
export MAX_NUM_SEQS=128
export MAX_NUM_BATCHED_TOKENS=32768
export GPU_MEMORY_UTILIZATION=0.90
export OUTPUT_ROOT="$PWD/validation_data/openr1_lighteval_aime24"
export RUN_NAME=openr1_aime24_base_t0p6_p0p95_n64_seed42

JOB_ID=$(
  sbatch --parsable \
    --export=ALL,REPO_ROOT="$REPO_ROOT",CONDA_SH="$CONDA_SH",EVAL_ENV_PATH="$EVAL_ENV_PATH",MODEL_PATH="$MODEL_PATH",AIME24_FILE="$AIME24_FILE",EVAL_SEED="$EVAL_SEED",DATA_PARALLEL_SIZE="$DATA_PARALLEL_SIZE",MAX_MODEL_LENGTH="$MAX_MODEL_LENGTH",MAX_NEW_TOKENS="$MAX_NEW_TOKENS",MAX_NUM_SEQS="$MAX_NUM_SEQS",MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS",GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION",OUTPUT_ROOT="$OUTPUT_ROOT",RUN_NAME="$RUN_NAME" \
    examples/grpo_trainer/submit_openr1_aime24_lighteval_h100.slurm |
  cut -d';' -f1
)

echo "JOB_ID=$JOB_ID"
tail -F "slurm-openr1-aime24-${JOB_ID}.out" "slurm-openr1-aime24-${JOB_ID}.err"
```

LightEval and vLLM print their native progress bars. The run also saves:

- `protocol_manifest.json`: an explicit protocol/topology/model audit;
- `dataset/dataset_manifest.json`: the 30-problem deduplication audit;
- `lighteval/results/**/results_*.json`: aggregate LightEval metrics;
- `lighteval/details/**/details_*.parquet`: per-problem generations and scores.

The original Open-R1 task permits up to 32,768 generated tokens. Qwen2.5-Math-7B has a shorter context, so the launcher defaults to the existing comparison cap (`MAX_MODEL_LENGTH=4096`, `MAX_NEW_TOKENS=3072`). These two limits are recorded in the manifest and may be raised for a checkpoint that supports the full Open-R1 length; temperature, top-p, and `n=64` are not configurable.

The wrapper fixes `enforce_eager=True` for inference-engine reliability. This changes only vLLM execution optimization (TorchInductor/CUDA graph use), not the model distribution or evaluation protocol.
