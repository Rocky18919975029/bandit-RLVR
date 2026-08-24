# Offline Open-R1 LightEval AIME24

This directory runs the official LightEval CLI against a local Hugging Face
model or exported VERL actor without giving the HPC job network access.

The fixed protocol is:

- LightEval commit `24895519caecec2abeea53fa790021325ce7e59e`
- official LightEval AIME24 prompt and math metrics from that revision
- 30 unique AIME24 problems
- temperature `0.6`, top-p `0.95`
- 64 completions per problem (`1,920` total)
- official `AsyncVLLMModel` backend with vLLM-native data parallelism
- tensor parallel size one

There is no monkeypatch of `VLLMModel`, `_generate`, `LLM.generate`, Ray, or
metric computation. `openr1_aime24_task.py` changes only the dataset location
to a prepared local parquet because the compute nodes cannot access the Hub.

## Isolated environment

`build_offline_bundle.sh` builds a Linux wheel bundle on the networked Mac.
`create_offline_env.sh` installs the pinned official LightEval revision in a
separate HPC Conda prefix. The training environment is used only as the source
of the initial offline clone and is not modified.

The selected LightEval revision contains LightEval's own `AsyncVLLMModel`.
That backend passes `data_parallel_size` directly to vLLM instead of nesting a
Ray-backed vLLM executor inside LightEval Ray tasks.

## One-time bundle build

On the Mac:

```bash
cd "/Users/zeshenghong/Documents/ChatGPT/bandit RLVR"

export PYTHON_BIN=/opt/anaconda3/bin/python3.12
export BUNDLE_DIR="$PWD/offline_bundles/openr1-lighteval-24895519caecec2abeea53fa790021325ce7e59e"

bash examples/grpo_trainer/openr1_lighteval/build_offline_bundle.sh
```

Sync that bundle and the repository files to HPC. Then create or update the
isolated environment:

```bash
cd ~/bandit-RLVR

export CONDA_SH=/share/anaconda3/etc/profile.d/conda.sh
export BASE_ENV_PATH=/data/user/zhongal/.conda/envs/verl
export EVAL_ENV_PATH=/data/user/zhongal/.conda/envs/openr1-lighteval-bandit
export BUNDLE_DIR="$PWD/offline_bundles/openr1-lighteval-24895519caecec2abeea53fa790021325ce7e59e"

bash examples/grpo_trainer/openr1_lighteval/create_offline_env.sh
```

This setup uses `--no-index`; a missing dependency fails closed instead of
contacting PyPI.

## Evaluation outputs

`run_aime24_offline.sh`, normally invoked by the Slurm launcher, writes:

- `protocol_manifest.json`: model, protocol, versions, and topology
- `dataset/dataset_manifest.json`: source rows and 30-problem deduplication
- `lighteval/results/**/results_*.json`: aggregate LightEval metrics
- `lighteval/details/**/details_*.parquet`: generations and per-sample scores

The Open-R1 task allows much longer generations. This repository defaults to
the existing Qwen2.5-Math-7B comparison limits of model length 4096 and 3072
new tokens. Those limits are recorded in the protocol manifest; temperature,
top-p, and 64 samples per problem are fixed in code.
