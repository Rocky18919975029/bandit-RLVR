# Complete-trajectory block SIR experiment

This experiment tests whether sequence-level tempering of an early reasoning block improves the strict-boxed final
answer accuracy relative to vanilla sampling and token-level (myopic) temperature sampling.

## Design

For each prompt, `generate_pool.py` samples `N_max` **complete** trajectories at temperature 1 and stores every
chosen token log-probability plus the strict-boxed verifier result. For a block length `B`, `sweep_joint.py` computes

```text
L_i(B) = sum(token_log_probs_i[:B])
w_i(B, alpha) = softmax((alpha - 1) * L_i(B))
```

and reweights the already complete trajectories. Since the stored suffix was sampled from `p(rest | first B
tokens)`, resampling the whole trajectory targets

```text
p(first B tokens)^alpha * p(rest | first B tokens).
```

The same pool can therefore be reused for arbitrary block lengths, alphas, resampling seeds, and candidate counts up
to `N_max`. `generate_myopic.py` is necessarily separate: it samples the first `B` tokens at temperature `1 / alpha`
and then switches back to temperature 1 until EOS. Its JSONL can append newly requested `(B, alpha, repeat)`
configurations without regenerating completed ones.

## Default 128-problem experiment

The launcher deterministically samples 128 source rows with NumPy seed 42 from:

```text
/data/user/zhongal/data/reschedule/DAPO-Math-17k.filtered.seed42.sample1536.parquet
```

The exact source row indices are recorded in `pool.jsonl.manifest.json`. Generation is resumable: completed prompt or
myopic configuration rows are skipped after a restart.

From an allocated HPC GPU node, run:

```bash
cd ~/bandit-RLVR

MODEL_PATH=/data/user/zhongal/path/to/model \
TENSOR_PARALLEL_SIZE=8 \
bash scripts/joint_tempering/launch_experiment.sh
```

Or submit the included generic Slurm template, adding site-specific partition/account flags if needed:

```bash
sbatch \
  --export=ALL,MODEL_PATH=/data/user/zhongal/path/to/model \
  scripts/joint_tempering/run_experiment.slurm
```

The defaults are:

```text
num_problems=128
pool_candidates=16
block_lengths=16 32 64 128
alphas=1.25 1.5 2.0
candidate_counts=4 8 16
max_response_tokens=4096
myopic_repeats=1
```

When `--max-model-len` is set, each request is automatically capped at
`min(max_response_tokens, max_model_len - prompt_tokens)`. This keeps full
trajectories within the model context while giving every prompt all available
space to reach EOS.

Override them with environment variables, for example:

```bash
MODEL_PATH=/data/user/zhongal/models/Qwen2.5-7B-Instruct \
BLOCK_LENGTHS="32 64 96" \
ALPHAS="1.25 1.5" \
POOL_CANDIDATES=32 \
CANDIDATE_COUNTS="8 16 32" \
MYOPIC_REPEATS=4 \
bash scripts/joint_tempering/launch_experiment.sh
```

All Hugging Face offline environment variables are enabled by the launcher. It never downloads a model or dataset.

## Separate stages

The expensive temperature-1 pool is generated only once:

```bash
python examples/joint_tempering/generate_pool.py \
  --model /hpc/local/model \
  --num-candidates 16 \
  --num-problems 128 \
  --problem-seed 42 \
  --output outputs/joint_tempering/pool.jsonl
```

Joint/vanilla sweeps are CPU-only and do not load the model:

```bash
python examples/joint_tempering/sweep_joint.py \
  --pool outputs/joint_tempering/pool.jsonl \
  --block-lengths 16 32 64 128 \
  --alphas 1.25 1.5 2.0 \
  --candidate-counts 4 8 16
```

Exact myopic baselines require GPU generation for each `(B, alpha)`:

```bash
python examples/joint_tempering/generate_myopic.py \
  --model /hpc/local/model \
  --pool outputs/joint_tempering/pool.jsonl \
  --block-lengths 16 32 64 128 \
  --alphas 1.25 1.5 2.0
```

Finally:

```bash
python examples/joint_tempering/analyze_results.py \
  --joint outputs/joint_tempering/joint_sweep.jsonl \
  --myopic outputs/joint_tempering/myopic.jsonl
```

The summary includes paired bootstrap confidence intervals, SIR ESS, maximum weights, early-EOS diagnostics, and both
weighted expected accuracy and categorical-resampling accuracy.

## Length validity

A complete pool trajectory that terminates before `B` does not define a fixed `B`-token action. The sweep records it
as invalid instead of silently padding or comparing variable-length actions. Use `short_block_prompt_count` in the
summary to decide whether a requested block length is supported by the generated pool.
