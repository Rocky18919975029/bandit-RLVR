# Group Relative Policy Optimization (GRPO)

In reinforcement learning, classic algorithms like PPO rely on a "critic" model to estimate the value of actions, guiding the learning process. However, training this critic model can be resource-intensive.

GRPO simplifies this process by eliminating the need for a separate critic model. Instead, it operates as follows:
- Group Sampling: for a given problem, the model generates multiple possible solutions, forming a "group" of outputs.
- Reward Assignment: each solution is evaluated and assigned a reward based on its correctness or quality.
- Baseline Calculation: the average reward of the group serves as a baseline.
- Policy Update: the model updates its parameters by comparing each solution's reward to the group baseline, reinforcing better-than-average solutions and discouraging worse-than-average ones.

For more details, refer to the original paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300).

## Key Components

- No Value Function (Critic-less): unlike PPO, GRPO does not train a separate value network (critic).
- Group Sampling (Grouped Rollouts): instead of evaluating one rollout per input, GRPO generates multiple completions (responses) from the current policy for each prompt. This set of completions is referred to as a group.
- Relative Rewards: within each group, completions are scored (e.g., based on correctness), and rewards are normalized relative to the group.

## Important knobs

- `actor_rollout_ref.rollout.n`: per-prompt sample count (required >= 2 for GRPO).
- `data.train_batch_size`: prompts per global step. Total trajectories = `train_batch_size * rollout.n`.
- `actor_rollout_ref.actor.ppo_mini_batch_size`: global mini-batch for actor updates (must divide `train_batch_size * n`).
- `actor_rollout_ref.actor.ppo_epochs`: inner-loop epochs over the sampled trajectories.
- `actor_rollout_ref.actor.clip_ratio`: PPO clip range, default `0.2`.
- `actor_rollout_ref.actor.loss_agg_mode`: `token-mean` (default), `seq-mean-token-sum`, or `seq-mean-token-mean`.
- `actor_rollout_ref.actor.use_kl_loss=True` + `actor_rollout_ref.actor.kl_loss_coef` / `kl_loss_type`: regularise toward the reference policy via KL loss on the actor.
- `algorithm.adv_estimator=grpo`.

## Dr. GRPO

To enable Dr. GRPO (see [Understanding R1-Zero-Like Training](https://arxiv.org/pdf/2503.20783)), set on top of the canonical GRPO overrides:

```
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
actor_rollout_ref.actor.use_kl_loss=False
algorithm.norm_adv_by_std_in_grpo=False
```

## Optional joint-tempered SIR before GRPO

The Qwen2.5-Math Re-Schedule baseline launcher can optionally generate a pool
of `N` complete responses per prompt, resample `K` trajectories according to a
joint-tempered prefix action, and pass only those `K` rows into the ordinary
reward, GRPO advantage, and actor-update path.

The three sizes are deliberately separate:

- `ROLLOUT_N` / `actor_rollout_ref.rollout.n`: generated pool size `N`.
- `SIR_K` / `algorithm.sir.selected_count`: resampled GRPO group size `K`.
- `SIR_BLOCK_LENGTH` / `algorithm.sir.block_length`: prefix action horizon `B`.

`SIR_POOL_MODE=independent` preserves the original behavior and generates all
`N` responses independently. `SIR_POOL_MODE=branched_prefix` first generates
`K` complete responses per prompt. For each initial response it chooses
`N/K-1` random, nonterminal cut positions in its first `B` tokens and completes
each retained prefix once. Cuts are distinct when possible; a short response
reuses its available cuts so the pool still contains exactly `N` completions.
A one-token response falls back to the empty prefix. Branched mode requires
`N > K` and `N` divisible by `K`.

For candidate `i`, the launcher uses the rollout policy's chosen-token
log-probabilities to compute

```text
L_i = sum_{t=1}^{min(B,T_i)} log p(y_t | x,y_<t)
w_i = softmax((alpha - 1) L_i)
```

and draws `K` distinct trajectories by weighted sampling without replacement.
If EOS occurs before `B`, its probability is included in `L_i`; no length
normalization or virtual post-EOS probability is added. SIR is disabled by
default, so the canonical baseline is unchanged.

Example:

```bash
SIR_ENABLE=True \
SIR_POOL_MODE=branched_prefix \
ROLLOUT_N=32 \
SIR_K=8 \
SIR_BLOCK_LENGTH=64 \
SIR_ALPHA=1.5 \
bash examples/grpo_trainer/run_qwen2_5_math_7b_grpo_reschedule_baseline.sh
```

When enabled, the complete pool for step `s` is written to
`trainer.rollout_data_dir/sir_pool/s.jsonl`, one JSON object per prompt. Every
candidate contains its response, sampled token IDs, optional chosen-token
log-probabilities, available verifier score, prefix joint log-probability, normalized SIR weight,
`selected_count`, and `selected_draws`. The ordinary rollout dump continues to
contain only the `K` rows used for training and includes their source pool and
draw indices. Set `SIR_DUMP_TOKEN_LOG_PROBS=False` only when storage is more
important than post-hoc reweighting at different block lengths.

For an exact one-step control, set `SIR_ENABLE=False`, `ROLLOUT_N=SIR_K`, and
point `SIR_INITIAL_REPLAY_PATH` at the saved branched-SIR `sir_pool/1.jsonl`.
The trainer then uses each prompt's original `sir_pool_origin=initial`
trajectories as an ordinary GRPO group. It replays the saved token IDs and
behavior log-probabilities, recomputes strict rewards, and leaves the normal
GRPO/PPO update unchanged. Prompt order, chat-template output, ground truth,
data source, token/log-prob cardinality, K, and step are checked before any
actor update; a mismatch aborts the run.

Before enabling full-sequence tempering, use
`analyze_tempered_initial_pool.py` on a saved pool. It scans escort weights
`w_i(alpha) proportional to exp((alpha - 1) L_i)` over the original initial
trajectories and writes ESS, maximum-weight, tempered-accuracy, and
reward/log-probability-covariance curves. The covariance has the identity
`Cov_q(R, L) = d E_q[R] / d alpha`. The report also identifies the largest
scanned alpha for which a configured fraction of prompt groups retains a
specified ESS fraction; it does not change or train a model.

## Exact-replay tempered GRPO comparison

`TEMPERED_GRPO_ENABLE=True` changes only the GRPO advantage computed from an
exactly replayed initial group. It requires `SIR_INITIAL_REPLAY_PATH`, keeps
the saved trajectories, rewards, behavior log-probabilities, PPO loss, batch
order, and optimizer path fixed, and is currently restricted to one step.

For complete response `i`, including its EOS token when present, it computes

```text
L_i = sum_t log pi_rollout(y_i,t | x,y_i,<t)
w_i = softmax((tempering_beta - 1) L_i)
A_i = K w_i (R_i - sum_j w_j R_j) / weighted_sample_std(R)
```

The constant global factor `tempering_beta` from the exact escort-objective
gradient is absorbed into the learning-rate scale. At `tempering_beta=1`, the
implementation calls canonical GRPO directly, so the baseline is unchanged.
The configured ESS constraint uses fixed Rényi order 2; `tempering_beta` is a
different parameter. The run fails before the actor update if too few groups
meet the ESS budget.

For the saved 512-by-16 initial pool, the predeclared 95%-of-groups budget
selected `TEMPERING_BETA=1.0033251003215344`, with
`TEMPERED_GRPO_ESS_BUDGET_FRACTION=0.5` (ESS at least 8 of 16) and
`TEMPERED_GRPO_REQUIRED_GROUP_FRACTION=0.95`.

## Post-hoc AIME 2024 validation

`run_aime24_posthoc_validation_fsdp.sh` reuses VERL's ordinary validation generation
and DAPO strict-boxed reward manager, then reports sample accuracy and standard
unbiased pass@k. `MODEL_PATH` may point to an untrained Hugging Face base model,
an exported `actor/huggingface` directory, or its enclosing `global_step_*`
directory. The defaults generate 32 completions per AIME 2024 problem at
temperature 1.0 and top-p 1.0 with a 3072-token response limit. Because the
local AIME parquet may already contain 32 physical copies of every problem,
the launcher first deduplicates it by `prompt` and requires exactly 30 unique
problems. It then applies VERL's validation `n=32`, for exactly 960 rollouts;
the source-row, unique-problem, and expected-rollout counts are printed before
model initialization. Validation displays a live `completed/total` rollout
progress bar with running accuracy. `VALIDATION_PROBLEM_BATCH_SIZE` defaults to
8 to keep eight data-parallel rollout replicas occupied.

```bash
MODEL_PATH=/path/to/global_step_10 \
EVAL_N=32 \
PASS_KS=1,2,4,8,16,32 \
bash examples/grpo_trainer/run_aime24_posthoc_validation_fsdp.sh
```

Raw VERL generations are retained under `OUTPUT_DIR/raw`. The script writes
`summary.json`, `summary.csv`, and `per_problem.csv` beside them. For a problem
with `c` correct completions among `n`, it computes
`pass@k = 1 - C(n-c,k) / C(n,k)` and then averages across the 30 problems.

For repeated evaluations of two checkpoints, use
`analyze_paired_aime24_runs.py`. It aligns problems by stable input and ground
truth rather than the run-local random UID, pools all completions for each
problem, recomputes unbiased pass@k, and reports a paired percentile-bootstrap
interval by resampling the 30 AIME problems. The output directory contains
`summary.json`, `summary.csv`, `per_problem.csv`, and `per_seed.csv`. The
bootstrap interval is conditional on the observed completions; its unit is the
AIME problem, not an individual rollout.

## Canonical scripts

All scripts in this directory follow the naming convention:

```
run_<model>_<train-backend>[_<platform-or-variant>].sh
```

Where:
- `<model>` is the canonical size for a model family
  (`qwen3_8b` for dense text, `qwen3_30b_a3b` for MoE, `qwen2_5_vl_7b` / `qwen3_vl_8b` for vision,
  `qwen3_235b_a22b` / `deepseek_v3_671b` / `deepseek_v4_flash` for scale demos).
- `<train-backend>` ∈ {`fsdp`, `megatron`, `mindspeed`}.
- `<platform-or-variant>` is used only for hardware-specific variants such as `gb200`, `fp8`, `veomni`,
  or MindSpeed NPU scripts.
- `INFER_BACKEND` selects rollout backend inside scripts that support multiple choices
  (`vllm`, `sglang`, or `trtllm`).
- `DEVICE` selects GPU/NPU paths inside scripts that support both platforms.

Every script exposes the commonly tuned knobs as environment variables at the top, so you can run:

```bash
MODEL_PATH=Qwen/Qwen3-14B \
NNODES=2 NGPUS_PER_NODE=8 \
INFER_BACKEND=sglang ROLLOUT_N=8 TRAIN_BATCH_SIZE=2048 \
bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh
```

### Defaults

- `dynamic batch size` and `sequence balancing` are enabled by default on all scripts.
- Text LLM scripts train on `gsm8k + math` by default; vision scripts train on `geo3k`.
- Scale-demo scripts (235B, 671B) train on `dapo-math-17k` / `aime-2024`.

### Matrix

| Model family          | `vllm` | `sglang` | `trtllm` | Train backend   | Platforms |
| --------------------- | :----: | :------: | :------: | --------------- | --------- |
| Qwen3-8B (dense)      | ✓      | ✓        | ✓        | FSDP, Megatron  | nvidia, npu (FSDP + MindSpeed), `_gb200` variant |
| Qwen2.5-VL-7B         | ✓      | ✓        | ✓        | FSDP, Megatron  | nvidia    |
| Qwen3-VL-8B           | ✓      |          |          | FSDP, Megatron  | nvidia, npu (FSDP) |
| Qwen3-VL-30B-A3B      | ✓      |          |          | FSDP, Megatron  | nvidia, npu (FSDP, VeOmni) |
| Qwen3-VL-235B-A22B    | ✓      |          |          | Megatron        | nvidia    |
| Qwen3-30B-A3B (MoE)   | ✓      | ✓        | ✓        | FSDP, Megatron  | nvidia, npu (MindSpeed, VeOmni) |
| Qwen3-235B-A22B       | ✓      |          | ✓        | Megatron        | nvidia, npu |
| Qwen3-Next-80B-A3B    | ✓      |          |          | FSDP            | npu       |
| Qwen3.5-27B (dense)   | ✓      |          |          | FSDP2           | nvidia, npu |
| Qwen3.5-35B (dense)   | ✓      |          |          | FSDP2, Megatron | nvidia, npu |
| Qwen3.5-35B-A3B (MoE) |        | ✓        |          | VeOmni          | nvidia    |
| Qwen3.5-122B-A10B     | ✓      |          |          | Megatron        | nvidia    |
| DeepSeek-V3 671B      | ✓      |          |          | Megatron        | nvidia    |
| DeepSeek-V4-Flash     | ✓      |          |          | Megatron        | nvidia, amd    |
| GLM-4.1V-9B           | ✓      |          |          | FSDP            | nvidia    |
| MiniCPM-o-2.6         | ✓      |          |          | FSDP            | nvidia    |
| Moonlight-16B-A3B     | ✓      |          |          | Megatron        | nvidia    |
| Nemotron-Nano-v3-30B-A3B | ✓   |          |          | Megatron        | nvidia    |
| Seed-OSS-36B          | ✓      |          |          | FSDP2           | nvidia    |
| GPT-OSS-20B           |        | ✓        |          | FSDP            | nvidia    |
| Mistral-Nemo-12B (RM demo) | ✓ |          |          | FSDP            | nvidia    |

LoRA variants live in `examples/tuning/lora/`, profiling variants in `examples/profile/`.
Scale / hardware-specific demos (e.g. `run_qwen3_8b_fsdp_gb200.sh`, FP8 variants, VeOmni) keep a trailing suffix to stay discoverable.

## Reference

- See [verl baselines](https://verl.readthedocs.io/en/latest/algo/baseline.html) for reference metrics.
- Qwen2.5 GRPO training log: [experiments/gsm8k/qwen2-7b-fsdp2.log](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/qwen2-7b-fsdp2.log).
