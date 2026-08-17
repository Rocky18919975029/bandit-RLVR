# Copyright 2026 The verl team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from verl import DataProto


@dataclass
class HPFMaskedBatch:
    batch: DataProto
    metrics: dict[str, float]
    prefix_mask: torch.Tensor | None = None
    suffix_mask: torch.Tensor | None = None


@dataclass
class HPFTransitionPrefixPlan:
    current_prefix_ids: list[list[int]]
    next_prefix_ids: list[list[int]]
    current_prefix_lengths: np.ndarray
    next_prefix_lengths: np.ndarray
    current_needs_suffix: np.ndarray
    next_needs_suffix: np.ndarray
    request_source_rows: np.ndarray
    request_cut_indices: np.ndarray


@dataclass(frozen=True)
class HPFTransitionReturnEstimate:
    current_mean: float
    next_mean: float
    delta_mean: float
    delta_std: float
    delta_positive_frac: float
    num_pairs: int

    def metrics(self) -> dict[str, float]:
        return {
            "hpf/transition_current_reward_mean": self.current_mean,
            "hpf/transition_next_reward_mean": self.next_mean,
            "hpf/transition_current_return_mean": self.current_mean,
            "hpf/transition_next_return_mean": self.next_mean,
            "hpf/transition_return_delta_mean": self.delta_mean,
            "hpf/transition_return_delta_std": self.delta_std,
            "hpf/transition_return_delta_positive_frac": self.delta_positive_frac,
            "hpf/transition_return_num_pairs": float(self.num_pairs),
        }


def configure_hpf_transition_behavior_log_probs(
    current_batch: DataProto,
    next_batch: DataProto,
    *,
    reuse_rollout_log_probs: bool,
    recompute_fn: Callable[[DataProto], DataProto] | None = None,
) -> str:
    """Select the configured behavior log-prob source for both transition cuts."""
    batches = {"current": current_batch, "next": next_batch}
    for cut, batch in batches.items():
        if "old_log_probs" not in batch.batch:
            raise ValueError(f"Transition {cut}-cut batch is missing rollout old_log_probs.")

    if reuse_rollout_log_probs:
        return "vllm_tree_rollout"
    if recompute_fn is None:
        raise ValueError("Transition behavior log-prob recomputation requires recompute_fn.")

    for cut, batch in batches.items():
        recomputed = recompute_fn(batch)
        if "old_log_probs" not in recomputed.batch:
            raise ValueError(f"Recomputed transition {cut}-cut batch is missing old_log_probs.")
        batch.batch["old_log_probs"] = recomputed.batch["old_log_probs"]
    return "actor_mixed_temperature_forward"


def estimate_hpf_transition_return(
    current_batch: DataProto,
    next_batch: DataProto,
    *,
    current_reward_tensor: torch.Tensor,
    next_reward_tensor: torch.Tensor,
) -> HPFTransitionReturnEstimate:
    """Estimate ``J_{h_{i+1}} - J_{h_i}`` from paired behavior rollouts."""
    pair_key = "hpf_transition_pair_uid"
    if pair_key not in current_batch.non_tensor_batch or pair_key not in next_batch.non_tensor_batch:
        raise ValueError(f"Transition return estimation requires {pair_key!r} in both cut batches.")

    current_pair_ids = [str(value) for value in current_batch.non_tensor_batch[pair_key]]
    next_pair_ids = [str(value) for value in next_batch.non_tensor_batch[pair_key]]
    if len(current_pair_ids) != current_reward_tensor.shape[0]:
        raise ValueError(
            "Current transition pair IDs and rewards have different row counts: "
            f"{len(current_pair_ids)} != {current_reward_tensor.shape[0]}."
        )
    if len(next_pair_ids) != next_reward_tensor.shape[0]:
        raise ValueError(
            "Next transition pair IDs and rewards have different row counts: "
            f"{len(next_pair_ids)} != {next_reward_tensor.shape[0]}."
        )
    if len(set(current_pair_ids)) != len(current_pair_ids):
        raise ValueError("Current-cut transition pair IDs must be unique.")
    if len(set(next_pair_ids)) != len(next_pair_ids):
        raise ValueError("Next-cut transition pair IDs must be unique.")
    current_pair_set = set(current_pair_ids)
    next_pair_set = set(next_pair_ids)
    if current_pair_set != next_pair_set:
        missing_next = sorted(current_pair_set - next_pair_set)[:3]
        missing_current = sorted(next_pair_set - current_pair_set)[:3]
        raise ValueError(
            "Current- and next-cut transition pair IDs do not match: "
            f"missing_from_next={missing_next}, missing_from_current={missing_current}."
        )
    if not current_pair_ids:
        raise ValueError("Transition return estimation requires at least one rollout pair.")

    current_returns = current_reward_tensor.sum(dim=-1).detach().float().cpu()
    next_returns = next_reward_tensor.sum(dim=-1).detach().float().cpu()
    next_index_by_pair = {pair_id: index for index, pair_id in enumerate(next_pair_ids)}
    next_order = torch.tensor(
        [next_index_by_pair[pair_id] for pair_id in current_pair_ids],
        dtype=torch.long,
    )
    aligned_next_returns = next_returns.index_select(0, next_order)
    return_delta = aligned_next_returns - current_returns

    return HPFTransitionReturnEstimate(
        current_mean=float(current_returns.mean().item()),
        next_mean=float(aligned_next_returns.mean().item()),
        delta_mean=float(return_delta.mean().item()),
        delta_std=float(return_delta.std(unbiased=False).item()),
        delta_positive_frac=float((return_delta > 0).float().mean().item()),
        num_pairs=len(current_pair_ids),
    )


def build_hpf_transition_prefix_plan(
    high_token_ids: list[list[int]],
    *,
    current_horizon: int,
    next_horizon: int,
    max_response_length: int,
) -> HPFTransitionPrefixPlan:
    """Derive paired cut prefixes and a source-major low-temperature request plan."""
    if current_horizon < 0:
        raise ValueError(f"current_horizon must be non-negative, got {current_horizon}.")
    if next_horizon < current_horizon:
        raise ValueError(
            f"next_horizon must be at least current_horizon, got {next_horizon} < {current_horizon}."
        )
    if next_horizon > max_response_length:
        raise ValueError(
            f"next_horizon must not exceed max_response_length, got {next_horizon} > {max_response_length}."
        )

    high_lengths = np.asarray([len(token_ids) for token_ids in high_token_ids], dtype=np.int32)
    current_prefix_ids = [token_ids[:current_horizon] for token_ids in high_token_ids]
    next_prefix_ids = [token_ids[:next_horizon] for token_ids in high_token_ids]
    current_prefix_lengths = np.asarray([len(token_ids) for token_ids in current_prefix_ids], dtype=np.int32)
    next_prefix_lengths = np.asarray([len(token_ids) for token_ids in next_prefix_ids], dtype=np.int32)
    # The high-temperature request runs past the current cut. If it stops at
    # exactly the current horizon, that prefix already contains its terminal
    # token and must not be continued by the low-temperature policy.
    current_needs_suffix = (high_lengths > current_horizon) & (current_horizon < max_response_length)
    next_needs_suffix = (high_lengths >= next_horizon) & (next_horizon < max_response_length)

    current_rows = np.nonzero(current_needs_suffix)[0]
    next_rows = np.nonzero(next_needs_suffix)[0]
    source_rows = np.concatenate([current_rows, next_rows])
    cut_indices = np.concatenate(
        [np.zeros(len(current_rows), dtype=np.int32), np.ones(len(next_rows), dtype=np.int32)]
    )
    if len(source_rows):
        request_order = np.lexsort((cut_indices, source_rows))
        source_rows = source_rows[request_order]
        cut_indices = cut_indices[request_order]

    return HPFTransitionPrefixPlan(
        current_prefix_ids=current_prefix_ids,
        next_prefix_ids=next_prefix_ids,
        current_prefix_lengths=current_prefix_lengths,
        next_prefix_lengths=next_prefix_lengths,
        current_needs_suffix=current_needs_suffix,
        next_needs_suffix=next_needs_suffix,
        request_source_rows=source_rows,
        request_cut_indices=cut_indices,
    )


def _normalize_group_scores(
    scores: torch.Tensor, group_ids: np.ndarray, epsilon: float, std_normalize: bool
) -> torch.Tensor:
    out = torch.zeros_like(scores, dtype=torch.float32)
    unique_ids = np.unique(group_ids)
    for group_id in unique_ids:
        idx_np = np.nonzero(group_ids == group_id)[0]
        if len(idx_np) <= 1:
            continue
        idx = torch.as_tensor(idx_np, device=scores.device, dtype=torch.long)
        group_scores = scores[idx].float()
        centered = group_scores - group_scores.mean()
        if std_normalize:
            std = group_scores.std(unbiased=True)
            if torch.isfinite(std) and std > 0:
                centered = centered / (std + epsilon)
            else:
                centered = torch.zeros_like(centered)
        out[idx] = centered
    return out


def _normalize_repeated_prefix_scores(
    scores: torch.Tensor,
    problem_ids: np.ndarray,
    prefix_group_ids: np.ndarray,
    epsilon: float,
    std_normalize: bool,
) -> torch.Tensor:
    out = torch.zeros_like(scores, dtype=torch.float32)
    for problem_id in np.unique(problem_ids):
        problem_idx_np = np.nonzero(problem_ids == problem_id)[0]
        problem_prefix_ids = prefix_group_ids[problem_idx_np]
        unique_prefix_ids = np.unique(problem_prefix_ids)
        if len(unique_prefix_ids) <= 1:
            continue

        prefix_rows = []
        prefix_scores = []
        for prefix_group_id in unique_prefix_ids:
            rows_np = problem_idx_np[np.nonzero(problem_prefix_ids == prefix_group_id)[0]]
            rows = torch.as_tensor(rows_np, device=scores.device, dtype=torch.long)
            prefix_rows.append(rows)
            prefix_scores.append(scores[rows[0]].float())

        group_scores = torch.stack(prefix_scores)
        centered = group_scores - group_scores.mean()
        if std_normalize:
            std = group_scores.std(unbiased=True)
            if torch.isfinite(std) and std > 0:
                centered = centered / (std + epsilon)
            else:
                centered = torch.zeros_like(centered)

        for rows, value in zip(prefix_rows, centered, strict=True):
            out[rows] = value
    return out


def _group_ids(*arrays: np.ndarray) -> np.ndarray:
    return np.array(["::".join(map(str, values)) for values in zip(*arrays, strict=True)], dtype=object)


def _sequence_scores(token_level_rewards: torch.Tensor) -> torch.Tensor:
    return token_level_rewards.sum(dim=-1).float()


def _compute_horizon_masks(
    batch: DataProto, round_index: int, progressive_block_size: int, max_response_length: int
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    response_mask = batch.batch["response_mask"]
    response_len = response_mask.shape[-1]
    horizon = min(int(round_index) * int(progressive_block_size), int(max_response_length), response_len)
    prefix_lengths = torch.full((response_mask.shape[0],), horizon, dtype=torch.long, device=response_mask.device)
    prefix_lengths = torch.minimum(prefix_lengths, response_mask.sum(dim=-1).long())
    prefix_mask, suffix_mask = _make_prefix_suffix_masks(response_mask, prefix_lengths)
    return horizon, prefix_lengths, prefix_mask, suffix_mask


def _make_prefix_suffix_masks(
    response_mask: torch.Tensor, prefix_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    response_len = response_mask.shape[-1]
    positions = torch.arange(response_len, device=response_mask.device).unsqueeze(0)
    prefix_lengths = prefix_lengths.to(device=response_mask.device).long().clamp(min=0, max=response_len).unsqueeze(1)
    prefix_mask = (positions < prefix_lengths) & response_mask.bool()
    suffix_mask = (positions >= prefix_lengths) & response_mask.bool()
    return prefix_mask.to(response_mask.dtype), suffix_mask.to(response_mask.dtype)


def _clone_for_masked_update(
    batch: DataProto,
    pg_mask: torch.Tensor,
    scalar_advantages: torch.Tensor,
    old_log_probs: torch.Tensor | None = None,
    kl_mask: torch.Tensor | None = None,
    kl_ref_log_probs: torch.Tensor | None = None,
) -> DataProto:
    update_batch = batch.select(
        batch_keys=list(batch.batch.keys()),
        non_tensor_batch_keys=list(batch.non_tensor_batch.keys()),
        meta_info_keys=list(batch.meta_info.keys()),
        deepcopy=True,
    )
    if old_log_probs is not None:
        update_batch.batch["old_log_probs"] = old_log_probs.to(device=pg_mask.device, dtype=torch.float32)
    update_batch.batch["hpf_pg_mask"] = pg_mask
    update_batch.batch["advantages"] = scalar_advantages.unsqueeze(-1).to(pg_mask.device) * pg_mask
    update_batch.batch["returns"] = update_batch.batch["advantages"]
    if kl_mask is not None and kl_ref_log_probs is not None:
        update_batch.batch["hpf_kl_mask"] = kl_mask
        update_batch.batch["hpf_kl_ref_log_prob"] = kl_ref_log_probs.to(device=pg_mask.device, dtype=torch.float32)
    return update_batch


def build_hpf_mixed_policy_grpo_batch(
    batch: DataProto,
    prefix_horizon: int,
    suffix_window_size: int | None,
    leader_old_log_probs: torch.Tensor,
    follower_old_log_probs: torch.Tensor,
) -> HPFMaskedBatch:
    """Build one GRPO update under a position-dependent mixed policy.

    The rollout reward and GRPO advantage come from the complete trajectory.
    ``prefix_horizon`` determines the high/low-temperature cut independently
    from ``suffix_window_size``. When the suffix window is ``None``, every
    valid response token after the prefix is trained. The PPO anchor uses the
    high-temperature prefix policy on prefix tokens and the low-temperature
    follower policy on suffix tokens.
    """
    if "response_mask" not in batch.batch:
        raise ValueError("response_mask is required before building HPF masks")
    if "advantages" not in batch.batch:
        raise ValueError("advantages are required before building mixed-policy GRPO")
    if prefix_horizon < 0:
        raise ValueError(f"prefix_horizon must be non-negative, got {prefix_horizon}.")
    if suffix_window_size is not None and suffix_window_size <= 0:
        raise ValueError(f"suffix_window_size must be positive or None, got {suffix_window_size}.")

    response_mask = batch.batch["response_mask"]
    response_len = response_mask.shape[-1]
    prefix_horizon = min(int(prefix_horizon), response_len)
    prefix_lengths = torch.full(
        (response_mask.shape[0],), prefix_horizon, dtype=torch.long, device=response_mask.device
    )
    prefix_lengths = torch.minimum(prefix_lengths, response_mask.sum(dim=-1).long())
    prefix_mask, full_suffix_mask = _make_prefix_suffix_masks(response_mask, prefix_lengths)
    positions = torch.arange(response_len, device=response_mask.device).unsqueeze(0)
    if suffix_window_size is None:
        suffix_update_mask = full_suffix_mask
    else:
        suffix_ends = (prefix_lengths + int(suffix_window_size)).clamp(max=response_len).unsqueeze(1)
        suffix_update_mask = full_suffix_mask.bool() & (positions < suffix_ends)
        suffix_update_mask = suffix_update_mask.to(response_mask.dtype)
    update_mask = (prefix_mask.bool() | suffix_update_mask.bool()).to(response_mask.dtype)

    mixed_old_log_probs = torch.where(
        prefix_mask.bool(),
        leader_old_log_probs.to(device=response_mask.device, dtype=torch.float32),
        follower_old_log_probs.to(device=response_mask.device, dtype=torch.float32),
    )
    update_batch = _clone_for_masked_update(
        batch,
        update_mask,
        torch.zeros(response_mask.shape[0], device=response_mask.device, dtype=torch.float32),
        old_log_probs=mixed_old_log_probs,
    )
    # Preserve the standard GRPO token-level advantages computed from the
    # complete-trajectory rewards; only the PG mask limits trained tokens.
    update_batch.batch["advantages"] = batch.batch["advantages"] * update_mask.to(batch.batch["advantages"])
    update_batch.batch["returns"] = update_batch.batch["advantages"]

    suffix_nonempty = suffix_update_mask.sum(dim=-1) > 0
    metrics = {
        "hpf/mixed_policy_grpo_enabled": 1.0,
        "hpf/mixed_policy_grpo_full_suffix_tail": float(suffix_window_size is None),
        # Compatibility alias for existing dashboards; this is the prefix cut,
        # not the independently configured suffix-window length.
        "hpf/mixed_policy_grpo_horizon_tokens": float(prefix_horizon),
        "hpf/mixed_policy_grpo_prefix_horizon_tokens": float(prefix_horizon),
        "hpf/mixed_policy_grpo_suffix_window_size": float(
            suffix_window_size if suffix_window_size is not None else -1
        ),
        "hpf/mixed_policy_grpo_prefix_tokens_mean": float(prefix_mask.sum(dim=-1).float().mean().item()),
        "hpf/mixed_policy_grpo_suffix_tokens_mean": float(
            suffix_update_mask.sum(dim=-1).float().mean().item()
        ),
        "hpf/mixed_policy_grpo_update_tokens_mean": float(update_mask.sum(dim=-1).float().mean().item()),
        "hpf/mixed_policy_grpo_suffix_empty_frac": float((~suffix_nonempty).float().mean().item()),
    }
    return HPFMaskedBatch(
        batch=update_batch,
        metrics=metrics,
        prefix_mask=prefix_mask,
        suffix_mask=suffix_update_mask,
    )


def _masked_sequence_correction(
    updated_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    mask: torch.Tensor,
    correction_clip: float,
    metric_prefix: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw_delta = ((updated_log_probs - old_log_probs).to(mask.device) * mask).sum(dim=-1)
    delta = raw_delta
    clipped_upper = torch.zeros_like(raw_delta, dtype=torch.bool)
    clipped_lower = torch.zeros_like(raw_delta, dtype=torch.bool)
    if np.isfinite(correction_clip):
        clipped_upper = raw_delta > correction_clip
        clipped_lower = raw_delta < -correction_clip
        delta = delta.clamp(min=-correction_clip, max=correction_clip)
    correction = torch.exp(delta).detach()
    return correction, {
        f"{metric_prefix}_clip_upper_frac": float(clipped_upper.float().mean().item()),
        f"{metric_prefix}_clip_lower_frac": float(clipped_lower.float().mean().item()),
        f"{metric_prefix}_clip_frac": float((clipped_upper | clipped_lower).float().mean().item()),
        f"{metric_prefix}_log_ratio_mean": float(delta.mean().item()),
        f"{metric_prefix}_log_ratio_std": float(delta.std(unbiased=True).item()),
        f"{metric_prefix}_ratio_mean": float(correction.mean().item()),
        f"{metric_prefix}_ratio_max": float(correction.max().item()),
        f"{metric_prefix}_ratio_min": float(correction.min().item()),
    }


def build_hpf_corrected_leader_batch(
    batch: DataProto,
    round_index: int,
    progressive_block_size: int,
    max_response_length: int,
    leader_old_log_probs: torch.Tensor,
    leader_post_follower_log_probs: torch.Tensor,
    follower_old_log_probs: torch.Tensor,
    follower_post_follower_log_probs: torch.Tensor,
    correction_clip: float,
) -> HPFMaskedBatch:
    """Build the Algorithm-2 leader batch after the follower update.

    The leader phase is prefix-level. We use prefix correction to map rollout
    prefixes from the round-start leader to the post-follower leader, and suffix
    correction to estimate the post-follower value of each sampled prefix.
    """
    if "response_mask" not in batch.batch:
        raise ValueError("response_mask is required before building HPF masks")
    if "token_level_rewards" not in batch.batch:
        raise ValueError("token_level_rewards is required before building HPF masks")
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("uid is required before building HPF advantages")

    horizon, _, prefix_mask, suffix_mask = _compute_horizon_masks(
        batch, round_index, progressive_block_size, max_response_length
    )
    device = prefix_mask.device
    rewards = _sequence_scores(batch.batch["token_level_rewards"])
    uid = batch.non_tensor_batch["uid"]
    problem_ids = batch.non_tensor_batch.get("hpf_problem_uid", uid)
    prefix_group_ids = batch.non_tensor_batch.get("hpf_prefix_uid")
    has_tree_groups = prefix_group_ids is not None
    if prefix_group_ids is None:
        prefix_ids = np.arange(len(uid), dtype=object)
        prefix_group_ids = _group_ids(uid, prefix_ids)

    prefix_correction, prefix_metrics = _masked_sequence_correction(
        updated_log_probs=leader_post_follower_log_probs,
        old_log_probs=leader_old_log_probs,
        mask=prefix_mask,
        correction_clip=correction_clip,
        metric_prefix="hpf/prefix_correction",
    )
    suffix_correction, suffix_metrics = _masked_sequence_correction(
        updated_log_probs=follower_post_follower_log_probs,
        old_log_probs=follower_old_log_probs,
        mask=suffix_mask,
        correction_clip=correction_clip,
        metric_prefix="hpf/suffix_correction",
    )

    prefix_q = torch.zeros_like(rewards, dtype=torch.float32)
    prefix_weight = torch.zeros_like(rewards, dtype=torch.float32)
    unique_prefix_ids = np.unique(prefix_group_ids)
    for prefix_group_id in unique_prefix_ids:
        idx_np = np.nonzero(prefix_group_ids == prefix_group_id)[0]
        idx = torch.as_tensor(idx_np, device=device, dtype=torch.long)
        weights = suffix_correction[idx].float()
        denom = weights.sum().clamp_min(1e-12)
        q_value = (weights * rewards[idx].float()).sum() / denom
        prefix_q[idx] = q_value
        prefix_weight[idx] = prefix_correction[idx[0]].float()

    baseline = torch.zeros_like(rewards, dtype=torch.float32)
    for problem_id in np.unique(problem_ids):
        problem_idx_np = np.nonzero(problem_ids == problem_id)[0]
        problem_prefix_ids = prefix_group_ids[problem_idx_np]
        unique_problem_prefix_ids = np.unique(problem_prefix_ids)
        if len(unique_problem_prefix_ids) == 0:
            continue
        prefix_rows = []
        q_values = []
        c_values = []
        for prefix_group_id in unique_problem_prefix_ids:
            rows_np = problem_idx_np[np.nonzero(problem_prefix_ids == prefix_group_id)[0]]
            rows = torch.as_tensor(rows_np, device=device, dtype=torch.long)
            prefix_rows.append(rows)
            q_values.append(prefix_q[rows[0]].float())
            c_values.append(prefix_weight[rows[0]].float())
        q_tensor = torch.stack(q_values)
        c_tensor = torch.stack(c_values)
        value = (c_tensor * q_tensor).sum() / c_tensor.sum().clamp_min(1e-12)
        for rows in prefix_rows:
            baseline[rows] = value

    leader_adv = prefix_weight * (prefix_q - baseline)
    leader_batch = _clone_for_masked_update(
        batch,
        prefix_mask,
        leader_adv,
        old_log_probs=leader_post_follower_log_probs,
    )
    suffix_nonempty = suffix_mask.sum(dim=-1) > 0
    metrics = {
        "hpf/enabled": 1.0,
        "hpf/round_index": float(round_index),
        "hpf/horizon_tokens": float(horizon),
        "hpf/prefix_tokens_mean": float(prefix_mask.sum(dim=-1).float().mean().item()),
        "hpf/suffix_tokens_mean": float(suffix_mask.sum(dim=-1).float().mean().item()),
        "hpf/suffix_empty_frac": float((~suffix_nonempty).float().mean().item()),
        "hpf/leader_adv_mean": float(leader_adv.mean().item()),
        "hpf/leader_adv_std": float(leader_adv.std(unbiased=True).item()),
        "hpf/leader_prefix_value_mean": float(prefix_q.mean().item()),
        "hpf/leader_prefix_value_std": float(prefix_q.std(unbiased=True).item()),
        "hpf/leader_baseline_mean": float(baseline.mean().item()),
        "hpf/leader_baseline_std": float(baseline.std(unbiased=True).item()),
        "hpf/minimal_grouping": 0.0 if has_tree_groups else 1.0,
        "hpf/leader_prefix_groups": float(len(unique_prefix_ids)),
    }
    metrics.update(prefix_metrics)
    metrics.update(suffix_metrics)
    # Backward-compatible metric aliases for the existing dashboard.
    for key, value in suffix_metrics.items():
        metrics[key.replace("hpf/suffix_correction", "hpf/correction")] = value
    return HPFMaskedBatch(batch=leader_batch, metrics=metrics, prefix_mask=prefix_mask, suffix_mask=suffix_mask)


def build_hpf_fresh_leader_batch(
    batch: DataProto,
    round_index: int,
    progressive_block_size: int,
    max_response_length: int,
    epsilon: float,
    std_normalize: bool,
    leader_old_log_probs: torch.Tensor,
) -> HPFMaskedBatch:
    """Build the Algorithm-3 leader batch from a fresh post-follower tree."""
    if "response_mask" not in batch.batch:
        raise ValueError("response_mask is required before building HPF masks")
    if "token_level_rewards" not in batch.batch:
        raise ValueError("token_level_rewards is required before building HPF masks")
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("uid is required before building HPF advantages")

    horizon, _, prefix_mask, suffix_mask = _compute_horizon_masks(
        batch, round_index, progressive_block_size, max_response_length
    )
    device = prefix_mask.device
    rewards = _sequence_scores(batch.batch["token_level_rewards"])
    uid = batch.non_tensor_batch["uid"]
    problem_ids = batch.non_tensor_batch.get("hpf_problem_uid", uid)
    prefix_group_ids = batch.non_tensor_batch.get("hpf_prefix_uid")
    has_tree_groups = prefix_group_ids is not None
    if prefix_group_ids is None:
        prefix_ids = np.arange(len(uid), dtype=object)
        prefix_group_ids = _group_ids(uid, prefix_ids)

    prefix_q = torch.zeros_like(rewards, dtype=torch.float32)
    unique_prefix_ids = np.unique(prefix_group_ids)
    for prefix_group_id in unique_prefix_ids:
        idx_np = np.nonzero(prefix_group_ids == prefix_group_id)[0]
        idx = torch.as_tensor(idx_np, device=device, dtype=torch.long)
        prefix_q[idx] = rewards[idx].float().mean()

    leader_adv = torch.zeros_like(rewards, dtype=torch.float32)
    for problem_id in np.unique(problem_ids):
        problem_idx_np = np.nonzero(problem_ids == problem_id)[0]
        problem_prefix_ids = prefix_group_ids[problem_idx_np]
        unique_problem_prefix_ids = np.unique(problem_prefix_ids)
        if len(unique_problem_prefix_ids) <= 1:
            continue

        prefix_rows = []
        q_values = []
        for prefix_group_id in unique_problem_prefix_ids:
            rows_np = problem_idx_np[np.nonzero(problem_prefix_ids == prefix_group_id)[0]]
            rows = torch.as_tensor(rows_np, device=device, dtype=torch.long)
            prefix_rows.append(rows)
            q_values.append(prefix_q[rows[0]].float())

        q_tensor = torch.stack(q_values)
        centered = q_tensor - q_tensor.mean()
        if std_normalize:
            std = q_tensor.std(unbiased=True)
            if torch.isfinite(std) and std > 0:
                centered = centered / (std + epsilon)
            else:
                centered = torch.zeros_like(centered)

        for rows, value in zip(prefix_rows, centered, strict=True):
            leader_adv[rows] = value

    suffix_nonempty = suffix_mask.sum(dim=-1) > 0
    leader_batch = _clone_for_masked_update(
        batch,
        prefix_mask,
        leader_adv,
        old_log_probs=leader_old_log_probs,
    )
    metrics = {
        "hpf/enabled": 1.0,
        "hpf/fresh_leader_tree_enabled": 1.0,
        "hpf/round_index": float(round_index),
        "hpf/horizon_tokens": float(horizon),
        "hpf/prefix_tokens_mean": float(prefix_mask.sum(dim=-1).float().mean().item()),
        "hpf/suffix_tokens_mean": float(suffix_mask.sum(dim=-1).float().mean().item()),
        "hpf/suffix_empty_frac": float((~suffix_nonempty).float().mean().item()),
        "hpf/leader_adv_mean": float(leader_adv.mean().item()),
        "hpf/leader_adv_std": float(leader_adv.std(unbiased=True).item()),
        "hpf/leader_prefix_value_mean": float(prefix_q.mean().item()),
        "hpf/leader_prefix_value_std": float(prefix_q.std(unbiased=True).item()),
        "hpf/minimal_grouping": 0.0 if has_tree_groups else 1.0,
        "hpf/leader_prefix_groups": float(len(unique_prefix_ids)),
    }
    return HPFMaskedBatch(batch=leader_batch, metrics=metrics, prefix_mask=prefix_mask, suffix_mask=suffix_mask)


def build_hpf_masked_batches(
    batch: DataProto,
    round_index: int,
    progressive_block_size: int,
    max_response_length: int,
    epsilon: float = 1e-6,
    std_normalize: bool = True,
    follower_old_log_probs: torch.Tensor | None = None,
    leader_old_log_probs: torch.Tensor | None = None,
) -> tuple[HPFMaskedBatch | None, HPFMaskedBatch]:
    """Build suffix/follower and prefix/leader masked update batches.

    This builds the follower suffix update and leader prefix update for HPF-RLVR.
    When tree rollout metadata is available, follower advantages are normalized
    within each sampled prefix and leader prefix rewards use any-correct over the
    suffixes under that prefix. Without tree metadata, it falls back to the
    earlier masked full-response smoke behavior.
    """
    if "response_mask" not in batch.batch:
        raise ValueError("response_mask is required before building HPF masks")
    if "token_level_rewards" not in batch.batch:
        raise ValueError("token_level_rewards is required before building HPF masks")
    if "uid" not in batch.non_tensor_batch:
        raise ValueError("uid is required before building HPF advantages")

    horizon, _, prefix_mask, suffix_mask = _compute_horizon_masks(
        batch, round_index, progressive_block_size, max_response_length
    )
    device = prefix_mask.device

    rewards = _sequence_scores(batch.batch["token_level_rewards"])
    correct = (rewards > 0).float()
    uid = batch.non_tensor_batch["uid"]

    problem_ids = batch.non_tensor_batch.get("hpf_problem_uid", uid)
    prefix_group_ids = batch.non_tensor_batch.get("hpf_prefix_uid")
    has_tree_groups = prefix_group_ids is not None
    if prefix_group_ids is None:
        prefix_ids = np.arange(len(uid), dtype=object)
        prefix_group_ids = _group_ids(uid, prefix_ids)

    follower_adv = _normalize_group_scores(rewards, prefix_group_ids, epsilon, std_normalize)

    leader_reward = torch.zeros_like(correct)
    for prefix_group_id in np.unique(prefix_group_ids):
        idx_np = np.nonzero(prefix_group_ids == prefix_group_id)[0]
        idx = torch.as_tensor(idx_np, device=device, dtype=torch.long)
        leader_reward[idx] = correct[idx].max()
    leader_adv = _normalize_repeated_prefix_scores(
        leader_reward,
        problem_ids,
        prefix_group_ids,
        epsilon,
        std_normalize,
    )

    suffix_nonempty = suffix_mask.sum(dim=-1) > 0
    follower_batch = None
    if bool(suffix_nonempty.any()):
        follower_update_batch = _clone_for_masked_update(
            batch,
            suffix_mask,
            follower_adv,
            old_log_probs=follower_old_log_probs,
            kl_mask=prefix_mask,
            kl_ref_log_probs=leader_old_log_probs,
        )[
            suffix_nonempty.detach().cpu().numpy()
        ]
        follower_batch = HPFMaskedBatch(
            batch=follower_update_batch,
            metrics={
                "hpf/follower_batch_size": float(len(follower_update_batch)),
                "hpf/follower_nonempty_frac": float(suffix_nonempty.float().mean().item()),
                "hpf/follower_adv_mean": float(follower_adv.mean().item()),
                "hpf/follower_adv_std": float(follower_adv.std(unbiased=True).item()),
            },
            prefix_mask=prefix_mask[suffix_nonempty],
            suffix_mask=suffix_mask[suffix_nonempty],
        )

    leader_batch = HPFMaskedBatch(
        batch=_clone_for_masked_update(batch, prefix_mask, leader_adv, old_log_probs=leader_old_log_probs),
        metrics={
            "hpf/enabled": 1.0,
            "hpf/round_index": float(round_index),
            "hpf/horizon_tokens": float(horizon),
            "hpf/prefix_tokens_mean": float(prefix_mask.sum(dim=-1).float().mean().item()),
            "hpf/suffix_tokens_mean": float(suffix_mask.sum(dim=-1).float().mean().item()),
            "hpf/suffix_empty_frac": float((~suffix_nonempty).float().mean().item()),
            "hpf/leader_adv_mean": float(leader_adv.mean().item()),
            "hpf/leader_adv_std": float(leader_adv.std(unbiased=True).item()),
            "hpf/leader_prefix_any_correct_rate": float(leader_reward.mean().item()),
            "hpf/minimal_grouping": 0.0 if has_tree_groups else 1.0,
            "hpf/leader_prefix_groups": float(len(np.unique(prefix_group_ids))),
        },
        prefix_mask=prefix_mask,
        suffix_mask=suffix_mask,
    )
    return follower_batch, leader_batch
