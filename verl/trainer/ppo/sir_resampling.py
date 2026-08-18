# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sampling-importance-resampling helpers for GRPO rollout groups.

The proposal distribution is the rollout policy ``p``. Tempering the first
``B`` generated tokens targets ``q(a) ∝ p(a) ** alpha``, so the categorical
SIR weights are proportional to ``p(a) ** (alpha - 1)``. A response that emits
EOS before ``B`` contributes every generated-token log-probability, including
the EOS probability, and has no synthetic post-EOS terms.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SIRGroupPlan:
    """SIR diagnostics and draws for one prompt's contiguous rollout pool."""

    group_index: int
    seed: int
    response_lengths: np.ndarray
    prefix_joint_log_probs: np.ndarray
    weights: np.ndarray
    selected_local_indices: np.ndarray
    selected_counts: np.ndarray
    selected_draws: tuple[tuple[int, ...], ...]
    effective_sample_size: float


@dataclass(frozen=True)
class SIRSelectionPlan:
    """Global row selection plus per-prompt diagnostics."""

    pool_size: int
    selected_count: int
    block_length: int
    alpha: float
    selected_global_indices: np.ndarray
    selected_pool_indices: np.ndarray
    selected_draw_indices: np.ndarray
    groups: tuple[SIRGroupPlan, ...]

    def metrics(self) -> dict[str, float]:
        """Return scalar diagnostics suitable for the trainer metrics logger."""
        ess = np.asarray([group.effective_sample_size for group in self.groups], dtype=np.float64)
        max_weights = np.asarray([group.weights.max() for group in self.groups], dtype=np.float64)
        unique_counts = np.asarray([np.count_nonzero(group.selected_counts) for group in self.groups], dtype=np.float64)
        response_lengths = np.concatenate([group.response_lengths for group in self.groups])
        joint_log_probs = np.concatenate([group.prefix_joint_log_probs for group in self.groups])
        return {
            "sir/enabled": 1.0,
            "sir/pool_size": float(self.pool_size),
            "sir/selected_count": float(self.selected_count),
            "sir/block_length": float(self.block_length),
            "sir/alpha": float(self.alpha),
            "sir/ess_mean": float(ess.mean()),
            "sir/ess_fraction_mean": float((ess / self.pool_size).mean()),
            "sir/max_weight_mean": float(max_weights.mean()),
            "sir/unique_selected_mean": float(unique_counts.mean()),
            "sir/unique_selected_fraction_mean": float((unique_counts / self.pool_size).mean()),
            "sir/unique_draw_fraction_mean": float((unique_counts / self.selected_count).mean()),
            "sir/duplicate_draw_fraction_mean": float(1.0 - (unique_counts / self.selected_count).mean()),
            "sir/short_response_fraction": float(np.mean(response_lengths < self.block_length)),
            "sir/prefix_joint_log_prob_mean": float(joint_log_probs.mean()),
            "sir/prefix_joint_log_prob_min": float(joint_log_probs.min()),
            "sir/prefix_joint_log_prob_max": float(joint_log_probs.max()),
        }


def stable_seed(*parts: object) -> int:
    """Derive a reproducible uint32 seed without Python's salted hash."""
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="big", signed=False)


def tempered_sir_weights(prefix_joint_log_probs: np.ndarray, alpha: float) -> np.ndarray:
    """Normalize weights proportional to ``exp((alpha - 1) * log p)``."""
    values = np.asarray(prefix_joint_log_probs, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("prefix_joint_log_probs must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("prefix_joint_log_probs contains non-finite values")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"alpha must be finite and positive, got {alpha}")

    logits = (float(alpha) - 1.0) * values
    logits -= logits.max()
    unnormalized = np.exp(logits)
    return unnormalized / unnormalized.sum(dtype=np.float64)


def _prefix_statistics(
    rollout_log_probs: np.ndarray,
    response_mask: np.ndarray,
    block_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute valid response lengths and B-capped joint log-probabilities."""
    lengths = response_mask.astype(bool).sum(axis=1, dtype=np.int64)
    if np.any(lengths == 0):
        empty_rows = np.flatnonzero(lengths == 0).tolist()
        raise ValueError(f"SIR received responses without generated tokens at rows {empty_rows[:8]}")

    joint_log_probs = np.empty(rollout_log_probs.shape[0], dtype=np.float64)
    for row_index in range(rollout_log_probs.shape[0]):
        valid_log_probs = rollout_log_probs[row_index][response_mask[row_index].astype(bool)]
        prefix = np.asarray(valid_log_probs[:block_length], dtype=np.float64)
        if not np.all(np.isfinite(prefix)):
            raise ValueError(f"SIR rollout row {row_index} contains non-finite chosen-token log-probabilities")
        joint_log_probs[row_index] = prefix.sum(dtype=np.float64)
    return lengths, joint_log_probs


def build_sir_selection_plan(
    rollout_log_probs: np.ndarray,
    response_mask: np.ndarray,
    *,
    pool_size: int,
    selected_count: int,
    block_length: int,
    alpha: float,
    seed: int,
    global_step: int,
) -> SIRSelectionPlan:
    """Draw ``selected_count`` rows per contiguous ``pool_size`` prompt group.

    Sampling is categorical with replacement, which is the standard SIR
    resampling step. The returned indices preserve prompt-group order and draw
    order, so duplicate source trajectories remain explicit GRPO rows.
    """
    log_probs = np.asarray(rollout_log_probs)
    mask = np.asarray(response_mask)
    if log_probs.ndim != 2 or mask.ndim != 2 or log_probs.shape != mask.shape:
        raise ValueError(
            "rollout_log_probs and response_mask must be equal-shape two-dimensional arrays; "
            f"got {log_probs.shape=} and {mask.shape=}"
        )
    if pool_size < 2:
        raise ValueError(f"pool_size must be at least 2 for GRPO SIR, got {pool_size}")
    if selected_count < 2 or selected_count > pool_size:
        raise ValueError(
            f"selected_count must satisfy 2 <= selected_count <= pool_size; got {selected_count} and {pool_size}"
        )
    if block_length <= 0:
        raise ValueError(f"block_length must be positive, got {block_length}")
    if log_probs.shape[0] % pool_size != 0:
        raise ValueError(
            f"rollout row count {log_probs.shape[0]} is not divisible by pool_size={pool_size}; "
            "SIR requires contiguous equal-size prompt groups"
        )

    lengths, joint_log_probs = _prefix_statistics(log_probs, mask, block_length)
    num_groups = log_probs.shape[0] // pool_size
    selected_global_indices: list[int] = []
    selected_pool_indices: list[int] = []
    selected_draw_indices: list[int] = []
    groups: list[SIRGroupPlan] = []

    for group_index in range(num_groups):
        start = group_index * pool_size
        stop = start + pool_size
        group_lengths = lengths[start:stop]
        group_joint_log_probs = joint_log_probs[start:stop]
        weights = tempered_sir_weights(group_joint_log_probs, alpha)
        group_seed = stable_seed(seed, global_step, group_index)
        selected_local = (
            np.random.default_rng(group_seed)
            .choice(
                pool_size,
                size=selected_count,
                replace=True,
                p=weights,
            )
            .astype(np.int64)
        )
        selected_counts = np.bincount(selected_local, minlength=pool_size).astype(np.int64)
        draws_by_candidate: list[list[int]] = [[] for _ in range(pool_size)]
        for draw_index, local_index in enumerate(selected_local.tolist()):
            selected_global_indices.append(start + local_index)
            selected_pool_indices.append(local_index)
            selected_draw_indices.append(draw_index)
            draws_by_candidate[local_index].append(draw_index)

        groups.append(
            SIRGroupPlan(
                group_index=group_index,
                seed=group_seed,
                response_lengths=group_lengths.copy(),
                prefix_joint_log_probs=group_joint_log_probs.copy(),
                weights=weights,
                selected_local_indices=selected_local,
                selected_counts=selected_counts,
                selected_draws=tuple(tuple(draws) for draws in draws_by_candidate),
                effective_sample_size=float(1.0 / np.square(weights).sum(dtype=np.float64)),
            )
        )

    return SIRSelectionPlan(
        pool_size=pool_size,
        selected_count=selected_count,
        block_length=block_length,
        alpha=float(alpha),
        selected_global_indices=np.asarray(selected_global_indices, dtype=np.int64),
        selected_pool_indices=np.asarray(selected_pool_indices, dtype=np.int64),
        selected_draw_indices=np.asarray(selected_draw_indices, dtype=np.int64),
        groups=tuple(groups),
    )
