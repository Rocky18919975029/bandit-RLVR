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
import json
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class BranchedPrefixPlan:
    """Random prefix cuts used to expand K initial trajectories into an N-way pool."""

    pool_size: int
    initial_count: int
    branches_per_initial: int
    parent_global_indices: np.ndarray
    parent_local_indices: np.ndarray
    branch_local_indices: np.ndarray
    cut_positions: np.ndarray
    cut_with_replacement: np.ndarray


@dataclass(frozen=True)
class InitialRolloutReplay:
    """Exact initial trajectories recovered from a branched-SIR pool dump."""

    source_path: str
    prompt_count: int
    initial_count: int
    response_token_ids: tuple[tuple[int, ...], ...]
    response_log_probs: tuple[tuple[float, ...], ...]
    source_pool_indices: np.ndarray


def load_initial_rollout_replay(
    path: str | Path,
    *,
    expected_prompts: list[str],
    expected_ground_truths: list[object],
    expected_data_sources: list[object],
    initial_count: int,
    expected_step: int = 1,
) -> InitialRolloutReplay:
    """Load and fail-closed validate the initial K rows from a saved SIR pool."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"SIR initial-rollout replay file does not exist: {source_path}")
    if initial_count < 2:
        raise ValueError(f"initial_count must be at least 2, got {initial_count}")
    expected_count = len(expected_prompts)
    if len(expected_ground_truths) != expected_count or len(expected_data_sources) != expected_count:
        raise ValueError("Expected prompt, ground-truth, and data-source lists must have equal lengths")

    response_token_ids: list[tuple[int, ...]] = []
    response_log_probs: list[tuple[float, ...]] = []
    source_pool_indices: list[int] = []
    row_count = 0
    with source_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if row_count >= expected_count:
                raise ValueError(
                    f"Replay pool has more than the expected {expected_count} prompt rows: {source_path}"
                )
            row = json.loads(line)
            if int(row.get("step", -1)) != expected_step:
                raise ValueError(
                    f"Replay pool step mismatch at line {line_number}: "
                    f"expected {expected_step}, got {row.get('step')!r}"
                )
            if row.get("pool_mode") != "branched_prefix":
                raise ValueError(
                    f"Replay pool line {line_number} is not branched_prefix: {row.get('pool_mode')!r}"
                )
            if int(row.get("selected_count", -1)) != initial_count:
                raise ValueError(
                    f"Replay pool K mismatch at line {line_number}: "
                    f"expected {initial_count}, got {row.get('selected_count')!r}"
                )
            if str(row.get("prompt")) != expected_prompts[row_count]:
                raise ValueError(
                    f"Replay prompt mismatch at prompt index {row_count}; "
                    "the data order or chat template differs from the source SIR run"
                )
            if str(row.get("ground_truth")) != str(expected_ground_truths[row_count]):
                raise ValueError(f"Replay ground truth mismatch at prompt index {row_count}")
            if str(row.get("data_source")) != str(expected_data_sources[row_count]):
                raise ValueError(f"Replay data source mismatch at prompt index {row_count}")

            initial_candidates = [
                candidate
                for candidate in row.get("candidates", [])
                if candidate.get("sir_pool_origin") == "initial"
            ]
            initial_candidates.sort(key=lambda candidate: int(candidate.get("sir_parent_index", -1)))
            parent_indices = [int(candidate.get("sir_parent_index", -1)) for candidate in initial_candidates]
            if parent_indices != list(range(initial_count)):
                raise ValueError(
                    f"Replay prompt {row_count} has invalid initial parent indices: {parent_indices}"
                )

            for candidate in initial_candidates:
                sampled_ids = tuple(int(token_id) for token_id in candidate.get("sampled_token_ids", []))
                response_ids = tuple(int(token_id) for token_id in candidate.get("response_token_ids", []))
                log_probs = tuple(float(log_prob) for log_prob in candidate.get("sampled_token_log_probs", []))
                if not sampled_ids:
                    raise ValueError(f"Replay prompt {row_count} contains an empty initial trajectory")
                if sampled_ids != response_ids:
                    raise ValueError(
                        f"Replay prompt {row_count} initial candidate {candidate.get('pool_index')} "
                        "contains non-sampled response tokens and cannot be replayed as ordinary single-turn GRPO"
                    )
                if len(log_probs) != len(sampled_ids):
                    raise ValueError(
                        f"Replay prompt {row_count} initial candidate {candidate.get('pool_index')} has "
                        f"{len(sampled_ids)} tokens but {len(log_probs)} log-probabilities"
                    )
                if not np.all(np.isfinite(np.asarray(log_probs, dtype=np.float64))):
                    raise ValueError(f"Replay prompt {row_count} contains non-finite behavior log-probabilities")
                response_token_ids.append(sampled_ids)
                response_log_probs.append(log_probs)
                source_pool_indices.append(int(candidate["pool_index"]))
            row_count += 1

    if row_count != expected_count:
        raise ValueError(f"Replay pool has {row_count} prompt rows, expected {expected_count}: {source_path}")
    expected_trajectories = expected_count * initial_count
    if len(response_token_ids) != expected_trajectories:
        raise ValueError(
            f"Replay pool yielded {len(response_token_ids)} initial trajectories, expected {expected_trajectories}"
        )
    return InitialRolloutReplay(
        source_path=str(source_path),
        prompt_count=expected_count,
        initial_count=initial_count,
        response_token_ids=tuple(response_token_ids),
        response_log_probs=tuple(response_log_probs),
        source_pool_indices=np.asarray(source_pool_indices, dtype=np.int64),
    )


def stable_seed(*parts: object) -> int:
    """Derive a reproducible uint32 seed without Python's salted hash."""
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="big", signed=False)


def build_branched_prefix_plan(
    response_lengths: np.ndarray,
    *,
    pool_size: int,
    initial_count: int,
    block_length: int,
    seed: int,
    global_step: int,
) -> BranchedPrefixPlan:
    """Choose nonterminal prefix cuts for every initial trajectory.

    ``response_lengths`` must contain K initial trajectories contiguously for
    each prompt. A cut position ``c`` retains tokens ``[:c]``. The terminal
    boundary is excluded so a branch always generates at least one fresh token.
    Cuts are distinct when enough positions exist; short responses sample the
    available cuts with replacement. A one-token response falls back to cut 0.
    """
    lengths = np.asarray(response_lengths, dtype=np.int64)
    if lengths.ndim != 1 or lengths.size == 0:
        raise ValueError("response_lengths must be a non-empty one-dimensional array")
    if pool_size <= initial_count or pool_size % initial_count != 0:
        raise ValueError(
            "branched SIR requires pool_size to be an integer multiple of initial_count and larger than it; "
            f"got N={pool_size}, K={initial_count}"
        )
    if initial_count < 2:
        raise ValueError(f"initial_count must be at least 2, got {initial_count}")
    if block_length <= 0:
        raise ValueError(f"block_length must be positive, got {block_length}")
    if lengths.size % initial_count != 0:
        raise ValueError(f"initial response count {lengths.size} is not divisible by initial_count={initial_count}")

    branches_per_initial = pool_size // initial_count - 1
    parent_global_indices: list[int] = []
    parent_local_indices: list[int] = []
    branch_local_indices: list[int] = []
    cut_positions: list[int] = []
    cut_with_replacement: list[bool] = []
    for parent_global_index, response_length in enumerate(lengths.tolist()):
        prompt_index, parent_local_index = divmod(parent_global_index, initial_count)
        max_cut = min(block_length, response_length - 1)
        eligible_cuts = np.arange(1, max_cut + 1, dtype=np.int64) if max_cut >= 1 else np.asarray([0], dtype=np.int64)
        use_replacement = eligible_cuts.size < branches_per_initial
        parent_seed = stable_seed(seed, global_step, prompt_index, parent_local_index, "prefix-cuts")
        parent_cuts = np.random.default_rng(parent_seed).choice(
            eligible_cuts,
            size=branches_per_initial,
            replace=use_replacement,
        )
        for branch_local_index, cut_position in enumerate(parent_cuts.tolist()):
            parent_global_indices.append(parent_global_index)
            parent_local_indices.append(parent_local_index)
            branch_local_indices.append(branch_local_index)
            cut_positions.append(cut_position)
            cut_with_replacement.append(use_replacement)

    return BranchedPrefixPlan(
        pool_size=pool_size,
        initial_count=initial_count,
        branches_per_initial=branches_per_initial,
        parent_global_indices=np.asarray(parent_global_indices, dtype=np.int64),
        parent_local_indices=np.asarray(parent_local_indices, dtype=np.int64),
        branch_local_indices=np.asarray(branch_local_indices, dtype=np.int64),
        cut_positions=np.asarray(cut_positions, dtype=np.int64),
        cut_with_replacement=np.asarray(cut_with_replacement, dtype=bool),
    )


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

    Sampling is weighted without replacement. The returned indices preserve
    prompt-group order and selection order, and every selected GRPO group
    therefore contains distinct source trajectories.
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
        # Gumbel-top-k samples an ordered, weighted subset without replacement.
        # Use the unnormalized log weights instead of ``log(weights)`` so that
        # candidates remain selectable even when normalized weights underflow
        # to exactly zero for highly concentrated groups.
        selection_logits = (float(alpha) - 1.0) * group_joint_log_probs
        gumbels = np.random.default_rng(group_seed).gumbel(size=pool_size)
        selected_local = np.argsort(selection_logits + gumbels)[::-1][:selected_count].astype(np.int64)
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
