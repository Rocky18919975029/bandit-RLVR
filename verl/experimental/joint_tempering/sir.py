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

"""Pure NumPy helpers for sampling-importance-resampling (SIR)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


def stable_seed(*parts: object) -> int:
    """Derive a reproducible uint32 seed without relying on Python's salted hash."""
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder="big", signed=False)


def prefix_joint_log_probs(token_log_probs: Sequence[Sequence[float]], block_length: int) -> np.ndarray:
    """Sum each candidate's chosen-token log-probabilities over its first block."""
    if block_length <= 0:
        raise ValueError(f"block_length must be positive, got {block_length}")
    if not token_log_probs:
        raise ValueError("token_log_probs must contain at least one candidate")

    joint_log_probs = []
    for candidate_index, candidate_log_probs in enumerate(token_log_probs):
        if len(candidate_log_probs) < block_length:
            raise ValueError(
                f"candidate {candidate_index} has {len(candidate_log_probs)} tokens, "
                f"fewer than block_length={block_length}"
            )
        prefix = np.asarray(candidate_log_probs[:block_length], dtype=np.float64)
        if not np.all(np.isfinite(prefix)):
            raise ValueError(f"candidate {candidate_index} contains non-finite log-probabilities")
        joint_log_probs.append(float(prefix.sum(dtype=np.float64)))

    return np.asarray(joint_log_probs, dtype=np.float64)


def sir_weights(joint_log_probs: Sequence[float], alpha: float) -> np.ndarray:
    """Return normalized SIR weights proportional to p(action) ** (alpha - 1)."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"alpha must be finite and positive, got {alpha}")

    log_probs = np.asarray(joint_log_probs, dtype=np.float64)
    if log_probs.ndim != 1 or log_probs.size == 0:
        raise ValueError("joint_log_probs must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(log_probs)):
        raise ValueError("joint_log_probs contains non-finite values")

    logits = (float(alpha) - 1.0) * log_probs
    logits -= logits.max()
    unnormalized = np.exp(logits)
    normalizer = unnormalized.sum(dtype=np.float64)
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("failed to normalize SIR weights")
    return unnormalized / normalizer


def effective_sample_size(weights: Sequence[float]) -> float:
    """Compute 1 / sum(w^2) after validating normalized non-negative weights."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative")
    total = values.sum(dtype=np.float64)
    if total <= 0:
        raise ValueError("weights must have positive mass")
    normalized = values / total
    return float(1.0 / np.square(normalized).sum(dtype=np.float64))


def resample_index(weights: Sequence[float], seed: int) -> int:
    """Draw one categorical index reproducibly."""
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative")
    total = values.sum(dtype=np.float64)
    if total <= 0:
        raise ValueError("weights must have positive mass")
    normalized = values / total
    return int(np.random.default_rng(seed).choice(values.size, p=normalized))
