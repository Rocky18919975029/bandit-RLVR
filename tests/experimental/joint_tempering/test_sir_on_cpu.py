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

import numpy as np
import pytest

from examples.joint_tempering.common import deterministic_problem_indices, response_token_budget
from examples.joint_tempering.sweep_joint import evaluate_configuration
from verl.experimental.joint_tempering.sir import (
    effective_sample_size,
    prefix_joint_log_probs,
    resample_index,
    sir_weights,
    stable_seed,
)


def test_alpha_one_produces_uniform_weights():
    weights = sir_weights([-100.0, -10.0, -1.0], alpha=1.0)

    np.testing.assert_allclose(weights, np.full(3, 1 / 3))
    assert effective_sample_size(weights) == pytest.approx(3.0)


def test_alpha_above_one_favors_higher_joint_probability():
    weights = sir_weights([-8.0, -2.0, -5.0], alpha=1.5)

    assert weights[1] > weights[2] > weights[0]
    assert weights.sum() == pytest.approx(1.0)


def test_prefix_joint_log_probs_requires_full_fixed_block():
    with pytest.raises(ValueError, match="fewer than block_length"):
        prefix_joint_log_probs([[-1.0, -2.0], [-1.0]], block_length=2)


def test_stable_sampling_is_reproducible():
    seed = stable_seed(42, "prompt-7", 64, 1.5)
    weights = [0.1, 0.2, 0.7]

    assert seed == stable_seed(42, "prompt-7", 64, 1.5)
    assert resample_index(weights, seed) == resample_index(weights, seed)


def test_problem_sampling_selects_exactly_128_unique_sorted_rows():
    indices = deterministic_problem_indices(dataset_size=1536, num_problems=128, seed=42)

    assert len(indices) == 128
    assert len(set(indices)) == 128
    assert indices == sorted(indices)
    assert indices == deterministic_problem_indices(dataset_size=1536, num_problems=128, seed=42)


def test_response_budget_respects_model_context_and_requested_cap():
    assert response_token_budget(prompt_token_count=512, requested_tokens=4096, max_model_len=4096) == 3584
    assert response_token_budget(prompt_token_count=512, requested_tokens=2048, max_model_len=4096) == 2048
    assert response_token_budget(prompt_token_count=512, requested_tokens=4096, max_model_len=None) == 4096


def test_response_budget_rejects_prompt_that_fills_context():
    with pytest.raises(ValueError, match="leaving no response capacity"):
        response_token_budget(prompt_token_count=4096, requested_tokens=4096, max_model_len=4096)


def _pool_row(candidate_lengths=(4, 4, 4)):
    log_probs = [
        [-1.0, -1.0, -1.0, -1.0],
        [-0.1, -0.1, -0.1, -0.1],
        [-2.0, -2.0, -2.0, -2.0],
    ]
    return {
        "prompt_id": "problem-1",
        "source_row_index": 9,
        "candidates": [
            {
                "token_ids": list(range(length)),
                "token_log_probs": candidate_log_probs[:length],
                "acc": candidate_index == 1,
                "score": 1.0 if candidate_index == 1 else -1.0,
            }
            for candidate_index, (length, candidate_log_probs) in enumerate(
                zip(candidate_lengths, log_probs, strict=True)
            )
        ],
    }


def test_complete_pool_is_reweighted_without_regeneration():
    result = evaluate_configuration(
        pool_row=_pool_row(),
        block_length=2,
        alpha=2.0,
        candidate_count=3,
        resample_seed=7,
    )

    assert result["valid"] is True
    assert result["prefix_joint_log_probs"] == pytest.approx([-2.0, -0.2, -4.0])
    assert result["joint_expected_acc"] > result["vanilla_expected_acc"]
    assert 1.0 <= result["sir_ess"] <= 3.0


def test_short_complete_trajectory_marks_configuration_invalid():
    result = evaluate_configuration(
        pool_row=_pool_row(candidate_lengths=(4, 1, 4)),
        block_length=2,
        alpha=1.5,
        candidate_count=3,
        resample_seed=7,
    )

    assert result["valid"] is False
    assert result["short_candidate_count"] == 1
    assert result["invalid_reason"] == "candidate_shorter_than_block"
