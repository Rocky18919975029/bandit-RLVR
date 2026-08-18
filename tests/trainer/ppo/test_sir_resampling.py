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

from verl.trainer.ppo.sir_resampling import build_sir_selection_plan, tempered_sir_weights


def test_tempered_weights_use_proposal_correction():
    joint_log_probs = np.asarray([-8.0, -2.0, -5.0])

    weights = tempered_sir_weights(joint_log_probs, alpha=1.5)

    expected = np.exp(0.5 * joint_log_probs - np.max(0.5 * joint_log_probs))
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected)
    assert weights[1] > weights[2] > weights[0]


def test_alpha_one_is_uniform():
    weights = tempered_sir_weights(np.asarray([-100.0, -10.0, -1.0]), alpha=1.0)

    np.testing.assert_allclose(weights, np.full(3, 1 / 3))


def test_selection_uses_valid_prefix_and_keeps_early_terminal_probability():
    log_probs = np.asarray(
        [
            [-1.0, -2.0, 0.0, 0.0],
            [-0.1, 0.0, 0.0, 0.0],
            [-2.0, -2.0, -2.0, 0.0],
        ]
    )
    response_mask = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
        ]
    )

    plan = build_sir_selection_plan(
        log_probs,
        response_mask,
        pool_size=3,
        selected_count=2,
        block_length=2,
        alpha=1.5,
        seed=7,
        global_step=3,
    )

    group = plan.groups[0]
    np.testing.assert_array_equal(group.response_lengths, [2, 1, 3])
    np.testing.assert_allclose(group.prefix_joint_log_probs, [-3.0, -0.1, -4.0])
    assert group.weights[1] > group.weights[0] > group.weights[2]


def test_selection_is_reproducible_grouped_and_has_no_duplicate_draws():
    log_probs = np.asarray(
        [
            [-9.0, 0.0],
            [-0.01, 0.0],
            [-9.0, 0.0],
            [-0.01, 0.0],
        ]
    )
    response_mask = np.asarray([[1, 0], [1, 0], [1, 0], [1, 0]])
    kwargs = dict(
        pool_size=2,
        selected_count=2,
        block_length=1,
        alpha=10.0,
        seed=42,
        global_step=1,
    )

    first = build_sir_selection_plan(log_probs, response_mask, **kwargs)
    second = build_sir_selection_plan(log_probs, response_mask, **kwargs)

    np.testing.assert_array_equal(first.selected_global_indices, second.selected_global_indices)
    np.testing.assert_array_equal(first.selected_global_indices, [1, 0, 3, 2])
    np.testing.assert_array_equal(first.selected_pool_indices, [1, 0, 1, 0])
    np.testing.assert_array_equal(first.selected_draw_indices, [0, 1, 0, 1])
    assert first.groups[0].selected_draws == ((1,), (0,))
    assert first.groups[0].selected_counts.tolist() == [1, 1]
    for group in first.groups:
        assert len(set(group.selected_local_indices.tolist())) == 2
        assert group.selected_counts.max() == 1


def test_selection_without_replacement_survives_normalized_weight_underflow():
    log_probs = np.asarray([[-1.0], [-1000.0], [-2000.0], [-3000.0]])
    response_mask = np.ones_like(log_probs)

    plan = build_sir_selection_plan(
        log_probs,
        response_mask,
        pool_size=4,
        selected_count=3,
        block_length=1,
        alpha=2.0,
        seed=42,
        global_step=1,
    )

    group = plan.groups[0]
    assert np.count_nonzero(group.weights) == 1
    assert len(set(group.selected_local_indices.tolist())) == 3
    assert group.selected_counts.max() == 1


def test_selection_rejects_invalid_group_sizes():
    log_probs = np.zeros((3, 2))
    response_mask = np.ones((3, 2))

    with pytest.raises(ValueError, match="not divisible"):
        build_sir_selection_plan(
            log_probs,
            response_mask,
            pool_size=2,
            selected_count=2,
            block_length=1,
            alpha=1.5,
            seed=1,
            global_step=1,
        )
