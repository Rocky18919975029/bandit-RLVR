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

import json

import numpy as np
import pytest

from verl.trainer.ppo.sir_resampling import (
    build_branched_prefix_plan,
    build_sir_selection_plan,
    load_initial_rollout_replay,
    tempered_sir_weights,
)


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


def test_branched_prefix_plan_expands_k_initial_trajectories_to_n():
    kwargs = dict(
        pool_size=8,
        initial_count=2,
        block_length=6,
        seed=42,
        global_step=3,
    )

    first = build_branched_prefix_plan(np.asarray([10, 8, 9, 7]), **kwargs)
    second = build_branched_prefix_plan(np.asarray([10, 8, 9, 7]), **kwargs)

    assert first.branches_per_initial == 3
    assert len(first.cut_positions) == 12
    assert not first.cut_with_replacement.any()
    np.testing.assert_array_equal(first.cut_positions, second.cut_positions)
    np.testing.assert_array_equal(first.parent_global_indices, np.repeat(np.arange(4), 3))
    np.testing.assert_array_equal(first.parent_local_indices, np.tile(np.repeat(np.arange(2), 3), 2))
    np.testing.assert_array_equal(first.branch_local_indices, np.tile(np.arange(3), 4))
    for parent_index in range(4):
        parent_cuts = first.cut_positions[first.parent_global_indices == parent_index]
        assert len(set(parent_cuts.tolist())) == 3
        assert np.all((1 <= parent_cuts) & (parent_cuts <= 6))


def test_branched_prefix_plan_reuses_cuts_for_short_initial_trajectory():
    plan = build_branched_prefix_plan(
        np.asarray([2, 1]),
        pool_size=8,
        initial_count=2,
        block_length=6,
        seed=42,
        global_step=1,
    )

    np.testing.assert_array_equal(plan.cut_positions[:3], [1, 1, 1])
    np.testing.assert_array_equal(plan.cut_positions[3:], [0, 0, 0])
    assert plan.cut_with_replacement.all()


def test_load_initial_rollout_replay_selects_exact_initial_k(tmp_path):
    replay_path = tmp_path / "1.jsonl"
    rows = []
    for prompt_index in range(2):
        candidates = []
        for parent_index in range(2):
            token_ids = [10 + prompt_index, 20 + parent_index]
            candidates.append(
                {
                    "pool_index": parent_index,
                    "sir_pool_origin": "initial",
                    "sir_parent_index": parent_index,
                    "sampled_token_ids": token_ids,
                    "response_token_ids": token_ids,
                    "sampled_token_log_probs": [-0.1, -0.2],
                }
            )
        candidates.append(
            {
                "pool_index": 2,
                "sir_pool_origin": "branch",
                "sir_parent_index": 0,
                "sampled_token_ids": [99],
                "response_token_ids": [99],
                "sampled_token_log_probs": [-9.0],
            }
        )
        rows.append(
            {
                "step": 1,
                "pool_mode": "branched_prefix",
                "selected_count": 2,
                "prompt": f"prompt-{prompt_index}",
                "ground_truth": str(prompt_index),
                "data_source": "math_dapo",
                "candidates": candidates,
            }
        )
    replay_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    replay = load_initial_rollout_replay(
        replay_path,
        expected_prompts=["prompt-0", "prompt-1"],
        expected_ground_truths=["0", "1"],
        expected_data_sources=["math_dapo", "math_dapo"],
        initial_count=2,
    )

    assert replay.prompt_count == 2
    assert replay.initial_count == 2
    assert replay.response_token_ids == ((10, 20), (10, 21), (11, 20), (11, 21))
    assert replay.response_log_probs == ((-0.1, -0.2),) * 4
    np.testing.assert_array_equal(replay.source_pool_indices, [0, 1, 0, 1])


def test_load_initial_rollout_replay_rejects_prompt_mismatch(tmp_path):
    replay_path = tmp_path / "1.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "step": 1,
                "pool_mode": "branched_prefix",
                "selected_count": 2,
                "prompt": "wrong",
                "ground_truth": "0",
                "data_source": "math_dapo",
                "candidates": [],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="prompt mismatch"):
        load_initial_rollout_replay(
            replay_path,
            expected_prompts=["expected"],
            expected_ground_truths=["0"],
            expected_data_sources=["math_dapo"],
            initial_count=2,
        )
