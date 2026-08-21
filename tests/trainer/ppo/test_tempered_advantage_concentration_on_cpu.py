# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json

import numpy as np

from examples.grpo_trainer.analyze_tempered_advantage_concentration import analyze_pool, write_results


def _candidate(index: int, log_prob: float, reward: float, length: int):
    return {
        "pool_index": index,
        "sir_pool_origin": "initial",
        "sir_parent_index": index,
        "sampled_token_ids": list(range(length)),
        "sampled_token_log_probs": [log_prob / length] * length,
        "acc": reward > 0,
    }


def test_advantage_concentration_uses_exact_tokens_and_zeroes_homogeneous_groups(tmp_path):
    pool = tmp_path / "pool.jsonl"
    rows = [
        {
            "prompt_id": "mixed",
            "source_row_index": 10,
            "candidates": [
                _candidate(0, -1.0, 1.0, 1),
                _candidate(1, -2.0, -1.0, 2),
                _candidate(2, -3.0, -1.0, 3),
                _candidate(3, -4.0, -1.0, 4),
            ],
        },
        {
            "prompt_id": "all-wrong",
            "source_row_index": 11,
            "candidates": [_candidate(index, -float(index + 1), -1.0, index + 1) for index in range(4)],
        },
    ]
    pool.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary, problems, trajectories = analyze_pool(
        pool,
        initial_count=4,
        beta=1.1,
    )

    assert summary["problem_count"] == 2
    assert summary["trajectory_count"] == 8
    assert summary["mixed_reward_problem_count"] == 1
    assert summary["zero_advantage_problem_count"] == 1
    assert problems[0]["update_active"] is True
    assert problems[1]["update_active"] is False
    assert problems[1]["canonical_advantage_tensor_norm2"] == 0.0
    assert problems[1]["tempered_advantage_tensor_norm2"] == 0.0

    mixed_trajectories = [row for row in trajectories if row["prompt_id"] == "mixed"]
    weights = np.asarray([row["escort_weight"] for row in mixed_trajectories])
    expected_weights = np.exp(0.1 * np.asarray([-1.0, -2.0, -3.0, -4.0]))
    expected_weights /= expected_weights.sum()
    np.testing.assert_allclose(weights, expected_weights)
    assert np.isclose(sum(row["tempered_token_norm2_share"] for row in mixed_trajectories), 1.0)
    assert problems[0]["tempered_token_norm_high_logp_top1_share"] > 0.0

    output_dir = tmp_path / "analysis"
    write_results(output_dir, summary, problems, trajectories)
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "per_problem.csv").is_file()
    assert (output_dir / "per_trajectory.csv").is_file()
