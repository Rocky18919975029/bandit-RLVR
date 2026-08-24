# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json

import numpy as np
import pytest

from examples.grpo_trainer.analyze_tempered_initial_pool import load_initial_groups


def _candidate(index, token_log_probs, *, ends_with_eos, correct):
    token_ids = [100 + index] * len(token_log_probs)
    if ends_with_eos:
        token_ids[-1] = 99
    return {
        "pool_index": index,
        "sir_pool_origin": "initial",
        "sir_parent_index": index,
        "sampled_token_ids": token_ids,
        "sampled_token_log_probs": token_log_probs,
        "ends_with_eos": ends_with_eos,
        "acc": correct,
    }


def test_pool_scan_uses_group_shortest_non_eos_prefix(tmp_path):
    pool = tmp_path / "pool.jsonl"
    row = {
        "candidates": [
            _candidate(0, [-1.0, -1000.0], ends_with_eos=True, correct=True),
            _candidate(1, [-2.0, -20.0, -2000.0], ends_with_eos=True, correct=False),
        ]
    }
    pool.write_text(json.dumps(row) + "\n", encoding="utf-8")

    joint_log_probs, rewards, metadata = load_initial_groups(pool, initial_count=2)

    # Non-EOS lengths are one and two, so both candidates use only token 1.
    np.testing.assert_array_equal(joint_log_probs, np.asarray([[-1.0, -2.0]]))
    np.testing.assert_array_equal(rewards, np.asarray([[1.0, -1.0]]))
    assert metadata["joint_log_prob_horizon"] == "group_shortest_non_eos_prefix"
    assert metadata["common_horizon_min"] == 1
    assert metadata["common_horizon_median"] == 1.0
    assert metadata["eos_fraction"] == 1.0


def test_pool_scan_fails_closed_without_eos_metadata(tmp_path):
    pool = tmp_path / "pool.jsonl"
    candidates = [
        _candidate(0, [-1.0, -10.0], ends_with_eos=True, correct=True),
        _candidate(1, [-2.0, -20.0], ends_with_eos=True, correct=False),
    ]
    del candidates[0]["ends_with_eos"]
    pool.write_text(json.dumps({"candidates": candidates}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing ends_with_eos"):
        load_initial_groups(pool, initial_count=2)
