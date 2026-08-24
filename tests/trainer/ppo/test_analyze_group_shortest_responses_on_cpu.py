import json

from examples.grpo_trainer.analyze_group_shortest_responses import build_report, load_groups


def test_group_minimum_excludes_terminal_eos(tmp_path):
    pool = tmp_path / "pool.jsonl"
    row = {
        "prompt_uid": "p0",
        "prompt": "question",
        "ground_truth": "7",
        "candidates": [
            {
                "pool_index": 0,
                "sir_pool_origin": "initial",
                "sir_parent_index": 0,
                "sampled_token_ids": [10, 11, 99],
                "sampled_token_log_probs": [-1.0, -2.0, -0.1],
                "ends_with_eos": True,
                "response": "short",
                "acc": True,
            },
            {
                "pool_index": 1,
                "sir_pool_origin": "initial",
                "sir_parent_index": 1,
                "sampled_token_ids": [20, 21, 22, 23],
                "sampled_token_log_probs": [-1.0, -1.0, -1.0, -1.0],
                "ends_with_eos": False,
                "response": "longer",
                "acc": False,
            },
        ],
    }
    pool.write_text(json.dumps(row) + "\n", encoding="utf-8")

    groups = load_groups(pool, initial_count=2, expected_groups=1)
    assert groups[0]["minimum_non_eos_length"] == 2
    assert groups[0]["shortest"][0]["response"] == "short"

    report = build_report(pool, groups, global_count=1, random_count=1, seed=42, text_limit=100)
    assert "non_eos_tokens=2" in report
    assert "RESPONSE:\nshort" in report
