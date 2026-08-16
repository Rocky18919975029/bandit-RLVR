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

"""Reuse a complete trajectory pool to sweep joint-tempering configurations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.joint_tempering.common import alpha_key, read_jsonl, write_json, write_jsonl  # noqa: E402
from verl.experimental.joint_tempering.sir import (  # noqa: E402
    effective_sample_size,
    prefix_joint_log_probs,
    resample_index,
    sir_weights,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="outputs/joint_tempering/pool.jsonl")
    parser.add_argument("--output", default="outputs/joint_tempering/joint_sweep.jsonl")
    parser.add_argument("--block-lengths", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.25, 1.5, 2.0])
    parser.add_argument("--candidate-counts", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--resample-seed", type=int, default=2026)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(block_length <= 0 for block_length in args.block_lengths):
        raise ValueError("all --block-lengths must be positive")
    if any(not np.isfinite(alpha) or alpha <= 0 for alpha in args.alphas):
        raise ValueError("all --alphas must be finite and positive")
    if any(candidate_count <= 0 for candidate_count in args.candidate_counts):
        raise ValueError("all --candidate-counts must be positive")
    if len(set(args.block_lengths)) != len(args.block_lengths):
        raise ValueError("--block-lengths contains duplicates")
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("--alphas contains duplicates")
    if len(set(args.candidate_counts)) != len(args.candidate_counts):
        raise ValueError("--candidate-counts contains duplicates")


def evaluate_configuration(
    pool_row: dict,
    block_length: int,
    alpha: float,
    candidate_count: int,
    resample_seed: int,
) -> dict:
    all_candidates = pool_row["candidates"]
    if candidate_count > len(all_candidates):
        raise ValueError(
            f"prompt {pool_row['prompt_id']} has {len(all_candidates)} candidates, "
            f"fewer than candidate_count={candidate_count}"
        )
    candidates = all_candidates[:candidate_count]
    token_counts = [len(candidate["token_ids"]) for candidate in candidates]
    short_candidate_count = sum(token_count < block_length for token_count in token_counts)
    base = {
        "prompt_id": pool_row["prompt_id"],
        "source_row_index": pool_row["source_row_index"],
        "block_length": block_length,
        "alpha": float(alpha),
        "candidate_count": candidate_count,
        "short_candidate_count": short_candidate_count,
        "min_candidate_tokens": min(token_counts),
    }
    if short_candidate_count:
        return {**base, "valid": False, "invalid_reason": "candidate_shorter_than_block"}

    joint_log_probs = prefix_joint_log_probs(
        [candidate["token_log_probs"] for candidate in candidates], block_length=block_length
    )
    weights = sir_weights(joint_log_probs, alpha=alpha)
    accuracies = np.asarray([float(bool(candidate["acc"])) for candidate in candidates], dtype=np.float64)
    scores = np.asarray([float(candidate["score"]) for candidate in candidates], dtype=np.float64)

    joint_seed = stable_seed(
        resample_seed,
        pool_row["prompt_id"],
        block_length,
        alpha_key(alpha),
        candidate_count,
        "joint",
    )
    vanilla_seed = stable_seed(resample_seed, pool_row["prompt_id"], candidate_count, "vanilla")
    joint_selected_index = resample_index(weights, joint_seed)
    vanilla_selected_index = resample_index(np.ones(candidate_count), vanilla_seed)

    return {
        **base,
        "valid": True,
        "prefix_joint_log_probs": joint_log_probs.tolist(),
        "sir_weights": weights.tolist(),
        "sir_ess": effective_sample_size(weights),
        "sir_ess_fraction": effective_sample_size(weights) / candidate_count,
        "sir_max_weight": float(weights.max()),
        "sir_logit_range": float(((alpha - 1.0) * joint_log_probs).max() - ((alpha - 1.0) * joint_log_probs).min()),
        "joint_expected_acc": float(np.dot(weights, accuracies)),
        "joint_expected_score": float(np.dot(weights, scores)),
        "vanilla_expected_acc": float(accuracies.mean()),
        "vanilla_expected_score": float(scores.mean()),
        "joint_selected_index": joint_selected_index,
        "joint_selected_acc": bool(candidates[joint_selected_index]["acc"]),
        "joint_selected_score": float(candidates[joint_selected_index]["score"]),
        "vanilla_selected_index": vanilla_selected_index,
        "vanilla_selected_acc": bool(candidates[vanilla_selected_index]["acc"]),
        "vanilla_selected_score": float(candidates[vanilla_selected_index]["score"]),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    pool_rows = read_jsonl(args.pool)
    if not pool_rows:
        raise ValueError(f"pool is empty: {args.pool}")

    output_rows = []
    for pool_row in pool_rows:
        for candidate_count in args.candidate_counts:
            for block_length in args.block_lengths:
                for alpha in args.alphas:
                    output_rows.append(
                        evaluate_configuration(
                            pool_row=pool_row,
                            block_length=block_length,
                            alpha=alpha,
                            candidate_count=candidate_count,
                            resample_seed=args.resample_seed,
                        )
                    )

    write_jsonl(args.output, output_rows)
    manifest = {
        "pool": args.pool,
        "output": args.output,
        "pool_problem_count": len(pool_rows),
        "block_lengths": args.block_lengths,
        "alphas": args.alphas,
        "candidate_counts": args.candidate_counts,
        "resample_seed": args.resample_seed,
        "result_row_count": len(output_rows),
        "valid_row_count": sum(bool(row["valid"]) for row in output_rows),
    }
    write_json(str(args.output) + ".manifest.json", manifest)
    print(
        f"Wrote {manifest['valid_row_count']}/{manifest['result_row_count']} valid sweep rows to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
