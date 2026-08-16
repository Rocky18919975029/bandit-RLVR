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

"""Summarize joint, vanilla, and exact myopic accuracy with paired bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.joint_tempering.common import alpha_key, read_jsonl, write_json  # noqa: E402
from verl.experimental.joint_tempering.sir import stable_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", default="outputs/joint_tempering/joint_sweep.jsonl")
    parser.add_argument("--myopic", default="outputs/joint_tempering/myopic.jsonl")
    parser.add_argument("--output-prefix", default="outputs/joint_tempering/summary")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


def mean_or_none(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def paired_bootstrap_interval(
    left: np.ndarray,
    right: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if left.size == 0 or samples <= 0:
        return None, None
    if left.shape != right.shape:
        raise ValueError("paired bootstrap arrays must have matching shapes")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, left.size, size=(samples, left.size))
    differences = (left[indices] - right[indices]).mean(axis=1)
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")

    joint_rows = read_jsonl(args.joint)
    if not joint_rows:
        raise ValueError(f"joint sweep is empty: {args.joint}")
    myopic_rows = read_jsonl(args.myopic) if Path(args.myopic).exists() else []

    myopic_accs = defaultdict(list)
    myopic_eos = defaultdict(list)
    for row in myopic_rows:
        key = (row["prompt_id"], int(row["block_length"]), alpha_key(row["alpha"]))
        myopic_accs[key].append(float(bool(row["acc"])))
        myopic_eos[key].append(float(bool(row.get("eos_in_block", False))))

    grouped_joint = defaultdict(list)
    for row in joint_rows:
        key = (int(row["block_length"]), alpha_key(row["alpha"]), int(row["candidate_count"]))
        grouped_joint[key].append(row)

    summaries = []
    for (block_length, alpha_text, candidate_count), rows in sorted(grouped_joint.items()):
        valid_rows = [row for row in rows if row["valid"]]
        records = []
        for row in valid_rows:
            myopic_key = (row["prompt_id"], block_length, alpha_text)
            if myopic_key not in myopic_accs:
                continue
            records.append(
                (
                    float(row["joint_expected_acc"]),
                    float(row["joint_selected_acc"]),
                    float(row["vanilla_expected_acc"]),
                    float(row["vanilla_selected_acc"]),
                    float(np.mean(myopic_accs[myopic_key])),
                    float(row["sir_ess"]),
                    float(row["sir_ess_fraction"]),
                    float(row["sir_max_weight"]),
                    float(np.mean(myopic_eos[myopic_key])),
                )
            )

        values = np.asarray(records, dtype=np.float64)
        if values.size:
            joint_expected = values[:, 0]
            joint_selected = values[:, 1]
            vanilla_expected = values[:, 2]
            vanilla_selected = values[:, 3]
            myopic = values[:, 4]
            ess = values[:, 5]
            ess_fraction = values[:, 6]
            max_weight = values[:, 7]
            eos_rate = values[:, 8]
        else:
            joint_expected = joint_selected = vanilla_expected = vanilla_selected = np.asarray([], dtype=np.float64)
            myopic = ess = ess_fraction = max_weight = eos_rate = np.asarray([], dtype=np.float64)

        config_seed = stable_seed(
            args.bootstrap_seed,
            block_length,
            alpha_text,
            candidate_count,
            "bootstrap",
        )
        joint_myopic_low, joint_myopic_high = paired_bootstrap_interval(
            joint_expected, myopic, args.bootstrap_samples, config_seed
        )
        joint_vanilla_low, joint_vanilla_high = paired_bootstrap_interval(
            joint_expected, vanilla_expected, args.bootstrap_samples, config_seed + 1
        )
        myopic_vanilla_low, myopic_vanilla_high = paired_bootstrap_interval(
            myopic, vanilla_expected, args.bootstrap_samples, config_seed + 2
        )

        summaries.append(
            {
                "block_length": block_length,
                "alpha": float(alpha_text),
                "candidate_count": candidate_count,
                "pool_prompt_count": len(rows),
                "valid_pool_prompt_count": len(valid_rows),
                "compared_prompt_count": int(values.shape[0]) if values.ndim == 2 else 0,
                "short_block_prompt_count": len(rows) - len(valid_rows),
                "joint_expected_acc": mean_or_none(joint_expected),
                "joint_resampled_acc": mean_or_none(joint_selected),
                "vanilla_expected_acc": mean_or_none(vanilla_expected),
                "vanilla_resampled_acc": mean_or_none(vanilla_selected),
                "myopic_acc": mean_or_none(myopic),
                "joint_minus_myopic": mean_or_none(joint_expected - myopic),
                "joint_minus_myopic_ci_low": joint_myopic_low,
                "joint_minus_myopic_ci_high": joint_myopic_high,
                "joint_minus_vanilla": mean_or_none(joint_expected - vanilla_expected),
                "joint_minus_vanilla_ci_low": joint_vanilla_low,
                "joint_minus_vanilla_ci_high": joint_vanilla_high,
                "myopic_minus_vanilla": mean_or_none(myopic - vanilla_expected),
                "myopic_minus_vanilla_ci_low": myopic_vanilla_low,
                "myopic_minus_vanilla_ci_high": myopic_vanilla_high,
                "sir_ess_mean": mean_or_none(ess),
                "sir_ess_median": float(np.median(ess)) if ess.size else None,
                "sir_ess_fraction_mean": mean_or_none(ess_fraction),
                "sir_max_weight_mean": mean_or_none(max_weight),
                "myopic_eos_in_block_rate": mean_or_none(eos_rate),
            }
        )

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_prefix.with_suffix(".json"), summaries)
    csv_path = output_prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {len(summaries)} configurations to {output_prefix.with_suffix('.json')} and {csv_path}")


if __name__ == "__main__":
    main()
