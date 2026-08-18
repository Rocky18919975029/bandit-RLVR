#!/usr/bin/env python3
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
"""Compute exact sample accuracy and unbiased pass@k from VERL validation dumps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def unbiased_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Return the standard unbiased pass@k estimator for one problem."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if not 0 <= num_correct <= num_samples:
        raise ValueError(f"num_correct must be between 0 and num_samples, got {num_correct}/{num_samples}")
    if not 1 <= k <= num_samples:
        raise ValueError(f"k must be between 1 and num_samples, got k={k}, n={num_samples}")
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - math.comb(num_samples - num_correct, k) / math.comb(num_samples, k)


def _is_correct(row: dict[str, Any]) -> bool:
    if "acc" in row and row["acc"] is not None:
        value = row["acc"]
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float) and not isinstance(value, bool):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"Expected acc in [0, 1], got {value!r}")
            return float(value) >= 0.5
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValueError(f"Unsupported acc value: {value!r}")

    if "score" not in row:
        raise ValueError("Validation row contains neither 'acc' nor 'score'")
    return float(row["score"]) > 0.0


def _validation_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Validation dump does not exist: {input_path}")
    files = sorted(input_path.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL validation dumps found in {input_path}")
    return files


def load_validation_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _validation_files(input_path):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}")
                rows.append(row)
    if not rows:
        raise ValueError(f"Validation dump is empty: {input_path}")
    return rows


def summarize_validation(
    rows: list[dict[str, Any]],
    pass_ks: list[int],
    *,
    expected_problems: int | None = None,
    expected_samples_per_problem: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    used_uid_fallback = False
    for row in rows:
        data_source = str(row.get("data_source", "unknown"))
        uid = row.get("uid")
        if uid is None:
            uid = row.get("input")
            used_uid_fallback = True
        if uid is None:
            raise ValueError("Validation row contains neither 'uid' nor 'input'")
        grouped[(data_source, str(uid))].append(row)

    if expected_problems is not None and len(grouped) != expected_problems:
        raise ValueError(f"Expected {expected_problems} problems, found {len(grouped)}")

    sample_counts = {len(group_rows) for group_rows in grouped.values()}
    if expected_samples_per_problem is not None and sample_counts != {expected_samples_per_problem}:
        distribution: dict[int, int] = defaultdict(int)
        for group_rows in grouped.values():
            distribution[len(group_rows)] += 1
        raise ValueError(
            f"Expected {expected_samples_per_problem} samples per problem, found {dict(sorted(distribution.items()))}"
        )

    min_samples = min(sample_counts)
    valid_pass_ks = sorted({k for k in pass_ks if 1 <= k <= min_samples})
    if not valid_pass_ks:
        raise ValueError(f"No requested pass@k is valid for the minimum sample count {min_samples}")
    skipped_pass_ks = sorted({k for k in pass_ks if k < 1 or k > min_samples})

    per_problem: list[dict[str, Any]] = []
    total_correct = 0
    total_samples = 0
    for (data_source, uid), group_rows in sorted(grouped.items()):
        outcomes = [_is_correct(row) for row in group_rows]
        num_samples = len(outcomes)
        num_correct = sum(outcomes)
        total_correct += num_correct
        total_samples += num_samples
        item: dict[str, Any] = {
            "data_source": data_source,
            "uid": uid,
            "num_samples": num_samples,
            "num_correct": num_correct,
            "accuracy": num_correct / num_samples,
            "observed_pass_at_n": float(num_correct > 0),
        }
        for k in valid_pass_ks:
            item[f"pass@{k}"] = unbiased_pass_at_k(num_samples, num_correct, k)
        per_problem.append(item)

    metrics = {
        "accuracy": total_correct / total_samples,
        **{f"pass@{k}": sum(item[f"pass@{k}"] for item in per_problem) / len(per_problem) for k in valid_pass_ks},
    }
    summary: dict[str, Any] = {
        "problem_count": len(per_problem),
        "total_completion_count": total_samples,
        "samples_per_problem": sorted(sample_counts),
        "correct_completion_count": total_correct,
        "metrics": metrics,
        "pass_at_k_estimator": "mean_problem[1 - C(n-c,k) / C(n,k)]",
        "uid_fallback_to_input": used_uid_fallback,
    }
    if skipped_pass_ks:
        summary["skipped_pass_k"] = skipped_pass_ks
    return summary, per_problem


def _parse_pass_ks(value: str) -> list[int]:
    result = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not result:
        raise argparse.ArgumentTypeError("--pass-k must contain at least one integer")
    return result


def write_results(output_dir: Path, summary: dict[str, Any], per_problem: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric, value in summary["metrics"].items():
            writer.writerow([metric, value])

    fieldnames = list(per_problem[0])
    with (output_dir / "per_problem.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_problem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Validation JSONL file or directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pass-k", type=_parse_pass_ks, default=_parse_pass_ks("1,2,4,8,16,32"))
    parser.add_argument("--expected-problems", type=int, default=30)
    parser.add_argument("--expected-samples-per-problem", type=int)
    parser.add_argument("--model-path")
    parser.add_argument("--requested-model-path")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--strict-boxed-verifier", action="store_true")
    args = parser.parse_args()

    rows = load_validation_rows(args.input)
    summary, per_problem = summarize_validation(
        rows,
        args.pass_k,
        expected_problems=args.expected_problems,
        expected_samples_per_problem=args.expected_samples_per_problem,
    )
    summary.update(
        {
            key: value
            for key, value in {
                "model_path": args.model_path,
                "requested_model_path": args.requested_model_path,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "strict_boxed_verifier": args.strict_boxed_verifier,
            }.items()
            if value is not None
        }
    )
    write_results(args.output_dir, summary, per_problem)

    print(f"AIME24 problems: {summary['problem_count']}")
    print(f"Completions: {summary['total_completion_count']}")
    for metric, value in summary["metrics"].items():
        print(f"{metric}: {100.0 * value:.2f}%")
    if summary.get("skipped_pass_k"):
        print(f"Skipped k larger than available samples: {summary['skipped_pass_k']}")
    print(f"Saved summary to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
