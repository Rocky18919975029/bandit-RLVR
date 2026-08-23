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
"""Pool repeated AIME evaluations and compare two checkpoints problem-by-problem."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from examples.grpo_trainer.analyze_aime24_validation import _is_correct, unbiased_pass_at_k
else:
    from analyze_aime24_validation import _is_correct, unbiased_pass_at_k


@dataclass(frozen=True)
class RunSpec:
    variant: str
    seed: int
    output_dir: Path


def _parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def _parse_run(value: str) -> RunSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--run must be VARIANT:SEED:OUTPUT_DIR")
    variant, seed, output_dir = parts
    if not variant:
        raise argparse.ArgumentTypeError("--run variant must not be empty")
    return RunSpec(variant=variant, seed=int(seed), output_dir=Path(output_dir))


def _canonical_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _problem_key(row: dict[str, Any]) -> tuple[str, str, str]:
    if row.get("input") is None:
        raise ValueError("raw validation row has no input and cannot be paired across runs")
    return (
        str(row.get("data_source", "unknown")),
        str(row["input"]),
        _canonical_value(row.get("gts")),
    )


def _raw_files(output_dir: Path) -> list[Path]:
    raw_dir = output_dir / "raw"
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no raw validation JSONL files under {raw_dir}")
    return files


def _load_run_outcomes(
    spec: RunSpec,
    *,
    expected_problems: int,
    expected_samples: int,
    expected_temperature: float,
    expected_top_p: float,
) -> tuple[dict[tuple[str, str, str], list[bool]], dict[str, Any]]:
    summary_path = spec.output_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_summary = {
        "problem_count": expected_problems,
        "total_completion_count": expected_problems * expected_samples,
        "samples_per_problem": [expected_samples],
        "seed": spec.seed,
        "temperature": expected_temperature,
        "top_p": expected_top_p,
        "strict_boxed_verifier": True,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"{spec.variant} seed={spec.seed} summary mismatch for {key}: "
                f"expected {expected!r}, got {summary.get(key)!r}"
            )

    grouped: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    raw_count = 0
    for path in _raw_files(spec.output_dir):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"expected JSON object at {path}:{line_number}")
                grouped[_problem_key(row)].append(_is_correct(row))
                raw_count += 1

    if len(grouped) != expected_problems:
        raise ValueError(
            f"{spec.variant} seed={spec.seed} raw dump has {len(grouped)} problems, expected {expected_problems}"
        )
    count_distribution: dict[int, int] = defaultdict(int)
    for outcomes in grouped.values():
        count_distribution[len(outcomes)] += 1
    if count_distribution != {expected_samples: expected_problems}:
        raise ValueError(
            f"{spec.variant} seed={spec.seed} raw samples/problem mismatch: {dict(sorted(count_distribution.items()))}"
        )
    if raw_count != expected_problems * expected_samples:
        raise ValueError(f"{spec.variant} seed={spec.seed} raw row count mismatch: {raw_count}")
    correct_count = sum(sum(outcomes) for outcomes in grouped.values())
    if correct_count != int(summary["correct_completion_count"]):
        raise ValueError(
            f"{spec.variant} seed={spec.seed} raw/summary correct-count mismatch: "
            f"{correct_count} != {summary['correct_completion_count']}"
        )
    return dict(grouped), summary


def _load_jobs_file(path: Path) -> list[RunSpec]:
    if not path.is_file():
        raise FileNotFoundError(f"jobs file does not exist: {path}")
    specs = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            specs.append(
                RunSpec(
                    variant=str(row["variant"]),
                    seed=int(row["seed"]),
                    output_dir=Path(row["output_dir"]),
                )
            )
    return specs


def _paired_bootstrap(
    differences: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if differences.ndim != 1 or differences.size < 2:
        raise ValueError("paired bootstrap requires a one-dimensional array with at least two problems")
    if samples <= 0:
        raise ValueError(f"bootstrap samples must be positive, got {samples}")
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    chunk_size = 10_000
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(0, differences.size, size=(stop - start, differences.size))
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    positive_probability = np.mean(bootstrap_means > 0.0)
    return float(low), float(high), float(positive_probability)


def analyze_runs(
    specs: list[RunSpec],
    *,
    variants: tuple[str, str],
    expected_seeds: list[int],
    expected_problems: int,
    expected_samples_per_run: int,
    pass_ks: list[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
    expected_temperature: float = 1.0,
    expected_top_p: float = 1.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    expected_keys = {(variant, seed) for variant in variants for seed in expected_seeds}
    spec_by_key = {(spec.variant, spec.seed): spec for spec in specs}
    if len(spec_by_key) != len(specs):
        raise ValueError("duplicate variant/seed entries were provided")
    if set(spec_by_key) != expected_keys:
        missing = sorted(expected_keys - set(spec_by_key))
        unexpected = sorted(set(spec_by_key) - expected_keys)
        raise ValueError(f"run matrix mismatch; missing={missing}, unexpected={unexpected}")

    run_outcomes = {}
    run_summaries = {}
    reference_keys = None
    for key in sorted(spec_by_key):
        outcomes, run_summary = _load_run_outcomes(
            spec_by_key[key],
            expected_problems=expected_problems,
            expected_samples=expected_samples_per_run,
            expected_temperature=expected_temperature,
            expected_top_p=expected_top_p,
        )
        current_keys = set(outcomes)
        if reference_keys is None:
            reference_keys = current_keys
        elif current_keys != reference_keys:
            missing = sorted(reference_keys - current_keys)
            unexpected = sorted(current_keys - reference_keys)
            raise ValueError(f"problem set mismatch for {key}; missing={len(missing)}, unexpected={len(unexpected)}")
        run_outcomes[key] = outcomes
        run_summaries[key] = run_summary

    assert reference_keys is not None
    problem_keys = sorted(reference_keys)
    pooled_samples = expected_samples_per_run * len(expected_seeds)
    valid_pass_ks = sorted({k for k in pass_ks if 1 <= k <= pooled_samples})
    if not valid_pass_ks:
        raise ValueError(f"no requested pass@k is valid for pooled sample count {pooled_samples}")

    per_seed_rows = []
    for variant in variants:
        for seed in expected_seeds:
            metrics = run_summaries[(variant, seed)]["metrics"]
            per_seed_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "correct_completions": int(run_summaries[(variant, seed)]["correct_completion_count"]),
                    "total_completions": expected_problems * expected_samples_per_run,
                    **{metric: float(value) for metric, value in metrics.items()},
                }
            )

    per_problem_rows = []
    variant_metric_values: dict[str, dict[str, list[float]]] = {variant: defaultdict(list) for variant in variants}
    first_variant, second_variant = variants
    for problem_index, problem_key in enumerate(problem_keys):
        data_source, input_text, ground_truth = problem_key
        counts = {}
        problem_metrics = {}
        for variant in variants:
            pooled_outcomes = [
                outcome for seed in expected_seeds for outcome in run_outcomes[(variant, seed)][problem_key]
            ]
            if len(pooled_outcomes) != pooled_samples:
                raise AssertionError("internal pooled sample-count mismatch")
            correct = int(sum(pooled_outcomes))
            counts[variant] = correct
            metrics = {"accuracy": correct / pooled_samples}
            metrics.update({f"pass@{k}": unbiased_pass_at_k(pooled_samples, correct, k) for k in valid_pass_ks})
            problem_metrics[variant] = metrics
            for metric, value in metrics.items():
                variant_metric_values[variant][metric].append(value)

        accuracy_difference = problem_metrics[second_variant]["accuracy"] - problem_metrics[first_variant]["accuracy"]
        classification = "improved" if accuracy_difference > 0 else "regressed" if accuracy_difference < 0 else "tied"
        problem_id = hashlib.sha256("\x1f".join(problem_key).encode("utf-8")).hexdigest()[:16]
        row = {
            "problem_index": problem_index,
            "problem_id": problem_id,
            "data_source": data_source,
            "input": input_text,
            "ground_truth": ground_truth,
            "pooled_samples_per_variant": pooled_samples,
            f"{first_variant}_correct": counts[first_variant],
            f"{second_variant}_correct": counts[second_variant],
            f"{first_variant}_accuracy": problem_metrics[first_variant]["accuracy"],
            f"{second_variant}_accuracy": problem_metrics[second_variant]["accuracy"],
            f"{second_variant}_minus_{first_variant}_accuracy": accuracy_difference,
            "classification": classification,
        }
        for k in valid_pass_ks:
            metric = f"pass@{k}"
            row[f"{first_variant}_{metric}"] = problem_metrics[first_variant][metric]
            row[f"{second_variant}_{metric}"] = problem_metrics[second_variant][metric]
            row[f"{second_variant}_minus_{first_variant}_{metric}"] = (
                problem_metrics[second_variant][metric] - problem_metrics[first_variant][metric]
            )
        per_problem_rows.append(row)

    metric_names = ["accuracy", *(f"pass@{k}" for k in valid_pass_ks)]
    metric_rows = []
    for metric_index, metric in enumerate(metric_names):
        first_values = np.asarray(variant_metric_values[first_variant][metric], dtype=np.float64)
        second_values = np.asarray(variant_metric_values[second_variant][metric], dtype=np.float64)
        differences = second_values - first_values
        ci_low, ci_high, positive_probability = _paired_bootstrap(
            differences,
            samples=bootstrap_samples,
            seed=bootstrap_seed + metric_index,
        )
        metric_rows.append(
            {
                "metric": metric,
                first_variant: float(first_values.mean()),
                second_variant: float(second_values.mean()),
                f"{second_variant}_minus_{first_variant}": float(differences.mean()),
                "paired_bootstrap_ci_low": ci_low,
                "paired_bootstrap_ci_high": ci_high,
                "paired_bootstrap_positive_probability": positive_probability,
            }
        )

    first_counts = np.asarray([row[f"{first_variant}_correct"] for row in per_problem_rows])
    second_counts = np.asarray([row[f"{second_variant}_correct"] for row in per_problem_rows])
    summary = {
        "variants": list(variants),
        "seeds": expected_seeds,
        "problem_count": expected_problems,
        "samples_per_problem_per_seed": expected_samples_per_run,
        "pooled_samples_per_problem_per_variant": pooled_samples,
        "total_completions_per_variant": expected_problems * pooled_samples,
        "temperature": expected_temperature,
        "top_p": expected_top_p,
        "strict_boxed_verifier": True,
        "pass_at_k_estimator": "mean_problem[1 - C(n-c,k) / C(n,k)]",
        "paired_bootstrap": {
            "unit": "AIME problem",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "conditional_on_observed_completions": True,
        },
        "problem_correct_count_comparison": {
            f"{second_variant}_greater": int(np.sum(second_counts > first_counts)),
            f"{first_variant}_greater": int(np.sum(first_counts > second_counts)),
            "tied": int(np.sum(first_counts == second_counts)),
            f"{first_variant}_zero_{second_variant}_positive": int(np.sum((first_counts == 0) & (second_counts > 0))),
            f"{second_variant}_zero_{first_variant}_positive": int(np.sum((second_counts == 0) & (first_counts > 0))),
            "both_zero": int(np.sum((first_counts == 0) & (second_counts == 0))),
            "both_positive": int(np.sum((first_counts > 0) & (second_counts > 0))),
        },
        "metrics": {row["metric"]: row for row in metric_rows},
        "runs": [
            {
                "variant": spec.variant,
                "seed": spec.seed,
                "output_dir": str(spec.output_dir),
                "model_path": run_summaries[(spec.variant, spec.seed)].get("model_path"),
            }
            for spec in sorted(specs, key=lambda item: (item.variant, item.seed))
        ],
    }
    return summary, per_problem_rows, per_seed_rows


def write_results(
    output_dir: Path,
    summary: dict[str, Any],
    per_problem_rows: list[dict[str, Any]],
    per_seed_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metric_rows = list(summary["metrics"].values())
    for filename, rows in (
        ("summary.csv", metric_rows),
        ("per_problem.csv", per_problem_rows),
        ("per_seed.csv", per_seed_rows),
    ):
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _short_input(value: str, limit: int = 120) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-file", type=Path, action="append", default=[])
    parser.add_argument("--run", type=_parse_run, action="append", default=[])
    parser.add_argument("--variants", default="control,tempered")
    parser.add_argument("--expected-seeds", type=_parse_int_list, default=[42, 43, 44, 45])
    parser.add_argument("--expected-problems", type=int, default=30)
    parser.add_argument("--expected-samples-per-run", type=int, default=32)
    parser.add_argument("--expected-temperature", type=float, default=1.0)
    parser.add_argument("--expected-top-p", type=float, default=1.0)
    parser.add_argument("--pass-k", type=_parse_int_list, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    variants = tuple(part.strip() for part in args.variants.split(",") if part.strip())
    if len(variants) != 2 or variants[0] == variants[1]:
        raise ValueError("--variants must name exactly two distinct variants")
    if args.expected_problems <= 1:
        raise ValueError("--expected-problems must be greater than one for paired bootstrap")
    if args.expected_samples_per_run <= 0:
        raise ValueError("--expected-samples-per-run must be positive")

    specs = list(args.run)
    for jobs_file in args.jobs_file:
        specs.extend(_load_jobs_file(jobs_file))
    summary, per_problem_rows, per_seed_rows = analyze_runs(
        specs,
        variants=variants,
        expected_seeds=args.expected_seeds,
        expected_problems=args.expected_problems,
        expected_samples_per_run=args.expected_samples_per_run,
        pass_ks=args.pass_k,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        expected_temperature=args.expected_temperature,
        expected_top_p=args.expected_top_p,
    )
    write_results(args.output_dir, summary, per_problem_rows, per_seed_rows)

    first_variant, second_variant = variants
    print("Configuration audit: PASS")
    print(
        f"Pooled {len(args.expected_seeds)} seeds x {args.expected_samples_per_run} samples "
        f"= {summary['pooled_samples_per_problem_per_variant']} samples/problem/variant"
    )
    print(f"Problems: {summary['problem_count']}")
    print("\nPooled metrics and paired problem-bootstrap intervals:")
    for row in summary["metrics"].values():
        print(
            f"  {row['metric']:8s}: {first_variant}={100 * row[first_variant]:.2f}% "
            f"{second_variant}={100 * row[second_variant]:.2f}% "
            f"delta={100 * row[f'{second_variant}_minus_{first_variant}']:+.2f} pp "
            f"95% CI=[{100 * row['paired_bootstrap_ci_low']:+.2f}, "
            f"{100 * row['paired_bootstrap_ci_high']:+.2f}] pp "
            f"P(delta>0)={row['paired_bootstrap_positive_probability']:.3f}"
        )

    comparison = summary["problem_correct_count_comparison"]
    print("\nPer-problem correct-count comparison:")
    for key, value in comparison.items():
        print(f"  {key}: {value}")

    ranked = sorted(
        per_problem_rows,
        key=lambda row: row[f"{second_variant}_minus_{first_variant}_accuracy"],
        reverse=True,
    )
    pooled_samples = summary["pooled_samples_per_problem_per_variant"]
    print(f"\nLargest {second_variant} improvements:")
    for row in ranked[:5]:
        print(
            f"  {row['problem_id']} {row[f'{first_variant}_correct']} -> "
            f"{row[f'{second_variant}_correct']}/{pooled_samples}: {_short_input(row['input'])}"
        )
    print(f"\nLargest {second_variant} regressions:")
    for row in reversed(ranked[-5:]):
        print(
            f"  {row['problem_id']} {row[f'{first_variant}_correct']} -> "
            f"{row[f'{second_variant}_correct']}/{pooled_samples}: {_short_input(row['input'])}"
        )
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    main()
