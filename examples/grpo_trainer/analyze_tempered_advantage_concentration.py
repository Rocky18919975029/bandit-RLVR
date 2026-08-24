#!/usr/bin/env python3
"""Measure whether tempered GRPO advantage norm concentrates on high-probability rollouts.

The input must be a saved branched-SIR pool containing exact chosen-token
log-probabilities.  Only ``sir_pool_origin=initial`` trajectories are used, so
canonical and tempered GRPO are compared on exactly the same rollout groups.

For scalar response advantage ``A_i`` and sampled-token count ``T_i``, the
response's exact share of the stored advantage tensor's squared L2 norm is

    A_i^2 T_i / sum_j A_j^2 T_j.

This is an advantage-tensor diagnostic, not a parameter-gradient norm: score
function gradients can differ across responses even when their advantages are
equal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from examples.grpo_trainer.analyze_tempered_initial_pool import (
        _candidate_reward,
        _group_shortest_non_eos_joint_log_probs,
    )
else:
    from analyze_tempered_initial_pool import _candidate_reward, _group_shortest_non_eos_joint_log_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-count", type=int, default=16)
    parser.add_argument("--tempering-beta", type=float, default=1.0033251003215344)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    return parser.parse_args()


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    unnormalized = np.exp(shifted)
    return unnormalized / np.sum(unnormalized)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return math.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _shares(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= 0:
        return np.full(values.shape, math.nan, dtype=np.float64)
    return values / total


def _effective_count(shares: np.ndarray) -> float:
    if not np.all(np.isfinite(shares)):
        return math.nan
    return float(1.0 / np.sum(np.square(shares)))


def _top_share(shares: np.ndarray, descending_log_prob_order: np.ndarray, count: int) -> float:
    if not np.all(np.isfinite(shares)):
        return math.nan
    return float(np.sum(shares[descending_log_prob_order[:count]]))


def _canonical_advantages(rewards: np.ndarray, epsilon: float) -> np.ndarray:
    centered = rewards - np.mean(rewards)
    sample_std = float(np.std(rewards, ddof=1))
    return centered / (sample_std + epsilon)


def _tempered_advantages(
    rewards: np.ndarray,
    joint_log_probs: np.ndarray,
    beta: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    weights = _softmax((beta - 1.0) * joint_log_probs)
    weighted_mean = float(np.sum(weights * rewards))
    centered = rewards - weighted_mean
    sum_weight_squared = float(np.sum(np.square(weights)))
    ess = 1.0 / sum_weight_squared
    if np.all(rewards == rewards[0]):
        return np.zeros_like(rewards), weights, ess
    denominator = 1.0 - sum_weight_squared
    if denominator <= np.finfo(np.float64).eps:
        scale = 0.0
    else:
        scale = math.sqrt(max(float(np.sum(weights * np.square(centered))) / denominator, 0.0))
    advantages = len(rewards) * weights * centered / (scale + epsilon)
    return advantages, weights, ess


def _method_metrics(
    advantages: np.ndarray,
    token_counts: np.ndarray,
    joint_log_probs: np.ndarray,
    order: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    coefficient_norm2 = np.square(advantages)
    token_norm2 = coefficient_norm2 * token_counts
    coefficient_shares = _shares(coefficient_norm2)
    token_shares = _shares(token_norm2)
    metrics = {
        "scalar_advantage_l2": float(np.linalg.norm(advantages)),
        "advantage_tensor_norm2": float(np.sum(token_norm2)),
        "coefficient_effective_trajectories": _effective_count(coefficient_shares),
        "token_norm_effective_trajectories": _effective_count(token_shares),
        "coefficient_high_logp_top1_share": _top_share(coefficient_shares, order, 1),
        "coefficient_high_logp_top2_share": _top_share(coefficient_shares, order, 2),
        "coefficient_high_logp_top4_share": _top_share(coefficient_shares, order, 4),
        "token_norm_high_logp_top1_share": _top_share(token_shares, order, 1),
        "token_norm_high_logp_top2_share": _top_share(token_shares, order, 2),
        "token_norm_high_logp_top4_share": _top_share(token_shares, order, 4),
        "token_norm_max_trajectory_share": (
            float(np.nanmax(token_shares)) if np.any(np.isfinite(token_shares)) else math.nan
        ),
        "joint_logp_token_norm_spearman": _spearman(joint_log_probs, token_norm2),
    }
    return metrics, coefficient_shares, token_shares


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(item) for item in value]
    return value


def _distribution(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "p05": None, "median": None, "p95": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _batch_problem_concentration(rows: list[dict[str, Any]], method: str) -> dict[str, float]:
    values = np.asarray([row[f"{method}_advantage_tensor_norm2"] for row in rows], dtype=np.float64)
    shares = values / np.sum(values)
    descending = np.sort(shares)[::-1]
    active_count = int(np.sum(values > 0))

    def top_fraction(fraction: float) -> float:
        count = max(1, math.ceil(active_count * fraction))
        return float(np.sum(descending[:count]))

    return {
        "active_problem_count": active_count,
        "effective_problem_count": float(1.0 / np.sum(np.square(shares))),
        "largest_problem_share": float(descending[0]),
        "top_1pct_problem_share": top_fraction(0.01),
        "top_5pct_problem_share": top_fraction(0.05),
        "top_10pct_problem_share": top_fraction(0.10),
    }


def analyze_pool(
    pool_path: Path,
    *,
    initial_count: int,
    beta: float,
    epsilon: float = 1e-6,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if initial_count < 2:
        raise ValueError("initial_count must be at least 2")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("tempering_beta must be finite and positive")

    problem_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []

    with pool_path.open(encoding="utf-8") as source:
        for problem_index, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            initial = [
                candidate for candidate in row.get("candidates", []) if candidate.get("sir_pool_origin") == "initial"
            ]
            initial.sort(key=lambda candidate: int(candidate.get("sir_parent_index", -1)))
            if len(initial) != initial_count:
                raise ValueError(
                    f"problem {problem_index} has {len(initial)} initial trajectories; expected {initial_count}"
                )

            token_counts = np.asarray([len(candidate.get("sampled_token_ids", [])) for candidate in initial])
            if np.any(token_counts <= 0):
                raise ValueError(f"problem {problem_index} contains an empty response")
            joint_log_probs, common_horizon, _ = _group_shortest_non_eos_joint_log_probs(
                initial,
                context=f"problem {problem_index}",
            )
            rewards = np.asarray([_candidate_reward(candidate) for candidate in initial], dtype=np.float64)

            canonical_advantages = _canonical_advantages(rewards, epsilon)
            tempered_advantages, weights, ess = _tempered_advantages(rewards, joint_log_probs, beta, epsilon)
            order = np.argsort(-joint_log_probs, kind="stable")
            joint_logp_ranks = np.empty(initial_count, dtype=np.int64)
            joint_logp_ranks[order] = np.arange(1, initial_count + 1)

            canonical_metrics, canonical_coefficient_shares, canonical_token_shares = _method_metrics(
                canonical_advantages, token_counts, joint_log_probs, order
            )
            tempered_metrics, tempered_coefficient_shares, tempered_token_shares = _method_metrics(
                tempered_advantages, token_counts, joint_log_probs, order
            )
            active = not np.all(rewards == rewards[0])
            prompt_id = row.get("prompt_id", f"problem-{problem_index}")
            source_row_index = row.get("source_row_index")
            problem_result: dict[str, Any] = {
                "problem_index": problem_index,
                "prompt_id": prompt_id,
                "source_row_index": source_row_index,
                "trajectory_count": initial_count,
                "correct_count": int(np.sum(rewards > 0)),
                "update_active": active,
                "token_count_mean": float(np.mean(token_counts)),
                "token_count_min": int(np.min(token_counts)),
                "token_count_max": int(np.max(token_counts)),
                "common_non_eos_horizon": int(common_horizon),
                "joint_log_prob_max": float(np.max(joint_log_probs)),
                "joint_log_prob_median": float(np.median(joint_log_probs)),
                "escort_ess": float(ess),
                "escort_max_weight": float(np.max(weights)),
                "tempered_expected_reward": float(np.sum(weights * rewards)),
            }
            for method, metrics in (("canonical", canonical_metrics), ("tempered", tempered_metrics)):
                problem_result.update({f"{method}_{key}": value for key, value in metrics.items()})
            for key in (
                "coefficient_high_logp_top1_share",
                "coefficient_high_logp_top2_share",
                "coefficient_high_logp_top4_share",
                "token_norm_high_logp_top1_share",
                "token_norm_high_logp_top2_share",
                "token_norm_high_logp_top4_share",
                "token_norm_effective_trajectories",
                "advantage_tensor_norm2",
            ):
                problem_result[f"tempered_minus_canonical_{key}"] = tempered_metrics[key] - canonical_metrics[key]
            problem_rows.append(problem_result)

            for trajectory_index, candidate in enumerate(initial):
                trajectory_rows.append(
                    {
                        "problem_index": problem_index,
                        "prompt_id": prompt_id,
                        "source_row_index": source_row_index,
                        "trajectory_index": trajectory_index,
                        "pool_index": candidate.get("pool_index"),
                        "joint_logp_rank": int(joint_logp_ranks[trajectory_index]),
                        "joint_log_prob": float(joint_log_probs[trajectory_index]),
                        "joint_log_prob_horizon": int(common_horizon),
                        "token_count": int(token_counts[trajectory_index]),
                        "reward": float(rewards[trajectory_index]),
                        "escort_weight": float(weights[trajectory_index]),
                        "canonical_advantage": float(canonical_advantages[trajectory_index]),
                        "tempered_advantage": float(tempered_advantages[trajectory_index]),
                        "canonical_coefficient_norm2_share": float(canonical_coefficient_shares[trajectory_index]),
                        "tempered_coefficient_norm2_share": float(tempered_coefficient_shares[trajectory_index]),
                        "canonical_token_norm2_share": float(canonical_token_shares[trajectory_index]),
                        "tempered_token_norm2_share": float(tempered_token_shares[trajectory_index]),
                    }
                )

    if not problem_rows:
        raise ValueError(f"no problems found in {pool_path}")
    active_rows = [row for row in problem_rows if row["update_active"]]
    zero_rows = [row for row in problem_rows if not row["update_active"]]
    if not active_rows:
        raise ValueError("pool has no mixed-reward groups and therefore no nonzero GRPO advantages")

    distribution_keys = [
        f"{method}_{metric}"
        for method in ("canonical", "tempered")
        for metric in (
            "token_norm_high_logp_top1_share",
            "token_norm_high_logp_top2_share",
            "token_norm_high_logp_top4_share",
            "token_norm_max_trajectory_share",
            "token_norm_effective_trajectories",
            "joint_logp_token_norm_spearman",
        )
    ]
    concentration_distributions = {key: _distribution(active_rows, key) for key in distribution_keys}
    delta_key = "tempered_minus_canonical_token_norm_high_logp_top4_share"
    delta_values = np.asarray([row[delta_key] for row in active_rows], dtype=np.float64)
    summary = {
        "pool": str(pool_path.resolve()),
        "tempering_beta": beta,
        "problem_count": len(problem_rows),
        "trajectory_count": len(trajectory_rows),
        "trajectories_per_problem": initial_count,
        "mixed_reward_problem_count": len(active_rows),
        "zero_advantage_problem_count": len(zero_rows),
        "zero_advantage_problem_fraction": len(zero_rows) / len(problem_rows),
        "joint_log_prob_horizon": "group_shortest_non_eos_prefix",
        "common_horizon_distribution": _distribution(problem_rows, "common_non_eos_horizon"),
        "analysis_scope": (
            "Exact advantage-tensor L2 concentration; this is not parameter-gradient norm. "
            "Only mixed-reward groups have nonzero GRPO updates."
        ),
        "concentration_distributions_over_mixed_groups": concentration_distributions,
        "high_logp_top4_share_delta": {
            "mean": float(np.mean(delta_values)),
            "median": float(np.median(delta_values)),
            "p05": float(np.quantile(delta_values, 0.05)),
            "p95": float(np.quantile(delta_values, 0.95)),
            "fraction_tempered_greater": float(np.mean(delta_values > 0)),
        },
        "batch_problem_norm_concentration": {
            method: _batch_problem_concentration(active_rows, method) for method in ("canonical", "tempered")
        },
    }
    ranked = sorted(
        active_rows,
        key=lambda row: row["tempered_token_norm_high_logp_top4_share"],
        reverse=True,
    )
    summary["most_concentrated_mixed_problems"] = [
        {
            "problem_index": row["problem_index"],
            "prompt_id": row["prompt_id"],
            "source_row_index": row["source_row_index"],
            "correct_count": row["correct_count"],
            "escort_ess": row["escort_ess"],
            "escort_max_weight": row["escort_max_weight"],
            "canonical_high_logp_top4_norm2_share": row["canonical_token_norm_high_logp_top4_share"],
            "tempered_high_logp_top4_norm2_share": row["tempered_token_norm_high_logp_top4_share"],
            "tempered_effective_trajectories": row["tempered_token_norm_effective_trajectories"],
        }
        for row in ranked[:20]
    ]
    return _finite_or_none(summary), problem_rows, trajectory_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_finite_or_none(rows))


def write_results(
    output_dir: Path,
    summary: dict[str, Any],
    problem_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "per_problem.csv", problem_rows)
    _write_csv(output_dir / "per_trajectory.csv", trajectory_rows)


def main() -> None:
    args = parse_args()
    summary, problem_rows, trajectory_rows = analyze_pool(
        args.pool,
        initial_count=args.initial_count,
        beta=args.tempering_beta,
        epsilon=args.epsilon,
    )
    write_results(args.output_dir, summary, problem_rows, trajectory_rows)

    print(f"Problems: {summary['problem_count']}")
    print(f"Trajectories: {summary['trajectory_count']}")
    print(
        f"Mixed-reward groups with nonzero update: {summary['mixed_reward_problem_count']}/{summary['problem_count']}"
    )
    print(
        "Homogeneous groups with zero advantage: "
        f"{summary['zero_advantage_problem_count']}/{summary['problem_count']} "
        f"({summary['zero_advantage_problem_fraction']:.2%})"
    )
    print(f"Tempering beta: {summary['tempering_beta']:.12g}")
    horizon = summary["common_horizon_distribution"]
    print(
        "Common non-EOS horizon: "
        f"min={min(row['common_non_eos_horizon'] for row in problem_rows)} "
        f"p05={horizon['p05']:.1f} median={horizon['median']:.1f} "
        f"mean={horizon['mean']:.1f} "
        f"max={max(row['common_non_eos_horizon'] for row in problem_rows)}"
    )
    distributions = summary["concentration_distributions_over_mixed_groups"]
    print("\nWithin-problem advantage-tensor norm concentration (mixed groups only):")
    for method in ("canonical", "tempered"):
        print(f"  {method}:")
        for count in (1, 2, 4):
            stats = distributions[f"{method}_token_norm_high_logp_top{count}_share"]
            print(
                f"    high-joint-logp top-{count} share: "
                f"mean={stats['mean']:.2%} median={stats['median']:.2%} p95={stats['p95']:.2%}"
            )
        effective = distributions[f"{method}_token_norm_effective_trajectories"]
        correlation = distributions[f"{method}_joint_logp_token_norm_spearman"]
        print(
            "    effective trajectories: "
            f"mean={effective['mean']:.2f} median={effective['median']:.2f} p05={effective['p05']:.2f}"
        )
        print(f"    joint-logp/norm Spearman: mean={correlation['mean']:.3f} median={correlation['median']:.3f}")
    delta = summary["high_logp_top4_share_delta"]
    print(
        "\nTempered minus canonical high-logp top-4 norm share: "
        f"mean={delta['mean']:+.2%} median={delta['median']:+.2%} "
        f"p05={delta['p05']:+.2%} p95={delta['p95']:+.2%} "
        f"tempered_greater={delta['fraction_tempered_greater']:.2%}"
    )
    print("\nAcross-problem advantage norm concentration (mixed groups only):")
    for method, metrics in summary["batch_problem_norm_concentration"].items():
        print(
            f"  {method}: effective_problems={metrics['effective_problem_count']:.1f}/"
            f"{metrics['active_problem_count']} top1%={metrics['top_1pct_problem_share']:.2%} "
            f"top5%={metrics['top_5pct_problem_share']:.2%} "
            f"top10%={metrics['top_10pct_problem_share']:.2%}"
        )
    print("\nMost concentrated mixed problems:")
    for row in summary["most_concentrated_mixed_problems"][:10]:
        print(
            f"  problem={row['problem_index']} source_row={row['source_row_index']} "
            f"correct={row['correct_count']}/{args.initial_count} ESS={row['escort_ess']:.2f} "
            f"top4_share={row['tempered_high_logp_top4_norm2_share']:.2%} "
            f"effective={row['tempered_effective_trajectories']:.2f}"
        )
    print(f"\nSummary: {args.output_dir / 'summary.json'}")
    print(f"Per problem: {args.output_dir / 'per_problem.csv'}")
    print(f"Per trajectory: {args.output_dir / 'per_trajectory.csv'}")


if __name__ == "__main__":
    main()
