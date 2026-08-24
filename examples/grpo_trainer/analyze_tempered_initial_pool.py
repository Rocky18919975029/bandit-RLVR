#!/usr/bin/env python3
"""Analyze group-shared-prefix tempering on saved initial SIR rollouts.

The script never loads a model.  It reads the chosen-token log-probabilities
already stored in a branched-SIR pool, keeps only ``sir_pool_origin=initial``
trajectories, and evaluates the empirical escort weights

    w_i(alpha) proportional to exp((alpha - 1) * L_i),

For group ``g``, the shared horizon is the shortest response length after
removing terminal EOS.  ``L_i`` sums only the first ``T_g`` chosen-token
log-probabilities, so every response is compared at the same length and EOS is
never included.  The reward/log-probability covariance is the derivative of
the empirical tempered reward with respect to alpha.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="Saved sir_pool/<step>.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-count", type=int, default=16)
    parser.add_argument(
        "--max-alpha-minus-one",
        type=float,
        default=0.5,
        help="Largest alpha-1 on the logarithmic scan (default: 0.5, i.e. alpha=1.5)",
    )
    parser.add_argument("--scan-points", type=int, default=401)
    parser.add_argument(
        "--ess-fraction-budget",
        type=float,
        default=0.5,
        help="ESS/K threshold used only for diagnostics and alpha recommendations",
    )
    parser.add_argument(
        "--required-group-fraction",
        type=float,
        default=0.95,
        help="Fraction of prompt groups that must meet the ESS budget",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def _candidate_reward(candidate: dict[str, Any]) -> float:
    """Return the DAPO +/-1 verifier reward stored in a candidate."""
    if "acc" in candidate:
        return 1.0 if bool(candidate["acc"]) else -1.0
    score = candidate.get("score")
    if score is None:
        raise ValueError(f"candidate {candidate.get('pool_index')} has neither acc nor score")
    score = float(score)
    if not math.isfinite(score):
        raise ValueError(f"candidate {candidate.get('pool_index')} has non-finite score {score}")
    return 1.0 if score > 0 else -1.0


def _group_shortest_non_eos_joint_log_probs(
    candidates: list[dict[str, Any]],
    *,
    context: str,
) -> tuple[np.ndarray, int, int]:
    """Return equal-horizon joint log-probabilities and terminal-EOS count."""
    token_log_prob_arrays: list[np.ndarray] = []
    non_eos_lengths: list[int] = []
    eos_count = 0

    for candidate in candidates:
        token_ids = candidate.get("sampled_token_ids", [])
        token_log_probs = candidate.get("sampled_token_log_probs", [])
        candidate_id = candidate.get("pool_index")
        if not token_ids:
            raise ValueError(f"{context} candidate {candidate_id} has no sampled tokens")
        if len(token_ids) != len(token_log_probs):
            raise ValueError(
                f"{context} candidate {candidate_id} has {len(token_ids)} tokens "
                f"but {len(token_log_probs)} log-probabilities"
            )
        if "ends_with_eos" not in candidate:
            raise ValueError(
                f"{context} candidate {candidate_id} is missing ends_with_eos; "
                "cannot exclude terminal EOS exactly"
            )
        values = np.asarray(token_log_probs, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{context} candidate {candidate_id} contains non-finite log-probabilities")

        ends_with_eos = bool(candidate["ends_with_eos"])
        non_eos_length = len(token_ids) - int(ends_with_eos)
        if non_eos_length < 0:
            raise ValueError(f"{context} candidate {candidate_id} has invalid non-EOS length")
        token_log_prob_arrays.append(values)
        non_eos_lengths.append(non_eos_length)
        eos_count += int(ends_with_eos)

    common_horizon = min(non_eos_lengths)
    joint_log_probs = np.asarray(
        [
            math.fsum(float(value) for value in values[:common_horizon])
            for values in token_log_prob_arrays
        ],
        dtype=np.float64,
    )
    return joint_log_probs, common_horizon, eos_count


def load_initial_groups(pool_path: Path, initial_count: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if initial_count < 2:
        raise ValueError(f"initial_count must be at least 2, got {initial_count}")

    group_log_probs: list[list[float]] = []
    group_rewards: list[list[float]] = []
    eos_count = 0
    trajectory_count = 0
    common_horizons: list[int] = []

    with pool_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            initial = [
                candidate
                for candidate in row.get("candidates", [])
                if candidate.get("sir_pool_origin") == "initial"
            ]
            initial.sort(key=lambda candidate: int(candidate.get("sir_parent_index", -1)))
            if len(initial) != initial_count:
                raise ValueError(
                    f"pool line {line_number} has {len(initial)} initial candidates; expected {initial_count}"
                )

            log_probs, common_horizon, group_eos_count = _group_shortest_non_eos_joint_log_probs(
                initial,
                context=f"pool line {line_number}",
            )
            rewards = [_candidate_reward(candidate) for candidate in initial]
            eos_count += group_eos_count
            trajectory_count += len(initial)
            common_horizons.append(common_horizon)

            group_log_probs.append(log_probs.tolist())
            group_rewards.append(rewards)

    if not group_log_probs:
        raise ValueError(f"no prompt groups found in {pool_path}")

    joint_log_probs = np.asarray(group_log_probs, dtype=np.float64)
    rewards = np.asarray(group_rewards, dtype=np.float64)
    horizons = np.asarray(common_horizons, dtype=np.float64)
    metadata = {
        "pool": str(pool_path.resolve()),
        "problem_count": int(joint_log_probs.shape[0]),
        "initial_count": int(joint_log_probs.shape[1]),
        "trajectory_count": trajectory_count,
        "eos_count": eos_count,
        "eos_fraction": eos_count / trajectory_count,
        "untempered_accuracy": float(np.mean(rewards > 0)),
        "joint_log_prob_horizon": "group_shortest_non_eos_prefix",
        "common_horizon_min": int(np.min(horizons)),
        "common_horizon_p05": float(np.quantile(horizons, 0.05)),
        "common_horizon_mean": float(np.mean(horizons)),
        "common_horizon_median": float(np.median(horizons)),
        "common_horizon_max": int(np.max(horizons)),
        "common_horizon_le_16_fraction": float(np.mean(horizons <= 16)),
        "common_horizon_le_32_fraction": float(np.mean(horizons <= 32)),
        "common_horizon_le_64_fraction": float(np.mean(horizons <= 64)),
    }
    return joint_log_probs, rewards, metadata


def escort_weights(joint_log_probs: np.ndarray, alpha: float) -> np.ndarray:
    delta = float(alpha) - 1.0
    if delta < 0:
        raise ValueError(f"this mode-focused scan requires alpha >= 1, got {alpha}")
    logits = delta * joint_log_probs
    logits -= np.max(logits, axis=1, keepdims=True)
    unnormalized = np.exp(logits)
    return unnormalized / np.sum(unnormalized, axis=1, keepdims=True)


def summarize_alpha(
    joint_log_probs: np.ndarray,
    rewards: np.ndarray,
    alpha: float,
    ess_budget: float,
) -> dict[str, float]:
    weights = escort_weights(joint_log_probs, alpha)
    group_size = joint_log_probs.shape[1]
    ess = 1.0 / np.sum(np.square(weights), axis=1)
    max_weight = np.max(weights, axis=1)

    reward_mean = np.sum(weights * rewards, axis=1)
    log_prob_mean = np.sum(weights * joint_log_probs, axis=1)
    reward_covariance = np.sum(
        weights * (rewards - reward_mean[:, None]) * (joint_log_probs - log_prob_mean[:, None]),
        axis=1,
    )
    reward_variance = np.sum(weights * np.square(rewards - reward_mean[:, None]), axis=1)
    tempered_accuracy = np.sum(weights * (rewards > 0), axis=1)

    def percentile(values: np.ndarray, q: float) -> float:
        return float(np.quantile(values, q))

    return {
        "alpha": float(alpha),
        "alpha_minus_one": float(alpha - 1.0),
        "ess_mean": float(np.mean(ess)),
        "ess_min": float(np.min(ess)),
        "ess_p05": percentile(ess, 0.05),
        "ess_p10": percentile(ess, 0.10),
        "ess_median": percentile(ess, 0.50),
        "ess_p90": percentile(ess, 0.90),
        "ess_fraction_mean": float(np.mean(ess) / group_size),
        "fraction_groups_meeting_ess_budget": float(np.mean(ess >= ess_budget)),
        "max_weight_mean": float(np.mean(max_weight)),
        "max_weight_p05": percentile(max_weight, 0.05),
        "max_weight_median": percentile(max_weight, 0.50),
        "max_weight_p90": percentile(max_weight, 0.90),
        "max_weight_p95": percentile(max_weight, 0.95),
        "max_weight_max": float(np.max(max_weight)),
        "tempered_accuracy_mean": float(np.mean(tempered_accuracy)),
        "tempered_reward_mean": float(np.mean(reward_mean)),
        "reward_variance_mean": float(np.mean(reward_variance)),
        "reward_covariance_mean": float(np.mean(reward_covariance)),
        "reward_covariance_abs_mean": float(np.mean(np.abs(reward_covariance))),
        "reward_covariance_p05": percentile(reward_covariance, 0.05),
        "reward_covariance_median": percentile(reward_covariance, 0.50),
        "reward_covariance_p95": percentile(reward_covariance, 0.95),
        "reward_covariance_positive_fraction": float(np.mean(reward_covariance > 0)),
    }


def make_alpha_grid(max_delta: float, points: int) -> np.ndarray:
    if max_delta <= 0:
        raise ValueError(f"max_alpha_minus_one must be positive, got {max_delta}")
    if points < 2:
        raise ValueError(f"scan_points must be at least 2, got {points}")
    positive = np.geomspace(1e-7, max_delta, num=points, dtype=np.float64)
    return np.concatenate((np.asarray([1.0]), 1.0 + positive))


def max_alpha_from_curve(
    rows: list[dict[str, float]], required_group_fraction: float
) -> dict[str, float] | None:
    eligible = [
        row
        for row in rows
        if row["fraction_groups_meeting_ess_budget"] >= required_group_fraction
    ]
    return eligible[-1] if eligible else None


def write_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(rows: list[dict[str, float]], output_path: Path, group_size: int, ess_budget: float) -> None:
    """Write a dependency-free four-panel SVG curve."""
    width, height = 1400, 940
    outer_x, outer_y = 70, 80
    gap_x, gap_y = 75, 80
    panel_width = (width - 2 * outer_x - gap_x) / 2
    panel_height = (height - outer_y - 70 - gap_y) / 2

    positive_delta = [row["alpha_minus_one"] for row in rows if row["alpha_minus_one"] > 0]
    min_delta = min(positive_delta)
    max_delta = max(positive_delta)
    plot_min_delta = min_delta / 10.0
    log_min = math.log10(plot_min_delta)
    log_max = math.log10(max_delta)

    def x_position(delta: float, left: float) -> float:
        value = max(delta, plot_min_delta)
        fraction = (math.log10(value) - log_min) / (log_max - log_min)
        return left + fraction * panel_width

    colors = {
        "primary": "#2563eb",
        "low": "#93c5fd",
        "high": "#60a5fa",
        "budget": "#dc2626",
        "baseline": "#111827",
        "grid": "#d1d5db",
        "text": "#111827",
    }
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.title{font-size:22px;font-weight:700}'
        '.panel-title{font-size:16px;font-weight:700}.axis{font-size:12px}.legend{font-size:12px}</style>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" class="title">'
        "Group-shared shortest non-EOS prefix tempering</text>",
    ]

    panels = [
        {
            "title": "Effective sample size",
            "series": [
                ("ess_p05", "p05", colors["low"]),
                ("ess_p90", "p90", colors["high"]),
                ("ess_median", "median", colors["primary"]),
            ],
            "y_min": 0.0,
            "y_max": float(group_size),
            "reference": (ess_budget, f"budget={ess_budget:g}", colors["budget"]),
        },
        {
            "title": "Largest trajectory weight",
            "series": [
                ("max_weight_p05", "p05", colors["low"]),
                ("max_weight_p95", "p95", colors["high"]),
                ("max_weight_median", "median", colors["primary"]),
            ],
            "y_min": 0.0,
            "y_max": 1.0,
            "reference": (1.0 / group_size, "uniform", colors["baseline"]),
        },
        {
            "title": "Tempered expected accuracy",
            "series": [("tempered_accuracy_mean", "mean", colors["primary"])],
            "y_min": 0.0,
            "y_max": max(0.25, max(row["tempered_accuracy_mean"] for row in rows) * 1.15),
            "reference": (rows[0]["tempered_accuracy_mean"], "alpha=1", colors["baseline"]),
        },
        {
            "title": "Reward/log-prob covariance",
            "series": [
                ("reward_covariance_p05", "p05", colors["low"]),
                ("reward_covariance_p95", "p95", colors["high"]),
                ("reward_covariance_mean", "mean", colors["primary"]),
            ],
            "y_min": min(0.0, min(row["reward_covariance_p05"] for row in rows)),
            "y_max": max(0.0, max(row["reward_covariance_p95"] for row in rows)),
            "reference": (0.0, "zero", colors["baseline"]),
        },
    ]

    for panel_index, panel in enumerate(panels):
        column = panel_index % 2
        row_index = panel_index // 2
        left = outer_x + column * (panel_width + gap_x)
        top = outer_y + row_index * (panel_height + gap_y)
        bottom = top + panel_height
        y_min = float(panel["y_min"])
        y_max = float(panel["y_max"])
        if math.isclose(y_min, y_max):
            y_min -= 1.0
            y_max += 1.0

        def y_position(
            value: float,
            current_y_min: float = y_min,
            current_y_max: float = y_max,
            current_bottom: float = bottom,
        ) -> float:
            fraction = (float(value) - current_y_min) / (current_y_max - current_y_min)
            return current_bottom - fraction * panel_height

        svg.append(
            f'<text x="{left + panel_width / 2:.1f}" y="{top - 20:.1f}" '
            f'text-anchor="middle" class="panel-title">{escape(str(panel["title"]))}</text>'
        )
        for tick_index in range(6):
            value = y_min + (y_max - y_min) * tick_index / 5
            y = y_position(value)
            svg.append(
                f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + panel_width:.1f}" y2="{y:.1f}" '
                f'stroke="{colors["grid"]}" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{left - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.3g}</text>'
            )
        svg.append(
            f'<rect x="{left:.1f}" y="{top:.1f}" width="{panel_width:.1f}" height="{panel_height:.1f}" '
            'fill="none" stroke="#6b7280" stroke-width="1"/>'
        )

        min_power = math.ceil(log_min)
        max_power = math.floor(log_max)
        for power in range(min_power, max_power + 1):
            delta_tick = 10.0**power
            x = x_position(delta_tick, left)
            svg.append(
                f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{bottom:.1f}" '
                f'stroke="{colors["grid"]}" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{x:.1f}" y="{bottom + 18:.1f}" text-anchor="middle" class="axis">1e{power}</text>'
            )
        svg.append(
            f'<text x="{left:.1f}" y="{bottom + 18:.1f}" text-anchor="middle" class="axis">alpha=1</text>'
        )
        svg.append(
            f'<text x="{left + panel_width / 2:.1f}" y="{bottom + 42:.1f}" '
            'text-anchor="middle" class="axis">alpha - 1 (log scale)</text>'
        )

        reference_value, reference_label, reference_color = panel["reference"]
        reference_y = y_position(float(reference_value))
        if top <= reference_y <= bottom:
            svg.append(
                f'<line x1="{left:.1f}" y1="{reference_y:.1f}" x2="{left + panel_width:.1f}" '
                f'y2="{reference_y:.1f}" stroke="{reference_color}" stroke-width="1.5" stroke-dasharray="7,5"/>'
            )

        legend_x = left + 10
        legend_y = top + 18
        for series_index, (key, label, color) in enumerate(panel["series"]):
            points = " ".join(
                f'{x_position(row["alpha_minus_one"], left):.2f},{y_position(row[key]):.2f}' for row in rows
            )
            line_width = 2.5 if label in {"median", "mean"} else 1.3
            svg.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                f'stroke-width="{line_width}" stroke-linejoin="round"/>'
            )
            current_x = legend_x + series_index * 95
            svg.append(
                f'<line x1="{current_x:.1f}" y1="{legend_y:.1f}" x2="{current_x + 22:.1f}" '
                f'y2="{legend_y:.1f}" stroke="{color}" stroke-width="{line_width}"/>'
            )
            svg.append(
                f'<text x="{current_x + 28:.1f}" y="{legend_y + 4:.1f}" class="legend">{escape(label)}</text>'
            )
        reference_x = legend_x + len(panel["series"]) * 95
        svg.append(
            f'<line x1="{reference_x:.1f}" y1="{legend_y:.1f}" x2="{reference_x + 22:.1f}" '
            f'y2="{legend_y:.1f}" stroke="{reference_color}" stroke-width="1.5" stroke-dasharray="7,5"/>'
        )
        svg.append(
            f'<text x="{reference_x + 28:.1f}" y="{legend_y + 4:.1f}" class="legend">'
            f"{escape(str(reference_label))}</text>"
        )

    svg.append("</svg>")
    output_path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not (0 < args.ess_fraction_budget <= 1):
        raise ValueError("ess_fraction_budget must be in (0, 1]")
    if not (0 < args.required_group_fraction <= 1):
        raise ValueError("required_group_fraction must be in (0, 1]")

    joint_log_probs, rewards, metadata = load_initial_groups(args.pool, args.initial_count)
    group_size = int(joint_log_probs.shape[1])
    ess_budget = args.ess_fraction_budget * group_size
    alpha_grid = make_alpha_grid(args.max_alpha_minus_one, args.scan_points)
    rows = [summarize_alpha(joint_log_probs, rewards, float(alpha), ess_budget) for alpha in alpha_grid]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "alpha_ess_reward_covariance.csv"
    plot_path = args.output_dir / "alpha_ess_reward_covariance.svg"
    summary_path = args.output_dir / "summary.json"
    write_csv(rows, csv_path)
    if not args.no_plot:
        plot_curve(rows, plot_path, group_size, ess_budget)

    selected = max_alpha_from_curve(rows, args.required_group_fraction)
    strict = max_alpha_from_curve(rows, 1.0)
    summary = {
        **metadata,
        "scan": {
            "max_alpha_minus_one": args.max_alpha_minus_one,
            "scan_points_excluding_alpha_one": args.scan_points,
            "ess_fraction_budget": args.ess_fraction_budget,
            "ess_budget": ess_budget,
            "required_group_fraction": args.required_group_fraction,
        },
        "largest_scanned_alpha_meeting_required_group_fraction": selected,
        "largest_scanned_alpha_meeting_budget_for_all_groups": strict,
        "csv": str(csv_path.resolve()),
        "plot": None if args.no_plot else str(plot_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Problems: {metadata['problem_count']}")
    print(f"Initial trajectories/problem: {group_size}")
    print(f"Untempered accuracy: {metadata['untempered_accuracy']:.2%}")
    print(f"EOS fraction: {metadata['eos_fraction']:.2%}")
    print(
        "Common non-EOS horizon: "
        f"min={metadata['common_horizon_min']} "
        f"p05={metadata['common_horizon_p05']:.1f} "
        f"median={metadata['common_horizon_median']:.1f} "
        f"mean={metadata['common_horizon_mean']:.1f} "
        f"max={metadata['common_horizon_max']}"
    )
    print(f"ESS budget: {ess_budget:g}/{group_size} ({args.ess_fraction_budget:.0%})")
    print(f"Required groups meeting budget: {args.required_group_fraction:.0%}")
    if selected is None:
        print("No scanned alpha met the requested group-fraction budget")
    else:
        print(
            "Largest scanned alpha meeting requested budget: "
            f"{selected['alpha']:.10g} "
            f"(delta={selected['alpha_minus_one']:.4g}, "
            f"groups={selected['fraction_groups_meeting_ess_budget']:.2%}, "
            f"median ESS={selected['ess_median']:.3f}, "
            f"p05 ESS={selected['ess_p05']:.3f}, "
            f"median max-weight={selected['max_weight_median']:.3f})"
        )
    if strict is not None:
        print(
            "Largest scanned alpha meeting budget for every group: "
            f"{strict['alpha']:.10g} (delta={strict['alpha_minus_one']:.4g})"
        )
    print(f"CSV: {csv_path}")
    if not args.no_plot:
        print(f"Plot: {plot_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
