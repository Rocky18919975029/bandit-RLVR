#!/usr/bin/env python3
"""Audit the shortest initial response in every saved rollout group."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _percentile(sorted_values: list[int], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def load_groups(pool_path: Path, initial_count: int, expected_groups: int | None) -> list[dict[str, Any]]:
    groups = []
    with pool_path.open(encoding="utf-8") as handle:
        for group_index, line in enumerate(handle):
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
                    f"group {group_index}: expected {initial_count} initial responses, found {len(initial)}"
                )

            candidates = []
            for candidate in initial:
                token_ids = candidate.get("sampled_token_ids", [])
                token_log_probs = candidate.get("sampled_token_log_probs", [])
                if not token_ids:
                    raise ValueError(f"group {group_index}, pool_index={candidate.get('pool_index')}: empty response")
                if len(token_ids) != len(token_log_probs):
                    raise ValueError(
                        f"group {group_index}, pool_index={candidate.get('pool_index')}: "
                        f"{len(token_ids)} tokens != {len(token_log_probs)} log-probs"
                    )
                ends_with_eos = bool(candidate.get("ends_with_eos", False))
                candidates.append(
                    {
                        "group_index": group_index,
                        "prompt_uid": row.get("prompt_uid"),
                        "prompt": row.get("prompt", ""),
                        "ground_truth": row.get("ground_truth"),
                        "pool_index": candidate.get("pool_index"),
                        "parent_index": candidate.get("sir_parent_index"),
                        "raw_length": len(token_ids),
                        "non_eos_length": len(token_ids) - int(ends_with_eos),
                        "ends_with_eos": ends_with_eos,
                        "acc": bool(candidate.get("acc", False)),
                        "score": candidate.get("score"),
                        "response": candidate.get("response", ""),
                    }
                )

            minimum = min(candidate["non_eos_length"] for candidate in candidates)
            groups.append(
                {
                    "group_index": group_index,
                    "minimum_non_eos_length": minimum,
                    "shortest": [candidate for candidate in candidates if candidate["non_eos_length"] == minimum],
                }
            )

    if expected_groups is not None and len(groups) != expected_groups:
        raise ValueError(f"expected {expected_groups} groups, found {len(groups)}")
    if not groups:
        raise ValueError(f"no groups found in {pool_path}")
    return groups


def build_report(
    pool_path: Path,
    groups: list[dict[str, Any]],
    *,
    global_count: int,
    random_count: int,
    seed: int,
    text_limit: int,
) -> str:
    lines: list[str] = []

    def emit(value: str = "") -> None:
        lines.append(value)

    def emit_candidate(label: str, candidate: dict[str, Any]) -> None:
        emit()
        emit("-" * 120)
        emit(label)
        emit(
            f"group={candidate['group_index']} uid={candidate['prompt_uid']} "
            f"pool_index={candidate['pool_index']} parent_index={candidate['parent_index']} "
            f"raw_tokens={candidate['raw_length']} non_eos_tokens={candidate['non_eos_length']} "
            f"ends_with_eos={candidate['ends_with_eos']} correct={candidate['acc']} score={candidate['score']}"
        )
        emit("\nPROMPT:")
        emit(str(candidate["prompt"])[:text_limit])
        emit("\nGROUND TRUTH:")
        emit(str(candidate["ground_truth"]))
        emit("\nRESPONSE:")
        response = str(candidate["response"])
        emit(response[:text_limit])
        if len(response) > text_limit:
            emit(f"\n... [display truncated; total characters={len(response)}]")

    minima = [int(group["minimum_non_eos_length"]) for group in groups]
    sorted_minima = sorted(minima)
    shortest = [candidate for group in groups for candidate in group["shortest"]]
    distribution = Counter(minima)

    emit("=" * 120)
    emit("INITIAL-ROLLOUT GROUP-SHORTEST RESPONSE AUDIT")
    emit("=" * 120)
    emit(f"pool: {pool_path.resolve()}")
    emit(f"groups: {len(groups)}")
    emit("length: sampled tokens minus the terminal EOS token, when present")
    emit("proposed group horizon: minimum non-EOS length within the group")
    emit()
    emit("Per-group minimum non-EOS length:")
    emit(f"  min={min(minima)}")
    emit(f"  p01={_percentile(sorted_minima, 0.01):.2f}")
    emit(f"  p05={_percentile(sorted_minima, 0.05):.2f}")
    emit(f"  p10={_percentile(sorted_minima, 0.10):.2f}")
    emit(f"  mean={statistics.mean(minima):.2f}")
    emit(f"  median={statistics.median(minima):.2f}")
    emit(f"  p90={_percentile(sorted_minima, 0.90):.2f}")
    emit(f"  max={max(minima)}")
    emit()
    for threshold in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256):
        count = sum(length <= threshold for length in minima)
        emit(f"  groups with minimum <= {threshold:3d}: {count:3d}/{len(groups)} ({count / len(groups):6.2%})")
    emit()
    emit("Exact distribution for minimum lengths <=128:")
    for length, count in sorted(distribution.items()):
        if length <= 128:
            emit(f"  T={length:3d}: {count:3d} groups")
    emit()
    emit("Candidates attaining their group minimum:")
    emit(f"  count={len(shortest)}")
    emit(f"  EOS rate={sum(item['ends_with_eos'] for item in shortest) / len(shortest):.2%}")
    emit(f"  accuracy={sum(item['acc'] for item in shortest) / len(shortest):.2%}")
    emit(f"  non-EOS/truncated rate={sum(not item['ends_with_eos'] for item in shortest) / len(shortest):.2%}")

    emit()
    emit("=" * 120)
    emit(f"GLOBAL {global_count} SHORTEST GROUP-MINIMUM RESPONSES")
    emit("=" * 120)
    ordered = sorted(
        shortest,
        key=lambda item: (item["non_eos_length"], item["group_index"], int(item["pool_index"])),
    )
    for rank, candidate in enumerate(ordered[:global_count], start=1):
        emit_candidate(f"GLOBAL SHORTEST #{rank}/{global_count}", candidate)

    emit()
    emit("=" * 120)
    actual_random_count = min(random_count, len(groups))
    emit(f"RANDOM {actual_random_count} GROUPS: ONE SHORTEST RESPONSE PER GROUP")
    emit("=" * 120)
    rng = random.Random(seed)
    for sample_index, group in enumerate(rng.sample(groups, actual_random_count), start=1):
        candidate = min(group["shortest"], key=lambda item: int(item["pool_index"]))
        emit_candidate(f"RANDOM GROUP {sample_index}/{actual_random_count}", candidate)

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-count", type=int, default=16)
    parser.add_argument("--expected-groups", type=int, default=512)
    parser.add_argument("--global-count", type=int, default=30)
    parser.add_argument("--random-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-limit", type=int, default=1600)
    args = parser.parse_args()

    groups = load_groups(args.pool, args.initial_count, args.expected_groups)
    report = build_report(
        args.pool,
        groups,
        global_count=args.global_count,
        random_count=args.random_count,
        seed=args.seed,
        text_limit=args.text_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(args.output)

    minima = [group["minimum_non_eos_length"] for group in groups]
    print(f"Audit complete: groups={len(groups)} min={min(minima)} median={statistics.median(minima):.2f}")
    print(f"Saved non-empty report ({args.output.stat().st_size} bytes): {args.output.resolve()}")


if __name__ == "__main__":
    main()
