#!/usr/bin/env python3
"""Generate one AIME request per problem with ``SamplingParams(n=N)``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.grpo_trainer.prepare_aime24_validation import canonical_prompt
from examples.joint_tempering.common import (
    make_tokens_prompt,
    response_token_budget,
    to_builtin,
    tokenize_problem_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replica-index", type=int, required=True)
    parser.add_argument("--num-replicas", type=int, required=True)
    parser.add_argument("--samples-per-problem", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-response-tokens", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.replica_index < args.num_replicas:
        raise ValueError("replica index is outside [0, num_replicas)")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from verl.utils.reward_score import default_compute_score

    frame = pd.read_parquet(args.data).reset_index(drop=True)
    indices = [index for index in range(len(frame)) if index % args.num_replicas == args.replica_index]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)

    problems = []
    prompts = []
    sampling_params = []
    for index in indices:
        row = frame.iloc[index]
        prompt = to_builtin(row["prompt"])
        reward_model = to_builtin(row["reward_model"])
        if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
            raise ValueError(f"AIME row {index} has no reward_model.ground_truth")
        prompt_ids = tokenize_problem_prompt(tokenizer, prompt)
        max_tokens = response_token_budget(len(prompt_ids), args.max_response_tokens, args.max_model_len)
        problems.append(
            {
                "index": index,
                "uid": hashlib.sha256(canonical_prompt(prompt).encode()).hexdigest()[:16],
                "input": tokenizer.decode(prompt_ids, skip_special_tokens=True),
                "ground_truth": str(reward_model["ground_truth"]),
                "data_source": str(to_builtin(row.get("data_source", "aime24"))),
            }
        )
        prompts.append(make_tokens_prompt(prompt_ids))
        sampling_params.append(
            SamplingParams(
                n=args.samples_per_problem,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=-1,
                max_tokens=max_tokens,
                seed=args.seed,
            )
        )

    print(
        f"Replica {args.replica_index}/{args.num_replicas}: {len(problems)} problems, "
        f"one request/problem, n={args.samples_per_problem}, seed={args.seed}",
        flush=True,
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        seed=args.seed,
        generation_config="vllm",
        enable_prefix_caching=True,
    )
    request_outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    if len(request_outputs) != len(problems):
        raise RuntimeError(f"vLLM returned {len(request_outputs)} requests for {len(problems)} problems")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for problem, request_output in zip(problems, request_outputs, strict=True):
            if len(request_output.outputs) != args.samples_per_problem:
                raise RuntimeError(
                    f"problem {problem['index']} returned {len(request_output.outputs)} outputs; "
                    f"expected {args.samples_per_problem}"
                )
            for sample_index, completion in enumerate(request_output.outputs):
                response = tokenizer.decode(completion.token_ids, skip_special_tokens=True)
                score = default_compute_score(
                    "math_dapo", response, problem["ground_truth"], strict_box_verify=True
                )
                if not isinstance(score, dict):
                    raise TypeError(f"strict math verifier returned {type(score).__name__}, expected dict")
                row = {
                    "input": problem["input"],
                    "output": response,
                    "gts": problem["ground_truth"],
                    "score": float(score["score"]),
                    "acc": bool(score["acc"]),
                    "pred": score.get("pred"),
                    "step": 0,
                    "uid": problem["uid"],
                    "data_source": problem["data_source"],
                    "sample_index": sample_index,
                    "replica_index": args.replica_index,
                    "request_seed": args.seed,
                    "request_n": args.samples_per_problem,
                }
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"Replica {args.replica_index}: saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
