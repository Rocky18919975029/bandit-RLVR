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

"""Generate a reusable pool of complete temperature-1 math trajectories."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.joint_tempering.common import (  # noqa: E402
    DEFAULT_DATA_PATH,
    POOL_SCHEMA_VERSION,
    append_jsonl,
    build_llm,
    extract_vllm_completion,
    load_sampled_problems,
    make_tokens_prompt,
    read_jsonl,
    tokenize_problem_prompt,
    write_json,
)
from verl.experimental.joint_tempering.sir import stable_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HPC-local model or checkpoint path.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="HPC-local DAPO parquet path.")
    parser.add_argument("--output", default="outputs/joint_tempering/pool.jsonl")
    parser.add_argument("--num-problems", type=int, default=128)
    parser.add_argument("--problem-seed", type=int, default=42)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--max-response-tokens", type=int, default=4096)
    parser.add_argument("--prompt-batch-size", type=int, default=2)
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_candidates <= 0:
        raise ValueError("--num-candidates must be positive")
    if args.max_response_tokens <= 0:
        raise ValueError("--max-response-tokens must be positive")
    if args.prompt_batch_size <= 0:
        raise ValueError("--prompt-batch-size must be positive")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")


def expected_manifest(args: argparse.Namespace, selected_indices: list[int]) -> dict:
    return {
        "schema_version": POOL_SCHEMA_VERSION,
        "model": args.model,
        "data": args.data,
        "num_problems": args.num_problems,
        "problem_seed": args.problem_seed,
        "selected_source_row_indices": selected_indices,
        "num_candidates": args.num_candidates,
        "max_response_tokens": args.max_response_tokens,
        "generation_seed": args.generation_seed,
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "ignore_eos": False,
            "logprobs": 0,
        },
    }


def ensure_manifest(path: Path, expected: dict) -> None:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != expected:
            raise ValueError(
                f"existing manifest {path} does not match this run; use a new --output path or restore the old config"
            )
    else:
        write_json(path, expected)


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    problems, selected_indices = load_sampled_problems(args.data, args.num_problems, args.problem_seed)
    output_path = Path(args.output)
    manifest_path = Path(str(output_path) + ".manifest.json")
    ensure_manifest(manifest_path, expected_manifest(args, selected_indices))

    existing_rows = read_jsonl(output_path) if output_path.exists() else []
    completed_prompt_ids = {row["prompt_id"] for row in existing_rows}
    if len(completed_prompt_ids) != len(existing_rows):
        raise ValueError(f"duplicate prompt_id rows found in {output_path}")
    expected_prompt_ids = {problem["prompt_id"] for problem in problems}
    unexpected_prompt_ids = completed_prompt_ids - expected_prompt_ids
    if unexpected_prompt_ids:
        raise ValueError(f"unexpected prompt IDs found in {output_path}: {sorted(unexpected_prompt_ids)}")
    for row in existing_rows:
        if row.get("schema_version") != POOL_SCHEMA_VERSION:
            raise ValueError(f"unsupported pool schema for prompt {row['prompt_id']}")
        if len(row.get("candidates", [])) != args.num_candidates:
            raise ValueError(f"incomplete candidate pool for prompt {row['prompt_id']}")
    pending_problems = [problem for problem in problems if problem["prompt_id"] not in completed_prompt_ids]
    if not pending_problems:
        print(f"Pool already contains all {len(problems)} sampled problems: {output_path}", flush=True)
        return

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from verl.utils.reward_score import default_compute_score

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    for problem in pending_problems:
        problem["prompt_token_ids"] = tokenize_problem_prompt(tokenizer, problem["prompt"])

    llm = build_llm(
        model_path=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
        seed=args.generation_seed,
    )

    for batch_start in range(0, len(pending_problems), args.prompt_batch_size):
        batch = pending_problems[batch_start : batch_start + args.prompt_batch_size]
        prompts = [make_tokens_prompt(problem["prompt_token_ids"]) for problem in batch]
        sampling_params = [
            SamplingParams(
                n=args.num_candidates,
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                repetition_penalty=1.0,
                max_tokens=args.max_response_tokens,
                ignore_eos=False,
                logprobs=0,
                seed=stable_seed(args.generation_seed, problem["prompt_id"], "pool"),
            )
            for problem in batch
        ]
        request_outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
        if len(request_outputs) != len(batch):
            raise RuntimeError(f"vLLM returned {len(request_outputs)} requests for a batch of {len(batch)}")

        output_rows = []
        for problem, request_output in zip(batch, request_outputs, strict=True):
            if len(request_output.outputs) != args.num_candidates:
                raise RuntimeError(
                    f"prompt {problem['prompt_id']} returned {len(request_output.outputs)} candidates; "
                    f"expected {args.num_candidates}"
                )
            candidates = []
            for candidate_index, raw_completion in enumerate(request_output.outputs):
                candidate = extract_vllm_completion(raw_completion)
                response_text = tokenizer.decode(candidate["token_ids"], skip_special_tokens=True)
                score_result = default_compute_score(
                    "math_dapo",
                    response_text,
                    problem["ground_truth"],
                    strict_box_verify=True,
                )
                candidate.update(
                    {
                        "candidate_id": candidate_index,
                        "response_text": response_text,
                        "score": float(score_result["score"]),
                        "acc": bool(score_result["acc"]),
                        "pred": score_result["pred"],
                    }
                )
                candidates.append(candidate)

            output_rows.append(
                {
                    "schema_version": POOL_SCHEMA_VERSION,
                    "prompt_id": problem["prompt_id"],
                    "source_row_index": problem["source_row_index"],
                    "prompt": problem["prompt"],
                    "prompt_token_ids": problem["prompt_token_ids"],
                    "ground_truth": problem["ground_truth"],
                    "data_source": problem["data_source"],
                    "extra_info": problem["extra_info"],
                    "candidates": candidates,
                }
            )
        append_jsonl(output_path, output_rows)
        completed = len(completed_prompt_ids) + batch_start + len(batch)
        print(f"Saved {completed}/{len(problems)} problems to {output_path}", flush=True)


if __name__ == "__main__":
    main()
