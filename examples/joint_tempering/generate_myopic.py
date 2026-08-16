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

"""Generate exact myopic-temperature prefixes followed by temperature-1 completions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.joint_tempering.common import (  # noqa: E402
    alpha_key,
    append_jsonl,
    build_llm,
    extract_vllm_completion,
    make_tokens_prompt,
    read_jsonl,
    response_token_budget,
    write_json,
)
from verl.experimental.joint_tempering.sir import stable_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HPC-local model or checkpoint path.")
    parser.add_argument("--pool", default="outputs/joint_tempering/pool.jsonl")
    parser.add_argument("--output", default="outputs/joint_tempering/myopic.jsonl")
    parser.add_argument("--block-lengths", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.25, 1.5, 2.0])
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--max-response-tokens", type=int, default=4096)
    parser.add_argument("--request-batch-size", type=int, default=16)
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(block_length <= 0 for block_length in args.block_lengths):
        raise ValueError("all --block-lengths must be positive")
    if any(block_length >= args.max_response_tokens for block_length in args.block_lengths):
        raise ValueError("every block length must be smaller than --max-response-tokens")
    if any(not np.isfinite(alpha) or alpha <= 0 for alpha in args.alphas):
        raise ValueError("all --alphas must be finite and positive")
    if args.num_repeats <= 0:
        raise ValueError("--num-repeats must be positive")
    if args.request_batch_size <= 0:
        raise ValueError("--request-batch-size must be positive")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive")
    if args.max_model_len is not None and args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")


def expected_manifest(args: argparse.Namespace, prompt_ids: list[str]) -> dict:
    return {
        "model": args.model,
        "pool": args.pool,
        "prompt_ids": prompt_ids,
        "max_response_tokens": args.max_response_tokens,
        "max_model_len": args.max_model_len,
        "generation_seed": args.generation_seed,
        "prefix_sampling": {
            "temperature": "1 / alpha",
            "top_p": 1.0,
            "top_k": -1,
            "ignore_eos": True,
            "logprobs": 0,
        },
        "continuation_sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
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


def request_key(row: dict) -> tuple[str, int, str, int]:
    return (row["prompt_id"], int(row["block_length"]), alpha_key(row["alpha"]), int(row["repeat"]))


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    pool_rows = read_jsonl(args.pool)
    if not pool_rows:
        raise ValueError(f"pool is empty: {args.pool}")
    prompt_ids = [row["prompt_id"] for row in pool_rows]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError(f"duplicate prompt IDs found in {args.pool}")

    output_path = Path(args.output)
    manifest_path = Path(str(output_path) + ".manifest.json")
    ensure_manifest(manifest_path, expected_manifest(args, prompt_ids))
    existing_rows = read_jsonl(output_path) if output_path.exists() else []
    completed_keys = {request_key(row) for row in existing_rows}
    if len(completed_keys) != len(existing_rows):
        raise ValueError(f"duplicate myopic rows found in {output_path}")

    desired_requests = []
    for block_length in args.block_lengths:
        for alpha in args.alphas:
            for repeat in range(args.num_repeats):
                for pool_row in pool_rows:
                    desired_requests.append(
                        {
                            "pool_row": pool_row,
                            "block_length": block_length,
                            "alpha": float(alpha),
                            "repeat": repeat,
                        }
                    )
    unexpected_prompt_ids = {key[0] for key in completed_keys} - set(prompt_ids)
    if unexpected_prompt_ids:
        raise ValueError(f"unexpected prompt IDs found in {output_path}: {sorted(unexpected_prompt_ids)}")
    pending_requests = [
        request
        for request in desired_requests
        if (
            request["pool_row"]["prompt_id"],
            request["block_length"],
            alpha_key(request["alpha"]),
            request["repeat"],
        )
        not in completed_keys
    ]
    if not pending_requests:
        print(f"Myopic output already contains all requested rows: {output_path}", flush=True)
        return

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from verl.utils.reward_score import default_compute_score

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
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

    for batch_start in range(0, len(pending_requests), args.request_batch_size):
        batch = pending_requests[batch_start : batch_start + args.request_batch_size]
        for request in batch:
            request["response_token_budget"] = response_token_budget(
                len(request["pool_row"]["prompt_token_ids"]),
                args.max_response_tokens,
                args.max_model_len,
            )
            if request["block_length"] >= request["response_token_budget"]:
                raise ValueError(
                    f"prompt {request['pool_row']['prompt_id']} has total response budget "
                    f"{request['response_token_budget']}, which must exceed block length {request['block_length']}"
                )
        prefix_prompts = [make_tokens_prompt(request["pool_row"]["prompt_token_ids"]) for request in batch]
        prefix_params = [
            SamplingParams(
                n=1,
                temperature=1.0 / request["alpha"],
                top_p=1.0,
                top_k=-1,
                repetition_penalty=1.0,
                max_tokens=request["block_length"],
                ignore_eos=True,
                logprobs=0,
                seed=stable_seed(
                    args.generation_seed,
                    request["pool_row"]["prompt_id"],
                    request["block_length"],
                    alpha_key(request["alpha"]),
                    request["repeat"],
                    "myopic-prefix",
                ),
            )
            for request in batch
        ]
        prefix_outputs = llm.generate(prefix_prompts, sampling_params=prefix_params, use_tqdm=True)

        blocks = []
        continuation_prompts = []
        continuation_params = []
        for request, request_output in zip(batch, prefix_outputs, strict=True):
            if len(request_output.outputs) != 1:
                raise RuntimeError("myopic prefix request did not return exactly one output")
            block = extract_vllm_completion(request_output.outputs[0])
            if len(block["token_ids"]) != request["block_length"]:
                raise RuntimeError(
                    f"prompt {request['pool_row']['prompt_id']} generated {len(block['token_ids'])} prefix tokens; "
                    f"expected exactly {request['block_length']}"
                )
            blocks.append(block)
            continuation_prompts.append(
                make_tokens_prompt(request["pool_row"]["prompt_token_ids"] + block["token_ids"])
            )
            continuation_params.append(
                SamplingParams(
                    n=1,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    repetition_penalty=1.0,
                    max_tokens=request["response_token_budget"] - request["block_length"],
                    ignore_eos=False,
                    logprobs=0,
                    seed=stable_seed(
                        args.generation_seed,
                        request["pool_row"]["prompt_id"],
                        request["repeat"],
                        "shared-continuation",
                    ),
                )
            )

        continuation_outputs = llm.generate(
            continuation_prompts,
            sampling_params=continuation_params,
            use_tqdm=True,
        )
        output_rows = []
        for request, block, request_output in zip(batch, blocks, continuation_outputs, strict=True):
            if len(request_output.outputs) != 1:
                raise RuntimeError("myopic continuation request did not return exactly one output")
            suffix = extract_vllm_completion(request_output.outputs[0])
            full_token_ids = block["token_ids"] + suffix["token_ids"]
            full_token_log_probs = block["token_log_probs"] + suffix["token_log_probs"]
            response_text = tokenizer.decode(full_token_ids, skip_special_tokens=True)
            score_result = default_compute_score(
                "math_dapo",
                response_text,
                request["pool_row"]["ground_truth"],
                strict_box_verify=True,
            )
            eos_token_id = tokenizer.eos_token_id
            output_rows.append(
                {
                    "prompt_id": request["pool_row"]["prompt_id"],
                    "source_row_index": request["pool_row"]["source_row_index"],
                    "block_length": request["block_length"],
                    "alpha": request["alpha"],
                    "repeat": request["repeat"],
                    "response_token_budget": request["response_token_budget"],
                    "block_token_ids": block["token_ids"],
                    "block_token_log_probs": block["token_log_probs"],
                    "block_finish_reason": block["finish_reason"],
                    "eos_in_block": eos_token_id is not None and eos_token_id in block["token_ids"],
                    "suffix_token_ids": suffix["token_ids"],
                    "suffix_token_log_probs": suffix["token_log_probs"],
                    "finish_reason": suffix["finish_reason"],
                    "stop_reason": suffix["stop_reason"],
                    "token_ids": full_token_ids,
                    "token_log_probs": full_token_log_probs,
                    "response_text": response_text,
                    "score": float(score_result["score"]),
                    "acc": bool(score_result["acc"]),
                    "pred": score_result["pred"],
                }
            )
        append_jsonl(output_path, output_rows)
        completed = len(completed_keys) + batch_start + len(batch)
        total = len(completed_keys) + len(pending_requests)
        print(f"Saved {completed}/{total} myopic rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
