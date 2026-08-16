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

"""Shared I/O, data, and vLLM helpers for joint-tempering examples."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATA_PATH = "/data/user/zhongal/data/reschedule/DAPO-Math-17k.filtered.seed42.sample1536.parquet"
POOL_SCHEMA_VERSION = 1


def to_builtin(value: Any) -> Any:
    """Recursively convert pandas/NumPy values into JSON-compatible Python objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_builtin(item) for item in value]
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} at line {line_number}") from exc
    return rows


def append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary_path.replace(output_path)


def write_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_path.replace(output_path)


def deterministic_problem_indices(dataset_size: int, num_problems: int, seed: int) -> list[int]:
    if dataset_size <= 0:
        raise ValueError(f"dataset_size must be positive, got {dataset_size}")
    if num_problems <= 0:
        raise ValueError(f"num_problems must be positive, got {num_problems}")
    if num_problems > dataset_size:
        raise ValueError(f"requested {num_problems} problems from a dataset with only {dataset_size} rows")
    rng = np.random.default_rng(seed)
    return sorted(int(index) for index in rng.choice(dataset_size, size=num_problems, replace=False))


def load_sampled_problems(data_path: str, num_problems: int, seed: int) -> tuple[list[dict[str, Any]], list[int]]:
    """Read a local parquet and deterministically select the requested rows."""
    import pandas as pd

    frame = pd.read_parquet(data_path)
    selected_indices = deterministic_problem_indices(len(frame), num_problems, seed)
    problems = []
    dataset_name = Path(data_path).stem
    for source_index in selected_indices:
        row = frame.iloc[source_index]
        prompt = to_builtin(row["prompt"])
        reward_model = to_builtin(row["reward_model"])
        if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
            raise ValueError(f"row {source_index} has no reward_model.ground_truth")
        problems.append(
            {
                "prompt_id": f"{dataset_name}:row-{source_index}",
                "source_row_index": source_index,
                "prompt": prompt,
                "ground_truth": str(reward_model["ground_truth"]),
                "data_source": str(to_builtin(row.get("data_source", "math_dapo"))),
                "extra_info": to_builtin(row.get("extra_info", {})),
            }
        )
    return problems, selected_indices


def normalize_token_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, dict):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        if len(tokenized) != 1:
            raise ValueError("expected one tokenized prompt")
        tokenized = tokenized[0]
    return [int(token_id) for token_id in tokenized]


def tokenize_problem_prompt(tokenizer: Any, prompt: Any) -> list[int]:
    if isinstance(prompt, str):
        return normalize_token_ids(tokenizer.encode(prompt, add_special_tokens=True))
    if not isinstance(prompt, list):
        raise TypeError(f"prompt must be a chat-message list or string, got {type(prompt).__name__}")
    tokenized = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
    return normalize_token_ids(tokenized)


def make_tokens_prompt(prompt_token_ids: list[int]) -> Any:
    """Use vLLM's typed token prompt, with the repo's compatibility fallback."""
    prompt_kwargs = {"prompt_token_ids": prompt_token_ids}
    try:
        from vllm.inputs import TokensPrompt

        return TokensPrompt(**prompt_kwargs)
    except (ImportError, TypeError):
        return prompt_kwargs


def build_llm(
    model_path: str,
    tensor_parallel_size: int,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    trust_remote_code: bool,
    enforce_eager: bool,
    seed: int,
) -> Any:
    from vllm import LLM

    kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": trust_remote_code,
        "enforce_eager": enforce_eager,
        "seed": seed,
        "generation_config": "vllm",
        "enable_prefix_caching": True,
        "disable_log_stats": True,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    return LLM(**kwargs)


def extract_vllm_completion(completion: Any) -> dict[str, Any]:
    token_ids = [int(token_id) for token_id in completion.token_ids]
    if completion.logprobs is None:
        raise ValueError("vLLM returned no logprobs; SamplingParams.logprobs must be enabled")
    if len(completion.logprobs) != len(token_ids):
        raise ValueError(
            f"vLLM returned {len(completion.logprobs)} logprob entries for {len(token_ids)} generated tokens"
        )

    token_log_probs = []
    for position, (token_id, logprob_candidates) in enumerate(zip(token_ids, completion.logprobs, strict=True)):
        if token_id not in logprob_candidates:
            raise ValueError(f"chosen token {token_id} missing from logprobs at position {position}")
        token_log_probs.append(float(logprob_candidates[token_id].logprob))

    return {
        "token_ids": token_ids,
        "token_log_probs": token_log_probs,
        "finish_reason": getattr(completion, "finish_reason", None),
        "stop_reason": to_builtin(getattr(completion, "stop_reason", None)),
    }


def alpha_key(alpha: float) -> str:
    return format(float(alpha), ".12g")
