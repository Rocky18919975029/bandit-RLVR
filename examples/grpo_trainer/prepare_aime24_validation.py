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

"""Deduplicate an AIME parquet before VERL applies validation ``n`` sampling."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def canonical_prompt(prompt) -> str:
    """Return a stable key for nested chat prompts loaded from parquet."""
    return json.dumps(
        _to_jsonable(prompt),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def prepare_unique_aime24(input_path: Path, output_path: Path, expected_problems: int) -> dict:
    dataframe = pd.read_parquet(input_path)
    if "prompt" not in dataframe.columns:
        raise ValueError(f"AIME parquet has no 'prompt' column: {input_path}")

    prompt_keys = dataframe["prompt"].map(canonical_prompt)
    repetition_counts = prompt_keys.value_counts(sort=False)
    unique_count = int(len(repetition_counts))
    if unique_count != expected_problems:
        raise ValueError(
            f"Expected {expected_problems} unique AIME prompts, found {unique_count} "
            f"among {len(dataframe)} rows in {input_path}"
        )

    unique_dataframe = dataframe.loc[~prompt_keys.duplicated(keep="first")].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_dataframe.to_parquet(output_path, index=False)

    summary = {
        "source_rows": int(len(dataframe)),
        "unique_problems": unique_count,
        "minimum_source_repetitions": int(repetition_counts.min()),
        "maximum_source_repetitions": int(repetition_counts.max()),
        "output_rows": int(len(unique_dataframe)),
        "output_path": str(output_path),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-problems", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_unique_aime24(args.input, args.output, args.expected_problems)
    print("AIME24 dataset preflight")
    print(f"  source rows: {summary['source_rows']}")
    print(f"  unique problems: {summary['unique_problems']}")
    print(
        "  source repetitions/problem: "
        f"min={summary['minimum_source_repetitions']} max={summary['maximum_source_repetitions']}"
    )
    print(f"  deduplicated rows: {summary['output_rows']}")
    print(f"  deduplicated parquet: {summary['output_path']}")


if __name__ == "__main__":
    main()
