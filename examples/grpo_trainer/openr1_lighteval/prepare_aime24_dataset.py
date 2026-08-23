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

"""Convert a repeated VERL AIME24 parquet into a local LightEval dataset."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _extract_problem(row: dict[str, Any]) -> str:
    for key in ("problem", "question"):
        value = _to_builtin(row.get(key))
        if isinstance(value, str) and value.strip():
            return value.strip()

    prompt = _to_builtin(row.get("prompt"))
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    if isinstance(prompt, list):
        user_messages = [
            message.get("content") for message in prompt if isinstance(message, dict) and message.get("role") == "user"
        ]
        user_messages = [message.strip() for message in user_messages if isinstance(message, str) and message.strip()]
        if user_messages:
            return user_messages[-1]

    raise ValueError("Could not extract an AIME problem from columns problem/question/prompt")


def _extract_answer(row: dict[str, Any]) -> str:
    reward_model = _to_builtin(row.get("reward_model"))
    if isinstance(reward_model, dict):
        value = reward_model.get("ground_truth")
        if value is not None and str(value).strip():
            return str(value).strip()

    for key in ("answer", "ground_truth", "target"):
        value = _to_builtin(row.get(key))
        if value is not None and str(value).strip():
            return str(value).strip()

    raise ValueError("Could not extract an AIME answer from reward_model.ground_truth/answer/ground_truth/target")


def prepare_lighteval_aime24(input_path: Path, output_dir: Path, expected_problems: int = 30) -> dict[str, Any]:
    dataframe = pd.read_parquet(input_path)
    records: dict[str, str] = {}

    for row_number, raw_row in enumerate(dataframe.to_dict(orient="records")):
        problem = _extract_problem(raw_row)
        answer = _extract_answer(raw_row)
        previous_answer = records.get(problem)
        if previous_answer is not None and previous_answer != answer:
            raise ValueError(
                f"Conflicting answers for repeated problem at source row {row_number}: "
                f"{previous_answer!r} != {answer!r}"
            )
        records[problem] = answer

    if len(records) != expected_problems:
        raise ValueError(
            f"Expected {expected_problems} unique AIME24 problems, found {len(records)} "
            f"among {len(dataframe)} source rows in {input_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train.parquet"
    pd.DataFrame(
        [{"problem": problem, "answer": answer} for problem, answer in records.items()],
        columns=["problem", "answer"],
    ).to_parquet(output_path, index=False)

    source_repetitions = len(dataframe) / len(records)
    summary = {
        "source_path": str(input_path.resolve()),
        "source_rows": int(len(dataframe)),
        "unique_problems": int(len(records)),
        "average_source_repetitions": source_repetitions,
        "output_path": str(output_path.resolve()),
        "columns": ["problem", "answer"],
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-problems", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_lighteval_aime24(args.input, args.output_dir, args.expected_problems)
    print("Offline LightEval AIME24 dataset prepared")
    print(f"  source rows: {summary['source_rows']}")
    print(f"  unique problems: {summary['unique_problems']}")
    print(f"  local dataset: {summary['output_path']}")


if __name__ == "__main__":
    main()
