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

import csv
import json
from pathlib import Path

import pytest

from examples.grpo_trainer.analyze_aime24_validation import (
    load_validation_rows,
    summarize_validation,
    unbiased_pass_at_k,
    write_results,
)
from examples.grpo_trainer.prepare_aime24_validation import canonical_prompt, prepare_unique_aime24


def test_unbiased_pass_at_k():
    assert unbiased_pass_at_k(4, 0, 2) == 0.0
    assert unbiased_pass_at_k(4, 1, 2) == 0.5
    assert unbiased_pass_at_k(4, 2, 2) == pytest.approx(5 / 6)
    assert unbiased_pass_at_k(4, 4, 2) == 1.0


def test_summarize_and_write_validation(tmp_path):
    correct_counts = [0, 1, 2, 4]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    dump_path = raw_dir / "0.jsonl"
    with dump_path.open("w", encoding="utf-8") as handle:
        for problem_index, correct_count in enumerate(correct_counts):
            for sample_index in range(4):
                handle.write(
                    json.dumps(
                        {
                            "uid": f"problem-{problem_index}",
                            "data_source": "aime24",
                            "input": f"prompt {problem_index}",
                            "output": f"answer {sample_index}",
                            "acc": sample_index < correct_count,
                            "score": 1.0 if sample_index < correct_count else -1.0,
                        }
                    )
                    + "\n"
                )

    rows = load_validation_rows(raw_dir)
    summary, per_problem = summarize_validation(
        rows,
        [1, 2, 4, 8],
        expected_problems=4,
        expected_samples_per_problem=4,
    )

    assert summary["metrics"]["accuracy"] == pytest.approx(7 / 16)
    assert summary["metrics"]["pass@1"] == pytest.approx(7 / 16)
    assert summary["metrics"]["pass@2"] == pytest.approx((0 + 1 / 2 + 5 / 6 + 1) / 4)
    assert summary["metrics"]["pass@4"] == pytest.approx(3 / 4)
    assert summary["skipped_pass_k"] == [8]
    assert summary["uid_fallback_to_input"] is False

    output_dir = tmp_path / "results"
    write_results(output_dir, summary, per_problem)
    assert json.loads((output_dir / "summary.json").read_text())["problem_count"] == 4
    with (output_dir / "summary.csv").open(newline="") as handle:
        metrics = {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}
    assert metrics["pass@4"] == pytest.approx(0.75)


def test_validation_count_mismatch_is_rejected():
    rows = [{"uid": "one", "data_source": "aime24", "acc": True}]
    with pytest.raises(ValueError, match="Expected 2 problems"):
        summarize_validation(rows, [1], expected_problems=2)


def test_aime24_posthoc_uses_lighteval_shared_replica_seed_by_default():
    script = Path("examples/grpo_trainer/run_aime24_posthoc_validation_fsdp.sh").read_text()
    worker = Path("examples/grpo_trainer/generate_aime24_vllm_replica.py").read_text()
    assert "ROLLOUT_REPLICA_SEED_MODE=${ROLLOUT_REPLICA_SEED_MODE:-shared}" in script
    assert "EVAL_N=${EVAL_N:-64}" in script
    assert "EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-0.6}" in script
    assert "EVAL_TOP_P=${EVAL_TOP_P:-0.95}" in script
    assert '--samples-per-problem "${EVAL_N}"' in script
    assert '--seed "${EVAL_SEED}"' in script
    assert "n=args.samples_per_problem" in worker
    assert "seed=args.seed" in worker


def test_prepare_unique_aime24_removes_physical_repetitions(tmp_path):
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    prompts = [
        [{"role": "user", "content": "problem zero"}],
        [{"role": "user", "content": "problem one"}],
        [{"role": "user", "content": "problem two"}],
    ]
    source = tmp_path / "repeated.parquet"
    output = tmp_path / "unique.parquet"
    pandas.DataFrame(
        [
            {"prompt": prompt, "reward_model": {"ground_truth": str(index)}}
            for index, prompt in enumerate(prompts)
            for _ in range(4)
        ]
    ).to_parquet(source, index=False)

    summary = prepare_unique_aime24(source, output, expected_problems=3)
    deduplicated = pandas.read_parquet(output)

    assert summary["source_rows"] == 12
    assert summary["unique_problems"] == 3
    assert summary["minimum_source_repetitions"] == 4
    assert summary["maximum_source_repetitions"] == 4
    assert len(deduplicated) == 3
    assert deduplicated["prompt"].map(canonical_prompt).nunique() == 3
