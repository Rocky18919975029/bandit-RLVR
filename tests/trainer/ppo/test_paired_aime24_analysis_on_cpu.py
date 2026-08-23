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

import json

import pytest

from examples.grpo_trainer.analyze_paired_aime24_runs import RunSpec, analyze_runs, write_results


def _write_run(output_dir, *, variant, seed, correct_counts, samples=4, temperature=1.0, top_p=1.0):
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    correct_total = sum(correct_counts)
    summary = {
        "problem_count": len(correct_counts),
        "total_completion_count": len(correct_counts) * samples,
        "samples_per_problem": [samples],
        "correct_completion_count": correct_total,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "strict_boxed_verifier": True,
        "model_path": f"/{variant}",
        "metrics": {"accuracy": correct_total / (len(correct_counts) * samples)},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary))
    with (raw_dir / "0.jsonl").open("w") as handle:
        for problem_index, correct_count in enumerate(correct_counts):
            for sample_index in range(samples):
                handle.write(
                    json.dumps(
                        {
                            "data_source": "aime24",
                            "uid": f"random-{variant}-{seed}-{problem_index}",
                            "input": f"problem {problem_index}",
                            "gts": str(problem_index),
                            "acc": sample_index < correct_count,
                            "score": 1.0 if sample_index < correct_count else -1.0,
                        }
                    )
                    + "\n"
                )


def test_pooled_paired_analysis_uses_stable_problem_content(tmp_path):
    variants = ("control", "tempered")
    seeds = [42, 43]
    count_matrix = {
        ("control", 42): [0, 2, 4],
        ("control", 43): [0, 2, 4],
        ("tempered", 42): [1, 3, 4],
        ("tempered", 43): [1, 3, 4],
    }
    specs = []
    for (variant, seed), counts in count_matrix.items():
        output_dir = tmp_path / f"{variant}-{seed}"
        _write_run(
            output_dir,
            variant=variant,
            seed=seed,
            correct_counts=counts,
            temperature=0.6,
            top_p=0.95,
        )
        specs.append(RunSpec(variant, seed, output_dir))

    summary, per_problem, per_seed = analyze_runs(
        specs,
        variants=variants,
        expected_seeds=seeds,
        expected_problems=3,
        expected_samples_per_run=4,
        pass_ks=[1, 2, 4, 8],
        bootstrap_samples=2_000,
        bootstrap_seed=7,
        expected_temperature=0.6,
        expected_top_p=0.95,
    )

    assert summary["pooled_samples_per_problem_per_variant"] == 8
    assert summary["temperature"] == 0.6
    assert summary["top_p"] == 0.95
    assert summary["metrics"]["accuracy"]["control"] == pytest.approx(0.5)
    assert summary["metrics"]["accuracy"]["tempered"] == pytest.approx(2 / 3)
    assert summary["metrics"]["accuracy"]["tempered_minus_control"] == pytest.approx(1 / 6)
    assert summary["metrics"]["accuracy"]["paired_bootstrap_ci_low"] >= 0
    assert summary["problem_correct_count_comparison"] == {
        "tempered_greater": 2,
        "control_greater": 0,
        "tied": 1,
        "control_zero_tempered_positive": 1,
        "tempered_zero_control_positive": 0,
        "both_zero": 0,
        "both_positive": 2,
    }
    assert len(per_problem) == 3
    assert len(per_seed) == 4

    result_dir = tmp_path / "result"
    write_results(result_dir, summary, per_problem, per_seed)
    assert (result_dir / "summary.json").is_file()
    assert (result_dir / "summary.csv").is_file()
    assert (result_dir / "per_problem.csv").is_file()
    assert (result_dir / "per_seed.csv").is_file()


def test_paired_analysis_rejects_problem_set_mismatch(tmp_path):
    specs = []
    for variant in ("control", "tempered"):
        output_dir = tmp_path / variant
        _write_run(output_dir, variant=variant, seed=42, correct_counts=[1, 1, 1])
        specs.append(RunSpec(variant, 42, output_dir))

    raw_path = specs[-1].output_dir / "raw" / "0.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    rows[-1]["input"] = "different problem"
    raw_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="raw dump has 4 problems"):
        analyze_runs(
            specs,
            variants=("control", "tempered"),
            expected_seeds=[42],
            expected_problems=3,
            expected_samples_per_run=4,
            pass_ks=[1],
            bootstrap_samples=100,
            bootstrap_seed=7,
        )
