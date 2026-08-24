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

from pathlib import Path

import pytest

from examples.grpo_trainer.openr1_lighteval.prepare_aime24_dataset import prepare_lighteval_aime24

REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_DIR = REPO_ROOT / "examples" / "grpo_trainer" / "openr1_lighteval"


def test_prepare_offline_dataset_deduplicates_and_extracts_chat_prompts(tmp_path):
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    source = tmp_path / "aime24.parquet"
    output_dir = tmp_path / "dataset"
    pandas.DataFrame(
        [
            {
                "prompt": [{"role": "system", "content": "system"}, {"role": "user", "content": problem}],
                "reward_model": {"style": "rule", "ground_truth": answer},
            }
            for problem, answer in (("problem zero", "0"), ("problem one", "1"), ("problem two", "2"))
            for _ in range(4)
        ]
    ).to_parquet(source, index=False)

    summary = prepare_lighteval_aime24(source, output_dir, expected_problems=3)
    result = pandas.read_parquet(output_dir / "train.parquet")

    assert summary["source_rows"] == 12
    assert summary["unique_problems"] == 3
    assert result.to_dict(orient="records") == [
        {"problem": "problem zero", "answer": "0"},
        {"problem": "problem one", "answer": "1"},
        {"problem": "problem two", "answer": "2"},
    ]
    assert (output_dir / "dataset_manifest.json").is_file()


def test_prepare_offline_dataset_rejects_conflicting_duplicate_answers(tmp_path):
    pandas = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    source = tmp_path / "aime24.parquet"
    pandas.DataFrame(
        [
            {"problem": "same problem", "answer": "1"},
            {"problem": "same problem", "answer": "2"},
        ]
    ).to_parquet(source, index=False)

    with pytest.raises(ValueError, match="Conflicting answers"):
        prepare_lighteval_aime24(source, tmp_path / "dataset", expected_problems=1)


def test_wrapper_protocol_is_fixed_and_offline():
    wrapper = (WRAPPER_DIR / "run_aime24_offline.sh").read_text(encoding="utf-8")
    task = (WRAPPER_DIR / "openr1_aime24_task.py").read_text(encoding="utf-8")
    launcher = (REPO_ROOT / "examples/grpo_trainer/submit_openr1_aime24_lighteval_h100.slurm").read_text(
        encoding="utf-8"
    )

    assert "EVAL_TEMPERATURE=0.6" in wrapper
    assert "EVAL_TOP_P=0.95" in wrapper
    assert "EVAL_N=64" in wrapper
    assert "EXPECTED_PROBLEMS=30" in wrapper
    assert "Metrics.math_pass_at_1_64n" in task
    assert "generation_size=32768" in task
    assert "HF_HUB_OFFLINE=1" in wrapper
    assert "HF_DATASETS_OFFLINE=1" in wrapper
    assert "TRANSFORMERS_OFFLINE=1" in wrapper
    assert "unset PYTHONPATH" in wrapper
    assert 'PYTHONPATH="${SCRIPT_DIR}" lighteval vllm' in wrapper
    assert "--custom-tasks openr1_aime24_task" in wrapper
    assert '"trust_remote_code" in inspect.signature(load_dataset).parameters' in task
    assert "lighteval_task_module.download_dataset_worker = _offline_download_dataset_worker" in task
    assert "#SBATCH --gres=gpu:8" in launcher
    assert "DATA_PARALLEL_SIZE != ALLOCATED_GPUS" in launcher
    assert "tensor_parallel_size=1" in wrapper


def test_environment_setup_refuses_training_prefix():
    setup_script = (WRAPPER_DIR / "create_offline_env.sh").read_text(encoding="utf-8")
    assert 'if [ "${BASE_ENV_PATH}" = "${EVAL_ENV_PATH}" ]' in setup_script
    assert "conda create --offline" in setup_script
    assert '--clone "${BASE_ENV_PATH}"' in setup_script
    assert "--no-index" in setup_script
