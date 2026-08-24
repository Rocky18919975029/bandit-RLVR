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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    vllm_runner = (WRAPPER_DIR / "run_lighteval_vllm.py").read_text(encoding="utf-8")
    launcher = (REPO_ROOT / "examples/grpo_trainer/submit_openr1_aime24_lighteval_h100.slurm").read_text(
        encoding="utf-8"
    )

    assert "EVAL_TEMPERATURE=0.6" in wrapper
    assert "EVAL_TOP_P=0.95" in wrapper
    assert "EVAL_N=64" in wrapper
    assert "EXPECTED_PROBLEMS=30" in wrapper
    assert "TASK_NAME='lighteval|aime24|0|0'" in wrapper
    assert 'name="aime24"' in task
    assert 'suite=["lighteval"]' in task
    assert "Metrics.math_pass_at_1_64n" in task
    assert "generation_size=32768" in task
    assert "HF_HUB_OFFLINE=1" in wrapper
    assert "HF_DATASETS_OFFLINE=1" in wrapper
    assert "TRANSFORMERS_OFFLINE=1" in wrapper
    assert "unset PYTHONPATH" in wrapper
    assert 'PYTHONPATH="${SCRIPT_DIR}" python "${SCRIPT_DIR}/run_lighteval_vllm.py"' in wrapper
    assert "--custom-tasks openr1_aime24_task" in wrapper
    assert '"trust_remote_code" in inspect.signature(load_dataset).parameters' in task
    assert "lighteval_task_module.download_dataset_worker = _offline_download_dataset_worker" in task
    assert "#SBATCH --gres=gpu:8" in launcher
    assert "DATA_PARALLEL_SIZE != ALLOCATED_GPUS" in launcher
    assert "tensor_parallel_size=1" in wrapper
    assert '"vllm_distributed_executor_backend": "uni"' in wrapper
    assert "executor=uni" in wrapper
    assert "enforce_eager=True" in wrapper
    assert 'self.model_args["enforce_eager"] = True' in vllm_runner
    assert 'self.model_args["distributed_executor_backend"] = "uni"' in vllm_runner
    assert "TokensPrompt(prompt_token_ids=token_ids)" in vllm_runner
    assert "llm.generate(prompts=prompts" in vllm_runner
    assert "VLLMModel._generate = generate" in vllm_runner
    assert "run_vllm(" in vllm_runner
    assert 'parser.add_argument("--max-samples", type=int)' in vllm_runner
    assert '"reportable": not bool("${SMOKE_MAX_SAMPLES}")' in wrapper
    assert "SMOKE TEST ONLY" in wrapper
    assert "VLLM_DISABLE_COMPILE_CACHE=1" in launcher
    assert "/tmp/vllm_${USER}_${SLURM_JOB_ID}" in launcher


def test_environment_setup_refuses_training_prefix():
    setup_script = (WRAPPER_DIR / "create_offline_env.sh").read_text(encoding="utf-8")
    assert 'if [ "${BASE_ENV_PATH}" = "${EVAL_ENV_PATH}" ]' in setup_script
    assert "conda create --offline" in setup_script
    assert '--clone "${BASE_ENV_PATH}"' in setup_script
    assert "--no-index" in setup_script


def test_vllm_011_compatibility_preserves_interleaved_order(monkeypatch):
    calls = {"llm_args": [], "prompt_shards": [], "shutdowns": 0}

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeTokensPrompt(dict):
        def __init__(self, *, prompt_token_ids):
            super().__init__(prompt_token_ids=list(prompt_token_ids))

    class FakeLLM:
        def __init__(self, **kwargs):
            calls["llm_args"].append(kwargs)

        def generate(self, *, prompts, sampling_params):
            calls["prompt_shards"].append(prompts)
            assert sampling_params.temperature == 0.6
            assert sampling_params.top_p == 0.95
            assert sampling_params.n == 1
            assert sampling_params.max_tokens == 128
            return [tuple(prompt["prompt_token_ids"]) for prompt in prompts]

    class FakeRemoteFunction:
        def __init__(self, function):
            self.function = function

        def remote(self, *args):
            return self.function(*args)

    class FakeRay:
        def remote(self, **options):
            assert options == {"num_gpus": 1}
            return FakeRemoteFunction

        def get(self, values):
            return values

        def shutdown(self):
            calls["shutdowns"] += 1

    class FakeVLLMModel:
        def _create_auto_model(self, config):
            # Match pinned LightEval: data parallel mode asks each outer Ray
            # worker to start another Ray-backed vLLM executor.
            self.model_args = {
                "model": "fake-model",
                "distributed_executor_backend": "ray",
            }
            return None

    def distribute(count, values):
        values = list(values)
        return [iter(values[index::count]) for index in range(count)]

    fake_runner_calls = []
    fake_main_vllm = ModuleType("lighteval.main_vllm")
    fake_main_vllm.vllm = lambda **kwargs: fake_runner_calls.append(kwargs)

    fake_vllm_model = ModuleType("lighteval.models.vllm.vllm_model")
    fake_vllm_model.VLLMModel = FakeVLLMModel
    fake_vllm_model.SamplingParams = FakeSamplingParams
    fake_vllm_model.ray = FakeRay()
    fake_vllm_model.distribute = distribute

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.__version__ = "0.11.0-test"
    fake_vllm_inputs = ModuleType("vllm.inputs")
    fake_vllm_inputs.TokensPrompt = FakeTokensPrompt

    for package_name in ("lighteval", "lighteval.models", "lighteval.models.vllm"):
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    monkeypatch.setitem(sys.modules, "lighteval.main_vllm", fake_main_vllm)
    monkeypatch.setitem(
        sys.modules,
        "lighteval.models.vllm.vllm_model",
        fake_vllm_model,
    )
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", fake_vllm_inputs)

    runner_path = WRAPPER_DIR / "run_lighteval_vllm.py"
    spec = importlib.util.spec_from_file_location("openr1_lighteval_runner_test", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    runner.enable_eager_vllm_for_data_parallel()

    model = FakeVLLMModel()
    config = SimpleNamespace(data_parallel_size=2, tensor_parallel_size=1)
    assert model._create_auto_model(config) is None
    assert model.model_args["enforce_eager"] is True
    assert model.model_args["distributed_executor_backend"] == "uni"

    model.data_parallel_size = 2
    model.tensor_parallel_size = 1
    model._config = SimpleNamespace(
        generation_parameters=SimpleNamespace(to_vllm_dict=lambda: {"temperature": 0.6, "top_p": 0.95, "seed": 42})
    )

    outputs = model._generate(
        [[11], [22], [33], [44]],
        max_new_tokens=128,
        num_samples=1,
    )

    assert outputs == [(11,), (22,), (33,), (44,)]
    assert calls["prompt_shards"] == [
        [{"prompt_token_ids": [11]}, {"prompt_token_ids": [33]}],
        [{"prompt_token_ids": [22]}, {"prompt_token_ids": [44]}],
    ]
    assert calls["llm_args"] == [
        {
            "model": "fake-model",
            "enforce_eager": True,
            "distributed_executor_backend": "uni",
        },
        {
            "model": "fake-model",
            "enforce_eager": True,
            "distributed_executor_backend": "uni",
        },
    ]
    assert calls["shutdowns"] == 1

    runner.audit_vllm_runtime()
