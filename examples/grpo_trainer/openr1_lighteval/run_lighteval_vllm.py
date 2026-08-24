# SPDX-License-Identifier: Apache-2.0
"""Invoke pinned LightEval with reliable multi-replica vLLM settings."""

import argparse
import itertools

import lighteval.models.vllm.vllm_model as vllm_model_module
from lighteval.main_vllm import vllm as run_vllm
from lighteval.models.vllm.vllm_model import VLLMModel


def enable_eager_vllm_for_data_parallel() -> None:
    """Bridge the pinned LightEval backend to vLLM 0.11."""
    original_create_auto_model = VLLMModel._create_auto_model

    def create_auto_model(self, config):
        model = original_create_auto_model(self, config)
        if config.data_parallel_size <= 1:
            raise RuntimeError("This wrapper requires data_parallel_size > 1")
        self.model_args["enforce_eager"] = True
        return model

    VLLMModel._create_auto_model = create_auto_model

    def generate(
        self,
        inputs,
        max_new_tokens=None,
        stop_tokens=None,
        returns_logits=False,
        num_samples=1,
        generate=True,
    ):
        sampling_params = vllm_model_module.SamplingParams(**self._config.generation_parameters.to_vllm_dict())
        if generate:
            sampling_params.n = num_samples
            sampling_params.max_tokens = max_new_tokens
            sampling_params.stop = stop_tokens
            sampling_params.logprobs = 1 if returns_logits else 0
        else:
            sampling_params.temperature = 0
            sampling_params.prompt_logprobs = 1
            sampling_params.max_tokens = 1
            sampling_params.detokenize = False

        if self.data_parallel_size > 1:

            @vllm_model_module.ray.remote(num_gpus=1 if self.tensor_parallel_size == 1 else None)
            def run_inference_one_model(model_args, worker_sampling_params, requests):
                from vllm import LLM
                from vllm.inputs import TokensPrompt

                llm = LLM(**model_args)
                prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in requests]
                return llm.generate(prompts=prompts, sampling_params=worker_sampling_params)

            requests = [list(shard) for shard in vllm_model_module.distribute(self.data_parallel_size, inputs)]
            remote_inputs = ((self.model_args, sampling_params, request) for request in requests)
            object_refs = [run_inference_one_model.remote(*remote_input) for remote_input in remote_inputs]
            results = vllm_model_module.ray.get(object_refs)
            vllm_model_module.ray.shutdown()
            return [
                output
                for output in itertools.chain.from_iterable(
                    itertools.zip_longest(*[list(result) for result in results])
                )
                if output is not None
            ]

        from vllm.inputs import TokensPrompt

        prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in inputs]
        return self.model.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=True)

    VLLMModel._generate = generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-args", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--custom-tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    enable_eager_vllm_for_data_parallel()
    run_vllm(
        model_args=args.model_args,
        tasks=args.tasks,
        custom_tasks=args.custom_tasks,
        use_chat_template=True,
        save_details=True,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
