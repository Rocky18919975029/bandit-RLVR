# SPDX-License-Identifier: Apache-2.0
"""Invoke pinned LightEval with reliable multi-replica vLLM settings."""

import argparse

from lighteval.main_vllm import vllm as run_vllm
from lighteval.models.vllm.vllm_model import VLLMModel


def enable_eager_vllm_for_data_parallel() -> None:
    """Add an LLM kwarg omitted by this pinned LightEval model config."""
    original_create_auto_model = VLLMModel._create_auto_model

    def create_auto_model(self, config):
        model = original_create_auto_model(self, config)
        if config.data_parallel_size <= 1:
            raise RuntimeError("This wrapper requires data_parallel_size > 1")
        self.model_args["enforce_eager"] = True
        return model

    VLLMModel._create_auto_model = create_auto_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-args", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--custom-tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    enable_eager_vllm_for_data_parallel()
    run_vllm(
        model_args=args.model_args,
        tasks=args.tasks,
        custom_tasks=args.custom_tasks,
        use_chat_template=True,
        save_details=True,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
