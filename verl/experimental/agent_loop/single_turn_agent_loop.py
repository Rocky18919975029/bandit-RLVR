# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template and tokenize
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        prefix_ids = kwargs.get("hpf_prefix_ids")
        if prefix_ids is not None:
            if hasattr(prefix_ids, "tolist"):
                prefix_ids = prefix_ids.tolist()
            prefix_ids = [int(token_id) for token_id in prefix_ids]
        generation_prompt_ids = prompt_ids + (prefix_ids or [])

        replay_response_ids = kwargs.get("sir_replay_response_token_ids")
        replay_response_log_probs = kwargs.get("sir_replay_response_log_probs")
        if replay_response_ids is not None:
            if prefix_ids:
                raise ValueError("Exact SIR initial-rollout replay cannot be combined with a retained prefix")
            if replay_response_log_probs is None:
                raise ValueError("Exact SIR initial-rollout replay requires saved behavior log-probabilities")
            if hasattr(replay_response_ids, "tolist"):
                replay_response_ids = replay_response_ids.tolist()
            if hasattr(replay_response_log_probs, "tolist"):
                replay_response_log_probs = replay_response_log_probs.tolist()
            replay_response_ids = [int(token_id) for token_id in replay_response_ids]
            replay_response_log_probs = [float(log_prob) for log_prob in replay_response_log_probs]
            if not replay_response_ids:
                raise ValueError("Exact SIR initial-rollout replay received an empty response")
            if len(replay_response_ids) != len(replay_response_log_probs):
                raise ValueError(
                    "Exact SIR initial-rollout replay token/log-probability length mismatch: "
                    f"{len(replay_response_ids)} != {len(replay_response_log_probs)}"
                )
            if len(replay_response_ids) > self.response_length:
                raise ValueError(
                    f"Replayed response length {len(replay_response_ids)} exceeds "
                    f"rollout response_length={self.response_length}"
                )
            replay_output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=replay_response_ids,
                response_mask=[1] * len(replay_response_ids),
                response_logprobs=replay_response_log_probs,
                multi_modal_data=multi_modal_data,
                mm_processor_kwargs=mm_processor_kwargs,
                num_turns=2,
                metrics=AgentLoopMetrics(),
                extra_fields={"turn_scores": [], "tool_rewards": [], "sir_exact_replay": True},
            )
            return replay_output

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=generation_prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
                audio_data=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_ids = (prefix_ids or []) + output.token_ids
        response_mask = [1] * len(response_ids)
        response_logprobs = None
        if output.log_probs:
            response_logprobs = ([0.0] * len(prefix_ids or [])) + output.log_probs

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs is not None else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
