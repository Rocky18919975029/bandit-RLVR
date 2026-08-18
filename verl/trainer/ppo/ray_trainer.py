# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import math
import os
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.hpf_schedule import get_hpf_role_phase
from verl.trainer.ppo.hpf_utils import (
    build_hpf_corrected_leader_batch,
    build_hpf_fresh_leader_batch,
    build_hpf_masked_batches,
    build_hpf_mixed_policy_grpo_batch,
    build_hpf_transition_prefix_plan,
    configure_hpf_transition_behavior_log_probs,
    estimate_hpf_transition_return,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.sir_resampling import (
    SIRSelectionPlan,
    build_branched_prefix_plan,
    build_sir_selection_plan,
)
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    create_rl_dataset,
    create_rl_sampler,
    need_critic,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import deprecated, load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.skip.skip_manager import SkipManager
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_spec_decode_metrics(
    spec_drafts,
    spec_accepts,
    spec_verifies,
    non_padding_mask=None,
) -> dict:
    """Aggregate per-request speculative decoding stats.

    Ratios are computed per request and then averaged, so long and short
    responses have equal metric weight.

    The three inputs come from the rollout engine (vLLM request spec-decode
    stats or sglang ``meta_info["spec_*"]`` keys). Either all three are ``None``
    (caller didn't fetch them, e.g. spec rollout disabled) and the function
    is a no-op, or all three are populated; mixed state is a programmer error.

    ``non_padding_mask`` is a numpy bool array used by sync PPO to drop padded
    placeholder samples; pass ``None`` for async PPO.
    """
    if spec_drafts is None and spec_accepts is None and spec_verifies is None:
        return {}
    assert spec_drafts is not None and spec_accepts is not None and spec_verifies is not None, (
        "spec_decode metrics require all three of spec_num_draft_tokens / "
        "spec_num_accepted_tokens / spec_num_verify_steps; got partial inputs"
    )

    drafts = spec_drafts.tolist() if hasattr(spec_drafts, "tolist") else list(spec_drafts)
    accepts = spec_accepts.tolist() if hasattr(spec_accepts, "tolist") else list(spec_accepts)
    verifies = spec_verifies.tolist() if hasattr(spec_verifies, "tolist") else list(spec_verifies)

    if non_padding_mask is not None:
        drafts = [d for d, keep in zip(drafts, non_padding_mask, strict=True) if keep]
        accepts = [a for a, keep in zip(accepts, non_padding_mask, strict=True) if keep]
        verifies = [v for v, keep in zip(verifies, non_padding_mask, strict=True) if keep]

    if len(drafts) == 0:
        return {}

    # Treat zero-denominator samples as 0.0 and keep them in the mean.
    per_sample_accept_rate = [(a / d) if d > 0 else 0.0 for a, d in zip(accepts, drafts, strict=True)]
    per_sample_accept_length = [(1.0 + a / v) if v > 0 else 0.0 for a, v in zip(accepts, verifies, strict=True)]

    n = len(drafts)
    return {
        "rollout/spec_accept_rate": float(sum(per_sample_accept_rate) / n),
        "rollout/spec_accept_length": float(sum(per_sample_accept_length) / n),
    }


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # old_log_probs needed for path-variance proxy: w_t = 1 - 2*exp(old_log_probs) + sum_pi_squared
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


@deprecated("Legacy trainer is deprecated, and wil be removed in v0.9.0. Please use `trainer.use_v1=True` instead.")
class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None
        self._init_dump_executor()

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        hpf_config = self.config.algorithm.get("hpf_rlvr", {})
        role_phased_config = hpf_config.get("role_phased_training", {})
        role_phased_training = self._parse_hpf_bool(role_phased_config.get("enable", False), False)
        mixed_policy_config = hpf_config.get("mixed_policy_grpo", {})
        mixed_policy_grpo = self._parse_hpf_bool(mixed_policy_config.get("enable", False), False)
        transition_rollout_config = mixed_policy_config.get("transition_aware_rollout", {})
        transition_aware_rollout = self._parse_hpf_bool(
            transition_rollout_config.get("enable", False), False
        )
        transition_optimization_config = mixed_policy_config.get("transition_aware_optimization", {})
        transition_aware_optimization = self._parse_hpf_bool(
            transition_optimization_config.get("enable", False), False
        )
        if mixed_policy_grpo:
            tree_config = hpf_config.get("tree_rollout", {})
            if not self._parse_hpf_bool(hpf_config.get("enable", False), False) or not self._parse_hpf_bool(
                tree_config.get("enable", False), False
            ):
                raise ValueError("HPF mixed-policy GRPO requires HPF and tree rollout to be enabled.")
            if int(tree_config.get("num_suffixes", 1)) != 1:
                raise ValueError("HPF mixed-policy GRPO requires tree_rollout.num_suffixes=1.")
            if float(tree_config.get("prefix_top_p", 1.0)) != 1.0 or float(
                tree_config.get("suffix_top_p", 1.0)
            ) != 1.0:
                raise ValueError(
                    "HPF mixed-policy GRPO requires prefix_top_p=suffix_top_p=1.0 so the training "
                    "policy exactly matches the temperature-scaled rollout policy."
                )
            if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
                raise ValueError("HPF mixed-policy GRPO requires algorithm.adv_estimator=grpo.")
            if self._parse_hpf_bool(self.config.algorithm.get("use_kl_in_reward", False), False):
                raise ValueError("HPF mixed-policy GRPO requires algorithm.use_kl_in_reward=False.")
            if self._parse_hpf_bool(self.config.actor_rollout_ref.actor.get("use_kl_loss", False), False):
                raise ValueError("HPF mixed-policy GRPO is a pure GRPO update and requires actor.use_kl_loss=False.")
            if self._parse_hpf_bool(hpf_config.get("fresh_leader_tree", False), False):
                raise ValueError("HPF mixed-policy GRPO is a one-update path and requires fresh_leader_tree=False.")
            if role_phased_training:
                raise ValueError("HPF mixed-policy GRPO cannot be combined with role-phased training.")
            local_window_config = hpf_config.get("local_update_window", {})
            if self._parse_hpf_bool(local_window_config.get("enable", False), False):
                raise ValueError(
                    "HPF mixed-policy GRPO defines its own prefix-plus-suffix window; "
                    "disable local_update_window."
                )
            suffix_window_size = mixed_policy_config.get("suffix_window_size", None)
            if suffix_window_size is not None and str(suffix_window_size).lower() != "null":
                if int(suffix_window_size) <= 0:
                    raise ValueError(
                        "HPF mixed-policy GRPO suffix_window_size must be positive or null, "
                        f"got {suffix_window_size}."
                    )
            if transition_aware_rollout and int(tree_config.get("num_suffixes", 1)) != 1:
                raise ValueError(
                    "HPF transition-aware rollout requires tree_rollout.num_suffixes=1; "
                    "it produces one paired low-temperature completion per sampled prefix."
                )
            if transition_aware_rollout and not self.config.trainer.get("rollout_data_dir", None):
                raise ValueError(
                    "HPF transition-aware rollout requires trainer.rollout_data_dir so both cut batches are saved."
                )
            if transition_aware_optimization:
                if not transition_aware_rollout:
                    raise ValueError(
                        "HPF transition-aware mixed policy optimization requires transition-aware rollout."
                    )
                if self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla") != "vanilla":
                    raise ValueError(
                        "HPF transition-aware mixed policy optimization requires the vanilla clipped PPO loss."
                    )
                lambda_trans = float(transition_optimization_config.get("lambda_trans", 1.0))
                if not math.isfinite(lambda_trans) or lambda_trans < 0:
                    raise ValueError("HPF transition lambda_trans must be finite and non-negative.")
                diagnostics_config = transition_optimization_config.get("diagnostics", {})
                if self._parse_hpf_bool(diagnostics_config.get("enable", False), False):
                    sample_pairs = int(diagnostics_config.get("sample_pairs", 8))
                    if sample_pairs < 0:
                        raise ValueError("HPF transition diagnostic sample_pairs must be non-negative.")
                    if not self.config.trainer.get("rollout_data_dir", None):
                        raise ValueError("HPF transition diagnostics require trainer.rollout_data_dir.")
                if self._parse_hpf_bool(
                    self.config.actor_rollout_ref.actor.get("use_dynamic_bsz", False), False
                ):
                    raise ValueError(
                        "HPF transition-aware mixed policy optimization requires actor.use_dynamic_bsz=False "
                        "so paired cut rows stay together in each micro-batch."
                    )
                micro_batch_size_value = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
                if micro_batch_size_value is None:
                    raise ValueError(
                        "HPF transition-aware mixed policy optimization requires "
                        "actor.ppo_micro_batch_size_per_gpu to be set explicitly."
                    )
                micro_batch_size = int(micro_batch_size_value)
                if micro_batch_size < 2 or micro_batch_size % 2 != 0:
                    raise ValueError(
                        "HPF transition-aware mixed policy optimization requires an even "
                        "actor.ppo_micro_batch_size_per_gpu of at least 2 so both cuts are present "
                        f"in every micro-batch; got {micro_batch_size}."
                    )
                if float(self.config.actor_rollout_ref.actor.get("entropy_coeff", 0.0)) != 0.0:
                    raise ValueError(
                        "HPF transition-aware mixed policy optimization currently requires entropy_coeff=0."
                    )
        elif transition_aware_rollout or transition_aware_optimization:
            raise ValueError(
                "HPF transition-aware mixed policy optimization and rollout require "
                "mixed_policy_grpo.enable=True."
            )
        if role_phased_training:
            follower_epochs = int(role_phased_config.get("follower_epochs", 1))
            leader_epochs = int(role_phased_config.get("leader_epochs", 1))
            if follower_epochs <= 0 or leader_epochs <= 0:
                raise ValueError(
                    "hpf_rlvr.role_phased_training follower_epochs and leader_epochs must both be positive; "
                    f"got follower_epochs={follower_epochs}, leader_epochs={leader_epochs}."
                )
            tree_config = hpf_config.get("tree_rollout", {})
            if not self._parse_hpf_bool(hpf_config.get("enable", False), False) or not self._parse_hpf_bool(
                tree_config.get("enable", False), False
            ):
                raise ValueError("HPF role-phased training requires HPF and tree rollout to be enabled.")
            if not self._parse_hpf_bool(hpf_config.get("fresh_leader_tree", False), False):
                raise ValueError("HPF role-phased training requires hpf_rlvr.fresh_leader_tree=True.")
            if str(hpf_config.get("horizon_schedule", "epoch")).lower() != "epoch":
                raise ValueError("HPF role-phased training requires hpf_rlvr.horizon_schedule='epoch'.")
            phase_batch_size = int(self.config.data.get("gen_batch_size", self.config.data.train_batch_size))
            dropped_examples = len(self.train_dataset) % phase_batch_size
            if dropped_examples:
                raise ValueError(
                    "HPF role-phased training requires complete train-set passes, but the dataset size is not "
                    f"divisible by the dataloader batch size: dataset={len(self.train_dataset)}, "
                    f"batch_size={phase_batch_size}, remainder={dropped_examples}."
                )
            total_training_steps *= follower_epochs + leader_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    @staticmethod
    def _write_sir_pool(rows: list[dict[str, Any]], dump_path: str, global_step: int) -> None:
        """Write one JSON object per prompt, containing its complete SIR pool."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_step}.jsonl")
        with open(filename, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"[SIR] Dumped full rollout pool to {filename}", flush=True)

    def _dump_sir_pool(
        self,
        pool_batch: DataProto,
        plan: SIRSelectionPlan,
        *,
        dump_path: str,
        dump_token_log_probs: bool,
    ) -> None:
        """Materialize a compact, group-oriented audit record before resampling."""
        response_mask = pool_batch.batch["response_mask"].detach().cpu().bool().numpy()
        responses = pool_batch.batch["responses"].detach().cpu().numpy()
        rollout_log_probs = pool_batch.batch["rollout_log_probs"].detach().cpu().float().numpy()
        response_attention_mask = pool_batch.batch["attention_mask"][:, -responses.shape[1] :]
        response_attention_mask = response_attention_mask.detach().cpu().bool().numpy()
        prompts = pool_batch.batch["prompts"][:: plan.pool_size].detach().cpu()
        decoded_prompts = self.tokenizer.batch_decode(prompts, skip_special_tokens=True)

        eos_ids: set[int] = set()
        for value in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "eos_token_ids", None),
        ):
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                eos_ids.update(int(item) for item in value)
            else:
                eos_ids.add(int(value))

        rows: list[dict[str, Any]] = []
        for group in plan.groups:
            start = group.group_index * plan.pool_size
            first_item = pool_batch[start]
            reward_model = first_item.non_tensor_batch.get("reward_model", {})
            ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
            data_source = first_item.non_tensor_batch.get("data_source")
            prompt_uid = str(first_item.non_tensor_batch.get("uid", group.group_index))
            candidates = []
            for local_index in range(plan.pool_size):
                row_index = start + local_index
                sampled_positions = response_mask[row_index]
                valid_response_positions = response_attention_mask[row_index]
                sampled_token_ids = responses[row_index][sampled_positions].astype(np.int64).tolist()
                response_token_ids = responses[row_index][valid_response_positions].astype(np.int64).tolist()
                sampled_log_probs = rollout_log_probs[row_index][sampled_positions].astype(np.float64).tolist()
                score = (
                    float(pool_batch.batch["rm_scores"][row_index].sum().detach().cpu().item())
                    if "rm_scores" in pool_batch.batch.keys()
                    else None
                )
                candidate = {
                    "pool_index": local_index,
                    "request_id": (
                        str(pool_batch.non_tensor_batch["request_id"][row_index])
                        if "request_id" in pool_batch.non_tensor_batch
                        else None
                    ),
                    "response": self.tokenizer.decode(response_token_ids, skip_special_tokens=True),
                    "response_token_ids": response_token_ids,
                    "sampled_token_ids": sampled_token_ids,
                    "sampled_token_count": len(sampled_token_ids),
                    "ends_with_eos": bool(sampled_token_ids and sampled_token_ids[-1] in eos_ids),
                    "score": score,
                    "prefix_joint_log_prob": float(group.prefix_joint_log_probs[local_index]),
                    "sir_weight": float(group.weights[local_index]),
                    "selected_count": int(group.selected_counts[local_index]),
                    "selected_draws": list(group.selected_draws[local_index]),
                }
                for metadata_key in (
                    "sir_pool_origin",
                    "sir_parent_index",
                    "sir_branch_index",
                    "sir_cut_position",
                    "sir_cut_with_replacement",
                ):
                    if metadata_key in pool_batch.non_tensor_batch:
                        value = pool_batch.non_tensor_batch[metadata_key][row_index]
                        candidate[metadata_key] = value.item() if hasattr(value, "item") else value
                if dump_token_log_probs:
                    candidate["sampled_token_log_probs"] = sampled_log_probs
                for reward_key in ("acc", "pred", "overlong_reward", "overlong"):
                    if reward_key in pool_batch.non_tensor_batch:
                        value = pool_batch.non_tensor_batch[reward_key][row_index]
                        candidate[reward_key] = value.item() if hasattr(value, "item") else value
                candidates.append(candidate)

            rows.append(
                {
                    "step": self.global_steps,
                    "prompt_index": group.group_index,
                    "prompt_uid": prompt_uid,
                    "data_source": data_source,
                    "prompt": decoded_prompts[group.group_index],
                    "ground_truth": ground_truth,
                    "pool_size": plan.pool_size,
                    "pool_mode": str(self.config.algorithm.sir.get("pool_mode", "independent")),
                    "selected_count": plan.selected_count,
                    "block_length": plan.block_length,
                    "alpha": plan.alpha,
                    "resampling": "weighted_without_replacement",
                    "sir_seed": group.seed,
                    "sir_ess": group.effective_sample_size,
                    "selected_pool_indices": group.selected_local_indices.tolist(),
                    "candidates": candidates,
                }
            )

        future = self._dump_executor.submit(self._write_sir_pool, rows, dump_path, self.global_steps)
        self._dump_futures.append(future)
        still_pending = []
        for pending in self._dump_futures:
            if pending.done():
                pending.result()
            else:
                still_pending.append(pending)
        self._dump_futures = still_pending

    def _generate_sir_branched_pool(
        self,
        gen_batch: DataProto,
        initial_output: DataProto,
        *,
        pool_size: int,
        initial_count: int,
        block_length: int,
        seed: int,
    ) -> tuple[DataProto, dict[str, float]]:
        """Expand K full initial rollouts into an N-way pool from random prefixes."""
        expected_initial = len(gen_batch) * initial_count
        if len(initial_output) != expected_initial:
            raise ValueError(
                "branched SIR initial rollout cardinality mismatch: "
                f"expected {expected_initial}, got {len(initial_output)}"
            )
        for required_key in ("responses", "response_mask", "rollout_log_probs"):
            if required_key not in initial_output.batch:
                raise ValueError(f"branched SIR initial rollout is missing {required_key}")

        response_lengths = (
            initial_output.batch["response_mask"].detach().cpu().bool().sum(dim=-1).numpy().astype(np.int64)
        )
        prefix_plan = build_branched_prefix_plan(
            response_lengths,
            pool_size=pool_size,
            initial_count=initial_count,
            block_length=block_length,
            seed=seed,
            global_step=self.global_steps,
        )
        initial_token_ids = self._extract_response_token_ids(initial_output)
        initial_prompt_batch = gen_batch.repeat(repeat_times=initial_count, interleave=True)
        branch_source = initial_prompt_batch.select_idxs(prefix_plan.parent_global_indices)
        branch_prefix_ids = [
            initial_token_ids[parent_index][:cut_position]
            for parent_index, cut_position in zip(
                prefix_plan.parent_global_indices.tolist(),
                prefix_plan.cut_positions.tolist(),
                strict=True,
            )
        ]
        branch_source.non_tensor_batch["hpf_prefix_ids"] = self._object_array(branch_prefix_ids)
        max_response_length = int(self.config.data.max_response_length)
        branch_source.non_tensor_batch["__max_tokens__"] = (
            max_response_length - prefix_plan.cut_positions
        ).astype(np.int32)
        branch_source.meta_info["temperature"] = float(self.config.actor_rollout_ref.rollout.temperature)
        branch_source.meta_info["top_p"] = float(self.config.actor_rollout_ref.rollout.top_p)
        branch_source.meta_info["logprobs"] = True

        rollout_worker_divisor = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
        branch_source_padded, branch_pad_size = pad_dataproto_to_divisor(
            branch_source, rollout_worker_divisor
        )
        branch_start = time.perf_counter()
        print(
            "[SIR] branched suffix rollout start "
            f"step={self.global_steps} initial={len(initial_output)} branches={len(branch_source)} "
            f"cuts_per_initial={prefix_plan.branches_per_initial} pad={branch_pad_size}",
            flush=True,
        )
        branch_output = self.async_rollout_manager.generate_sequences(branch_source_padded)
        branch_output = unpad_dataproto(branch_output, branch_pad_size)
        branch_elapsed = time.perf_counter() - branch_start
        if len(branch_output) != len(prefix_plan.cut_positions):
            raise ValueError(
                "branched SIR suffix rollout cardinality mismatch: "
                f"expected {len(prefix_plan.cut_positions)}, got {len(branch_output)}"
            )
        if "rollout_log_probs" not in branch_output.batch:
            raise ValueError("branched SIR suffix rollout did not return rollout_log_probs")

        branch_responses = branch_output.batch["responses"]
        branch_log_probs = branch_output.batch["rollout_log_probs"]
        initial_responses = initial_output.batch["responses"]
        initial_log_probs = initial_output.batch["rollout_log_probs"]
        for branch_row, (parent_row, cut_position) in enumerate(
            zip(
                prefix_plan.parent_global_indices.tolist(),
                prefix_plan.cut_positions.tolist(),
                strict=True,
            )
        ):
            if not torch.equal(
                branch_responses[branch_row, :cut_position],
                initial_responses[parent_row, :cut_position],
            ):
                raise ValueError(
                    "branched SIR suffix response does not preserve its requested prefix: "
                    f"branch={branch_row}, parent={parent_row}, cut={cut_position}"
                )
            branch_log_probs[branch_row, :cut_position] = initial_log_probs[parent_row, :cut_position]

        internal_sampling_keys = [
            key
            for key in (
                "hpf_prefix_ids",
                "__temperature__",
                "__top_p__",
                "__top_k__",
                "__max_tokens__",
                "__max_new_tokens__",
                "__logprobs__",
            )
            if key in branch_output.non_tensor_batch
        ]
        if internal_sampling_keys:
            branch_output.pop(non_tensor_batch_keys=internal_sampling_keys)

        initial_output.non_tensor_batch["sir_pool_origin"] = np.full(len(initial_output), "initial", dtype=object)
        initial_output.non_tensor_batch["sir_parent_index"] = np.tile(
            np.arange(initial_count, dtype=np.int32), len(gen_batch)
        )
        initial_output.non_tensor_batch["sir_branch_index"] = np.full(len(initial_output), -1, dtype=np.int32)
        initial_output.non_tensor_batch["sir_cut_position"] = np.full(len(initial_output), -1, dtype=np.int32)
        initial_output.non_tensor_batch["sir_cut_with_replacement"] = np.zeros(len(initial_output), dtype=bool)
        branch_output.non_tensor_batch["sir_pool_origin"] = np.full(len(branch_output), "branch", dtype=object)
        branch_output.non_tensor_batch["sir_parent_index"] = prefix_plan.parent_local_indices.astype(np.int32)
        branch_output.non_tensor_batch["sir_branch_index"] = prefix_plan.branch_local_indices.astype(np.int32)
        branch_output.non_tensor_batch["sir_cut_position"] = prefix_plan.cut_positions.astype(np.int32)
        branch_output.non_tensor_batch["sir_cut_with_replacement"] = prefix_plan.cut_with_replacement.copy()

        initial_timing = initial_output.meta_info.pop("timing", {})
        branch_timing = branch_output.meta_info.pop("timing", {})
        combined = DataProto.concat([initial_output, branch_output])
        branch_offset = len(initial_output)
        branches_per_prompt = initial_count * prefix_plan.branches_per_initial
        pool_order: list[int] = []
        for prompt_index in range(len(gen_batch)):
            initial_start = prompt_index * initial_count
            branch_start_index = branch_offset + prompt_index * branches_per_prompt
            pool_order.extend(range(initial_start, initial_start + initial_count))
            pool_order.extend(range(branch_start_index, branch_start_index + branches_per_prompt))
        pool_output = combined.select_idxs(np.asarray(pool_order, dtype=np.int64))
        if len(pool_output) != len(gen_batch) * pool_size:
            raise ValueError(
                f"branched SIR constructed {len(pool_output)} rows, expected {len(gen_batch) * pool_size}"
            )
        timing = {f"sir_branched/initial/{key}": value for key, value in initial_timing.items()}
        timing.update({f"sir_branched/branch/{key}": value for key, value in branch_timing.items()})
        pool_output.meta_info["timing"] = timing
        metrics = {
            "sir/branched_pool_enabled": 1.0,
            "sir/initial_rollouts": float(len(initial_output)),
            "sir/branch_rollouts": float(len(branch_output)),
            "sir/branches_per_initial": float(prefix_plan.branches_per_initial),
            "sir/cut_position_mean": float(prefix_plan.cut_positions.mean()),
            "sir/cut_position_min": float(prefix_plan.cut_positions.min()),
            "sir/cut_position_max": float(prefix_plan.cut_positions.max()),
            "sir/cut_with_replacement_fraction": float(prefix_plan.cut_with_replacement.mean()),
            "timing_s/sir/branch_rollout_wall": float(branch_elapsed),
        }
        print(
            "[SIR] branched rollout pool done "
            f"step={self.global_steps} prompts={len(gen_batch)} initial_per_prompt={initial_count} "
            f"branches_per_initial={prefix_plan.branches_per_initial} pool_per_prompt={pool_size} "
            f"repeated_cut_frac={float(prefix_plan.cut_with_replacement.mean()):.4f} "
            f"branch_elapsed_s={branch_elapsed:.2f}",
            flush=True,
        )
        return pool_output, metrics

    def _apply_sir_resampling(
        self,
        prompt_batch: DataProto,
        gen_batch_output: DataProto,
        *,
        pool_size: int,
        sir_config: Any,
    ) -> tuple[DataProto, dict[str, float]]:
        """Build a full prompt/response pool, audit it, and return K SIR draws."""
        pool_batch = prompt_batch.repeat(repeat_times=pool_size, interleave=True)
        pool_batch = pool_batch.union(gen_batch_output)
        if "response_mask" not in pool_batch.batch.keys():
            pool_batch.batch["response_mask"] = compute_response_mask(pool_batch)
        if "rollout_log_probs" not in pool_batch.batch.keys():
            raise ValueError(
                "SIR requires chosen-token rollout_log_probs; set "
                "actor_rollout_ref.rollout.calculate_log_probs=True"
            )
        if "uid" in pool_batch.non_tensor_batch:
            if len(pool_batch.non_tensor_batch["uid"]) % pool_size != 0:
                raise ValueError("SIR rollout row count is not divisible by the configured pool size N")
            grouped_uids = np.asarray(pool_batch.non_tensor_batch["uid"], dtype=object).reshape(-1, pool_size)
            if any(len(set(group.tolist())) != 1 for group in grouped_uids):
                raise ValueError("SIR requires each prompt's N rollout rows to remain contiguous")

        selected_count = int(sir_config.get("selected_count"))
        block_length = int(sir_config.get("block_length"))
        alpha = float(sir_config.get("alpha"))
        seed = int(sir_config.get("seed", 42))
        plan = build_sir_selection_plan(
            pool_batch.batch["rollout_log_probs"].detach().cpu().float().numpy(),
            pool_batch.batch["response_mask"].detach().cpu().numpy(),
            pool_size=pool_size,
            selected_count=selected_count,
            block_length=block_length,
            alpha=alpha,
            seed=seed,
            global_step=self.global_steps,
        )
        if len(plan.groups) != len(prompt_batch.batch):
            raise ValueError(
                f"SIR formed {len(plan.groups)} groups for {len(prompt_batch.batch)} prompts; "
                "rollout output ordering or cardinality changed unexpectedly"
            )

        if self._parse_hpf_bool(sir_config.get("dump_pool", True), True):
            dump_path = sir_config.get("dump_dir", None)
            if not dump_path:
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if not rollout_data_dir:
                    raise ValueError(
                        "SIR pool recording is enabled but neither algorithm.sir.dump_dir nor "
                        "trainer.rollout_data_dir is configured"
                    )
                dump_path = os.path.join(str(rollout_data_dir), "sir_pool")
            self._dump_sir_pool(
                pool_batch,
                plan,
                dump_path=str(dump_path),
                dump_token_log_probs=self._parse_hpf_bool(
                    sir_config.get("dump_token_log_probs", True), True
                ),
            )

        selected_batch = pool_batch.select_idxs(plan.selected_global_indices)
        selected_joint_log_probs = []
        selected_weights = []
        for group in plan.groups:
            selected_joint_log_probs.extend(group.prefix_joint_log_probs[group.selected_local_indices].tolist())
            selected_weights.extend(group.weights[group.selected_local_indices].tolist())
        selected_batch.non_tensor_batch["sir_pool_index"] = plan.selected_pool_indices.copy()
        selected_batch.non_tensor_batch["sir_draw_index"] = plan.selected_draw_indices.copy()
        selected_batch.non_tensor_batch["sir_group_index"] = np.repeat(
            np.arange(len(plan.groups), dtype=np.int64), plan.selected_count
        )
        selected_batch.non_tensor_batch["sir_prefix_joint_log_prob"] = np.asarray(
            selected_joint_log_probs, dtype=np.float64
        )
        selected_batch.non_tensor_batch["sir_weight"] = np.asarray(selected_weights, dtype=np.float64)
        selected_batch.meta_info["sir_pool_size"] = pool_size
        selected_batch.meta_info["sir_selected_count"] = selected_count

        print(
            "[SIR] resampled rollout groups "
            f"step={self.global_steps} prompts={len(plan.groups)} N={pool_size} K={selected_count} "
            f"B={block_length} alpha={alpha} ess_mean={plan.metrics()['sir/ess_mean']:.3f}",
            flush=True,
        )
        return selected_batch, plan.metrics()

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    @staticmethod
    def _shutdown_dataloader_iterator(iterator):
        """Best-effort shutdown for multiprocessing dataloader iterators."""
        if iterator is None:
            return
        candidates = [
            iterator,
            getattr(iterator, "_iterator", None),
            getattr(iterator, "_main_iter", None),
            getattr(iterator, "_wrapped_iterator", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            for method_name in ("_shutdown_workers", "shutdown", "close"):
                shutdown = getattr(candidate, method_name, None)
                if callable(shutdown):
                    shutdown()
                    return

    def _iterate_train_dataloader(self):
        iterator = iter(self.train_dataloader)
        try:
            yield from iterator
        finally:
            self._shutdown_dataloader_iterator(iterator)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = {
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in reward_extra_infos_dict.items()
            }
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )
            for key in (
                "hpf_transition_pair_uid",
                "hpf_transition_cut",
                "hpf_transition_prefix_horizon",
                "hpf_prefix_ids",
                "hpf_transition_suffix_ids",
                "sir_pool_index",
                "sir_draw_index",
                "sir_group_index",
                "sir_prefix_joint_log_prob",
                "sir_weight",
                "sir_pool_origin",
                "sir_parent_index",
                "sir_branch_index",
                "sir_cut_position",
                "sir_cut_with_replacement",
            ):
                if key in batch.non_tensor_batch:
                    reward_extra_infos_to_dump.setdefault(key, batch.non_tensor_batch[key].tolist())

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        validation_rollout_count = len(self.val_dataset) * int(
            self.config.actor_rollout_ref.rollout.val_kwargs.n
        )
        validation_progress = tqdm(
            total=validation_rollout_count,
            desc="Validation Rollouts",
            unit="rollout",
            dynamic_ncols=True,
        )

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            validation_progress.update(len(scores))
            acc_values = reward_extra_infos_dict.get("acc", [])
            if acc_values:
                validation_progress.set_postfix(
                    acc=f"{sum(bool(value) for value in acc_values) / len(acc_values):.3f}"
                )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        validation_progress.close()

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            dump_extra_infos = dict(reward_extra_infos_dict)
            dump_extra_infos["uid"] = list(sample_uids)
            dump_extra_infos["data_source"] = np.concatenate(data_source_lst, axis=0).tolist()
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=dump_extra_infos,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # convert critic_cfg into TrainingWorkerConfig for the unified model engine worker
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
                extra_context=getattr(self, "_critic_extra_context", {}),
            )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/verl-project/verl/blob/master/examples/tutorial/ray/tutorial.ipynb
        # for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # initialize teacher loop manager
        if self.use_teacher_policy:
            from verl.experimental.teacher_loop import MultiTeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = MultiTeacherModelManager(
                config=self.config,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        self.llm_server_manager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        # To stream teacher computation with actor rollout, we instead pass the full manager so that the
        # teacher loop workers can sleep/wake together with rollout workers
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        # Support custom CheckpointEngineManager via config
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        hpf_config = self.config.algorithm.get("hpf_rlvr", {})
        phased_config = hpf_config.get("role_phased_training", {})
        if self._parse_hpf_bool(phased_config.get("enable", False), False):
            torch.save(
                {
                    "enabled": True,
                    "follower_epochs": int(phased_config.get("follower_epochs", 1)),
                    "leader_epochs": int(phased_config.get("leader_epochs", 1)),
                    "batches_per_pass": len(self.train_dataloader),
                },
                os.path.join(local_global_step_folder, "hpf_role_phase.pt"),
            )

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        hpf_config = self.config.algorithm.get("hpf_rlvr", {})
        phased_config = hpf_config.get("role_phased_training", {})
        if self._parse_hpf_bool(phased_config.get("enable", False), False):
            phase_state_path = os.path.join(global_step_folder, "hpf_role_phase.pt")
            if not os.path.exists(phase_state_path):
                raise ValueError(
                    "Cannot resume HPF role-phased training from a checkpoint without hpf_role_phase.pt. "
                    "Use a checkpoint created by the same role-phased mode."
                )
            phase_state = torch.load(phase_state_path, weights_only=False)
            expected_phase_state = {
                "enabled": True,
                "follower_epochs": int(phased_config.get("follower_epochs", 1)),
                "leader_epochs": int(phased_config.get("leader_epochs", 1)),
                "batches_per_pass": len(self.train_dataloader),
            }
            if phase_state != expected_phase_state:
                raise ValueError(
                    "HPF role-phased resume configuration does not match the checkpoint: "
                    f"checkpoint={phase_state}, current={expected_phase_state}."
                )

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            steps_per_epoch = len(self.train_dataloader)
            at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
            if at_epoch_boundary:
                print(
                    f"Skipping dataloader state restore: global_steps={self.global_steps} "
                    f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                    f"The saved state marks the dataloader as exhausted. "
                    f"Next epoch will iterate from scratch."
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        # step 4. No padding to padding
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)

        return ref_log_prob

    def _compute_old_log_prob(
        self,
        batch: DataProto,
        *,
        temperature: float | None = None,
        calculate_entropy: bool = True,
    ):
        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        calculate_sum_pi_squared = self.config.actor_rollout_ref.actor.get("calculate_sum_pi_squared", False)
        metadata = {}
        if temperature is not None:
            metadata["temperature"] = float(temperature)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            compute_loss=False,
            **metadata,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        # gather output
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        sum_pi_squared = tu.get(output, "sum_pi_squared") if calculate_sum_pi_squared else None

        old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
        # step 4. No padding to padding
        if entropy is not None:
            entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        if sum_pi_squared is not None:
            sum_pi_squared = no_padding_2_padding(sum_pi_squared, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        result = {"old_log_probs": log_probs.float()}
        if entropy is not None:
            result["entropys"] = entropy.float()
        if routed_experts is not None:
            result["routed_experts"] = routed_experts
        if sum_pi_squared is not None:
            result["sum_pi_squared"] = sum_pi_squared.float()
        old_log_prob = tu.get_tensordict(result)
        old_log_prob = DataProto.from_tensordict(old_log_prob)
        return old_log_prob, old_log_prob_mfu

    def _update_actor(
        self,
        batch: DataProto,
        *,
        progress_label: str | None = None,
        progress_log_interval: int | None = None,
        temperature: float | None = None,
        mini_batch_size_multiplier: int = 1,
        responses_per_prompt: int | None = None,
        shuffle_override: bool | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        if temperature is not None:
            batch.meta_info["temperature"] = float(temperature)
        elif "temperature" not in batch.batch:
            batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        if responses_per_prompt is None:
            responses_per_prompt = self.config.actor_rollout_ref.rollout.n
        ppo_mini_batch_size = ppo_mini_batch_size * int(responses_per_prompt)
        ppo_mini_batch_size *= int(mini_batch_size_multiplier)
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = (
            self.config.actor_rollout_ref.actor.shuffle if shuffle_override is None else bool(shuffle_override)
        )
        actor_update_metadata = dict(
            calculate_entropy=calculate_entropy,
            distillation_use_topk=distillation_use_topk,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            compute_loss=True,
        )
        if progress_label is not None:
            actor_update_metadata["progress_label"] = progress_label
            actor_update_metadata["progress_log_interval"] = (
                1 if progress_log_interval is None else int(progress_log_interval)
            )
        if extra_metadata:
            actor_update_metadata.update(extra_metadata)
        tu.assign_non_tensor(batch_td, **actor_update_metadata)
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

        return actor_output

    @staticmethod
    def _set_hpf_token_temperatures(
        batch: DataProto,
        *,
        pg_temperature: float,
        kl_temperature: float,
    ) -> None:
        response_mask = batch.batch["response_mask"]
        temperature = torch.ones_like(response_mask, dtype=torch.float32)
        if "hpf_pg_mask" in batch.batch:
            temperature = torch.where(
                batch.batch["hpf_pg_mask"].bool(), torch.full_like(temperature, float(pg_temperature)), temperature
            )
        if "hpf_kl_mask" in batch.batch:
            temperature = torch.where(
                batch.batch["hpf_kl_mask"].bool(), torch.full_like(temperature, float(kl_temperature)), temperature
            )
        batch.batch["temperature"] = temperature
        batch.meta_info.pop("temperature", None)

    @staticmethod
    def _parse_hpf_float(value: Any, default: float) -> float:
        if value is None:
            return default
        if isinstance(value, str) and value.lower() in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        return float(value)

    def _get_hpf_round_index(self, epoch: int | None) -> int:
        hpf_config = self.config.algorithm.get("hpf_rlvr", {})
        schedule = str(hpf_config.get("horizon_schedule", "epoch")).lower()
        if schedule == "epoch":
            if epoch is not None:
                return int(epoch) + 1
            steps_per_epoch = max(len(self.train_dataloader), 1)
            return max((self.global_steps - 1) // steps_per_epoch + 1, 1)
        if schedule in {"step", "global_step"}:
            interval = int(hpf_config.get("horizon_update_interval_steps", 1))
            if interval <= 0:
                raise ValueError(f"hpf_rlvr.horizon_update_interval_steps must be positive, got {interval}")
            return max((self.global_steps - 1) // interval + 1, 1)
        raise ValueError(f"Unsupported HPF horizon_schedule={schedule!r}; expected 'epoch' or 'step'")

    def _get_hpf_role_phase(self, physical_epoch: int) -> tuple[str | None, int | None, int | None]:
        """Return role, role-local epoch, and fixed-horizon round for a dataloader pass."""
        hpf_config = self.config.algorithm.get("hpf_rlvr", {})
        phased_config = hpf_config.get("role_phased_training", {})
        if not self._parse_hpf_bool(phased_config.get("enable", False), False):
            return None, None, None
        if not self._parse_hpf_bool(hpf_config.get("fresh_leader_tree", False), False):
            raise ValueError("HPF role-phased training requires hpf_rlvr.fresh_leader_tree=True.")
        if str(hpf_config.get("horizon_schedule", "epoch")).lower() != "epoch":
            raise ValueError("HPF role-phased training requires hpf_rlvr.horizon_schedule='epoch'.")

        follower_epochs = int(phased_config.get("follower_epochs", 1))
        leader_epochs = int(phased_config.get("leader_epochs", 1))
        if follower_epochs <= 0 or leader_epochs <= 0:
            raise ValueError(
                "hpf_rlvr.role_phased_training follower_epochs and leader_epochs must both be positive; "
                f"got follower_epochs={follower_epochs}, leader_epochs={leader_epochs}."
            )
        return get_hpf_role_phase(int(physical_epoch), follower_epochs, leader_epochs)

    @staticmethod
    def _compute_hpf_suffix_correction(
        *,
        updated_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        suffix_mask: torch.Tensor,
        correction_clip: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raw_delta = ((updated_log_probs - old_log_probs).to(suffix_mask.device) * suffix_mask).sum(dim=-1)
        delta = raw_delta
        clipped_upper = torch.zeros_like(raw_delta, dtype=torch.bool)
        clipped_lower = torch.zeros_like(raw_delta, dtype=torch.bool)
        if np.isfinite(correction_clip):
            clipped_upper = raw_delta > correction_clip
            clipped_lower = raw_delta < -correction_clip
            delta = delta.clamp(min=-correction_clip, max=correction_clip)
        correction = torch.exp(delta).detach()
        return correction, {
            "hpf/correction_clip_upper_frac": float(clipped_upper.float().mean().item()),
            "hpf/correction_clip_lower_frac": float(clipped_lower.float().mean().item()),
            "hpf/correction_clip_frac": float((clipped_upper | clipped_lower).float().mean().item()),
            "hpf/correction_log_ratio_mean": float(delta.mean().item()),
            "hpf/correction_log_ratio_std": float(delta.std(unbiased=True).item()),
            "hpf/correction_ratio_mean": float(correction.mean().item()),
            "hpf/correction_ratio_max": float(correction.max().item()),
            "hpf/correction_ratio_min": float(correction.min().item()),
        }

    @staticmethod
    def _parse_hpf_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _compute_hpf_local_suffix_window(
        response_mask: torch.Tensor,
        *,
        horizon: int,
        window_size: int,
    ) -> torch.Tensor:
        response_len = response_mask.shape[-1]
        start = min(max(int(horizon), 0), response_len)
        end = min(start + max(int(window_size), 0), response_len)
        positions = torch.arange(response_len, device=response_mask.device).unsqueeze(0)
        return ((positions >= start) & (positions < end) & response_mask.bool()).to(response_mask.dtype)

    @staticmethod
    def _truncate_hpf_update_batch_response(batch: DataProto, response_cutoff: int) -> int:
        response_len = batch.batch["responses"].shape[-1]
        cutoff = min(max(int(response_cutoff), 1), response_len)
        if cutoff >= response_len:
            return response_len

        prompt_len = batch.batch["prompts"].shape[-1]
        seq_len = prompt_len + response_len
        seq_cutoff = prompt_len + cutoff
        response_keys = {
            "responses",
            "response_mask",
            "old_log_probs",
            "advantages",
            "returns",
            "hpf_pg_mask",
            "hpf_kl_mask",
            "hpf_kl_ref_log_prob",
            "ref_log_prob",
            "rollout_log_probs",
            "token_level_scores",
            "token_level_rewards",
            "temperature",
        }
        sequence_keys = {"input_ids", "attention_mask", "position_ids"}

        for key in response_keys:
            value = batch.batch.get(key, None)
            if isinstance(value, torch.Tensor) and value.shape[-1] == response_len:
                batch.batch[key] = value[..., :cutoff].contiguous()

        for key in sequence_keys:
            value = batch.batch.get(key, None)
            if isinstance(value, torch.Tensor) and value.shape[-1] == seq_len:
                batch.batch[key] = value[..., :seq_cutoff].contiguous()

        if "attention_mask" in batch.batch:
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        return cutoff

    def _apply_hpf_local_update_window(
        self,
        *,
        batch: DataProto,
        hpf_round_index: int,
        progressive_block_size: int,
        max_response_length: int,
        window_size: int,
        phase: str,
        metrics: dict[str, float],
    ) -> DataProto:
        response_mask = batch.batch["response_mask"]
        response_len = response_mask.shape[-1]
        horizon = min(int(hpf_round_index) * int(progressive_block_size), int(max_response_length), response_len)
        local_suffix_mask = self._compute_hpf_local_suffix_window(
            response_mask,
            horizon=horizon,
            window_size=window_size,
        )
        old_pg_tokens = (
            batch.batch["hpf_pg_mask"].sum().item() if "hpf_pg_mask" in batch.batch else response_mask.sum().item()
        )
        old_kl_tokens = batch.batch["hpf_kl_mask"].sum().item() if "hpf_kl_mask" in batch.batch else 0.0

        if phase == "follower":
            # Follower PG only trains the current suffix block. Its prefix KL,
            # when enabled, remains on the prefix up to the horizon.
            batch.batch["hpf_pg_mask"] = local_suffix_mask
            batch.batch["advantages"] = batch.batch["advantages"] * local_suffix_mask.to(batch.batch["advantages"])
            batch.batch["returns"] = batch.batch["advantages"]
        elif phase == "leader":
            # Leader PG remains prefix-level; only the suffix KL is restricted
            # to the current local suffix block.
            if "hpf_kl_mask" in batch.batch:
                batch.batch["hpf_kl_mask"] = local_suffix_mask
        else:
            raise ValueError(f"Unsupported HPF local update phase: {phase}")

        local_pg_tokens = (
            batch.batch["hpf_pg_mask"].sum().item() if "hpf_pg_mask" in batch.batch else response_mask.sum().item()
        )
        local_kl_tokens = batch.batch["hpf_kl_mask"].sum().item() if "hpf_kl_mask" in batch.batch else 0.0
        local_nonempty_frac = float((local_suffix_mask.sum(dim=-1) > 0).float().mean().item())
        truncated_response_len = self._truncate_hpf_update_batch_response(batch, horizon + window_size)
        prefix = f"hpf/local_update_window/{phase}"
        metrics[f"{prefix}_enabled"] = 1.0
        metrics[f"{prefix}_horizon_tokens"] = float(horizon)
        metrics[f"{prefix}_window_size"] = float(window_size)
        metrics[f"{prefix}_response_len_after_truncate"] = float(truncated_response_len)
        metrics[f"{prefix}_pg_tokens_before"] = float(old_pg_tokens)
        metrics[f"{prefix}_pg_tokens_after"] = float(local_pg_tokens)
        metrics[f"{prefix}_kl_tokens_before"] = float(old_kl_tokens)
        metrics[f"{prefix}_kl_tokens_after"] = float(local_kl_tokens)
        metrics[f"{prefix}_nonempty_frac"] = local_nonempty_frac
        print(
            "[HPF] local update window applied "
            f"step={self.global_steps} phase={phase} horizon={horizon} window={window_size} "
            f"pg_tokens={float(old_pg_tokens):.0f}->{float(local_pg_tokens):.0f} "
            f"kl_tokens={float(old_kl_tokens):.0f}->{float(local_kl_tokens):.0f} "
            f"response_len={response_len}->{truncated_response_len} "
            f"nonempty_frac={local_nonempty_frac:.4f}",
            flush=True,
        )
        return batch

    def _prepare_hpf_mixed_policy_grpo_update_batch(
        self,
        batch: DataProto,
        *,
        hpf_round_index: int | None,
        cut_horizon: int | None = None,
        truncate_response: bool = True,
        pad_to_mini_batch: bool = True,
    ) -> tuple[DataProto, dict[str, float]]:
        """Build one cut-specific mixed-policy GRPO batch without updating the actor.

        In transition-aware mode this prepares only the current-cut term
        ``L_{i,h_i}``. The paired next-cut trajectories are deliberately kept
        out of this batch so rewards and group-relative advantages cannot mix
        across cuts.
        """
        hpf_config = self.config.algorithm.get("hpf_rlvr", {})
        tree_config = hpf_config.get("tree_rollout", {})
        mixed_policy_config = hpf_config.get("mixed_policy_grpo", {})
        progressive_block_size = int(hpf_config.get("progressive_block_size", 256))
        max_response_length = int(hpf_config.get("max_response_length", self.config.data.max_response_length))
        prefix_temperature = float(
            tree_config.get("prefix_temperature", self.config.actor_rollout_ref.rollout.temperature)
        )
        suffix_temperature = float(
            tree_config.get("suffix_temperature", self.config.actor_rollout_ref.rollout.temperature)
        )
        suffix_window_size_value = mixed_policy_config.get("suffix_window_size", None)
        suffix_window_size = (
            None
            if suffix_window_size_value is None or str(suffix_window_size_value).lower() == "null"
            else int(suffix_window_size_value)
        )
        if suffix_window_size is not None and suffix_window_size <= 0:
            raise ValueError(
                "hpf_rlvr.mixed_policy_grpo.suffix_window_size must be positive or null, "
                f"got {suffix_window_size}."
            )
        if hpf_round_index is None:
            hpf_round_index = self._get_hpf_round_index(None)
        prefix_horizon = (
            min(int(hpf_round_index) * progressive_block_size, max_response_length)
            if cut_horizon is None
            else min(int(cut_horizon), max_response_length)
        )
        transition_aware_rollout = self._parse_hpf_bool(
            mixed_policy_config.get("transition_aware_rollout", {}).get("enable", False), False
        )
        next_horizon = min(prefix_horizon + progressive_block_size, max_response_length)
        transition_width = next_horizon - prefix_horizon
        if (
            transition_aware_rollout
            and suffix_window_size is not None
            and suffix_window_size < transition_width
        ):
            raise ValueError(
                "HPF transition-aware suffix_window_size must cover the next-cut transition: "
                f"suffix_window_size={suffix_window_size}, "
                f"next_horizon-current_horizon={transition_width}."
            )
        if (
            "hpf_leader_rollout_old_log_probs" not in batch.batch
            or "hpf_follower_rollout_old_log_probs" not in batch.batch
        ):
            raise ValueError("HPF mixed-policy GRPO requires tree rollout log probabilities.")

        mixed = build_hpf_mixed_policy_grpo_batch(
            batch=batch,
            prefix_horizon=prefix_horizon,
            suffix_window_size=suffix_window_size,
            leader_old_log_probs=batch.batch["hpf_leader_rollout_old_log_probs"],
            follower_old_log_probs=batch.batch["hpf_follower_rollout_old_log_probs"],
        )
        update_batch = mixed.batch
        diagnostics_enabled = self._parse_hpf_bool(
            mixed_policy_config.get("transition_aware_optimization", {})
            .get("diagnostics", {})
            .get("enable", False),
            False,
        )

        def masked_max_abs_error(lhs: torch.Tensor, rhs: torch.Tensor, mask: torch.Tensor) -> float:
            if not bool(mask.any().item()):
                return 0.0
            return float((lhs - rhs).abs().masked_select(mask).max().item())

        diagnostic_metrics = {}
        if diagnostics_enabled:
            source_leader_old = batch.batch["hpf_leader_rollout_old_log_probs"].to(
                device=update_batch.batch["old_log_probs"].device, dtype=torch.float32
            )
            source_follower_old = batch.batch["hpf_follower_rollout_old_log_probs"].to(
                device=update_batch.batch["old_log_probs"].device, dtype=torch.float32
            )
            diagnostic_metrics.update(
                {
                    "hpf/mixed_policy_grpo_prefix_old_logprob_max_abs_error": masked_max_abs_error(
                        update_batch.batch["old_log_probs"], source_leader_old, mixed.prefix_mask.bool()
                    ),
                    "hpf/mixed_policy_grpo_suffix_old_logprob_max_abs_error": masked_max_abs_error(
                        update_batch.batch["old_log_probs"], source_follower_old, mixed.suffix_mask.bool()
                    ),
                }
            )
        tree_log_prob_keys = [
            key
            for key in ("hpf_leader_rollout_old_log_probs", "hpf_follower_rollout_old_log_probs")
            if key in update_batch.batch
        ]
        if tree_log_prob_keys:
            update_batch.pop(batch_keys=tree_log_prob_keys)
        response_len_before = update_batch.batch["responses"].shape[-1]
        if suffix_window_size is None:
            update_length = int(update_batch.batch["response_mask"].sum(dim=-1).max().item())
        else:
            update_length = prefix_horizon + suffix_window_size
        response_len_after = (
            self._truncate_hpf_update_batch_response(update_batch, update_length)
            if truncate_response
            else response_len_before
        )

        response_mask = update_batch.batch["response_mask"]
        prefix_mask = mixed.prefix_mask[..., :response_len_after].bool()
        suffix_mask = mixed.suffix_mask[..., :response_len_after].bool()
        temperature = torch.ones_like(response_mask, dtype=torch.float32)
        temperature = torch.where(
            prefix_mask, torch.full_like(temperature, prefix_temperature), temperature
        )
        temperature = torch.where(
            suffix_mask, torch.full_like(temperature, suffix_temperature), temperature
        )
        update_batch.batch["temperature"] = temperature
        update_batch.meta_info.pop("temperature", None)
        if diagnostics_enabled:
            diagnostic_metrics.update(
                {
                    "hpf/mixed_policy_grpo_prefix_temperature_max_abs_error": masked_max_abs_error(
                        temperature, torch.full_like(temperature, prefix_temperature), prefix_mask
                    ),
                    "hpf/mixed_policy_grpo_suffix_temperature_max_abs_error": masked_max_abs_error(
                        temperature, torch.full_like(temperature, suffix_temperature), suffix_mask
                    ),
                    "hpf/mixed_policy_grpo_pg_tokens_outside_response": float(
                        (update_batch.batch["hpf_pg_mask"].bool() & ~response_mask.bool()).sum().item()
                    ),
                }
            )

        mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * int(
            self.config.actor_rollout_ref.rollout.n
        )
        if pad_to_mini_batch:
            update_batch, pad_size = pad_dataproto_to_divisor(update_batch, mini_batch_size)
        else:
            pad_size = 0
        metrics = dict(mixed.metrics)
        metrics.update(
            {
                "hpf/mixed_policy_grpo_prefix_temperature": prefix_temperature,
                "hpf/mixed_policy_grpo_suffix_temperature": suffix_temperature,
                "hpf/mixed_policy_grpo_full_suffix_tail": float(suffix_window_size is None),
                "hpf/mixed_policy_grpo_suffix_window_size": float(
                    suffix_window_size if suffix_window_size is not None else -1
                ),
                "hpf/mixed_policy_grpo_response_len_before": float(response_len_before),
                "hpf/mixed_policy_grpo_response_len_after": float(response_len_after),
                "hpf/mixed_policy_grpo_pad_size": float(pad_size),
                "hpf/mixed_policy_grpo_optimizer_steps": float(
                    math.ceil(len(update_batch) / mini_batch_size)
                    * self.config.actor_rollout_ref.actor.ppo_epochs
                ),
                **diagnostic_metrics,
            }
        )
        return update_batch, metrics

    def _update_actor_hpf_mixed_policy_grpo(
        self,
        batch: DataProto,
        *,
        hpf_round_index: int | None,
    ) -> DataProto:
        """Run one standard GRPO update under the tree rollout's mixed policy."""
        hpf_config = self.config.algorithm.get("hpf_rlvr", {})
        update_batch, metrics = self._prepare_hpf_mixed_policy_grpo_update_batch(
            batch,
            hpf_round_index=hpf_round_index,
        )
        print(
            "[HPF] mixed-policy GRPO actor update start "
            f"step={self.global_steps} batch={len(update_batch)} "
            f"pad={int(metrics['hpf/mixed_policy_grpo_pad_size'])} "
            f"prefix_horizon={int(metrics['hpf/mixed_policy_grpo_prefix_horizon_tokens'])} "
            f"response_len={int(metrics['hpf/mixed_policy_grpo_response_len_before'])}"
            f"->{int(metrics['hpf/mixed_policy_grpo_response_len_after'])} "
            f"suffix_window_size={int(metrics['hpf/mixed_policy_grpo_suffix_window_size'])} "
            f"prefix_temp={metrics['hpf/mixed_policy_grpo_prefix_temperature']} "
            f"suffix_temp={metrics['hpf/mixed_policy_grpo_suffix_temperature']}",
            flush=True,
        )
        update_start = time.perf_counter()
        actor_output = self._update_actor(
            update_batch,
            progress_label=f"hpf/mixed-policy-grpo/step-{self.global_steps}",
            progress_log_interval=int(hpf_config.get("progress_log_interval", 1)),
        )
        elapsed = time.perf_counter() - update_start
        print(
            f"[HPF] mixed-policy GRPO actor update done step={self.global_steps} elapsed_s={elapsed:.2f}",
            flush=True,
        )
        metrics["timing_s/hpf/mixed_policy_grpo_update_actor"] = float(elapsed)
        actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
        metrics.update(rename_dict(actor_metrics, "hpf/mixed_policy_grpo/"))
        return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})

    def _compute_hpf_behavior_grpo_objective(self, batch: DataProto) -> float:
        """Compute ``L_h(theta_i)`` from stored behavior log-probabilities."""
        actor_config = self.config.actor_rollout_ref.actor
        rollout_is_weights = batch.batch.get("rollout_is_weights", None)
        behavior_pg_loss, _ = core_algos.compute_policy_loss_vanilla(
            old_log_prob=batch.batch["old_log_probs"],
            log_prob=batch.batch["old_log_probs"],
            advantages=batch.batch["advantages"],
            response_mask=batch.batch["hpf_pg_mask"].bool(),
            loss_agg_mode=actor_config.loss_agg_mode,
            config=actor_config,
            rollout_is_weights=rollout_is_weights,
            global_batch_info={"loss_scale_factor": actor_config.loss_scale_factor},
        )
        return float((-behavior_pg_loss).detach().item())

    def _write_hpf_transition_diagnostics(
        self,
        *,
        current_batch: DataProto,
        next_batch: DataProto,
        current_metrics: dict[str, float],
        next_metrics: dict[str, float],
        actor_metrics: dict[str, float],
        current_horizon: int,
        next_horizon: int,
        transition_return: float,
        current_behavior_objective: float,
        next_behavior_objective: float,
        static_offset: float,
        lambda_trans: float,
        joint_batch_size: int,
        joint_pad_size: int,
        response_len_before: int,
        response_len_after: int,
    ) -> None:
        """Write a compact, model-free audit bundle for one transition update."""
        transition_config = (
            self.config.algorithm.get("hpf_rlvr", {})
            .get("mixed_policy_grpo", {})
            .get("transition_aware_optimization", {})
        )
        diagnostics_config = transition_config.get("diagnostics", {})
        if not self._parse_hpf_bool(diagnostics_config.get("enable", False), False):
            return

        sample_pairs = int(diagnostics_config.get("sample_pairs", 8))
        output_dir = os.path.join(
            str(self.config.trainer.rollout_data_dir), "transition_diagnostics"
        )
        os.makedirs(output_dir, exist_ok=True)
        step = int(self.global_steps)

        def as_strings(values: np.ndarray) -> list[str]:
            return [str(value) for value in values.tolist()]

        def summarize_cut(batch: DataProto, cut: str, horizon: int) -> tuple[list[dict], dict[str, int]]:
            response_mask = batch.batch["response_mask"].bool()
            pg_mask = batch.batch["hpf_pg_mask"].bool()
            positions = torch.arange(pg_mask.shape[-1], device=pg_mask.device).unsqueeze(0)
            prefix_mask = pg_mask & (positions < int(horizon))
            suffix_mask = pg_mask & (positions >= int(horizon))
            pg_counts = pg_mask.sum(dim=-1)
            safe_pg_counts = pg_counts.clamp_min(1)
            advantages = batch.batch["advantages"].float()
            advantage_means = (advantages * pg_mask).sum(dim=-1) / safe_pg_counts
            advantage_max = advantages.masked_fill(~pg_mask, float("-inf")).max(dim=-1).values
            advantage_min = advantages.masked_fill(~pg_mask, float("inf")).min(dim=-1).values
            advantage_spans = torch.where(
                pg_counts > 0, advantage_max - advantage_min, torch.zeros_like(advantage_max)
            )
            old_log_probs = batch.batch["old_log_probs"].float()
            temperatures = batch.batch["temperature"].float()
            scores = batch.batch["token_level_scores"].sum(dim=-1).float()

            def masked_row_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                counts = mask.sum(dim=-1)
                return torch.where(
                    counts > 0,
                    (values * mask).sum(dim=-1) / counts.clamp_min(1),
                    torch.zeros_like(counts, dtype=values.dtype),
                )

            pair_ids = as_strings(batch.non_tensor_batch["hpf_transition_pair_uid"])
            group_ids = as_strings(batch.non_tensor_batch["uid"])
            columns = {
                "score": scores.detach().cpu().tolist(),
                "advantage": advantage_means.detach().cpu().tolist(),
                "advantage_span": advantage_spans.detach().cpu().tolist(),
                "response_tokens": response_mask.sum(dim=-1).detach().cpu().tolist(),
                "pg_tokens": pg_counts.detach().cpu().tolist(),
                "prefix_pg_tokens": prefix_mask.sum(dim=-1).detach().cpu().tolist(),
                "suffix_pg_tokens": suffix_mask.sum(dim=-1).detach().cpu().tolist(),
                "prefix_old_logprob_mean": masked_row_mean(old_log_probs, prefix_mask).detach().cpu().tolist(),
                "suffix_old_logprob_mean": masked_row_mean(old_log_probs, suffix_mask).detach().cpu().tolist(),
                "prefix_temperature_mean": masked_row_mean(temperatures, prefix_mask).detach().cpu().tolist(),
                "suffix_temperature_mean": masked_row_mean(temperatures, suffix_mask).detach().cpu().tolist(),
            }
            rows = []
            for index, (pair_id, group_id) in enumerate(zip(pair_ids, group_ids, strict=True)):
                row = {"cut": cut, "pair_uid": pair_id, "group_uid": group_id}
                row.update({key: value[index] for key, value in columns.items()})
                rows.append(row)
            return rows, {pair_id: index for index, pair_id in enumerate(pair_ids)}

        current_rows, current_index = summarize_cut(current_batch, "current", current_horizon)
        next_rows, next_index = summarize_cut(next_batch, "next", next_horizon)
        if current_index.keys() != next_index.keys():
            raise ValueError("Cannot write transition diagnostics for mismatched pair IDs.")

        current_reward_mean = sum(float(row["score"]) for row in current_rows) / len(current_rows)
        next_reward_mean = sum(float(row["score"]) for row in next_rows) / len(next_rows)
        dynamic = {
            key.removeprefix("actor/transition_"): float(value)
            for key, value in actor_metrics.items()
            if key.startswith("actor/transition_")
        }
        current_pg_loss = dynamic.get("current_pg_loss")
        next_pg_loss = dynamic.get("next_pg_loss")
        surrogate = dynamic.get("surrogate")
        penalty = dynamic.get("penalty")
        objective_loss = dynamic.get("objective_loss")
        summary = {
            "schema_version": 1,
            "mode": "transition-aware mixed policy optimization",
            "step": step,
            "num_pairs": len(current_rows),
            "current_horizon": int(current_horizon),
            "next_horizon": int(next_horizon),
            "transition_width": int(next_horizon - current_horizon),
            "lambda_trans": float(lambda_trans),
            "loss_agg_mode": str(self.config.actor_rollout_ref.actor.loss_agg_mode),
            "loss_scale_factor": self.config.actor_rollout_ref.actor.loss_scale_factor,
            "sample_pairs": min(sample_pairs, len(current_rows)),
            "batch": {
                "current_rows": len(current_rows),
                "next_rows": len(next_rows),
                "joint_rows": int(joint_batch_size),
                "joint_pad_rows": int(joint_pad_size),
                "response_len_before": int(response_len_before),
                "response_len_after": int(response_len_after),
            },
            "rollout": {
                "pair_uid_sets_equal": current_index.keys() == next_index.keys(),
                "current_reward_mean": current_reward_mean,
                "next_reward_mean": next_reward_mean,
                "transition_return": float(transition_return),
                "transition_return_residual": float(
                    transition_return - (next_reward_mean - current_reward_mean)
                ),
            },
            "behavior": {
                "current_objective": float(current_behavior_objective),
                "next_objective": float(next_behavior_objective),
                "static_offset": float(static_offset),
                "static_offset_residual": float(
                    static_offset
                    - (transition_return + current_behavior_objective - next_behavior_objective)
                ),
            },
            "current_cut_preparation": current_metrics,
            "next_cut_preparation": next_metrics,
            "actor_metrics": {key: float(value) for key, value in actor_metrics.items()},
            "dynamic": dynamic,
            "identities": {
                "surrogate_residual": None
                if surrogate is None or current_pg_loss is None or next_pg_loss is None
                else float(surrogate - (static_offset + current_pg_loss - next_pg_loss)),
                "objective_loss_residual": None
                if objective_loss is None or current_pg_loss is None or penalty is None
                else float(objective_loss - (current_pg_loss + lambda_trans * penalty)),
            },
        }

        selected_pair_ids = list(current_index)[:sample_pairs]
        samples = []
        for pair_id in selected_pair_ids:
            for cut, batch, index in (
                ("current", current_batch, current_index[pair_id]),
                ("next", next_batch, next_index[pair_id]),
            ):
                valid_length = int(batch.batch["response_mask"][index].sum().item())
                samples.append(
                    {
                        "cut": cut,
                        "pair_uid": pair_id,
                        "response_ids": batch.batch["responses"][index, :valid_length].detach().cpu().tolist(),
                        "pg_mask": batch.batch["hpf_pg_mask"][index, :valid_length].bool().detach().cpu().tolist(),
                        "old_log_probs": batch.batch["old_log_probs"][index, :valid_length]
                        .float()
                        .detach()
                        .cpu()
                        .tolist(),
                        "temperature": batch.batch["temperature"][index, :valid_length]
                        .float()
                        .detach()
                        .cpu()
                        .tolist(),
                        "advantages": batch.batch["advantages"][index, :valid_length]
                        .float()
                        .detach()
                        .cpu()
                        .tolist(),
                    }
                )

        def write_json(path: str, value: Any) -> None:
            temporary = f"{path}.tmp"
            with open(temporary, "w") as file:
                json.dump(value, file, ensure_ascii=False, indent=2, default=str)
                file.write("\n")
            os.replace(temporary, path)

        def write_jsonl(path: str, values: list[dict]) -> None:
            temporary = f"{path}.tmp"
            with open(temporary, "w") as file:
                for value in values:
                    file.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
            os.replace(temporary, path)

        write_json(os.path.join(output_dir, f"step_{step}_summary.json"), summary)
        write_jsonl(os.path.join(output_dir, f"step_{step}_rows.jsonl"), current_rows + next_rows)
        write_jsonl(os.path.join(output_dir, f"step_{step}_samples.jsonl"), samples)
        print(
            "[HPF] transition diagnostic bundle written "
            f"step={step} dir={output_dir} rows={len(current_rows) + len(next_rows)} "
            f"sample_pairs={len(selected_pair_ids)}",
            flush=True,
        )

    def _update_actor_hpf_transition_aware_mixed_policy(
        self,
        current_batch: DataProto,
        next_batch: DataProto,
        *,
        hpf_round_index: int | None,
    ) -> DataProto:
        """Run one joint current-cut GRPO and transition-aware actor update."""
        hpf_config = self.config.algorithm.get("hpf_rlvr", {})
        mixed_config = hpf_config.get("mixed_policy_grpo", {})
        transition_config = mixed_config.get("transition_aware_optimization", {})
        progressive_block_size = int(hpf_config.get("progressive_block_size", 256))
        max_response_length = int(hpf_config.get("max_response_length", self.config.data.max_response_length))
        if hpf_round_index is None:
            hpf_round_index = self._get_hpf_round_index(None)
        current_horizon = min(int(hpf_round_index) * progressive_block_size, max_response_length)
        next_horizon = min(current_horizon + progressive_block_size, max_response_length)
        if next_horizon == current_horizon:
            return self._update_actor_hpf_mixed_policy_grpo(
                current_batch,
                hpf_round_index=hpf_round_index,
            )

        current_update, current_metrics = self._prepare_hpf_mixed_policy_grpo_update_batch(
            current_batch,
            hpf_round_index=hpf_round_index,
            cut_horizon=current_horizon,
            truncate_response=False,
            pad_to_mini_batch=False,
        )
        next_update, next_metrics = self._prepare_hpf_mixed_policy_grpo_update_batch(
            next_batch,
            hpf_round_index=hpf_round_index,
            cut_horizon=next_horizon,
            truncate_response=False,
            pad_to_mini_batch=False,
        )

        rollout_corr_config = self.config.algorithm.get("rollout_correction", {})
        reuse_rollout_log_probs = self._parse_hpf_bool(
            rollout_corr_config.get("bypass_mode", False), False
        )
        behavior_log_prob_start = time.perf_counter()
        behavior_log_prob_source = configure_hpf_transition_behavior_log_probs(
            current_update,
            next_update,
            reuse_rollout_log_probs=reuse_rollout_log_probs,
            recompute_fn=(
                None
                if reuse_rollout_log_probs
                else lambda update_batch: self._compute_old_log_prob(
                    update_batch, calculate_entropy=False
                )[0]
            ),
        )
        behavior_log_prob_elapsed = time.perf_counter() - behavior_log_prob_start
        print(
            "[HPF] transition behavior old_log_prob ready "
            f"step={self.global_steps} source={behavior_log_prob_source} "
            f"elapsed_s={behavior_log_prob_elapsed:.2f}",
            flush=True,
        )

        current_behavior_objective = self._compute_hpf_behavior_grpo_objective(current_update)
        next_behavior_objective = self._compute_hpf_behavior_grpo_objective(next_update)
        transition_return = float(current_batch.meta_info["hpf_transition_return_estimate"])
        static_offset = transition_return + current_behavior_objective - next_behavior_objective
        lambda_trans = float(transition_config.get("lambda_trans", 1.0))

        pair_key = "hpf_transition_pair_uid"
        current_pair_ids = [str(value) for value in current_update.non_tensor_batch[pair_key]]
        next_pair_ids = [str(value) for value in next_update.non_tensor_batch[pair_key]]
        next_index_by_pair = {pair_id: index for index, pair_id in enumerate(next_pair_ids)}
        if len(next_index_by_pair) != len(next_pair_ids) or set(current_pair_ids) != set(next_pair_ids):
            raise ValueError("Transition-aware actor update received mismatched current/next rollout pairs.")
        next_order = np.asarray([next_index_by_pair[pair_id] for pair_id in current_pair_ids], dtype=np.int64)
        next_update = next_update.select_idxs(next_order)

        current_update.batch["hpf_transition_cut_index"] = torch.zeros(
            len(current_update), dtype=torch.int32, device=current_update.batch["responses"].device
        )
        next_update.batch["hpf_transition_cut_index"] = torch.ones(
            len(next_update), dtype=torch.int32, device=next_update.batch["responses"].device
        )
        combined = DataProto.concat([current_update, next_update])
        num_pairs = len(current_update)
        interleave_order = np.column_stack(
            [np.arange(num_pairs, dtype=np.int64), np.arange(num_pairs, dtype=np.int64) + num_pairs]
        ).reshape(-1)
        combined = combined.select_idxs(interleave_order)

        suffix_window_value = mixed_config.get("suffix_window_size", None)
        suffix_window_size = (
            None
            if suffix_window_value is None or str(suffix_window_value).lower() == "null"
            else int(suffix_window_value)
        )
        if suffix_window_size is None:
            update_length = int(combined.batch["response_mask"].sum(dim=-1).max().item())
        else:
            update_length = next_horizon + suffix_window_size
        response_len_before = combined.batch["responses"].shape[-1]
        response_len_after = self._truncate_hpf_update_batch_response(combined, update_length)

        base_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * int(
            self.config.actor_rollout_ref.rollout.n
        )
        transition_mini_batch_size = base_mini_batch_size * 2
        combined, pad_size = pad_dataproto_to_divisor(combined, transition_mini_batch_size)
        if pad_size:
            raise ValueError(
                "Transition-aware mixed policy optimization requires the paired batch to be exactly "
                f"divisible by {transition_mini_batch_size}; padding {pad_size} rows would bias B_i."
            )
        metrics = {
            **rename_dict(current_metrics, "hpf/transition/current_cut/"),
            **rename_dict(next_metrics, "hpf/transition/next_cut/"),
            "hpf/transition_aware_optimization_enabled": 1.0,
            "hpf/transition_lambda": lambda_trans,
            "hpf/transition_behavior_current_objective": current_behavior_objective,
            "hpf/transition_behavior_next_objective": next_behavior_objective,
            "hpf/transition_static_offset": static_offset,
            "hpf/transition_joint_batch_size": float(len(combined)),
            "hpf/transition_joint_pad_size": float(pad_size),
            "hpf/transition_response_len_before": float(response_len_before),
            "hpf/transition_response_len_after": float(response_len_after),
            "hpf/transition_reuse_rollout_log_probs": float(reuse_rollout_log_probs),
            "timing_s/hpf/transition_behavior_old_log_prob": behavior_log_prob_elapsed,
        }
        print(
            "[HPF] transition-aware mixed policy optimization start "
            f"step={self.global_steps} pairs={num_pairs} batch={len(combined)} pad={pad_size} "
            f"current_horizon={current_horizon} next_horizon={next_horizon} "
            f"lambda_trans={lambda_trans:.6f} transition_return={transition_return:.6f} "
            f"behavior_current={current_behavior_objective:.6f} "
            f"behavior_next={next_behavior_objective:.6f} static_offset={static_offset:.6f}",
            flush=True,
        )
        update_start = time.perf_counter()
        actor_output = self._update_actor(
            combined,
            progress_label=f"hpf/transition-aware-mixed-policy/step-{self.global_steps}",
            progress_log_interval=int(hpf_config.get("progress_log_interval", 1)),
            mini_batch_size_multiplier=2,
            shuffle_override=False,
            extra_metadata={
                "hpf_transition_aware_optimization": True,
                "hpf_transition_static_offset": static_offset,
                "hpf_transition_lambda": lambda_trans,
            },
        )
        elapsed = time.perf_counter() - update_start
        print(
            "[HPF] transition-aware mixed policy optimization done "
            f"step={self.global_steps} elapsed_s={elapsed:.2f}",
            flush=True,
        )
        metrics["timing_s/hpf/transition_aware_mixed_policy_update_actor"] = float(elapsed)
        actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
        metrics.update(rename_dict(actor_metrics, "hpf/transition_aware_mixed_policy/"))
        self._write_hpf_transition_diagnostics(
            current_batch=current_update,
            next_batch=next_update,
            current_metrics=current_metrics,
            next_metrics=next_metrics,
            actor_metrics=actor_metrics,
            current_horizon=current_horizon,
            next_horizon=next_horizon,
            transition_return=transition_return,
            current_behavior_objective=current_behavior_objective,
            next_behavior_objective=next_behavior_objective,
            static_offset=static_offset,
            lambda_trans=lambda_trans,
            joint_batch_size=len(combined),
            joint_pad_size=pad_size,
            response_len_before=response_len_before,
            response_len_after=response_len_after,
        )
        return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})

    def _update_actor_hpf_masked_grpo(
        self,
        batch: DataProto,
        hpf_round_index: int | None = None,
        prompt_batch: DataProto | None = None,
        gen_batch: DataProto | None = None,
        role_phase: str = "both",
        role_responses_per_prompt: int | None = None,
    ) -> DataProto:
        hpf_update_start = time.perf_counter()
        hpf_config = self.config.algorithm.get("hpf_rlvr", {})
        if role_phase not in {"both", "follower", "leader"}:
            raise ValueError(f"Unsupported HPF role phase: {role_phase!r}")
        progressive_block_size = int(hpf_config.get("progressive_block_size", 256))
        max_response_length = int(hpf_config.get("max_response_length", self.config.data.max_response_length))
        epsilon = float(hpf_config.get("epsilon", 1e-6))
        progress_log_interval = int(hpf_config.get("progress_log_interval", 1))
        prefix_kl_coef = float(hpf_config.get("prefix_kl_coef", 0.0))
        suffix_kl_coef = float(hpf_config.get("suffix_kl_coef", 0.0))
        correction_clip = self._parse_hpf_float(hpf_config.get("correction_clip", float("inf")), float("inf"))
        local_window_config = hpf_config.get("local_update_window", {})
        local_update_window = self._parse_hpf_bool(local_window_config.get("enable", False), False)
        local_window_size_value = local_window_config.get("size", None)
        local_window_size = (
            progressive_block_size
            if local_window_size_value is None or str(local_window_size_value).lower() == "null"
            else int(local_window_size_value)
        )
        if local_update_window and local_window_size <= 0:
            raise ValueError(f"hpf_rlvr.local_update_window.size must be positive, got {local_window_size}.")
        fresh_leader_tree_value = hpf_config.get("fresh_leader_tree", False)
        fresh_leader_tree = (
            fresh_leader_tree_value
            if isinstance(fresh_leader_tree_value, bool)
            else str(fresh_leader_tree_value).lower() in {"1", "true", "yes", "on"}
        )
        std_normalize = bool(
            hpf_config.get("std_normalize", self.config.algorithm.get("norm_adv_by_std_in_grpo", True))
        )
        tree_config = hpf_config.get("tree_rollout", {})
        prefix_temperature = float(
            tree_config.get("prefix_temperature", self.config.actor_rollout_ref.rollout.temperature)
        )
        suffix_temperature = float(
            tree_config.get("suffix_temperature", self.config.actor_rollout_ref.rollout.temperature)
        )
        fresh_tree_config = hpf_config.get("fresh_tree_rollout", {})
        fresh_num_prefixes_value = fresh_tree_config.get("num_prefixes", None)
        fresh_num_suffixes_value = fresh_tree_config.get("num_suffixes", None)
        fresh_num_prefixes = int(
            tree_config.get("num_prefixes", 4) if fresh_num_prefixes_value is None else fresh_num_prefixes_value
        )
        fresh_num_suffixes = int(
            tree_config.get("num_suffixes", 2) if fresh_num_suffixes_value is None else fresh_num_suffixes_value
        )
        if fresh_num_prefixes <= 0:
            raise ValueError(f"hpf_rlvr.fresh_tree_rollout.num_prefixes must be positive, got {fresh_num_prefixes}.")
        if fresh_num_suffixes <= 0:
            raise ValueError(f"hpf_rlvr.fresh_tree_rollout.num_suffixes must be positive, got {fresh_num_suffixes}.")

        old_log_start = time.perf_counter()
        has_hpf_rollout_old_log_probs = (
            "hpf_follower_rollout_old_log_probs" in batch.batch
            and "hpf_leader_rollout_old_log_probs" in batch.batch
        )
        if has_hpf_rollout_old_log_probs:
            follower_old_log_prob = DataProto.from_single_dict(
                {"old_log_probs": batch.batch["hpf_follower_rollout_old_log_probs"]}
            )
            leader_old_log_prob = DataProto.from_single_dict(
                {"old_log_probs": batch.batch["hpf_leader_rollout_old_log_probs"]}
            )
            print(
                "[HPF] initial old_log_prob skipped "
                f"step={self.global_steps} source=tree_rollout_log_probs",
                flush=True,
            )
        else:
            follower_old_log_prob, _ = self._compute_old_log_prob(
                batch, temperature=suffix_temperature, calculate_entropy=False
            )
            leader_old_log_prob, _ = self._compute_old_log_prob(
                batch, temperature=prefix_temperature, calculate_entropy=False
            )
            print(
                "[HPF] initial old_log_prob recomputed "
                f"step={self.global_steps} source=actor_forward",
                flush=True,
            )
        role_old_log_elapsed = time.perf_counter() - old_log_start
        if hpf_round_index is None:
            hpf_round_index = self._get_hpf_round_index(None)
        if role_phase == "leader":
            follower_batch = None
            leader_batch = build_hpf_fresh_leader_batch(
                batch=batch,
                round_index=hpf_round_index,
                progressive_block_size=progressive_block_size,
                max_response_length=max_response_length,
                epsilon=epsilon,
                std_normalize=std_normalize,
                leader_old_log_probs=leader_old_log_prob.batch["old_log_probs"],
            )
        else:
            follower_batch, leader_batch = build_hpf_masked_batches(
                batch=batch,
                round_index=hpf_round_index,
                progressive_block_size=progressive_block_size,
                max_response_length=max_response_length,
                epsilon=epsilon,
                std_normalize=std_normalize,
                follower_old_log_probs=follower_old_log_prob.batch["old_log_probs"],
                leader_old_log_probs=leader_old_log_prob.batch["old_log_probs"],
            )
        metrics = dict(leader_batch.metrics)
        metrics["hpf/role_phase_follower"] = float(role_phase == "follower")
        metrics["hpf/role_phase_leader"] = float(role_phase == "leader")
        metrics["hpf/prefix_loss_temperature"] = prefix_temperature
        metrics["hpf/suffix_loss_temperature"] = suffix_temperature
        metrics["timing_s/hpf/role_old_log_prob"] = float(role_old_log_elapsed)
        metrics["hpf/initial_old_log_prob_skipped"] = float(has_hpf_rollout_old_log_probs)
        metrics["hpf/prefix_kl_coef"] = prefix_kl_coef
        metrics["hpf/suffix_kl_coef"] = suffix_kl_coef
        metrics["hpf/correction_clip"] = correction_clip
        metrics["hpf/fresh_leader_tree_enabled"] = float(fresh_leader_tree)
        metrics["hpf/local_update_window_enabled"] = float(local_update_window)
        metrics["hpf/local_update_window_size"] = float(local_window_size)
        if follower_batch is not None and role_phase in {"both", "follower"}:
            follower_mini_batch_size = (
                self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            )
            follower_phase_batch = follower_batch.batch
            if prefix_kl_coef > 0:
                follower_phase_batch.meta_info["hpf_kl_coef"] = float(prefix_kl_coef)
                follower_phase_batch.meta_info["hpf_kl_type"] = self.config.actor_rollout_ref.actor.kl_loss_type
            if local_update_window:
                follower_phase_batch = self._apply_hpf_local_update_window(
                    batch=follower_phase_batch,
                    hpf_round_index=hpf_round_index,
                    progressive_block_size=progressive_block_size,
                    max_response_length=max_response_length,
                    window_size=local_window_size,
                    phase="follower",
                    metrics=metrics,
                )
            self._set_hpf_token_temperatures(
                follower_phase_batch,
                pg_temperature=suffix_temperature,
                kl_temperature=prefix_temperature,
            )
            follower_phase_batch, follower_pad_size = pad_dataproto_to_divisor(
                follower_phase_batch, follower_mini_batch_size
            )
            metrics["hpf/follower_pad_size"] = float(follower_pad_size)
            metrics["hpf/follower_optimizer_steps"] = float(
                math.ceil(len(follower_phase_batch) / follower_mini_batch_size)
                * self.config.actor_rollout_ref.actor.ppo_epochs
            )
            follower_start = time.perf_counter()
            print(
                "[HPF] follower actor update start "
                f"step={self.global_steps} batch={len(follower_phase_batch)} pad={follower_pad_size}",
                flush=True,
            )
            follower_output = self._update_actor(
                follower_phase_batch,
                progress_label=f"hpf/follower/step-{self.global_steps}",
                progress_log_interval=progress_log_interval,
            )
            follower_elapsed = time.perf_counter() - follower_start
            print(
                "[HPF] follower actor update done "
                f"step={self.global_steps} elapsed_s={follower_elapsed:.2f}",
                flush=True,
            )
            metrics["timing_s/hpf/follower_update_actor"] = float(follower_elapsed)
            follower_metrics = reduce_metrics(follower_output.meta_info["metrics"])
            metrics.update(rename_dict(follower_metrics, "hpf/follower/"))
            metrics.update(follower_batch.metrics)

        if role_phase == "follower":
            metrics["timing_s/hpf/update_actor_total"] = float(time.perf_counter() - hpf_update_start)
            return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})

        follower_updated_log_prob = follower_old_log_prob
        leader_updated_log_prob = leader_old_log_prob
        if fresh_leader_tree and role_phase == "both":
            if prompt_batch is None or gen_batch is None:
                raise ValueError("HPF fresh leader tree requires prompt_batch and gen_batch.")
            fresh_start = time.perf_counter()
            print(
                "[HPF] fresh leader tree weight sync start "
                f"step={self.global_steps}",
                flush=True,
            )
            self.checkpoint_manager.update_weights(self.global_steps)
            print(
                "[HPF] fresh leader tree rollout start "
                f"step={self.global_steps} prefixes={fresh_num_prefixes} suffixes={fresh_num_suffixes}",
                flush=True,
            )
            fresh_gen_output, fresh_tree_metrics = self._generate_hpf_tree_sequences(
                gen_batch,
                hpf_round_index=hpf_round_index,
                num_prefixes=fresh_num_prefixes,
                num_suffixes=fresh_num_suffixes,
                require_rollout_n_match=False,
            )
            self.checkpoint_manager.sleep_replicas()
            prefixed_fresh_metrics = {}
            for key, value in fresh_tree_metrics.items():
                if key.startswith("timing_s/hpf/tree_"):
                    prefixed_fresh_metrics[key.replace("timing_s/hpf/tree_", "timing_s/hpf/fresh_leader_tree_")] = value
                elif key.startswith("timing_s/hpf/"):
                    prefixed_fresh_metrics[key.replace("timing_s/hpf/", "timing_s/hpf/fresh_leader_")] = value
                elif key.startswith("hpf/tree_"):
                    prefixed_fresh_metrics[key.replace("hpf/tree_", "hpf/fresh_leader_tree_")] = value
                else:
                    prefixed_fresh_metrics[f"hpf/fresh_leader/{key}"] = value
            metrics.update(prefixed_fresh_metrics)
            fresh_gen_output.meta_info.pop("timing", None)

            fresh_responses_per_prompt = fresh_num_prefixes * fresh_num_suffixes
            if len(fresh_gen_output) != len(prompt_batch) * fresh_responses_per_prompt:
                raise ValueError(
                    "HPF fresh leader tree rollout produced an unexpected number of responses: "
                    f"got {len(fresh_gen_output)}, expected {len(prompt_batch) * fresh_responses_per_prompt} "
                    f"({len(prompt_batch)} prompts x {fresh_num_prefixes} prefixes x {fresh_num_suffixes} suffixes)."
                )
            metrics["hpf/fresh_leader_tree_responses_per_prompt"] = float(fresh_responses_per_prompt)
            fresh_batch = prompt_batch.repeat(repeat_times=fresh_responses_per_prompt, interleave=True)
            fresh_batch = fresh_batch.union(fresh_gen_output)
            if "response_mask" not in fresh_batch.batch:
                fresh_batch.batch["response_mask"] = compute_response_mask(fresh_batch)
            if self.config.trainer.balance_batch:
                self._balance_batch(fresh_batch, metrics=metrics)
            fresh_batch.meta_info["global_token_num"] = torch.sum(
                fresh_batch.batch["attention_mask"], dim=-1
            ).tolist()

            fresh_reward_start = time.perf_counter()
            if self.use_rm and "rm_scores" not in fresh_batch.batch:
                fresh_batch_reward = self._compute_reward_colocate(fresh_batch)
                fresh_batch = fresh_batch.union(fresh_batch_reward)
            fresh_reward_tensor, fresh_reward_extra_infos_dict = extract_reward(fresh_batch)
            fresh_batch.batch["token_level_scores"] = fresh_reward_tensor
            if fresh_reward_extra_infos_dict:
                fresh_batch.non_tensor_batch.update(
                    {key: np.array(value) for key, value in fresh_reward_extra_infos_dict.items()}
                )
            fresh_batch.batch["token_level_rewards"] = fresh_batch.batch["token_level_scores"]
            metrics["timing_s/hpf/fresh_leader_reward"] = float(time.perf_counter() - fresh_reward_start)

            fresh_logprob_start = time.perf_counter()
            has_fresh_rollout_old_log_probs = (
                "hpf_follower_rollout_old_log_probs" in fresh_batch.batch
                and "hpf_leader_rollout_old_log_probs" in fresh_batch.batch
            )
            if has_fresh_rollout_old_log_probs:
                leader_updated_log_prob = DataProto.from_single_dict(
                    {"old_log_probs": fresh_batch.batch["hpf_leader_rollout_old_log_probs"]}
                )
                follower_updated_log_prob = DataProto.from_single_dict(
                    {"old_log_probs": fresh_batch.batch["hpf_follower_rollout_old_log_probs"]}
                )
                print(
                    "[HPF] fresh leader old_log_prob skipped "
                    f"step={self.global_steps} source=tree_rollout_log_probs",
                    flush=True,
                )
            else:
                leader_updated_log_prob, _ = self._compute_old_log_prob(
                    fresh_batch, temperature=prefix_temperature, calculate_entropy=False
                )
                follower_updated_log_prob, _ = self._compute_old_log_prob(
                    fresh_batch, temperature=suffix_temperature, calculate_entropy=False
                )
                print(
                    "[HPF] fresh leader old_log_prob recomputed "
                    f"step={self.global_steps} source=actor_forward",
                    flush=True,
                )
            metrics["timing_s/hpf/fresh_leader_role_old_log_prob"] = float(time.perf_counter() - fresh_logprob_start)
            metrics["hpf/fresh_leader_old_log_prob_skipped"] = float(has_fresh_rollout_old_log_probs)
            leader_batch = build_hpf_fresh_leader_batch(
                batch=fresh_batch,
                round_index=hpf_round_index,
                progressive_block_size=progressive_block_size,
                max_response_length=max_response_length,
                epsilon=epsilon,
                std_normalize=std_normalize,
                leader_old_log_probs=leader_updated_log_prob.batch["old_log_probs"],
            )
            metrics.update(leader_batch.metrics)
            metrics["timing_s/hpf/fresh_leader_total"] = float(time.perf_counter() - fresh_start)
        elif follower_batch is not None and leader_batch.suffix_mask is not None:
            correction_start = time.perf_counter()
            follower_updated_log_prob, _ = self._compute_old_log_prob(
                batch, temperature=suffix_temperature, calculate_entropy=False
            )
            leader_updated_log_prob, _ = self._compute_old_log_prob(
                batch, temperature=prefix_temperature, calculate_entropy=False
            )
            leader_batch = build_hpf_corrected_leader_batch(
                batch=batch,
                round_index=hpf_round_index,
                progressive_block_size=progressive_block_size,
                max_response_length=max_response_length,
                leader_old_log_probs=leader_old_log_prob.batch["old_log_probs"],
                leader_post_follower_log_probs=leader_updated_log_prob.batch["old_log_probs"],
                follower_old_log_probs=follower_old_log_prob.batch["old_log_probs"],
                follower_post_follower_log_probs=follower_updated_log_prob.batch["old_log_probs"],
                correction_clip=correction_clip,
            )
            metrics.update(leader_batch.metrics)
            metrics["timing_s/hpf/correction_log_prob"] = float(time.perf_counter() - correction_start)
            metrics["timing_s/hpf/suffix_correction_log_prob"] = metrics["timing_s/hpf/correction_log_prob"]
            metrics["timing_s/hpf/prefix_correction_log_prob"] = metrics["timing_s/hpf/correction_log_prob"]

        if suffix_kl_coef > 0 and leader_batch.suffix_mask is not None:
            suffix_ref_log_prob = follower_updated_log_prob.batch["old_log_probs"]
            leader_batch.batch.batch["hpf_kl_ref_log_prob"] = suffix_ref_log_prob.to(
                device=leader_batch.batch.batch["response_mask"].device, dtype=torch.float32
            )
            leader_batch.batch.batch["hpf_kl_mask"] = leader_batch.suffix_mask.to(
                device=leader_batch.batch.batch["response_mask"].device
            )
            leader_batch.batch.meta_info["hpf_kl_coef"] = float(suffix_kl_coef)
            leader_batch.batch.meta_info["hpf_kl_type"] = self.config.actor_rollout_ref.actor.kl_loss_type
            metrics["hpf/leader_suffix_kl_batch_size"] = float(len(leader_batch.batch))
        leader_phase_batch = leader_batch.batch
        if local_update_window:
            leader_phase_batch = self._apply_hpf_local_update_window(
                batch=leader_phase_batch,
                hpf_round_index=hpf_round_index,
                progressive_block_size=progressive_block_size,
                max_response_length=max_response_length,
                window_size=local_window_size,
                phase="leader",
                metrics=metrics,
            )
        self._set_hpf_token_temperatures(
            leader_phase_batch,
            pg_temperature=prefix_temperature,
            kl_temperature=suffix_temperature,
        )
        leader_mini_batch_size = (
            self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            * (
                int(role_responses_per_prompt)
                if role_phase == "leader" and role_responses_per_prompt is not None
                else self.config.actor_rollout_ref.rollout.n
            )
        )
        leader_phase_batch, leader_pad_size = pad_dataproto_to_divisor(leader_phase_batch, leader_mini_batch_size)
        metrics["hpf/leader_pad_size"] = float(leader_pad_size)
        metrics["hpf/leader_optimizer_steps"] = float(
            math.ceil(len(leader_phase_batch) / leader_mini_batch_size)
            * self.config.actor_rollout_ref.actor.ppo_epochs
        )
        leader_start = time.perf_counter()
        print(
            "[HPF] leader actor update start "
            f"step={self.global_steps} batch={len(leader_phase_batch)} pad={leader_pad_size}",
            flush=True,
        )
        leader_output = self._update_actor(
            leader_phase_batch,
            progress_label=f"hpf/leader/step-{self.global_steps}",
            progress_log_interval=progress_log_interval,
        )
        leader_elapsed = time.perf_counter() - leader_start
        print(
            "[HPF] leader actor update done "
            f"step={self.global_steps} elapsed_s={leader_elapsed:.2f}",
            flush=True,
        )
        metrics["timing_s/hpf/leader_update_actor"] = float(leader_elapsed)
        metrics["timing_s/hpf/update_actor_total"] = float(time.perf_counter() - hpf_update_start)
        leader_metrics = reduce_metrics(leader_output.meta_info["metrics"])
        metrics.update(rename_dict(leader_metrics, "hpf/leader/"))
        return DataProto.from_single_dict(data={}, meta_info={"metrics": metrics})

    @staticmethod
    def _object_array(values: list[Any]) -> np.ndarray:
        array = np.empty(len(values), dtype=object)
        array[:] = values
        return array

    def _hpf_tree_rollout_enabled(self) -> bool:
        hpf_config = self.config.algorithm.get("hpf_rlvr", {})
        tree_config = hpf_config.get("tree_rollout", {})
        return bool(hpf_config.get("enable", False)) and bool(tree_config.get("enable", False))

    def _extract_response_token_ids(self, rollout_output: DataProto) -> list[list[int]]:
        responses = rollout_output.batch["responses"].detach().cpu()
        response_mask = rollout_output.batch["response_mask"].detach().cpu().bool()
        token_ids = []
        for response, mask in zip(responses, response_mask, strict=True):
            token_ids.append(response[mask].tolist())
        return token_ids

    def _add_hpf_tree_metadata(
        self,
        output: DataProto,
        *,
        problem_uids: np.ndarray,
        prefix_indices: np.ndarray,
        suffix_indices: np.ndarray,
        prefix_token_ids: list[list[int]],
    ) -> None:
        prefix_uids = [
            f"{problem_uid}::prefix-{int(prefix_index)}"
            for problem_uid, prefix_index in zip(problem_uids, prefix_indices, strict=True)
        ]
        output.non_tensor_batch["hpf_problem_uid"] = np.asarray(problem_uids, dtype=object)
        output.non_tensor_batch["hpf_prefix_index"] = np.asarray(prefix_indices, dtype=np.int32)
        output.non_tensor_batch["hpf_suffix_index"] = np.asarray(suffix_indices, dtype=np.int32)
        output.non_tensor_batch["hpf_prefix_uid"] = np.asarray(prefix_uids, dtype=object)
        output.non_tensor_batch["hpf_prefix_ids"] = self._object_array(prefix_token_ids)

    @staticmethod
    def _drop_hpf_tree_unused_batch_keys(output: DataProto) -> None:
        # Tree suffix requests use prefix prefill, so rollout logprobs are not a
        # complete standalone old-logprob record. The HPF tree path extracts and
        # repacks them into explicit leader/follower old-logprob tensors.
        drop_keys = [key for key in ("rollout_log_probs",) if key in output.batch]
        if drop_keys:
            output.pop(batch_keys=drop_keys)

    def _generate_hpf_tree_sequences(
        self,
        gen_batch: DataProto,
        hpf_round_index: int | None = None,
        num_prefixes: int | None = None,
        num_suffixes: int | None = None,
        require_rollout_n_match: bool = True,
    ) -> tuple[DataProto, dict[str, float]]:
        tree_start = time.perf_counter()
        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise ValueError("HPF tree rollout does not support REMAX baseline generation.")
        if self.config.actor_rollout_ref.actor.get("use_rollout_log_probs", False):
            raise ValueError(
                "HPF tree rollout requires recomputing old_log_probs; disable actor.use_rollout_log_probs."
            )

        hpf_config = self.config.algorithm.get("hpf_rlvr", {})
        tree_config = hpf_config.get("tree_rollout", {})
        if num_prefixes is None:
            num_prefixes = int(tree_config.get("num_prefixes", 4))
        else:
            num_prefixes = int(num_prefixes)
        if num_suffixes is None:
            num_suffixes = int(tree_config.get("num_suffixes", 2))
        else:
            num_suffixes = int(num_suffixes)
        if num_prefixes <= 0:
            raise ValueError(f"HPF tree rollout num_prefixes must be positive, got {num_prefixes}.")
        if num_suffixes <= 0:
            raise ValueError(f"HPF tree rollout num_suffixes must be positive, got {num_suffixes}.")
        expected_rollout_n = num_prefixes * num_suffixes
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        if require_rollout_n_match and rollout_n != expected_rollout_n:
            raise ValueError(
                "HPF tree rollout requires actor_rollout_ref.rollout.n to equal "
                f"num_prefixes*num_suffixes ({expected_rollout_n}), got {rollout_n}."
            )

        max_response_length = int(hpf_config.get("max_response_length", self.config.data.max_response_length))
        progressive_block_size = int(hpf_config.get("progressive_block_size", 256))
        if hpf_round_index is None:
            hpf_round_index = self._get_hpf_round_index(None)
        horizon = min(int(hpf_round_index) * progressive_block_size, max_response_length)
        print(
            "[HPF] tree rollout start "
            f"step={self.global_steps} round={hpf_round_index} prompts={len(gen_batch)} prefixes={num_prefixes} "
            f"suffixes={num_suffixes} horizon={horizon} max_response={max_response_length}",
            flush=True,
        )

        prefix_batch = gen_batch.repeat(repeat_times=num_prefixes, interleave=True)
        prefix_batch.meta_info["temperature"] = float(tree_config.get("prefix_temperature", 1.0))
        prefix_batch.meta_info["top_p"] = float(tree_config.get("prefix_top_p", 1.0))
        prefix_batch.meta_info["max_tokens"] = horizon
        prefix_batch.meta_info["logprobs"] = True
        rollout_worker_divisor = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
        prefix_batch_padded, prefix_pad_size = pad_dataproto_to_divisor(prefix_batch, rollout_worker_divisor)
        prefix_start = time.perf_counter()
        print(
            "[HPF] prefix rollout start "
            f"step={self.global_steps} requests={len(prefix_batch)} pad={prefix_pad_size}",
            flush=True,
        )
        prefix_output = self.async_rollout_manager.generate_sequences(prefix_batch_padded)
        prefix_output = unpad_dataproto(prefix_output, prefix_pad_size)
        prefix_elapsed = time.perf_counter() - prefix_start
        print(
            "[HPF] prefix rollout done "
            f"step={self.global_steps} outputs={len(prefix_output)} elapsed_s={prefix_elapsed:.2f}",
            flush=True,
        )
        prefix_timing = prefix_output.meta_info.get("timing", {})
        prefix_output.meta_info.pop("timing", None)
        prefix_rollout_log_probs = prefix_output.batch.get("rollout_log_probs", None)
        if prefix_rollout_log_probs is None:
            raise ValueError("HPF tree prefix rollout did not return rollout_log_probs.")
        self._drop_hpf_tree_unused_batch_keys(prefix_output)

        prefix_token_ids = self._extract_response_token_ids(prefix_output)
        prefix_lengths = np.array([len(token_ids) for token_ids in prefix_token_ids], dtype=np.int32)
        needs_suffix = (prefix_lengths >= horizon) & (prefix_lengths < max_response_length)
        prefix_indices = np.tile(np.arange(num_prefixes, dtype=np.int32), len(gen_batch))
        problem_uids = np.asarray(prefix_batch.non_tensor_batch["uid"], dtype=object)

        suffix_output = None
        suffix_timing = {}
        suffix_cursor = 0
        suffix_elapsed = 0.0
        if bool(needs_suffix.any()):
            suffix_source = prefix_batch[needs_suffix]
            suffix_source = suffix_source.repeat(repeat_times=num_suffixes, interleave=True)
            source_indices = np.nonzero(needs_suffix)[0]
            repeated_source_indices = np.repeat(source_indices, num_suffixes)
            suffix_prefix_ids = [prefix_token_ids[index] for index in repeated_source_indices]
            suffix_source.non_tensor_batch["hpf_prefix_ids"] = self._object_array(suffix_prefix_ids)
            suffix_budgets = np.maximum(max_response_length - prefix_lengths[repeated_source_indices], 1)
            suffix_source.non_tensor_batch["__max_tokens__"] = suffix_budgets.astype(np.int32)
            suffix_source.meta_info["temperature"] = float(tree_config.get("suffix_temperature", 0.25))
            suffix_source.meta_info["top_p"] = float(tree_config.get("suffix_top_p", 1.0))
            suffix_source.meta_info["logprobs"] = True
            suffix_source_padded, suffix_pad_size = pad_dataproto_to_divisor(suffix_source, rollout_worker_divisor)
            suffix_start = time.perf_counter()
            print(
                "[HPF] suffix rollout start "
                f"step={self.global_steps} prefixes_needing_suffix={len(source_indices)} "
                f"requests={len(suffix_source)} pad={suffix_pad_size} "
                f"budget_mean={float(suffix_budgets.mean()):.1f} budget_max={int(suffix_budgets.max())}",
                flush=True,
            )
            suffix_output = self.async_rollout_manager.generate_sequences(suffix_source_padded)
            suffix_output = unpad_dataproto(suffix_output, suffix_pad_size)
            suffix_elapsed = time.perf_counter() - suffix_start
            print(
                "[HPF] suffix rollout done "
                f"step={self.global_steps} outputs={len(suffix_output)} elapsed_s={suffix_elapsed:.2f}",
                flush=True,
            )
            suffix_timing = suffix_output.meta_info.get("timing", {})
            suffix_output.meta_info.pop("timing", None)
            suffix_rollout_log_probs = suffix_output.batch.get("rollout_log_probs", None)
            if suffix_rollout_log_probs is None:
                raise ValueError("HPF tree suffix rollout did not return rollout_log_probs.")
            self._drop_hpf_tree_unused_batch_keys(suffix_output)
            internal_sampling_keys = [
                key
                for key in (
                    "hpf_prefix_ids",
                    "__temperature__",
                    "__top_p__",
                    "__top_k__",
                    "__max_tokens__",
                    "__max_new_tokens__",
                    "__logprobs__",
                )
                if key in suffix_output.non_tensor_batch
            ]
            if internal_sampling_keys:
                suffix_output.pop(non_tensor_batch_keys=internal_sampling_keys)
        else:
            suffix_pad_size = 0
            suffix_rollout_log_probs = None
            print(
                "[HPF] suffix rollout skipped "
                f"step={self.global_steps} prefixes_needing_suffix=0",
                flush=True,
            )

        ordered_outputs = []
        ordered_problem_uids = []
        ordered_prefix_indices = []
        ordered_suffix_indices = []
        ordered_prefix_token_ids = []
        ordered_leader_old_log_probs = []
        ordered_follower_old_log_probs = []
        for prefix_row in range(len(prefix_output)):
            if needs_suffix[prefix_row]:
                for suffix_idx in range(num_suffixes):
                    suffix_row = suffix_cursor + suffix_idx
                    ordered_outputs.append(suffix_output[suffix_row : suffix_row + 1])
                    ordered_problem_uids.append(problem_uids[prefix_row])
                    ordered_prefix_indices.append(prefix_indices[prefix_row])
                    ordered_suffix_indices.append(suffix_idx)
                    ordered_prefix_token_ids.append(prefix_token_ids[prefix_row])
                    follower_log_probs = suffix_rollout_log_probs[suffix_row].clone()
                    leader_log_probs = torch.zeros_like(follower_log_probs)
                    prefix_len = int(prefix_lengths[prefix_row])
                    if prefix_len > 0:
                        leader_log_probs[:prefix_len] = prefix_rollout_log_probs[prefix_row, :prefix_len]
                    ordered_leader_old_log_probs.append(leader_log_probs)
                    ordered_follower_old_log_probs.append(follower_log_probs)
                suffix_cursor += num_suffixes
            else:
                repeated_prefix_output = prefix_output[prefix_row : prefix_row + 1].repeat(
                    repeat_times=num_suffixes, interleave=True
                )
                for suffix_idx in range(num_suffixes):
                    ordered_outputs.append(repeated_prefix_output[suffix_idx : suffix_idx + 1])
                    ordered_problem_uids.append(problem_uids[prefix_row])
                    ordered_prefix_indices.append(prefix_indices[prefix_row])
                    ordered_suffix_indices.append(suffix_idx)
                    ordered_prefix_token_ids.append(prefix_token_ids[prefix_row])
                    leader_log_probs = prefix_rollout_log_probs[prefix_row].clone()
                    follower_log_probs = torch.zeros_like(leader_log_probs)
                    ordered_leader_old_log_probs.append(leader_log_probs)
                    ordered_follower_old_log_probs.append(follower_log_probs)

        tree_output = DataProto.concat(ordered_outputs)
        tree_output.batch["hpf_leader_rollout_old_log_probs"] = torch.stack(ordered_leader_old_log_probs, dim=0)
        tree_output.batch["hpf_follower_rollout_old_log_probs"] = torch.stack(ordered_follower_old_log_probs, dim=0)
        self._add_hpf_tree_metadata(
            tree_output,
            problem_uids=np.asarray(ordered_problem_uids, dtype=object),
            prefix_indices=np.asarray(ordered_prefix_indices, dtype=np.int32),
            suffix_indices=np.asarray(ordered_suffix_indices, dtype=np.int32),
            prefix_token_ids=ordered_prefix_token_ids,
        )
        tree_metrics = {
            "hpf/tree_rollout_enabled": 1.0,
            "hpf/horizon_round_index": float(hpf_round_index),
            "hpf/tree_num_prefixes": float(num_prefixes),
            "hpf/tree_num_suffixes": float(num_suffixes),
            "hpf/tree_responses_per_prompt": float(expected_rollout_n),
            "hpf/tree_horizon_tokens": float(horizon),
            "hpf/tree_prefix_tokens_mean": float(prefix_lengths.mean()) if len(prefix_lengths) else 0.0,
            "hpf/tree_prefix_stopped_frac": float((~needs_suffix).mean()) if len(needs_suffix) else 0.0,
            "hpf/tree_rollout_log_probs_available": 1.0,
            "hpf/tree_prefix_pad_size": float(prefix_pad_size),
            "hpf/tree_suffix_pad_size": float(suffix_pad_size) if suffix_output is not None else 0.0,
            "timing_s/hpf/tree_rollout_total_wall": float(time.perf_counter() - tree_start),
            "timing_s/hpf/prefix_rollout_wall": float(prefix_elapsed),
            "timing_s/hpf/suffix_rollout_wall": float(suffix_elapsed),
        }
        print(
            "[HPF] tree rollout done "
            f"step={self.global_steps} trajectories={len(tree_output)} "
            f"elapsed_s={tree_metrics['timing_s/hpf/tree_rollout_total_wall']:.2f}",
            flush=True,
        )
        tree_timing = {}
        for key, value in prefix_timing.items():
            tree_timing[f"hpf_tree/prefix/{key}"] = value
        for key, value in suffix_timing.items():
            tree_timing[f"hpf_tree/suffix/{key}"] = value
        tree_output.meta_info["timing"] = tree_timing
        return tree_output, tree_metrics

    def _generate_hpf_transition_aware_sequences(
        self,
        gen_batch: DataProto,
        *,
        hpf_round_index: int | None = None,
    ) -> tuple[DataProto, DataProto, dict[str, float]]:
        """Sample paired mixed-policy trajectories at the current and next cuts.

        A single high-temperature rollout produces each next-cut prefix. The
        current-cut prefix is its truncation. All current- and next-cut
        prefixes that require continuation are then submitted together in one
        low-temperature rollout batch. No replica affinity is assumed.
        """
        rollout_start = time.perf_counter()
        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise ValueError("HPF transition-aware rollout does not support REMAX baseline generation.")
        if self.config.actor_rollout_ref.actor.get("use_rollout_log_probs", False):
            raise ValueError(
                "HPF transition-aware rollout records its own mixed-policy log probabilities; "
                "disable actor.use_rollout_log_probs."
            )

        hpf_config = self.config.algorithm.get("hpf_rlvr", {})
        tree_config = hpf_config.get("tree_rollout", {})
        num_prefixes = int(tree_config.get("num_prefixes", 4))
        num_suffixes = int(tree_config.get("num_suffixes", 1))
        if num_prefixes <= 0:
            raise ValueError(f"HPF transition-aware rollout num_prefixes must be positive, got {num_prefixes}.")
        if num_suffixes != 1:
            raise ValueError(
                "HPF transition-aware rollout requires tree_rollout.num_suffixes=1, "
                f"got {num_suffixes}."
            )
        rollout_n = int(self.config.actor_rollout_ref.rollout.n)
        if rollout_n != num_prefixes:
            raise ValueError(
                "HPF transition-aware rollout requires actor_rollout_ref.rollout.n to equal "
                f"tree_rollout.num_prefixes ({num_prefixes}), got {rollout_n}."
            )

        max_response_length = int(hpf_config.get("max_response_length", self.config.data.max_response_length))
        progressive_block_size = int(hpf_config.get("progressive_block_size", 256))
        if hpf_round_index is None:
            hpf_round_index = self._get_hpf_round_index(None)
        current_horizon = min(int(hpf_round_index) * progressive_block_size, max_response_length)
        next_horizon = min(current_horizon + progressive_block_size, max_response_length)
        mixed_policy_config = hpf_config.get("mixed_policy_grpo", {})
        suffix_window_size_value = mixed_policy_config.get("suffix_window_size", None)
        suffix_window_size = (
            None
            if suffix_window_size_value is None or str(suffix_window_size_value).lower() == "null"
            else int(suffix_window_size_value)
        )
        transition_width = next_horizon - current_horizon
        if suffix_window_size is not None and suffix_window_size < transition_width:
            raise ValueError(
                "HPF transition-aware suffix_window_size must cover the next-cut transition before rollout: "
                f"suffix_window_size={suffix_window_size}, "
                f"next_horizon-current_horizon={transition_width}."
            )
        prefix_temperature = float(tree_config.get("prefix_temperature", 1.0))
        prefix_top_p = float(tree_config.get("prefix_top_p", 1.0))
        suffix_temperature = float(tree_config.get("suffix_temperature", 0.25))
        suffix_top_p = float(tree_config.get("suffix_top_p", 1.0))
        print(
            "[HPF] transition-aware rollout start "
            f"step={self.global_steps} round={hpf_round_index} prompts={len(gen_batch)} "
            f"prefixes={num_prefixes} current_horizon={current_horizon} "
            f"next_horizon={next_horizon} max_response={max_response_length}",
            flush=True,
        )

        rollout_worker_divisor = int(self.config.actor_rollout_ref.rollout.agent.num_workers)
        high_batch = gen_batch.repeat(repeat_times=num_prefixes, interleave=True)
        high_batch.meta_info["temperature"] = prefix_temperature
        high_batch.meta_info["top_p"] = prefix_top_p
        high_batch.meta_info["max_tokens"] = next_horizon
        high_batch.meta_info["logprobs"] = True
        high_batch_padded, high_pad_size = pad_dataproto_to_divisor(high_batch, rollout_worker_divisor)
        high_start = time.perf_counter()
        print(
            "[HPF] transition-aware high-temperature prefix rollout start "
            f"step={self.global_steps} requests={len(high_batch)} pad={high_pad_size} "
            f"max_tokens={next_horizon}",
            flush=True,
        )
        high_output = self.async_rollout_manager.generate_sequences(high_batch_padded)
        high_output = unpad_dataproto(high_output, high_pad_size)
        if len(high_output) != len(high_batch):
            raise ValueError(
                "HPF transition-aware high-temperature rollout returned an unexpected number of rows: "
                f"expected {len(high_batch)}, got {len(high_output)}."
            )
        high_elapsed = time.perf_counter() - high_start
        print(
            "[HPF] transition-aware high-temperature prefix rollout done "
            f"step={self.global_steps} outputs={len(high_output)} elapsed_s={high_elapsed:.2f}",
            flush=True,
        )
        high_timing = high_output.meta_info.get("timing", {})
        high_output.meta_info.pop("timing", None)
        high_rollout_log_probs = high_output.batch.get("rollout_log_probs", None)
        if high_rollout_log_probs is None:
            raise ValueError("HPF transition-aware high-temperature rollout did not return rollout_log_probs.")
        self._drop_hpf_tree_unused_batch_keys(high_output)

        high_token_ids = self._extract_response_token_ids(high_output)
        prefix_plan = build_hpf_transition_prefix_plan(
            high_token_ids,
            current_horizon=current_horizon,
            next_horizon=next_horizon,
            max_response_length=max_response_length,
        )
        current_source_rows = np.nonzero(prefix_plan.current_needs_suffix)[0]
        next_source_rows = np.nonzero(prefix_plan.next_needs_suffix)[0]
        request_source_rows = prefix_plan.request_source_rows
        request_cut_indices = prefix_plan.request_cut_indices
        request_prefix_ids = [
            (
                prefix_plan.current_prefix_ids[source_row]
                if cut_index == 0
                else prefix_plan.next_prefix_ids[source_row]
            )
            for source_row, cut_index in zip(request_source_rows, request_cut_indices, strict=True)
        ]
        low_output = None
        low_rollout_log_probs = None
        low_timing = {}
        low_elapsed = 0.0
        low_pad_size = 0
        low_row_by_cut_and_source: dict[tuple[int, int], int] = {}
        if len(request_source_rows):
            low_source = high_batch.select_idxs(request_source_rows)
            request_prefix_lengths = np.asarray([len(token_ids) for token_ids in request_prefix_ids], dtype=np.int32)
            low_source.non_tensor_batch["hpf_prefix_ids"] = self._object_array(request_prefix_ids)
            low_budgets = np.maximum(max_response_length - request_prefix_lengths, 1)
            low_source.non_tensor_batch["__max_tokens__"] = low_budgets.astype(np.int32)
            low_source.meta_info["temperature"] = suffix_temperature
            low_source.meta_info["top_p"] = suffix_top_p
            low_source.meta_info["logprobs"] = True
            low_source_padded, low_pad_size = pad_dataproto_to_divisor(low_source, rollout_worker_divisor)
            low_start = time.perf_counter()
            print(
                "[HPF] transition-aware combined low-temperature suffix rollout start "
                f"step={self.global_steps} current_requests={len(current_source_rows)} "
                f"next_requests={len(next_source_rows)} requests={len(low_source)} pad={low_pad_size} "
                f"budget_mean={float(low_budgets.mean()):.1f} budget_max={int(low_budgets.max())}",
                flush=True,
            )
            low_output = self.async_rollout_manager.generate_sequences(low_source_padded)
            low_output = unpad_dataproto(low_output, low_pad_size)
            if len(low_output) != len(low_source):
                raise ValueError(
                    "HPF transition-aware low-temperature rollout returned an unexpected number of rows: "
                    f"expected {len(low_source)}, got {len(low_output)}."
                )
            low_elapsed = time.perf_counter() - low_start
            print(
                "[HPF] transition-aware combined low-temperature suffix rollout done "
                f"step={self.global_steps} outputs={len(low_output)} elapsed_s={low_elapsed:.2f}",
                flush=True,
            )
            low_timing = low_output.meta_info.get("timing", {})
            low_output.meta_info.pop("timing", None)
            low_rollout_log_probs = low_output.batch.get("rollout_log_probs", None)
            if low_rollout_log_probs is None:
                raise ValueError("HPF transition-aware low-temperature rollout did not return rollout_log_probs.")
            self._drop_hpf_tree_unused_batch_keys(low_output)
            internal_sampling_keys = [
                key
                for key in (
                    "hpf_prefix_ids",
                    "__temperature__",
                    "__top_p__",
                    "__top_k__",
                    "__max_tokens__",
                    "__max_new_tokens__",
                    "__logprobs__",
                )
                if key in low_output.non_tensor_batch
            ]
            if internal_sampling_keys:
                low_output.pop(non_tensor_batch_keys=internal_sampling_keys)
            low_row_by_cut_and_source = {
                (int(cut_index), int(source_row)): low_row
                for low_row, (cut_index, source_row) in enumerate(
                    zip(request_cut_indices, request_source_rows, strict=True)
                )
            }
        else:
            print(
                "[HPF] transition-aware combined low-temperature suffix rollout skipped "
                f"step={self.global_steps} requests=0",
                flush=True,
            )

        problem_uids = np.asarray(high_batch.non_tensor_batch["uid"], dtype=object)
        prefix_indices = np.tile(np.arange(num_prefixes, dtype=np.int32), len(gen_batch))

        def assemble_cut_output(
            *,
            cut_index: int,
            prefix_ids: list[list[int]],
            prefix_lengths: np.ndarray,
        ) -> DataProto:
            ordered_outputs = []
            leader_old_log_probs = []
            follower_old_log_probs = []
            for source_row in range(len(high_output)):
                low_row = low_row_by_cut_and_source.get((cut_index, source_row))
                if low_row is None:
                    ordered_outputs.append(high_output[source_row : source_row + 1])
                    leader_log_probs = high_rollout_log_probs[source_row].clone()
                    follower_log_probs = torch.zeros_like(leader_log_probs)
                else:
                    ordered_outputs.append(low_output[low_row : low_row + 1])
                    follower_log_probs = low_rollout_log_probs[low_row].clone()
                    leader_log_probs = torch.zeros_like(follower_log_probs)
                    prefix_length = int(prefix_lengths[source_row])
                    if prefix_length > 0:
                        leader_log_probs[:prefix_length] = high_rollout_log_probs[source_row, :prefix_length]
                leader_old_log_probs.append(leader_log_probs)
                follower_old_log_probs.append(follower_log_probs)

            output = DataProto.concat(ordered_outputs)
            output.batch["hpf_leader_rollout_old_log_probs"] = torch.stack(leader_old_log_probs, dim=0)
            output.batch["hpf_follower_rollout_old_log_probs"] = torch.stack(follower_old_log_probs, dim=0)
            self._add_hpf_tree_metadata(
                output,
                problem_uids=problem_uids,
                prefix_indices=prefix_indices,
                suffix_indices=np.zeros(len(output), dtype=np.int32),
                prefix_token_ids=prefix_ids,
            )
            pair_uids = [
                f"{problem_uid}::prefix-{int(prefix_index)}"
                for problem_uid, prefix_index in zip(problem_uids, prefix_indices, strict=True)
            ]
            output.non_tensor_batch["hpf_transition_pair_uid"] = np.asarray(pair_uids, dtype=object)
            output.non_tensor_batch["hpf_transition_cut"] = np.asarray(
                ["current" if cut_index == 0 else "next"] * len(output), dtype=object
            )
            horizon = current_horizon if cut_index == 0 else next_horizon
            output.non_tensor_batch["hpf_transition_prefix_horizon"] = np.full(
                len(output), horizon, dtype=np.int32
            )
            response_ids = self._extract_response_token_ids(output)
            suffix_ids = [
                token_ids[len(prefix_token_ids) :]
                for token_ids, prefix_token_ids in zip(response_ids, prefix_ids, strict=True)
            ]
            output.non_tensor_batch["hpf_transition_suffix_ids"] = self._object_array(suffix_ids)
            return output

        current_output = assemble_cut_output(
            cut_index=0,
            prefix_ids=prefix_plan.current_prefix_ids,
            prefix_lengths=prefix_plan.current_prefix_lengths,
        )
        next_output = assemble_cut_output(
            cut_index=1,
            prefix_ids=prefix_plan.next_prefix_ids,
            prefix_lengths=prefix_plan.next_prefix_lengths,
        )
        total_elapsed = time.perf_counter() - rollout_start
        metrics = {
            "hpf/tree_rollout_enabled": 1.0,
            "hpf/transition_aware_rollout_enabled": 1.0,
            "hpf/horizon_round_index": float(hpf_round_index),
            "hpf/tree_num_prefixes": float(num_prefixes),
            "hpf/tree_num_suffixes": 1.0,
            "hpf/tree_responses_per_prompt": float(num_prefixes),
            "hpf/tree_horizon_tokens": float(current_horizon),
            "hpf/transition_next_horizon_tokens": float(next_horizon),
            "hpf/transition_current_prefix_tokens_mean": float(prefix_plan.current_prefix_lengths.mean()),
            "hpf/transition_next_prefix_tokens_mean": float(prefix_plan.next_prefix_lengths.mean()),
            "hpf/transition_current_suffix_empty_frac": float((~prefix_plan.current_needs_suffix).mean()),
            "hpf/transition_next_suffix_empty_frac": float((~prefix_plan.next_needs_suffix).mean()),
            "hpf/tree_rollout_log_probs_available": 1.0,
            "hpf/transition_high_prefix_pad_size": float(high_pad_size),
            "hpf/transition_low_suffix_pad_size": float(low_pad_size),
            "timing_s/hpf/transition_high_prefix_rollout_wall": float(high_elapsed),
            "timing_s/hpf/transition_combined_suffix_rollout_wall": float(low_elapsed),
            "timing_s/hpf/tree_rollout_total_wall": float(total_elapsed),
        }
        timing = {}
        for key, value in high_timing.items():
            timing[f"hpf_transition/high_prefix/{key}"] = value
        for key, value in low_timing.items():
            timing[f"hpf_transition/combined_suffix/{key}"] = value
        current_output.meta_info["timing"] = timing
        print(
            "[HPF] transition-aware rollout done "
            f"step={self.global_steps} current_trajectories={len(current_output)} "
            f"next_trajectories={len(next_output)} elapsed_s={total_elapsed:.2f}",
            flush=True,
        )
        return current_output, next_output, metrics

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        # modify key name
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)
        hpf_config = self.config.algorithm.get("hpf_rlvr", {})
        role_phased_config = hpf_config.get("role_phased_training", {})
        role_phased_training = self._parse_hpf_bool(role_phased_config.get("enable", False), False)
        mixed_policy_grpo = self._parse_hpf_bool(
            hpf_config.get("mixed_policy_grpo", {}).get("enable", False), False
        )
        transition_aware_rollout = self._parse_hpf_bool(
            hpf_config.get("mixed_policy_grpo", {})
            .get("transition_aware_rollout", {})
            .get("enable", False),
            False,
        )
        transition_aware_optimization = self._parse_hpf_bool(
            hpf_config.get("mixed_policy_grpo", {})
            .get("transition_aware_optimization", {})
            .get("enable", False),
            False,
        )
        sir_config = self.config.algorithm.get("sir", {})
        sir_enabled = self._parse_hpf_bool(sir_config.get("enable", False), False)
        sir_pool_mode = str(sir_config.get("pool_mode", "independent")).strip().lower()
        sir_branched_pool = False
        if sir_enabled:
            rollout_pool_size = int(self.config.actor_rollout_ref.rollout.n)
            sir_selected_count = int(sir_config.get("selected_count"))
            sir_block_length = int(sir_config.get("block_length"))
            sir_alpha = float(sir_config.get("alpha"))
            if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
                raise ValueError("algorithm.sir is only supported with algorithm.adv_estimator=grpo")
            if rollout_pool_size < 2:
                raise ValueError(f"SIR rollout pool N must be at least 2, got {rollout_pool_size}")
            if sir_selected_count < 2 or sir_selected_count > rollout_pool_size:
                raise ValueError(
                    "algorithm.sir.selected_count must satisfy 2 <= K <= rollout.n; "
                    f"got K={sir_selected_count}, N={rollout_pool_size}"
                )
            if sir_block_length <= 0 or sir_block_length > int(self.config.data.max_response_length):
                raise ValueError(
                    "algorithm.sir.block_length must satisfy 1 <= B <= data.max_response_length; "
                    f"got B={sir_block_length}, max={self.config.data.max_response_length}"
                )
            if not np.isfinite(sir_alpha) or sir_alpha <= 0:
                raise ValueError(f"algorithm.sir.alpha must be finite and positive, got {sir_alpha}")
            if sir_pool_mode not in {"independent", "branched_prefix"}:
                raise ValueError(
                    "algorithm.sir.pool_mode must be 'independent' or 'branched_prefix'; "
                    f"got {sir_pool_mode!r}"
                )
            sir_branched_pool = sir_pool_mode == "branched_prefix"
            if sir_branched_pool and (
                rollout_pool_size <= sir_selected_count
                or rollout_pool_size % sir_selected_count != 0
            ):
                raise ValueError(
                    "branched-prefix SIR requires N > K and N divisible by K; "
                    f"got N={rollout_pool_size}, K={sir_selected_count}"
                )
            if not self._parse_hpf_bool(
                self.config.actor_rollout_ref.rollout.get("calculate_log_probs", False), False
            ):
                raise ValueError(
                    "algorithm.sir requires actor_rollout_ref.rollout.calculate_log_probs=True"
                )
            if (
                self._parse_hpf_bool(hpf_config.get("enable", False), False)
                or self._parse_hpf_bool(hpf_config.get("tree_rollout", {}).get("enable", False), False)
                or mixed_policy_grpo
            ):
                raise ValueError("algorithm.sir is a GRPO-baseline mode and cannot be combined with HPF rollout paths")
            print(
                "[SIR] enabled "
                f"pool_mode={sir_pool_mode} N={rollout_pool_size} K={sir_selected_count} "
                f"B={sir_block_length} alpha={sir_alpha} "
                "resampling=weighted_without_replacement",
                flush=True,
            )
        physical_total_epochs = int(self.config.trainer.total_epochs)
        if role_phased_training:
            follower_phase_epochs = int(role_phased_config.get("follower_epochs", 1))
            leader_phase_epochs = int(role_phased_config.get("leader_epochs", 1))
            physical_total_epochs *= follower_phase_epochs + leader_phase_epochs
            physical_total_epochs = max(
                physical_total_epochs,
                math.ceil(self.total_training_steps / len(self.train_dataloader)),
            )
            print(
                "[HPF] role-phased training enabled "
                f"follower_epochs={follower_phase_epochs} leader_epochs={leader_phase_epochs} "
                f"batches_per_pass={len(self.train_dataloader)} physical_total_epochs={physical_total_epochs}",
                flush=True,
            )

        SkipManager.init(self.config)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                logger.finish()
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        SkipManager.set_step(self.global_steps)

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, physical_total_epochs):
            role_phase, role_epoch, role_round_index = self._get_hpf_role_phase(epoch)
            if role_phase is not None:
                print(
                    "[HPF] role phase start "
                    f"round={role_round_index} role={role_phase} role_epoch={role_epoch} "
                    f"physical_epoch={epoch + 1}",
                    flush=True,
                )
            for batch_dict in self._iterate_train_dataloader():
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_n = int(self.config.actor_rollout_ref.rollout.n)
                use_hpf_tree_rollout = self._hpf_tree_rollout_enabled()
                hpf_round_index = (
                    role_round_index
                    if use_hpf_tree_rollout and role_round_index is not None
                    else self._get_hpf_round_index(epoch)
                    if use_hpf_tree_rollout
                    else None
                )
                effective_rollout_n = rollout_n
                fresh_num_prefixes = None
                fresh_num_suffixes = None
                if use_hpf_tree_rollout and role_phase == "leader":
                    tree_config = hpf_config.get("tree_rollout", {})
                    fresh_tree_config = hpf_config.get("fresh_tree_rollout", {})
                    fresh_num_prefixes_value = fresh_tree_config.get("num_prefixes", None)
                    fresh_num_suffixes_value = fresh_tree_config.get("num_suffixes", None)
                    fresh_num_prefixes = int(
                        tree_config.get("num_prefixes", 4)
                        if fresh_num_prefixes_value is None
                        else fresh_num_prefixes_value
                    )
                    fresh_num_suffixes = int(
                        tree_config.get("num_suffixes", 2)
                        if fresh_num_suffixes_value is None
                        else fresh_num_suffixes_value
                    )
                    effective_rollout_n = fresh_num_prefixes * fresh_num_suffixes
                initial_generation_count = (
                    int(sir_config.get("selected_count")) if sir_branched_pool else rollout_n
                )
                gen_batch_output = gen_batch.repeat(
                    repeat_times=initial_generation_count, interleave=True
                )

                if use_hpf_tree_rollout:
                    combined_gen_batch = None
                    num_sampled_prompts = None
                elif self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # NOTE: REMAX needs one sampled rollout plus one greedy baseline per prompt.
                    # Keep them in a single agent-loop/vLLM request to avoid sending a second
                    # rollout after replicas have been put to sleep, which can leave async vLLM
                    # engines in an invalid state for multi-turn agent workloads.
                    gen_batch_output.non_tensor_batch["__do_sample__"] = np.ones(len(gen_batch_output), dtype=bool)
                    gen_baseline_batch = gen_batch.slice(0, None)
                    gen_baseline_batch.non_tensor_batch["__do_sample__"] = np.zeros(len(gen_baseline_batch), dtype=bool)
                    combined_gen_batch = DataProto.concat([gen_batch_output, gen_baseline_batch])
                    num_sampled_prompts = len(gen_batch_output)
                else:
                    combined_gen_batch = gen_batch_output
                    num_sampled_prompts = len(gen_batch_output)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    transition_next_gen_output = None
                    with marked_timer("gen", timing_raw, color="red"):
                        if curr_step_profile:
                            self.llm_server_manager.start_profile()
                        if use_hpf_tree_rollout:
                            if transition_aware_rollout:
                                (
                                    gen_batch_output,
                                    transition_next_gen_output,
                                    hpf_tree_metrics,
                                ) = self._generate_hpf_transition_aware_sequences(
                                    gen_batch,
                                    hpf_round_index=hpf_round_index,
                                )
                            else:
                                gen_batch_output, hpf_tree_metrics = self._generate_hpf_tree_sequences(
                                    gen_batch,
                                    hpf_round_index=hpf_round_index,
                                    num_prefixes=fresh_num_prefixes,
                                    num_suffixes=fresh_num_suffixes,
                                    require_rollout_n_match=role_phase != "leader",
                                )
                            metrics.update(hpf_tree_metrics)
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)
                        else:
                            if sir_branched_pool:
                                print(
                                    "[SIR] initial rollout start "
                                    f"step={self.global_steps} prompts={len(gen_batch)} "
                                    f"initial_per_prompt={initial_generation_count}",
                                    flush=True,
                                )
                            combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
                            if sir_branched_pool:
                                print(
                                    "[SIR] initial rollout done "
                                    f"step={self.global_steps} outputs={len(combined_gen_output)}",
                                    flush=True,
                                )
                                combined_gen_output, branched_metrics = self._generate_sir_branched_pool(
                                    gen_batch,
                                    combined_gen_output,
                                    pool_size=rollout_n,
                                    initial_count=initial_generation_count,
                                    block_length=int(sir_config.get("block_length")),
                                    seed=int(sir_config.get("seed", 42)),
                                )
                                metrics.update(branched_metrics)
                                num_sampled_prompts = len(combined_gen_output)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.llm_server_manager.stop_profile()

                        if not use_hpf_tree_rollout:
                            timing_raw.update(combined_gen_output.meta_info["timing"])
                            combined_gen_output.meta_info.pop("timing", None)

                    if not use_hpf_tree_rollout:
                        gen_batch_output = combined_gen_output.slice(0, num_sampled_prompts)
                        if "__do_sample__" in gen_batch_output.non_tensor_batch:
                            gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"])

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        gen_baseline_output = combined_gen_output.slice(num_sampled_prompts, None)
                        if "__do_sample__" in gen_baseline_output.non_tensor_batch:
                            gen_baseline_output.pop(non_tensor_batch_keys=["__do_sample__"])

                        if self.use_rm and "rm_scores" not in gen_baseline_output.batch.keys():
                            baseline_reward = self._compute_reward_colocate(gen_baseline_output)
                            gen_baseline_output = gen_baseline_output.union(baseline_reward)

                        reward_baseline_tensor = gen_baseline_output.batch["rm_scores"].sum(dim=-1)
                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_output
                    if use_hpf_tree_rollout:
                        del combined_gen_batch
                    else:
                        del combined_gen_batch, combined_gen_output
                    # Repeat prompts to align with generated responses. In SIR mode,
                    # first retain and record the full N-way pool, then pass only K
                    # resampled rows to the unchanged reward/GRPO update path.
                    prompt_batch_for_hpf = batch
                    transition_next_batch = None
                    if sir_enabled:
                        batch, sir_metrics = self._apply_sir_resampling(
                            batch,
                            gen_batch_output,
                            pool_size=rollout_n,
                            sir_config=sir_config,
                        )
                        effective_rollout_n = int(sir_config.get("selected_count"))
                        metrics.update(sir_metrics)
                    else:
                        if transition_next_gen_output is not None:
                            transition_next_batch = batch.repeat(
                                repeat_times=effective_rollout_n, interleave=True
                            )
                            transition_next_batch = transition_next_batch.union(transition_next_gen_output)
                        batch = batch.repeat(repeat_times=effective_rollout_n, interleave=True)
                        batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if transition_next_batch is not None and "response_mask" not in transition_next_batch.batch.keys():
                        transition_next_batch.batch["response_mask"] = compute_response_mask(transition_next_batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)
                        transition_next_reward_extra_infos_dict = {}
                        if transition_next_batch is not None:
                            if self.use_rm and "rm_scores" not in transition_next_batch.batch.keys():
                                transition_next_reward = self._compute_reward_colocate(transition_next_batch)
                                transition_next_batch = transition_next_batch.union(transition_next_reward)
                            (
                                transition_next_reward_tensor,
                                transition_next_reward_extra_infos_dict,
                            ) = extract_reward(transition_next_batch)
                            transition_next_batch.batch["token_level_scores"] = transition_next_reward_tensor
                            transition_return = estimate_hpf_transition_return(
                                batch,
                                transition_next_batch,
                                current_reward_tensor=reward_tensor,
                                next_reward_tensor=transition_next_reward_tensor,
                            )
                            metrics.update(transition_return.metrics())
                            batch.meta_info["hpf_transition_return_estimate"] = transition_return.delta_mean
                            transition_next_batch.meta_info[
                                "hpf_transition_return_estimate"
                            ] = transition_return.delta_mean
                            print(
                                "[HPF] transition return estimated "
                                f"step={self.global_steps} pairs={transition_return.num_pairs} "
                                f"current={transition_return.current_mean:.6f} "
                                f"next={transition_return.next_mean:.6f} "
                                f"delta={transition_return.delta_mean:.6f} "
                                f"delta_std={transition_return.delta_std:.6f}",
                                flush=True,
                            )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    transition_manages_old_log_prob = (
                        use_hpf_tree_rollout
                        and transition_aware_optimization
                        and not self.config.algorithm.use_kl_in_reward
                    )
                    skip_shared_old_log_prob = (
                        use_hpf_tree_rollout
                        and not self.config.algorithm.use_kl_in_reward
                        and (rollout_corr_config is None or transition_manages_old_log_prob)
                    )
                    if skip_shared_old_log_prob:
                        metrics["hpf/skipped_shared_old_log_prob"] = 1.0
                        if transition_manages_old_log_prob:
                            print(
                                "[HPF] shared old_log_prob skipped "
                                f"step={self.global_steps} source=transition_configured_behavior_log_prob",
                                flush=True,
                            )
                    elif bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    if not skip_shared_old_log_prob:
                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                            if transition_next_batch is not None:
                                transition_next_batch.batch["token_level_rewards"] = transition_next_batch.batch[
                                    "token_level_scores"
                                ]

                        if use_hpf_tree_rollout and not mixed_policy_grpo:
                            metrics["hpf/skipped_shared_advantage"] = 1.0
                        else:
                            # Compute rollout correction: IS weights, rejection sampling, and metrics
                            # Only runs in decoupled mode (computes once per batch using stable π_old)
                            # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs  # Only in decoupled mode
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import (
                                    compute_rollout_correction_and_add_to_batch,
                                )

                                # Compute IS weights, apply rejection sampling, compute metrics
                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(
                                    batch, rollout_corr_config
                                )
                                # IS and off-policy metrics already have rollout_corr/ prefix
                                metrics.update(is_metrics)

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=effective_rollout_n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )
                            if transition_next_batch is not None:
                                if transition_next_reward_extra_infos_dict:
                                    transition_next_batch.non_tensor_batch.update(
                                        {
                                            key: np.array(value)
                                            for key, value in transition_next_reward_extra_infos_dict.items()
                                        }
                                    )
                                transition_next_batch = compute_advantage(
                                    transition_next_batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                    num_repeat=effective_rollout_n,
                                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                    config=self.config.algorithm,
                                )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup > self.global_steps:
                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        self.checkpoint_manager.update_weights(self.global_steps)
                    else:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            if transition_aware_optimization:
                                if transition_next_batch is None:
                                    raise ValueError(
                                        "Transition-aware mixed policy optimization requires the next-cut batch."
                                    )
                                actor_output = self._update_actor_hpf_transition_aware_mixed_policy(
                                    batch,
                                    transition_next_batch,
                                    hpf_round_index=hpf_round_index,
                                )
                            elif mixed_policy_grpo:
                                actor_output = self._update_actor_hpf_mixed_policy_grpo(
                                    batch,
                                    hpf_round_index=hpf_round_index,
                                )
                            elif self.config.algorithm.get("hpf_rlvr", {}).get("enable", False):
                                actor_output = self._update_actor_hpf_masked_grpo(
                                    batch,
                                    hpf_round_index=hpf_round_index,
                                    prompt_batch=prompt_batch_for_hpf,
                                    gen_batch=gen_batch,
                                    role_phase=role_phase or "both",
                                    role_responses_per_prompt=effective_rollout_n,
                                )
                            else:
                                actor_progress_log_interval = int(
                                    self.config.trainer.get("actor_progress_log_interval", 0) or 0
                                )
                                actor_output = self._update_actor(
                                    batch,
                                    progress_label=(
                                        f"grpo/step-{self.global_steps}"
                                        if actor_progress_log_interval > 0
                                        else None
                                    ),
                                    progress_log_interval=actor_progress_log_interval,
                                    responses_per_prompt=effective_rollout_n,
                                )

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                        if transition_next_batch is not None:
                            self._log_rollout_data(
                                transition_next_batch,
                                transition_next_reward_extra_infos_dict,
                                timing_raw,
                                os.path.join(rollout_data_dir, "transition_next"),
                            )

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                if role_phase is not None:
                    metrics.update(
                        {
                            "hpf/role_phased_training_enabled": 1.0,
                            "hpf/role_phase_follower": float(role_phase == "follower"),
                            "hpf/role_phase_leader": float(role_phase == "leader"),
                            "hpf/role_phase_epoch": float(role_epoch),
                            "hpf/role_phase_round": float(role_round_index),
                        }
                    )
                if self.config.algorithm.get("hpf_rlvr", {}).get("enable", False) and "advantages" not in batch.batch:
                    metrics["hpf/metrics_placeholder_advantages"] = 1.0
                    batch.batch["advantages"] = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                    batch.batch["returns"] = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # Per-request spec decode metrics.
                metrics.update(
                    compute_spec_decode_metrics(
                        batch.non_tensor_batch.get("spec_num_draft_tokens", None),
                        batch.non_tensor_batch.get("spec_num_accepted_tokens", None),
                        batch.non_tensor_batch.get("spec_num_verify_steps", None),
                    )
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                SkipManager.set_step(self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    self._shutdown_dump_executor()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    logger.finish()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()
        logger.finish()
