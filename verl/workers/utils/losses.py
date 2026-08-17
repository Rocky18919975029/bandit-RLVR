# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_policy_loss_vanilla,
    compute_value_loss,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def _all_reduce_sum(value: torch.Tensor, dp_group) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM, group=dp_group)
    return value


def _hpf_transition_global_batch_info(
    mask: torch.Tensor,
    *,
    dp_size: int,
    loss_scale_factor,
    dp_group,
) -> dict:
    """Build cut-specific normalizers for one synchronized DP micro-batch."""
    global_tokens = _all_reduce_sum(mask.sum(dtype=torch.float32).detach().clone(), dp_group)
    global_sequences = _all_reduce_sum(
        (mask.sum(dim=-1) > 0).sum(dtype=torch.float32).detach().clone(), dp_group
    )
    if global_tokens.item() <= 0 or global_sequences.item() <= 0:
        raise ValueError(
            "Transition-aware mixed policy optimization requires both cuts in every DP micro-batch."
        )
    return {
        "dp_size": dp_size,
        "batch_num_tokens": global_tokens.item(),
        "global_batch_size": global_sequences.item(),
        "loss_scale_factor": loss_scale_factor,
    }


def _hpf_transition_global_scalar(local_loss: torch.Tensor, *, dp_size: int, dp_group) -> torch.Tensor:
    """Recover the global scalar represented by DP-scaled local loss contributions."""
    global_loss = _all_reduce_sum(local_loss.detach().clone(), dp_group)
    return global_loss / dp_size


def _hpf_transition_combine_policy_losses(
    current_pg_loss: torch.Tensor,
    next_pg_loss: torch.Tensor,
    *,
    static_offset: float,
    lambda_trans: float,
    dp_size: int,
    dp_group,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine two cut-specific minimizing losses into the transition objective."""
    current_global = _hpf_transition_global_scalar(current_pg_loss, dp_size=dp_size, dp_group=dp_group)
    next_global = _hpf_transition_global_scalar(next_pg_loss, dp_size=dp_size, dp_group=dp_group)
    # compute_policy_loss_vanilla returns -L. Thus the quantity inside the
    # desired ReLU is -(B + pg_current - pg_next).
    surrogate = static_offset + current_global - next_global
    active = (surrogate < 0).to(current_pg_loss.dtype)
    penalty = torch.relu(-surrogate)

    # The constant static offset selects the hinge branch but contributes no
    # gradient. Omitting it from this differentiable expression preserves the
    # exact ReLU subgradient while metrics below retain the full objective.
    loss = current_pg_loss + lambda_trans * active * (next_pg_loss - current_pg_loss)
    return loss, {
        "current_pg_loss": current_global,
        "next_pg_loss": next_global,
        "surrogate": surrogate,
        "penalty": penalty,
        "active": active,
        "objective": current_global + lambda_trans * penalty,
    }


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    transition_aware = bool(
        tu.get_non_tensor_data(data=data, key="hpf_transition_aware_optimization", default=False)
    )
    transition_static_offset = float(
        tu.get_non_tensor_data(data=data, key="hpf_transition_static_offset", default=0.0) or 0.0
    )
    transition_lambda = float(
        tu.get_non_tensor_data(data=data, key="hpf_transition_lambda", default=0.0) or 0.0
    )
    dp_size = int(data["dp_size"])

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    use_hpf_pg_mask = "hpf_pg_mask" in data
    if use_hpf_pg_mask:
        fields.append("hpf_pg_mask")
    if transition_aware:
        fields.append("hpf_transition_cut_index")
    hpf_kl_coef = float(tu.get_non_tensor_data(data=data, key="hpf_kl_coef", default=0.0) or 0.0)
    hpf_kl_type = tu.get_non_tensor_data(data=data, key="hpf_kl_type", default=config.kl_loss_type)
    if hpf_kl_coef > 0:
        fields.extend(["hpf_kl_ref_log_prob", "hpf_kl_mask"])
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    pg_mask = data["hpf_pg_mask"].to(bool) if use_hpf_pg_mask else response_mask
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    if transition_aware:
        if loss_mode != "vanilla":
            raise ValueError("Transition-aware mixed policy optimization supports only vanilla policy loss.")
        cut_index = data["hpf_transition_cut_index"]
        if cut_index.ndim > 1:
            cut_index = cut_index.reshape(cut_index.shape[0], -1)[:, 0]
        current_mask = pg_mask & (cut_index == 0).unsqueeze(-1)
        next_mask = pg_mask & (cut_index == 1).unsqueeze(-1)
        current_batch_info = _hpf_transition_global_batch_info(
            current_mask,
            dp_size=dp_size,
            loss_scale_factor=config.loss_scale_factor,
            dp_group=dp_group,
        )
        next_batch_info = _hpf_transition_global_batch_info(
            next_mask,
            dp_size=dp_size,
            loss_scale_factor=config.loss_scale_factor,
            dp_group=dp_group,
        )
        current_pg_loss, current_pg_metrics = compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=current_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
            global_batch_info=current_batch_info,
        )
        next_pg_loss, next_pg_metrics = compute_policy_loss_vanilla(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=next_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
            global_batch_info=next_batch_info,
        )

        # Each synchronized micro-batch is a stochastic estimate of the full
        # transition objective. The detached hinge gate is the exact ReLU
        # subgradient away from zero and avoids retaining graphs across all
        # gradient-accumulation micro-batches.
        pg_loss, transition_info = _hpf_transition_combine_policy_losses(
            current_pg_loss,
            next_pg_loss,
            static_offset=transition_static_offset,
            lambda_trans=transition_lambda,
            dp_size=dp_size,
            dp_group=dp_group,
        )
        pg_metrics = {
            **{
                f"actor/transition_current/{key.removeprefix('actor/')}": value
                for key, value in current_pg_metrics.items()
            },
            **{
                f"actor/transition_next/{key.removeprefix('actor/')}": value
                for key, value in next_pg_metrics.items()
            },
        }
    else:
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=pg_mask,
            loss_agg_mode=loss_agg_mode,
            config=config,
            rollout_is_weights=rollout_is_weights,
        )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    if transition_aware:
        metrics["actor/transition_current_pg_loss"] = Metric(
            value=transition_info["current_pg_loss"], aggregation=AggregationType.MEAN
        )
        metrics["actor/transition_next_pg_loss"] = Metric(
            value=transition_info["next_pg_loss"], aggregation=AggregationType.MEAN
        )
        metrics["actor/transition_surrogate"] = Metric(
            value=transition_info["surrogate"], aggregation=AggregationType.MEAN
        )
        metrics["actor/transition_penalty"] = Metric(
            value=transition_info["penalty"], aggregation=AggregationType.MEAN
        )
        metrics["actor/transition_active"] = Metric(
            value=transition_info["active"], aggregation=AggregationType.MEAN
        )
        metrics["actor/transition_objective_loss"] = Metric(
            value=transition_info["objective"], aggregation=AggregationType.MEAN
        )
        metrics["actor/transition_lambda"] = Metric(
            value=transition_lambda, aggregation=AggregationType.MEAN
        )
        metrics["actor/transition_static_offset"] = Metric(
            value=transition_static_offset, aggregation=AggregationType.MEAN
        )
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=pg_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info)
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    if hpf_kl_coef > 0:
        hpf_kl_ref_log_prob = data["hpf_kl_ref_log_prob"]
        hpf_kl_mask = data["hpf_kl_mask"].to(bool)
        hpf_kld = kl_penalty(logprob=log_prob, ref_logprob=hpf_kl_ref_log_prob, kl_penalty=hpf_kl_type)
        hpf_kl_loss = agg_loss(
            loss_mat=hpf_kld,
            loss_mask=hpf_kl_mask,
            loss_agg_mode=config.loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss += hpf_kl_coef * hpf_kl_loss
        metrics["actor/hpf_kl_loss"] = Metric(value=hpf_kl_loss, aggregation=metric_aggregation)
        metrics["actor/hpf_kl_coef"] = hpf_kl_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
