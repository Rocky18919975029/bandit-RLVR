# SPDX-License-Identifier: MIT
# Adapted from Hugging Face LightEval commit
# 24895519caecec2abeea53fa790021325ce7e59e.

"""Offline copy of Open-R1's LightEval AIME24 sampling task."""

import inspect
import os
from pathlib import Path

import lighteval.tasks.lighteval_task as lighteval_task_module
import lighteval.utils.utils as lighteval_utils_module
from datasets import load_dataset
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import aime_prompt_fn
from lighteval.tasks.lighteval_task import LightevalTaskConfig


def _offline_download_dataset_worker(
    dataset_path,
    dataset_config_name,
    trust_dataset,
    dataset_filter=None,
    revision=None,
):
    """Load the local task with either the old or new datasets API."""
    load_kwargs = {
        "path": dataset_path,
        "name": dataset_config_name,
        "data_dir": None,
        "cache_dir": None,
        "download_mode": None,
        "revision": revision,
    }
    if "trust_remote_code" in inspect.signature(load_dataset).parameters:
        load_kwargs["trust_remote_code"] = trust_dataset

    dataset = load_dataset(**load_kwargs)
    if dataset_filter is not None:
        dataset = dataset.filter(dataset_filter)
    return dataset


# The pinned Open-R1 LightEval revision predates datasets 5, which removed the
# trust_remote_code argument. Patch only its dataset-loading hook; prompts,
# generation parameters, metrics, and aggregation remain the pinned originals.
lighteval_task_module.download_dataset_worker = _offline_download_dataset_worker
lighteval_utils_module.download_dataset_worker = _offline_download_dataset_worker

dataset_dir_value = os.environ.get("OPENR1_AIME24_DATASET_DIR")
if not dataset_dir_value:
    raise RuntimeError("OPENR1_AIME24_DATASET_DIR must point to the prepared local AIME24 dataset")

dataset_dir = Path(dataset_dir_value).resolve()
if not (dataset_dir / "train.parquet").is_file():
    raise RuntimeError(f"Missing offline AIME24 train.parquet under {dataset_dir}")


# This mirrors the official `aime24` task in the pinned LightEval revision.
# The only change is hf_repo: it points at the local, deduplicated parquet
# directory so dataset loading never contacts the Hugging Face Hub.
aime24_openr1_offline = LightevalTaskConfig(
    name="aime24",
    suite=["lighteval"],
    prompt_function=aime_prompt_fn,
    hf_repo=str(dataset_dir),
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    metrics=[
        Metrics.math_pass_at_1_1n,
        Metrics.math_pass_at_1_4n,
        Metrics.math_pass_at_1_8n,
        Metrics.math_pass_at_1_16n,
        Metrics.math_pass_at_1_32n,
        Metrics.math_pass_at_1_64n,
    ],
    version=2,
)


TASKS_TABLE = [aime24_openr1_offline]
