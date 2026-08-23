# SPDX-License-Identifier: MIT
# Adapted from Hugging Face LightEval commit
# d3da6b9bbf38104c8b5e1acc86f83541f9a502d1.

"""Offline copy of Open-R1's LightEval AIME24 sampling task."""

import os
from pathlib import Path

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.default_prompts import aime_prompt_fn
from lighteval.tasks.lighteval_task import LightevalTaskConfig

dataset_dir_value = os.environ.get("OPENR1_AIME24_DATASET_DIR")
if not dataset_dir_value:
    raise RuntimeError("OPENR1_AIME24_DATASET_DIR must point to the prepared local AIME24 dataset")

dataset_dir = Path(dataset_dir_value).resolve()
if not (dataset_dir / "train.parquet").is_file():
    raise RuntimeError(f"Missing offline AIME24 train.parquet under {dataset_dir}")


# This intentionally mirrors the official `aime24` task in the pinned LightEval
# revision. The only change is hf_repo: it points at the local, deduplicated
# parquet directory so dataset loading never contacts the Hugging Face Hub.
aime24_openr1_offline = LightevalTaskConfig(
    name="aime24_openr1_offline",
    suite=["openr1_offline"],
    prompt_function=aime_prompt_fn,
    hf_repo=str(dataset_dir),
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[
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
