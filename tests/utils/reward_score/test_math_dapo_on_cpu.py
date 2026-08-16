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

from verl.utils.reward_score import default_compute_score


def test_math_dapo_uses_strict_boxed_verifier_by_default():
    unboxed = default_compute_score("math_dapo", "Answer: 4", "4")
    correct_box = default_compute_score("math_dapo", r"Therefore, \boxed{4}", "4")
    wrong_box = default_compute_score("math_dapo", r"Therefore, \boxed{5}", "4")

    assert unboxed == {"score": -1.0, "acc": False, "pred": None}
    assert correct_box == {"score": 1.0, "acc": True, "pred": "4"}
    assert wrong_box == {"score": -1.0, "acc": False, "pred": "5"}


def test_math_dapo_can_explicitly_use_legacy_verifier():
    result = default_compute_score("math_dapo", "Answer: 4", "4", strict_box_verify=False)

    assert result == {"score": 1.0, "acc": True, "pred": "4"}
