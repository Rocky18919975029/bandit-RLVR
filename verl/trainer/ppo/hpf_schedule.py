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


def get_hpf_role_phase(physical_epoch: int, follower_epochs: int, leader_epochs: int) -> tuple[str, int, int]:
    """Map a complete dataloader pass to its HPF role phase.

    Returns the role name, one-based epoch within that role, and one-based
    fixed-horizon round index.
    """
    if physical_epoch < 0:
        raise ValueError(f"physical_epoch must be nonnegative, got {physical_epoch}.")
    if follower_epochs <= 0 or leader_epochs <= 0:
        raise ValueError(
            "follower_epochs and leader_epochs must both be positive; "
            f"got follower_epochs={follower_epochs}, leader_epochs={leader_epochs}."
        )

    passes_per_round = follower_epochs + leader_epochs
    pass_in_round = physical_epoch % passes_per_round
    round_index = physical_epoch // passes_per_round + 1
    if pass_in_round < follower_epochs:
        return "follower", pass_in_round + 1, round_index
    return "leader", pass_in_round - follower_epochs + 1, round_index
