# Copyright 2026 InSynth-R1 Authors
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
"""Reward manager that exposes |y| and the training step to the rubric judge.

R_length (Eq. 5) needs the token length |y| of the response, and it is only activated in the later steps of RL
training. The stock naive reward manager passes neither to ``compute_score``, so this manager adds
``response_length`` and ``global_steps`` to ``extra_info`` before delegating to it.

Use with::

    reward.reward_manager.source=importlib
    reward.reward_manager.name=InSynthRewardManager
    reward.reward_manager.module.path=pkg://insynth_r1.reward.manager
"""

import numpy as np

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


class InSynthRewardManager(NaiveRewardManager):
    async def run_single(self, data: DataProto) -> dict:
        # The naive manager scores (and reads extra_info from) the last sequence only.
        data = data[-1:]
        item = data[0]
        if "response_len" in data.non_tensor_batch:
            response_length = int(data.non_tensor_batch["response_len"][0])
        else:
            response_length = int(item.batch["attention_mask"][-item.batch["responses"].shape[-1] :].sum())

        # Copy: the dataset's extra_info dict may be shared by all rollouts of a prompt.
        extra_info = dict(data.non_tensor_batch.get("extra_info", [None])[0] or {})
        extra_info["response_length"] = response_length
        if "global_steps" in data.non_tensor_batch:
            extra_info["global_steps"] = int(data.non_tensor_batch["global_steps"][0])
        extra_info_array = np.empty(1, dtype=object)
        extra_info_array[0] = extra_info
        data.non_tensor_batch["extra_info"] = extra_info_array
        return await super().run_single(data)
