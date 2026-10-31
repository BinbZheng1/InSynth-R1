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
"""Curriculum cache (paper Eq. 8): screen trivial groups out of the training batch.

verl's V1 replay buffer implements DAPO dynamic sampling: a finished group whose reward is identical across all
responses is evicted and replaced by freshly sampled prompts. The curriculum cache generalizes the test from
``std == 0`` to ``std < delta``, so that only informative groups with significant group-relative advantages
(``std{r} >= delta``) enter the batch, as the indicator in Eq. 8.
"""

from collections import Counter, defaultdict

import numpy as np
import transfer_queue as tq

from verl.trainer.ppo.v1.replay_buffer import FILTER_GROUPS_REWARD_METRIC, ReplayBuffer, ReplayBufferAsync


class CurriculumCacheMixin:
    """Overrides the DAPO group test with ``std{r_i} < cache_std_threshold`` (unbiased std, as in Eq. 7)."""

    cache_std_threshold: float = 0.0

    def _dapo_filtered_keys(self, partition_id: str) -> tuple[set[str], Counter]:
        if partition_id == "val" or self.filter_groups_metric is None:
            return set(), Counter()

        finished_uids = self.finished_keys[partition_id]
        classification_cache = self._dapo_classification_cache[partition_id]
        for uid in classification_cache.keys() - finished_uids:
            del classification_cache[uid]

        new_finished_uids = finished_uids - classification_cache.keys()
        trajectory_keys = [key for key in self.partitions[partition_id] if key.split("_")[0] in new_finished_uids]
        use_canonical_reward = self.filter_groups_metric == FILTER_GROUPS_REWARD_METRIC
        metrics_by_uid: dict[str, list[float]] = defaultdict(list)
        missing_metric_uids = new_finished_uids - {key.split("_")[0] for key in trajectory_keys}

        metric_data = []
        if trajectory_keys:
            data = tq.kv_batch_get(
                keys=trajectory_keys,
                partition_id=partition_id,
                select_fields=["rm_scores"] if use_canonical_reward else ["extra_fields"],
            )
            metric_data = list(data["rm_scores"] if use_canonical_reward else data["extra_fields"])

        for key, value in zip(trajectory_keys, metric_data, strict=True):
            uid = key.split("_")[0]
            if use_canonical_reward:
                metrics_by_uid[uid].append(float(value.sum().item()))
                continue
            extra_fields = getattr(value, "data", value)
            reward_extra_info = extra_fields.get("reward_extra_info", {}) if isinstance(extra_fields, dict) else {}
            if self.filter_groups_metric not in reward_extra_info:
                missing_metric_uids.add(uid)
            else:
                metrics_by_uid[uid].append(float(reward_extra_info[self.filter_groups_metric]))

        if missing_metric_uids:
            raise RuntimeError(
                f"Finished groups are missing metric {self.filter_groups_metric!r}: {sorted(missing_metric_uids)[:5]}"
            )

        for uid in new_finished_uids:
            values = metrics_by_uid[uid]
            trivial = len(values) > 1 and float(np.std(values, ddof=1)) < self.cache_std_threshold
            classification_cache[uid] = round(float(np.mean(values)), 4) if trivial else None

        filtered = {uid: reward for uid, reward in classification_cache.items() if reward is not None}
        return set(filtered), Counter(filtered.values())


class CurriculumCacheReplayBuffer(CurriculumCacheMixin, ReplayBuffer):
    def __init__(self, *args, cache_std_threshold: float = 0.0, **kwargs):
        self.cache_std_threshold = cache_std_threshold
        super().__init__(*args, **kwargs)


class CurriculumCacheReplayBufferAsync(CurriculumCacheMixin, ReplayBufferAsync):
    def __init__(self, *args, cache_std_threshold: float = 0.0, **kwargs):
        self.cache_std_threshold = cache_std_threshold
        super().__init__(*args, **kwargs)
