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
"""RCPO on top of verl's V1 PPO trainers.

:class:`RCPOTrainerMixin` changes two steps of any V1 trainer (sync / colocate_async / separate_async):

1. Sampling: the replay buffer becomes a curriculum cache that screens out groups with ``std{r} < delta`` and
   refills the batch with informative groups (:mod:`insynth_r1.trainer.curriculum_cache`).
2. Advantage: ``algorithm.adv_estimator=rcpo`` computes ``w_x * A^cache`` with the difficulty coefficient taken
   from the rubric judge's R_accuracy (:func:`insynth_r1.algo.rcpo.compute_rcpo_advantage`).

Only single-turn rollouts (one output per session) are supported, which is the InSynth-R1 setting.
"""

import numpy as np
import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from insynth_r1.algo.rcpo import RCPO_ESTIMATOR as RCPO
from insynth_r1.algo.rcpo import RCPOConfig, compute_rcpo_advantage
from insynth_r1.trainer.curriculum_cache import CurriculumCacheReplayBuffer, CurriculumCacheReplayBufferAsync
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import apply_kl_penalty
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer, ReplayBufferAsync
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.workers.utils.padding import response_to_nested


class RCPOTrainerMixin:
    """Mixin placed before a :class:`PPOTrainer` subclass in the MRO."""

    @property
    def rcpo_config(self) -> RCPOConfig:
        return RCPOConfig.from_config(self.config.algorithm.get("rcpo", None))

    def _build_replay_buffer(self):
        buffer = super()._build_replay_buffer()
        if self.config.algorithm.adv_estimator != RCPO:
            return buffer
        assert buffer.filter_groups_metric is not None, (
            "The curriculum cache refills through group filtering; launch with insynth_r1.main_rcpo, which enables "
            "algorithm.filter_groups automatically."
        )
        # Re-class the built buffer; the curriculum cache subclasses override the group test and add one attribute.
        if type(buffer) is ReplayBuffer:
            buffer.__class__ = CurriculumCacheReplayBuffer
        elif type(buffer) is ReplayBufferAsync:
            buffer.__class__ = CurriculumCacheReplayBufferAsync
        else:
            raise ValueError(f"Curriculum cache is incompatible with custom sampler {type(buffer).__name__}")
        buffer.cache_std_threshold = self.rcpo_config.cache_std_threshold
        return buffer

    def _difficulty_scores(self, batch: KVBatchMeta) -> torch.Tensor:
        """R_accuracy of every trajectory (NaN when missing), read from the reward extra info."""
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["extra_fields"])
        extra_fields = data["extra_fields"]
        extra_fields = extra_fields.tolist() if hasattr(extra_fields, "tolist") else list(extra_fields)
        scores = []
        for value in extra_fields:
            value = getattr(value, "data", value)
            info = value.get("reward_extra_info", {}) if isinstance(value, dict) else {}
            scores.append(float(info.get("acc", float("nan"))))
        return torch.tensor(scores, dtype=torch.float32)

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        if self.config.algorithm.adv_estimator != RCPO:
            return super()._compute_advantage(batch, metrics)

        rcpo = self.rcpo_config
        if any(not key.endswith("_0") for key in batch.keys):
            raise NotImplementedError("RCPO supports single-turn rollouts (one output per session) only")
        fields = ["uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "values"]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        response_mask = data["response_mask"]
        data = DataProto(batch=data.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        uids = np.array(data.batch.pop("uid").tolist(), dtype=object)
        data.non_tensor_batch["uid"] = uids

        # 1. [optional] KL penalty in reward (off; RCPO uses the KL loss of Eq. 11)
        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        # 2. [optional] rollout correction, as in the base trainer
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        rollout_correction = rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass
        if rollout_correction:
            data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
            metrics.update(is_metrics)

        # 3. RCPO advantages: w_x * A^cache (Eq. 7-10)
        advantages, returns, rcpo_metrics = compute_rcpo_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=uids,
            rcpo=rcpo,
            difficulty_scores=self._difficulty_scores(batch),
        )
        metrics.update(rcpo_metrics)
        data.batch["advantages"], data.batch["returns"] = advantages, returns

        # 4. write nested advantages and returns back to TransferQueue
        out_fields = ["advantages", "returns"]
        if self.config.algorithm.use_kl_in_reward:
            out_fields.append("token_level_rewards")
        if rollout_correction:
            out_fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                out_fields.append("rollout_is_weights")
        output = TensorDict(
            {field: response_to_nested(data.batch[field], response_mask) for field in out_fields},
            batch_size=len(batch),
        )
        return tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)


def build_rcpo_trainer_cls(base_cls: type[PPOTrainer]) -> type[PPOTrainer]:
    """Return ``RCPO<Base>`` with :class:`RCPOTrainerMixin` applied to a registered V1 trainer class."""
    return type(f"RCPO{base_cls.__name__}", (RCPOTrainerMixin, base_cls), {})
