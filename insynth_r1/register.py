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
"""Register RCPO into verl's advantage-estimator and policy-loss registries.

The policy loss runs inside the actor workers, so this module must be imported in every Ray process. verl does that
for modules listed in ``VERL_USE_EXTERNAL_MODULES`` (imported at the end of ``import verl``)::

    export VERL_USE_EXTERNAL_MODULES=insynth_r1.register

:mod:`insynth_r1.main_rcpo` sets it automatically and forwards it to Ray workers.
"""

from typing import Any, Optional

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from insynth_r1.algo.rcpo import (
    RCPO_ESTIMATOR,
    RCPOConfig,
    compute_rcpo_advantage,
    compute_rcpo_policy_loss_matrix,
)
from verl.trainer.ppo.core_algos import (
    ADV_ESTIMATOR_REGISTRY,
    POLICY_LOSS_REGISTRY,
    agg_loss,
    register_adv_est,
    register_policy_loss,
)

RCPO = RCPO_ESTIMATOR


def compute_rcpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Optional[Any] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``algorithm.adv_estimator=rcpo`` entry for verl's generic ``compute_advantage``.

    verl's generic path only provides r_total, so the difficulty coefficient uses normalized r_total here. The RCPO
    trainer (:mod:`insynth_r1.trainer.rcpo_trainer`) calls :func:`compute_rcpo_advantage` directly with R_accuracy
    and logs the curriculum metrics.
    """
    rcpo = RCPOConfig.from_config(config.get("rcpo", None) if config is not None else None)
    advantages, returns, _ = compute_rcpo_advantage(token_level_rewards, response_mask, index, rcpo)
    return advantages, returns


def compute_policy_loss_rcpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config=None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """RCPO clipped surrogate (Eq. 11-13) with asymmetric clipping and no dual-clip.

    ``advantages`` already carry the calibrated weight w_x (see :mod:`insynth_r1.algo.rcpo`). The paper's per-group
    objective averages tokens within a response and then responses (``loss_agg_mode=seq-mean-token-mean``).
    """
    assert config is not None
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    pg_losses, clipped, log_ratio = compute_rcpo_policy_loss_matrix(
        old_log_prob, log_prob, advantages, clip_ratio_low, clip_ratio_high
    )
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": verl_F.masked_mean(clipped, response_mask).detach().item(),
        "actor/ppo_kl": verl_F.masked_mean(-log_ratio, response_mask).detach().item(),
    }
    return pg_loss, pg_metrics


# Idempotent: the module may be imported more than once per process under different import paths.
if ADV_ESTIMATOR_REGISTRY.get(RCPO) is None:
    register_adv_est(RCPO)(compute_rcpo_outcome_advantage)
if RCPO not in POLICY_LOSS_REGISTRY:
    register_policy_loss(RCPO)(compute_policy_loss_rcpo)
