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
"""Rubric-augmented Curriculum Policy Optimization (RCPO), advantage-based dynamic curriculum (paper Sec. 3.3.2).

For a synthesis requirement x with G responses and rewards r_1..r_G:

    Eq. 7   A_i       = (r_i - mean{r}) / (std{r} + eps_adv)
    Eq. 8   A_i^cache = A_i * 1[std{r} >= delta]                          (curriculum cache)
    Eq. 9   rho_x     = 1/G * sum_i (1 - R_accuracy_i)                    (difficulty coefficient)
    Eq. 10  w_x       = w0 / (1 + exp(-k * (rho_x - rho0)))               (calibrated weight)
    Eq. 12  s3        = C_A(s1, s2) * w_x,  C_A = min(s1, s2) if A_i^cache >= 0 else max(s1, s2)
    Eq. 11  J(theta)  = E[ 1/G sum_i 1/|y_i| sum_t ( s3 * A_i^cache - beta * KL ) ]

``C_A(s1, s2) * A = min(s1 * A, s2 * A)`` and w_x > 0 is constant within a group, so the calibrated weight is
folded into the advantage returned here and the clipped surrogate (Eq. 11-13) is computed by
:func:`compute_rcpo_policy_loss_matrix`.

The difficulty coefficient of Eq. 9 is the average shortfall of the precursor-level accuracy reward R_accuracy in
[0, 1] (``difficulty_scores``, see :class:`insynth_r1.trainer.rcpo_trainer.RCPOTrainerMixin`); rows without it fall
back to r_total min-max normalized with ``reward_min`` / ``reward_max``.

This module only depends on torch/numpy so it can be unit-tested without a verl runtime.
"""

from dataclasses import dataclass

import numpy as np
import torch

RCPO_ESTIMATOR = "rcpo"


@dataclass
class RCPOConfig:
    """Hyper-parameters of the dynamic curriculum (``algorithm.rcpo`` in the trainer config).

    Attributes:
        cache_std_threshold: delta in Eq. 8.
        w0, rho0, k: logistic hyper-parameters of Eq. 10.
        reward_min, reward_max: bounds of r_total, used to normalize it when R_accuracy is unavailable.
        epsilon: eps_adv in Eq. 7, prevents division by zero.
    """

    cache_std_threshold: float = 0.001
    w0: float = 1.5
    rho0: float = 0.25
    k: float = 10.0
    reward_min: float = -2.0
    reward_max: float = 3.0
    epsilon: float = 1e-6

    @classmethod
    def from_config(cls, config) -> "RCPOConfig":
        """Build from a DictConfig / dict (``algorithm.rcpo``); missing keys keep their defaults."""
        if config is None:
            return cls()
        get = config.get if hasattr(config, "get") else lambda k, d=None: getattr(config, k, d)
        return cls(**{name: get(name, default) for name, default in cls().__dict__.items()})


def calibrated_weight(rho: torch.Tensor, w0: float, rho0: float, k: float) -> torch.Tensor:
    """Eq. 10."""
    return w0 / (1.0 + torch.exp(-k * (rho - rho0)))


def compute_rcpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    rcpo: RCPOConfig,
    difficulty_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute RCPO advantages ``w_x * A_i^cache`` broadcast over response tokens.

    Args:
        token_level_rewards: (bs, response_length) rewards; their sum per row is r_total.
        response_mask: (bs, response_length).
        index: (bs,) group id (prompt uid) of every row.
        rcpo: curriculum hyper-parameters.
        difficulty_scores: (bs,) per-response rewards in [0, 1] used for rho_x (Eq. 9); NaN entries and
            ``None`` fall back to r_total normalized with ``rcpo.reward_min`` / ``rcpo.reward_max``.

    Returns:
        advantages, returns (identical, outcome-only), and scalar metrics.
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1).float()
        device = scores.device
        _, inverse = np.unique(np.asarray(index, dtype=object).astype(str), return_inverse=True)
        group = torch.as_tensor(inverse, device=device, dtype=torch.long)
        num_groups = int(group.max().item()) + 1 if len(group) else 0

        count = torch.zeros(num_groups, device=device).scatter_add_(0, group, torch.ones_like(scores))
        mean = torch.zeros(num_groups, device=device).scatter_add_(0, group, scores) / count
        sq_dev = (scores - mean[group]) ** 2
        # Unbiased std, as torch.std in verl's GRPO estimator.
        var = torch.zeros(num_groups, device=device).scatter_add_(0, group, sq_dev) / (count - 1).clamp(min=1)
        std = var.sqrt()
        singleton = count <= 1
        # Same convention as verl's GRPO for singleton groups: mean 0, std 1.
        mean = torch.where(singleton, torch.zeros_like(mean), mean)
        std = torch.where(singleton, torch.ones_like(std), std)

        # Eq. 7
        adv = (scores - mean[group]) / (std[group] + rcpo.epsilon)

        # Eq. 8
        cached = std >= rcpo.cache_std_threshold
        adv = adv * cached[group].float()

        # Eq. 9
        normalized_total = ((scores - rcpo.reward_min) / (rcpo.reward_max - rcpo.reward_min)).clamp(0.0, 1.0)
        if difficulty_scores is None:
            difficulty_scores = normalized_total
        else:
            difficulty_scores = difficulty_scores.to(device=device, dtype=torch.float32)
            # Rows without the requested score (e.g. synthetic padding rows) fall back to normalized r_total.
            difficulty_scores = torch.where(torch.isnan(difficulty_scores), normalized_total, difficulty_scores)
        shortfall = 1.0 - difficulty_scores.clamp(0.0, 1.0)
        rho = torch.zeros(num_groups, device=device).scatter_add_(0, group, shortfall) / count

        # Eq. 10
        weight = calibrated_weight(rho, rcpo.w0, rcpo.rho0, rcpo.k)
        adv = adv * weight[group]

        advantages = adv.unsqueeze(-1) * response_mask

    metrics = {}
    if num_groups:
        metrics = {
            "rcpo/cache_kept_group_ratio": cached.float().mean().item(),
            "rcpo/difficulty_rho/mean": rho.mean().item(),
            "rcpo/weight/mean": weight.mean().item(),
            "rcpo/weight/max": weight.max().item(),
            "rcpo/weight/min": weight.min().item(),
        }
    return advantages, advantages, metrics


def compute_rcpo_policy_loss_matrix(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token RCPO surrogate loss ``-C_A(s1, s2) * A = -min(s1 * A, s2 * A)`` (Eq. 11-13), with
    ``A = w_x * A^cache``.

    Returns:
        loss matrix (bs, response_length), clipped-token indicator, and ``log_prob - old_log_prob``.
    """
    log_ratio = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    s1 = torch.exp(log_ratio)
    s2 = torch.clamp(s1, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    surrogate1, surrogate2 = s1 * advantages, s2 * advantages
    loss = -torch.minimum(surrogate1, surrogate2)
    clipped = (surrogate2 < surrogate1).float()
    return loss, clipped, log_ratio
