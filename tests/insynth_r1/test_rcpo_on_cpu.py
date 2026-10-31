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
import math

import numpy as np
import pytest
import torch

from insynth_r1.algo.rcpo import RCPOConfig, calibrated_weight, compute_rcpo_advantage, compute_rcpo_policy_loss_matrix


def _rewards(values, length=4):
    values = torch.tensor(values, dtype=torch.float32)
    rewards = torch.zeros(len(values), length)
    rewards[:, -1] = values
    return rewards, torch.ones(len(values), length)


def _grpo_reference(scores, index, eps=1e-6):
    """Loop implementation of verl's compute_grpo_outcome_advantage (Eq. 7)."""
    out = torch.zeros(len(scores))
    for uid in set(index):
        rows = [i for i, u in enumerate(index) if u == uid]
        group = torch.tensor([scores[i] for i in rows])
        mean, std = (group.mean(), group.std()) if len(rows) > 1 else (torch.tensor(0.0), torch.tensor(1.0))
        for i in rows:
            out[i] = (scores[i] - mean) / (std + eps)
    return out


def test_reduces_to_grpo_with_constant_weight():
    scores = [3.0, 1.0, 2.0, 0.5, 2.5, 2.5, 2.5, 2.5, 1.7]
    index = np.array(["a", "a", "a", "a", "b", "b", "b", "b", "c"], dtype=object)
    rewards, mask = _rewards(scores)
    # delta = 0 keeps every group; k = 0, w0 = 2 gives w_x = 1 for every group
    cfg = RCPOConfig(cache_std_threshold=0.0, k=0.0, w0=2.0)
    adv, ret, _ = compute_rcpo_advantage(rewards, mask, index, cfg)
    torch.testing.assert_close(adv[:, 0], _grpo_reference(scores, list(index)))
    torch.testing.assert_close(adv, ret)
    assert torch.all(adv == adv[:, :1])


def test_curriculum_cache_eq8():
    scores = [1.0, 0.0, 2.0, 2.0, 2.0, 2.005]
    index = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    rewards, mask = _rewards(scores)
    cfg = RCPOConfig(cache_std_threshold=0.01)
    adv, _, metrics = compute_rcpo_advantage(rewards, mask, index, cfg)
    assert torch.all(adv[:2, 0] != 0)
    # group b: std = 0; group c: std = 0.0035 < delta -> both screened out
    assert torch.all(adv[2:] == 0)
    assert metrics["rcpo/cache_kept_group_ratio"] == pytest.approx(1 / 3)


def test_difficulty_coefficient_and_weight_eq9_eq10():
    accuracy = torch.tensor([0.0, 0.5, 1.0, 1.0, 0.0, 0.0])
    scores = [0.0, 1.5, 3.0, 3.0, -1.0, 1.0]
    index = np.array(["a", "a", "a", "a", "b", "b"], dtype=object)
    rewards, mask = _rewards(scores)
    cfg = RCPOConfig(cache_std_threshold=0.0)
    adv, _, metrics = compute_rcpo_advantage(rewards, mask, index, cfg, difficulty_scores=accuracy)
    rho_a, rho_b = (1 + 0.5 + 0 + 0) / 4, 1.0
    w_a = cfg.w0 / (1 + math.exp(-cfg.k * (rho_a - cfg.rho0)))
    w_b = cfg.w0 / (1 + math.exp(-cfg.k * (rho_b - cfg.rho0)))
    base = _grpo_reference(scores, list(index))
    torch.testing.assert_close(adv[:4, 0], base[:4] * w_a)
    torch.testing.assert_close(adv[4:, 0], base[4:] * w_b)
    assert metrics["rcpo/difficulty_rho/mean"] == pytest.approx((rho_a + rho_b) / 2)


def test_harder_groups_get_larger_weights():
    cfg = RCPOConfig()
    rho = torch.linspace(0, 1, 11)
    weight = calibrated_weight(rho, cfg.w0, cfg.rho0, cfg.k)
    assert torch.all(weight[1:] > weight[:-1]) and torch.all(weight > 0) and torch.all(weight < cfg.w0)
    # rho = rho0 gives w0 / 2; with w0 = 1.5, rho0 = 0.25, k = 10: rho = 0.5 -> w ~ 1.386
    assert calibrated_weight(torch.tensor(cfg.rho0), cfg.w0, cfg.rho0, cfg.k).item() == pytest.approx(cfg.w0 / 2)
    assert calibrated_weight(torch.tensor(0.5), 1.5, 0.25, 10.0).item() == pytest.approx(1.386, abs=1e-3)


def test_difficulty_from_normalized_total_and_nan_fallback():
    scores = [3.0, -2.0]
    index = np.array(["a", "a"], dtype=object)
    rewards, mask = _rewards(scores)
    cfg = RCPOConfig(reward_min=-2.0, reward_max=3.0)
    _, _, m1 = compute_rcpo_advantage(rewards, mask, index, cfg)
    _, _, m2 = compute_rcpo_advantage(rewards, mask, index, cfg, difficulty_scores=torch.tensor([float("nan")] * 2))
    assert m1["rcpo/difficulty_rho/mean"] == pytest.approx(0.5)
    assert m2["rcpo/difficulty_rho/mean"] == pytest.approx(0.5)


def test_advantage_respects_response_mask():
    rewards, mask = _rewards([1.0, 0.0])
    mask[0, 2:] = 0
    adv, _, _ = compute_rcpo_advantage(rewards, mask, np.array(["a", "a"], dtype=object), RCPOConfig())
    assert torch.all(adv[0, 2:] == 0) and torch.all(adv[0, :2] != 0)


def test_policy_loss_folds_weight_into_advantage():
    """Eq. 12 with w_x > 0: the loss on w_x * A equals w_x times the loss on A."""
    torch.manual_seed(0)
    old = torch.randn(4, 6)
    new = old + 0.4 * torch.randn(4, 6)
    adv = torch.randn(4, 1).expand(4, 6)
    w = torch.tensor([[0.3], [1.0], [1.7], [2.0]])
    loss_w, _, _ = compute_rcpo_policy_loss_matrix(old, new, adv * w, 0.2, 0.28)
    loss, _, _ = compute_rcpo_policy_loss_matrix(old, new, adv, 0.2, 0.28)
    torch.testing.assert_close(loss_w, loss * w)


def test_policy_loss_matches_eq12_sign_dependent_form():
    """Eq. 12-13: s3 = C_A(s1, s2) * w_x, with C_A = min(s1, s2) if A^cache >= 0 and max(s1, s2) otherwise."""
    torch.manual_seed(0)
    old = torch.randn(6, 5)
    new = old + 0.5 * torch.randn(6, 5)
    adv_cache = torch.tensor([[1.2], [-0.7], [0.0], [2.5], [-1.9], [0.3]]).expand(6, 5)
    w = torch.tensor([[0.3], [1.0], [1.4], [0.8], [1.5], [0.1]])
    s1 = torch.exp(new - old)
    s2 = torch.clamp(s1, 1 - 0.2, 1 + 0.28)
    c = torch.where(adv_cache >= 0, torch.minimum(s1, s2), torch.maximum(s1, s2))
    loss, _, _ = compute_rcpo_policy_loss_matrix(old, new, w * adv_cache, 0.2, 0.28)
    torch.testing.assert_close(loss, -(c * w) * adv_cache)


def test_policy_loss_asymmetric_clip():
    old = torch.zeros(1, 4)
    new = torch.log(torch.tensor([[0.5, 0.9, 1.25, 1.5]]))
    pos = torch.ones(1, 4)
    loss, clipped, _ = compute_rcpo_policy_loss_matrix(old, new, pos, 0.2, 0.28)
    # A > 0: ratio is capped at 1 + eps_high = 1.28
    torch.testing.assert_close(loss, -torch.tensor([[0.5, 0.9, 1.25, 1.28]]))
    torch.testing.assert_close(clipped, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    loss, clipped, _ = compute_rcpo_policy_loss_matrix(old, new, -pos, 0.2, 0.28)
    # A < 0: ratio is floored at 1 - eps_low = 0.8
    torch.testing.assert_close(loss, torch.tensor([[0.8, 0.9, 1.25, 1.5]]))
    torch.testing.assert_close(clipped, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_config_from_dict():
    cfg = RCPOConfig.from_config({"w0": 3.0, "cache_std_threshold": 0.05})
    assert cfg.w0 == 3.0 and cfg.cache_std_threshold == 0.05 and cfg.k == RCPOConfig().k


def test_default_config_matches_trainer_yaml():
    from pathlib import Path

    from omegaconf import OmegaConf

    import insynth_r1

    yaml_path = Path(insynth_r1.__file__).parent / "config" / "rcpo_trainer.yaml"
    rcpo = OmegaConf.load(yaml_path).algorithm.rcpo
    assert RCPOConfig.from_config(rcpo) == RCPOConfig()
    assert (RCPOConfig().cache_std_threshold, RCPOConfig().w0, RCPOConfig().rho0, RCPOConfig().k) == (
        0.001,
        1.5,
        0.25,
        10.0,
    )
