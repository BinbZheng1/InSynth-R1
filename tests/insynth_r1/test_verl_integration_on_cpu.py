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
"""Integration checks against verl; skipped when the verl runtime dependencies are not installed."""

import asyncio

import numpy as np
import pytest
import torch

pytest.importorskip("hydra")
pytest.importorskip("tensordict")


def test_rcpo_registered_in_verl_registries():
    import insynth_r1.register  # noqa: F401
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn, get_policy_loss_fn

    assert get_adv_estimator_fn("rcpo") is insynth_r1.register.compute_rcpo_outcome_advantage
    assert get_policy_loss_fn("rcpo") is insynth_r1.register.compute_policy_loss_rcpo


def test_rcpo_through_verl_compute_advantage():
    import insynth_r1.register  # noqa: F401
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    rewards = torch.zeros(4, 3)
    rewards[:, -1] = torch.tensor([3.0, 1.0, 2.0, 2.0])
    data = DataProto.from_dict(
        tensors={"token_level_rewards": rewards, "response_mask": torch.ones(4, 3)},
        non_tensors={"uid": np.array(["a", "a", "b", "b"], dtype=object)},
    )
    config = {"rcpo": {"cache_std_threshold": 0.01}}
    out = compute_advantage(data, adv_estimator="rcpo", config=_AttrDict(config))
    adv = out.batch["advantages"][:, 0]
    assert adv[0] > 0 > adv[1] and torch.all(adv[2:] == 0)


def test_rcpo_policy_loss_matches_vanilla_without_dual_clip():
    import insynth_r1.register  # noqa: F401
    from omegaconf import OmegaConf

    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    torch.manual_seed(0)
    old, mask = torch.randn(3, 5), torch.ones(3, 5)
    new = old + 0.3 * torch.randn(3, 5)
    adv = torch.rand(3, 1).expand(3, 5).contiguous()  # positive advantages: dual-clip is inactive
    config = OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.28,
            "clip_ratio_c": 3.0,
            "global_batch_info": {},
        }
    )
    rcpo, _ = get_policy_loss_fn("rcpo")(old, new, adv, mask, "token-mean", config)
    vanilla, _ = get_policy_loss_fn("vanilla")(old, new, adv, mask, "token-mean", config)
    torch.testing.assert_close(rcpo, vanilla)


def test_reward_manager_injects_length_and_step():
    from insynth_r1.reward.manager import InSynthRewardManager
    from verl import DataProto

    captured = {}

    def fake_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
        captured.update(extra_info)
        return {"score": 1.0}

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "x"

    shared = {"target": "LiCoO2"}
    data = DataProto.from_dict(
        tensors={
            "prompts": torch.ones(1, 3, dtype=torch.long),
            "responses": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0]]),
        },
        non_tensors={
            "data_source": np.array(["insynth/precursor"], dtype=object),
            "reward_model": np.array([{"ground_truth": "A + B"}], dtype=object),
            "extra_info": np.array([shared], dtype=object),
            "global_steps": np.array([7]),
        },
    )
    manager = InSynthRewardManager.__new__(InSynthRewardManager)
    manager.tokenizer, manager.compute_score, manager.is_async_reward_score = _Tok(), fake_score, False
    manager.reward_router_address, manager.reward_model_tokenizer = None, None
    manager.loop = asyncio.new_event_loop()
    try:
        result = manager.loop.run_until_complete(manager.run_single(data))
    finally:
        manager.loop.close()
    assert result["reward_score"] == 1.0
    assert captured["response_length"] == 2 and captured["global_steps"] == 7 and captured["target"] == "LiCoO2"
    assert "response_length" not in shared


class _AttrDict(dict):
    def get(self, key, default=None):
        return super().get(key, default)
