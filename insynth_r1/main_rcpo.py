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
"""Entry point for RCPO training: ``python -m insynth_r1.main_rcpo [hydra overrides]``.

Same as ``verl.trainer.main_ppo`` (V1 trainer) except that

* the config is ``insynth_r1/config/rcpo_trainer.yaml`` (verl's ``ppo_trainer`` + RCPO / rubric defaults),
* the trainer class is wrapped with :class:`insynth_r1.trainer.rcpo_trainer.RCPOTrainerMixin`,
* ``insynth_r1.register`` is imported in every Ray process (RCPO policy loss in the actor workers).
"""

import os

EXTERNAL_MODULE = "insynth_r1.register"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _enable_external_module() -> str:
    modules = [m for m in os.environ.get("VERL_USE_EXTERNAL_MODULES", "").split(",") if m]
    if EXTERNAL_MODULE not in modules:
        modules.append(EXTERNAL_MODULE)
    os.environ["VERL_USE_EXTERNAL_MODULES"] = ",".join(modules)
    return os.environ["VERL_USE_EXTERNAL_MODULES"]


# Must run before verl is imported so that this driver process registers RCPO as well.
_enable_external_module()

import hydra  # noqa: E402
import ray  # noqa: E402
from omegaconf import DictConfig, OmegaConf, open_dict  # noqa: E402

import insynth_r1.register  # noqa: E402, F401
from insynth_r1.algo.rcpo import RCPO_ESTIMATOR  # noqa: E402
from insynth_r1.reward.key_step import (  # noqa: E402
    JUDGE_API_BASE_ENV,
    JUDGE_API_KEY_ENV,
    JUDGE_FALLBACK_MODELS_ENV,
    JUDGE_MODEL_ENV,
    split_models,
)
from verl.trainer.main_ppo import run_ppo  # noqa: E402
from verl.trainer.ppo.utils import need_critic, need_reference_policy  # noqa: E402
from verl.utils.config import validate_config  # noqa: E402
from verl.utils.device import auto_set_device  # noqa: E402
from verl.utils.import_utils import load_class_from_fqn  # noqa: E402
from verl.utils.logging_utils import configure_verl_logging  # noqa: E402


def _prepare_step_judge(reward_kwargs: DictConfig) -> None:
    """Resolve the LLM key-step judge endpoint from the environment and fail fast if it is incomplete.

    The base URL and the judge models (primary + ordered fallbacks) are written into the config so that every reward
    worker uses the same judges. The API
    key stays in the environment variable named by ``judge_api_key_env``; Ray workers inherit it on a single node, on
    multiple nodes it must be exported on every node.
    """
    if reward_kwargs.get("step_judge", "rule") != "llm":
        return
    reward_kwargs.judge_base_url = reward_kwargs.get("judge_base_url") or os.environ.get(JUDGE_API_BASE_ENV)
    reward_kwargs.judge_model = reward_kwargs.get("judge_model") or os.environ.get(JUDGE_MODEL_ENV)
    fallbacks = reward_kwargs.get("judge_fallback_models") or os.environ.get(JUDGE_FALLBACK_MODELS_ENV)
    reward_kwargs.judge_fallback_models = ",".join(split_models(fallbacks))
    if not reward_kwargs.judge_base_url or not reward_kwargs.judge_model:
        raise ValueError(
            f"step_judge=llm: export {JUDGE_API_BASE_ENV} and {JUDGE_MODEL_ENV} (or set reward_kwargs.judge_base_url / "
            "judge_model), or use step_judge=rule"
        )
    print(f"key-step judges: {[reward_kwargs.judge_model, *split_models(reward_kwargs.judge_fallback_models)]}")
    key_env = reward_kwargs.get("judge_api_key_env", JUDGE_API_KEY_ENV)
    if not os.environ.get(key_env):
        print(f"WARNING: {key_env} is not set; the key-step judge requests are sent without an API key")


def prepare_rcpo_config(config: DictConfig) -> None:
    """Wire RCPO requirements into the verl config in place."""
    if not config.trainer.use_v1:
        raise ValueError("insynth_r1.main_rcpo requires trainer.use_v1=True")
    if config.algorithm.adv_estimator != RCPO_ESTIMATOR:
        estimator = config.algorithm.adv_estimator
        raise ValueError(f"insynth_r1.main_rcpo trains RCPO; got algorithm.adv_estimator={estimator}")
    with open_dict(config):
        # The curriculum cache (Eq. 8) screens trivial groups out through verl's group-filtering refill path.
        filter_groups = config.algorithm.filter_groups
        if filter_groups.enable and filter_groups.metric != "reward":
            raise ValueError("The curriculum cache filters on the canonical reward; set filter_groups.metric=reward")
        filter_groups.enable = True
        filter_groups.metric = "reward"

        _prepare_step_judge(config.reward.custom_reward_function.reward_kwargs)

        # Ray workers must import insynth_r1 (reward manager, policy loss) and register RCPO.
        runtime_env = config.ray_kwargs.ray_init.runtime_env
        if runtime_env.get("env_vars", None) is None:
            runtime_env.env_vars = {}
        runtime_env.env_vars["VERL_USE_EXTERNAL_MODULES"] = os.environ["VERL_USE_EXTERNAL_MODULES"]
        if "PYTHONPATH" not in runtime_env.env_vars:
            paths = [REPO_ROOT, *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
            runtime_env.env_vars["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))


@ray.remote
class RCPOTaskRunner:
    """Mirror of ``verl.trainer.main_ppo.TaskRunnerV1`` with the RCPO trainer class."""

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            agent_loop_manager_cls = AgentLoopManagerTQ

        self.agent_loop_manager = agent_loop_manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig):
        configure_verl_logging()

        import transfer_queue as tq

        from insynth_r1.trainer.rcpo_trainer import build_rcpo_trainer_cls
        from verl.trainer.ppo.v1 import get_trainer_cls

        trainer_cls = build_rcpo_trainer_cls(get_trainer_cls(config.trainer.v1.trainer_mode))

        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        print(OmegaConf.to_yaml(config))
        self.config = config

        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = trainer_cls(config=config)
            self.trainer.init()
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path="config", config_name="rcpo_trainer", version_base=None)
def main(config: DictConfig):
    auto_set_device(config)
    prepare_rcpo_config(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=RCPOTaskRunner)


if __name__ == "__main__":
    main()
