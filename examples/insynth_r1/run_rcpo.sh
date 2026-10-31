#!/usr/bin/env bash
# InSynth-R1 RL stage: Rubric-augmented Curriculum Policy Optimization (paper Sec. 3.3, Sec. 4.2.1)
#
# Policy initialized from the merged MSFT checkpoint; single node with 8 GPUs; G=16 rollouts per prompt at
# temperature 1.0; batch size 128; 4 epochs; lr 1e-6 with cosine schedule; KL coefficient 0.001;
# eps_low=0.2 / eps_high=0.28; w_step = w_length = 1.0.
#
# Key-step reward R_step is judged by QwQ-32B at temperature 0 with the key-step evaluation template, served on an
# OpenAI-compatible endpoint (paper Sec. 3.3.1, Appendix "Prompt Templates"). Start the judge before training, e.g.
#   vllm serve Qwen/QwQ-32B --tensor-parallel-size 4 --port 8000
#   python3 -m insynth_r1.reward.key_step        # smoke-test the judge
# Optional: JUDGE_API_KEY and comma-separated JUDGE_FALLBACK_MODELS, tried in order
# when JUDGE_MODEL fails. Failed judge requests fall back to the rule-based judge; STEP_JUDGE=rule uses it only.
#
# Usage:
#   bash examples/insynth_r1/run_rcpo.sh [extra hydra overrides]
#   MODEL_SIZE=4b MODEL_PATH=ckpts/msft_merged_4b bash examples/insynth_r1/run_rcpo.sh

set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ---- user-adjustable ----
MODEL_SIZE=${MODEL_SIZE:-8b}
MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/ckpts/msft_merged}
DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data/insynth}
# RCPO trains precursor selection only (synthesizability prediction is learned in Stage-1);
# validation exact_match is Accuracy@1 (Fig. 6a)
TRAIN_FILES=${TRAIN_FILES:-"['${DATA_DIR}/rl/precursor_train.parquet']"}
VAL_FILES=${VAL_FILES:-"['${DATA_DIR}/rl/precursor_test.parquet']"}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
ROLLOUT_N=${ROLLOUT_N:-16}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
ACTOR_LR=${ACTOR_LR:-1e-6}
# Mini-batch size and max lengths; adjust to your GPU memory
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

case "${MODEL_SIZE}" in
    # FSDP: 4B shards optimizer states and gradients, 8B also reshards parameters
    4b) ROLLOUT_TP=${ROLLOUT_TP:-1}; PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}; RESHARD=${RESHARD:-False} ;;
    8b) ROLLOUT_TP=${ROLLOUT_TP:-2}; PARAM_OFFLOAD=${PARAM_OFFLOAD:-False}; RESHARD=${RESHARD:-True} ;;
    *) echo "Unsupported MODEL_SIZE=${MODEL_SIZE}" >&2; exit 1 ;;
esac
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}

STEP_JUDGE=${STEP_JUDGE:-llm}
export JUDGE_API_BASE=${JUDGE_API_BASE:-http://localhost:8000/v1}
# About train_batch_size x rollout.n = 2048 judge calls per training step
export JUDGE_MODEL=${JUDGE_MODEL:-Qwen/QwQ-32B}
export JUDGE_FALLBACK_MODELS=${JUDGE_FALLBACK_MODELS:-}

PROJECT_NAME=${PROJECT_NAME:-insynth_r1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-rcpo_qwen3_${MODEL_SIZE}}
SAVE_DIR=${SAVE_DIR:-${REPO_ROOT}/ckpts/rl/${EXPERIMENT_NAME}}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}
# ---- end user-adjustable ----

DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=${RESHARD}
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${PARAM_OFFLOAD}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    reward.custom_reward_function.reward_kwargs.step_judge=${STEP_JUDGE}
)

TRAINER=(
    trainer.use_v1=True
    trainer.v1.trainer_mode=sync
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.default_local_dir="${SAVE_DIR}"
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.val_before_train=True
)

python3 -m insynth_r1.main_rcpo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
