#!/usr/bin/env bash
# InSynth-R1 MSFT Stage-1: Domain Fine-Tuning (paper Sec. 3.2.1)
#
# LoRA (r=16) SFT on direct-answer synthesizability prediction and precursor selection data,
# peak lr 1e-5 with cosine decay (paper Sec. 4.2.1), on verl's SFT trainer.
#
# Usage: bash examples/insynth_r1/run_stage1_sft.sh [extra hydra overrides]

set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ---- user-adjustable ----
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data/insynth}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/sft/stage1_train.parquet}
# "null" disables validation
VAL_FILE=${VAL_FILE:-${DATA_DIR}/sft/stage1_test.parquet}
SAVE_DIR=${SAVE_DIR:-${REPO_ROOT}/ckpts/stage1_sft}
NGPUS=${NGPUS:-8}
LORA_RANK=${LORA_RANK:-16}
# LoRA alpha / warmup / epochs / batch size; adjust to your hardware
LORA_ALPHA=${LORA_ALPHA:-32}
LR=${LR:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}
PROJECT_NAME=${PROJECT_NAME:-insynth_r1}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-stage1_domain_sft}
# ---- end user-adjustable ----

torchrun --standalone --nnodes=1 --nproc_per_node="${NGPUS}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.messages_key=messages \
    data.custom_cls.path=pkg://insynth_r1.data.sft_dataset \
    data.custom_cls.name=SingleTurnSFTDataset \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.max_length="${MAX_LENGTH}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.truncation=error \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    model.lora_rank="${LORA_RANK}" \
    model.lora_alpha="${LORA_ALPHA}" \
    model.target_modules=all-linear \
    engine=fsdp \
    optim.lr="${LR}" \
    optim.lr_scheduler_type=cosine \
    optim.lr_warmup_steps_ratio="${WARMUP_RATIO}" \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console","tensorboard"]' \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.n_gpus_per_node="${NGPUS}" \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "$@"

# Export the last checkpoint and fold the LoRA adapter into the base weights:
#   bash examples/insynth_r1/merge_checkpoint.sh ${SAVE_DIR}/global_step_<N> ${REPO_ROOT}/ckpts/stage1_merged
