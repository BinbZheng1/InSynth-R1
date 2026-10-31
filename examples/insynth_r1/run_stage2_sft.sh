#!/usr/bin/env bash
# InSynth-R1 MSFT Stage-2: Rejective Fine-Tuning (paper Sec. 3.2.2, Algorithm 1)
#
# Continues SFT from the merged Stage-1 model on the complexity-utility-selected, expert-reviewed CoT trajectories
# (build them with examples/insynth_r1/build_coldstart.sh). Same LoRA / optimizer settings as Stage-1.
#
# Usage: bash examples/insynth_r1/run_stage2_sft.sh [extra hydra overrides]

set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data/insynth}

# ---- user-adjustable ----
export MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/ckpts/stage1_merged}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/sft/stage2_train.parquet}
export VAL_FILE=${VAL_FILE:-null}
export SAVE_DIR=${SAVE_DIR:-${REPO_ROOT}/ckpts/stage2_sft}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-stage2_rejective_sft}
# Reasoning trajectories are much longer than Stage-1 answers
export MAX_LENGTH=${MAX_LENGTH:-10240}
export MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-16384}
export MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-2}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
# ---- end user-adjustable ----

bash "${REPO_ROOT}/examples/insynth_r1/run_stage1_sft.sh" "$@"

# Then: bash examples/insynth_r1/merge_checkpoint.sh ${SAVE_DIR}/global_step_<N> ${REPO_ROOT}/ckpts/msft_merged
