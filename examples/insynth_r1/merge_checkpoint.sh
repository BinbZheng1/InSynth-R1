#!/usr/bin/env bash
# Convert a verl FSDP checkpoint to a HuggingFace model and fold LoRA adapters (if any) into the base weights.
#
# Usage:
#   bash examples/insynth_r1/merge_checkpoint.sh <ckpt_dir> <output_dir>
#   # SFT checkpoints: <save_dir>/global_step_<N>;  RL checkpoints: <save_dir>/global_step_<N>/actor

set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [ "$#" -ne 2 ]; then
    echo "Usage: merge_checkpoint.sh <ckpt_dir> <output_dir>" >&2
    exit 1
fi
CKPT_DIR=$1
OUTPUT_DIR=$2
TMP_DIR="${OUTPUT_DIR}.verl_export"
if [ -e "${OUTPUT_DIR}" ] || [ -e "${TMP_DIR}" ]; then
    echo "Refusing to overwrite existing ${OUTPUT_DIR} or ${TMP_DIR}" >&2
    exit 1
fi

python3 -m verl.model_merger merge --backend fsdp --local_dir "${CKPT_DIR}" --target_dir "${TMP_DIR}"

if [ -d "${TMP_DIR}/lora_adapter" ]; then
    python3 -m insynth_r1.tools.merge_lora --base_dir "${TMP_DIR}" --output_dir "${OUTPUT_DIR}"
    rm -rf -- "${TMP_DIR}"
else
    mv -- "${TMP_DIR}" "${OUTPUT_DIR}"
fi
echo "HuggingFace model saved to ${OUTPUT_DIR}"
