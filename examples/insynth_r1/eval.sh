#!/usr/bin/env bash
# Evaluate the final model on both tasks (paper Sec. 4.2.2, Appendix B), in reasoning mode with N = 5 responses,
# temperature 0.6, top-p 0.95, context window 32768 and up to 8192 new tokens (MAX_TOKENS=2048 for non-reasoning
# models).
#
# * Precursor selection (public test set and the temporally held-out in-house set): rejective fine-tuning prompt.
# * Synthesizability prediction: Stage-1 prompt; calibrated F1 under prior matching, with f(x) the P/U label
#   probability after the thinking block averaged over the N responses.
#
# Usage:
#   MODEL_PATH=ckpts/insynth-r1-8b bash examples/insynth_r1/eval.sh outputs/insynth-r1-8b
#   EVAL_SETS="precursor_test inhouse" MODEL_PATH=... bash examples/insynth_r1/eval.sh outputs/insynth-r1-8b

set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

OUT_DIR=${1:?usage: eval.sh <output_dir>}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data/insynth}
EVAL_SETS=${EVAL_SETS:-"precursor_test synthesizability_test inhouse"}
TP=${TP:-1}
N=${N:-5}
MAX_TOKENS=${MAX_TOKENS:-8192}
INVENTORY=${INVENTORY:-}

for NAME in ${EVAL_SETS}; do
    DATA="${DATA_DIR}/eval/${NAME}.parquet"
    if [ ! -f "${DATA}" ]; then
        echo "skip ${NAME}: ${DATA} not found"
        continue
    fi
    python3 -m insynth_r1.eval.generate --data "${DATA}" --model_path "${MODEL_PATH}" \
        --output "${OUT_DIR}/${NAME}.jsonl" --n "${N}" --temperature 0.6 --top_p 0.95 --max_tokens "${MAX_TOKENS}" \
        --max_model_len 32768 --tensor_parallel_size "${TP}"

    EVAL_ARGS=(--generations "${OUT_DIR}/${NAME}.jsonl" --output "${OUT_DIR}/${NAME}_metrics.json")
    if [ -n "${INVENTORY}" ]; then EVAL_ARGS+=(--inventory "${INVENTORY}"); fi
    python3 -m insynth_r1.eval.evaluate "${EVAL_ARGS[@]}"
done
