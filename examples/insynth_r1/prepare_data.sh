#!/usr/bin/env bash
# Build all datasets from raw records (see insynth_r1/data/preprocess.py for the expected columns).
#
# Usage:
#   REACTIONS=raw/reactions.jsonl SYNTHESIZABILITY=raw/synthesizability.csv \
#   EVAL_REACTIONS=inhouse_benchmark/data_sample.json bash examples/insynth_r1/prepare_data.sh

set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

REACTIONS=${REACTIONS:?set REACTIONS to the filtered reaction file}
SYNTHESIZABILITY=${SYNTHESIZABILITY:?set SYNTHESIZABILITY to the synthesized/hypothetical composition file}
EVAL_REACTIONS=${EVAL_REACTIONS:-}
SEED=${SEED:-42}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/data/insynth}

ARGS=(--reactions "${REACTIONS}" --synthesizability "${SYNTHESIZABILITY}" --seed "${SEED}" --out_dir "${OUT_DIR}")
if [ -n "${EVAL_REACTIONS}" ]; then ARGS+=(--eval_reactions "${EVAL_REACTIONS}" --eval_name inhouse); fi

python3 -m insynth_r1.data.preprocess "${ARGS[@]}"
