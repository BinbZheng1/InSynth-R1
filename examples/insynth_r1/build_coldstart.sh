#!/usr/bin/env bash
# Stage-2 cold-start data construction (paper Algorithm 1, Appendix "Stage-2 Trajectory Generation and Selection").
#
#   1. generate: DeepSeek-R1 rolls out one CoT trajectory per training reaction (T=0.6, top-p 0.95, 8192 tokens)
#   2. score:    DeepSeek-V3.1 rates complexity / utility on 1-5 scales (T=0.6, top-p 0.95, 1024 tokens),
#                S_x = 0.5 C~ + 0.5 U~
#   3. select:   top K = 50% by S_x, then exact-match answer verification; exports a review sheet
#   4. (manual)  two chemistry experts fill reviewer_1 / reviewer_2 in review.csv (1 = accept, 0 = reject)
#   5. build:    keep trajectories accepted by both experts and write the Stage-2 SFT parquet
#
# Usage:
#   GEN_BASE_URL=... GEN_MODEL=deepseek-ai/DeepSeek-R1 GEN_API_KEY_ENV=... \
#   EVAL_BASE_URL=... EVAL_MODEL=deepseek-ai/DeepSeek-V3.1 EVAL_API_KEY_ENV=... \
#   bash examples/insynth_r1/build_coldstart.sh <generate|score|select|build>
#
# API keys are read from the environment variables named by *_API_KEY_ENV.

set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

STEP=${1:?usage: build_coldstart.sh <generate|score|select|build>}
DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data/insynth}
WORK_DIR=${WORK_DIR:-${DATA_DIR}/coldstart}

case "${STEP}" in
    generate)
        python3 -m insynth_r1.coldstart.pipeline generate \
            --input "${WORK_DIR}/input.jsonl" --output "${WORK_DIR}/generated.jsonl" \
            --base_url "${GEN_BASE_URL:?}" --model "${GEN_MODEL:?}" --api_key_env "${GEN_API_KEY_ENV:?}" \
            --temperature 0.6 --top_p 0.95 --max_tokens 8192 --concurrency "${CONCURRENCY:-32}"
        ;;
    score)
        python3 -m insynth_r1.coldstart.pipeline score \
            --input "${WORK_DIR}/generated.jsonl" --output "${WORK_DIR}/scored.jsonl" \
            --base_url "${EVAL_BASE_URL:?}" --model "${EVAL_MODEL:?}" --api_key_env "${EVAL_API_KEY_ENV:?}" \
            --temperature 0.6 --top_p 0.95 --max_tokens 1024 --alpha 0.5 --concurrency "${CONCURRENCY:-32}"
        ;;
    select)
        python3 -m insynth_r1.coldstart.pipeline select \
            --input "${WORK_DIR}/scored.jsonl" --output "${WORK_DIR}/selected.jsonl" \
            --top_k 0.5 --review_csv "${WORK_DIR}/review.csv" --num_reviewers 2
        ;;
    build)
        python3 -m insynth_r1.coldstart.pipeline build \
            --input "${WORK_DIR}/selected.jsonl" --review_csv "${WORK_DIR}/review.csv" \
            --output "${DATA_DIR}/sft/stage2_train.parquet" --min_accept 2
        ;;
    *)
        echo "Unknown step ${STEP}" >&2
        exit 1
        ;;
esac
