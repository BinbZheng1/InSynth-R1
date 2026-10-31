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
"""Stage-2 cold-start data construction CLI (Algorithm 1).

Inputs are the training reactions of the precursor selection data (``coldstart/input.jsonl`` from preprocessing).

    python -m insynth_r1.coldstart.pipeline generate --input data/insynth/coldstart/input.jsonl \
        --output data/insynth/coldstart/generated.jsonl --base_url <endpoint serving DeepSeek-R1> \
        --model deepseek-ai/DeepSeek-R1 --api_key_env GENERATOR_API_KEY
    python -m insynth_r1.coldstart.pipeline score --input .../generated.jsonl --output .../scored.jsonl \
        --base_url <endpoint serving DeepSeek-V3.1> --model deepseek-ai/DeepSeek-V3.1 --api_key_env EVALUATOR_API_KEY
    python -m insynth_r1.coldstart.pipeline select --input .../scored.jsonl --output .../selected.jsonl \
        --review_csv .../review.csv
    # two chemistry experts fill reviewer_1 / reviewer_2 (1 = accept, 0 = reject) in review.csv
    python -m insynth_r1.coldstart.pipeline build --input .../selected.jsonl --review_csv .../review.csv \
        --output data/insynth/sft/stage2_train.parquet

Paper settings: DeepSeek-R1 generator (temperature 0.6, top-p 0.95, 8192 tokens, one trajectory per reaction);
DeepSeek-V3.1 evaluator (temperature 0.6, top-p 0.95, 1024 tokens, alpha = 0.5); top 50% + exact-match answer
verifier; a trajectory is kept only if both experts accept it.

Every API-calling step is resumable: records already present in ``--output`` are skipped.
"""

import argparse
import asyncio
import csv
import os

from insynth_r1.coldstart import rejection_sampling as rs
from insynth_r1.data import prompts
from insynth_r1.data.io import as_list, read_records, write_json, write_jsonl, write_parquet


def _load_done(path: str) -> dict[int, dict]:
    if not os.path.exists(path):
        return {}
    return {r["index"]: r for r in read_records(path)}


def _append_jsonl(records: list[dict], path: str) -> None:
    import json

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_in_chunks(records: list[dict], output: str, make_messages, handle_result, client, sampling, chunk: int):
    done = _load_done(output)
    todo = [r for r in records if r["index"] not in done]
    print(f"{len(done)} done, {len(todo)} to go")
    for start in range(0, len(todo), chunk):
        batch = todo[start : start + chunk]
        results = asyncio.run(client.chat_many([make_messages(r) for r in batch], **sampling))
        _append_jsonl([handle_result(r, res) for r, res in zip(batch, results, strict=True)], output)
        print(f"{min(start + chunk, len(todo))}/{len(todo)}")


def _load(path: str) -> list[dict]:
    return [{**r, "precursors": as_list(r["precursors"])} for r in read_records(path)]


def cmd_generate(args):
    from insynth_r1.coldstart.client import ChatClient

    records = _load(args.input)
    client = ChatClient(args.base_url, args.model, args.api_key_env, concurrency=args.concurrency)
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}

    def handle(record, result):
        if result is None:
            return {**record, "trajectory": None, "finish_reason": "error"}
        trajectory = rs.assemble_trajectory(result.content, result.reasoning_content, args.trajectory_source)
        return {**record, "trajectory": trajectory, "finish_reason": result.finish_reason}

    _run_in_chunks(
        records,
        args.output,
        lambda r: [{"role": "user", "content": prompts.rejection_sampling_prompt(r["target"])}],
        handle,
        client,
        sampling,
        args.chunk_size,
    )


def cmd_score(args):
    from insynth_r1.coldstart.client import ChatClient

    records = [r for r in read_records(args.input) if r.get("trajectory")]
    client = ChatClient(args.base_url, args.model, args.api_key_env, concurrency=args.concurrency)
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}

    def handle(record, result):
        verdict = rs.parse_evaluation(result.content if result else None)
        if verdict is None:
            return {**record, "complexity": None, "utility": None, "score": None}
        complexity, utility = verdict
        score = rs.unified_score(complexity, utility, args.alpha)
        return {**record, "complexity": complexity, "utility": utility, "score": score}

    _run_in_chunks(
        records,
        args.output,
        lambda r: [{"role": "user", "content": prompts.complexity_utility_prompt(r["target"], r["trajectory"])}],
        handle,
        client,
        sampling,
        args.chunk_size,
    )


def cmd_select(args):
    pool = _load(args.input)
    top, selected = rs.complexity_utility_selection(pool, args.top_k)
    stats = {
        "num_candidates": len(pool),
        "num_scored": sum(r.get("score") is not None for r in pool),
        "num_top_k": len(top),
        "num_verified": len(selected),
    }
    write_jsonl(selected, args.output)
    write_json(stats, os.path.splitext(args.output)[0] + "_stats.json")
    if args.review_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.review_csv)), exist_ok=True)
        with open(args.review_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            reviewers = [f"reviewer_{i + 1}" for i in range(args.num_reviewers)]
            writer.writerow(["index", "target", "reference", "trajectory", *reviewers])
            for r in selected:
                row = [r["index"], r["target"], " + ".join(r["precursors"]), r["trajectory"]]
                writer.writerow(row + [""] * len(reviewers))
    print(stats)


def _read_review(path: str) -> dict[int, list[bool]]:
    decisions = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            votes = [row[k].strip() for k in row if k.startswith("reviewer_") and row[k] and row[k].strip()]
            decisions[int(row["index"])] = [v.lower() in {"1", "y", "yes", "accept", "true"} for v in votes]
    return decisions


def cmd_build(args):
    selected = _load(args.input)
    stats = {"num_verified": len(selected)}
    if args.review_csv:
        selected = rs.human_review(selected, _read_review(args.review_csv), args.min_accept)
    else:
        print("WARNING: no --review_csv given, skipping expert review")
    rows = [row for row in (rs.to_sft_record(r, args.think_suffix) for r in selected) if row is not None]
    stats.update(num_reviewed=len(selected), num_final=len(rows))
    write_parquet(rows, args.output)
    write_json(stats, os.path.splitext(args.output)[0] + "_stats.json")
    print(stats)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def api_args(p, max_tokens, temperature=0.6, top_p=0.95):
        p.add_argument("--input", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--base_url", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--api_key_env", default=None, help="name of the env var holding the API key")
        p.add_argument("--concurrency", type=int, default=32)
        p.add_argument("--chunk_size", type=int, default=256)
        p.add_argument("--max_tokens", type=int, default=max_tokens)
        p.add_argument("--temperature", type=float, default=temperature)
        p.add_argument("--top_p", type=float, default=top_p)

    p = sub.add_parser("generate", help="roll out one candidate CoT trajectory per training reaction")
    api_args(p, 8192)
    p.add_argument("--trajectory_source", choices=["content", "reasoning"], default="content")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("score", help="complexity-utility evaluation")
    api_args(p, 1024)
    p.add_argument("--alpha", type=float, default=0.5)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("select", help="top-K selection + answer verification")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top_k", type=float, default=0.5, help="fraction (<=1) or count of the scored pool")
    p.add_argument("--review_csv", default=None, help="export a sheet for expert review")
    p.add_argument("--num_reviewers", type=int, default=2)
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("build", help="apply expert review and write the Stage-2 SFT parquet")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--review_csv", default=None)
    p.add_argument("--min_accept", type=int, default=2, help="accept votes needed (2 = both experts)")
    p.add_argument("--think_suffix", default=prompts.THINK_SUFFIX)
    p.set_defaults(func=cmd_build)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
