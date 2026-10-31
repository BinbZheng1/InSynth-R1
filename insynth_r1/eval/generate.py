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
"""Generate model outputs for evaluation (paper Sec. 4.2.1).

Both tasks are answered in reasoning mode (``{prompt} /think``, Table 1) with N = 5 responses per prompt, temperature
0.6, top-p 0.95 and up to 8192 new tokens (2048 for non-reasoning models, ``--disable_thinking``).

* Precursor selection (``insynth/precursor``): the rejective fine-tuning prompt.
* Synthesizability prediction (``insynth/synthesizability``): the Stage-1 prompt. For every response, the label
  probability p(P) / (p(P) + p(U)) is read from the log-probabilities of the P/U answer token after the thinking
  block; the synthesizability score f(x) used by the PU metrics is its mean over the N responses.

    python -m insynth_r1.eval.generate --data data/insynth/eval/precursor_test.parquet \\
        --model_path ckpts/insynth-r1-8b --output outputs/insynth-r1-8b/precursor_test.jsonl
    python -m insynth_r1.eval.generate --data data/insynth/eval/synthesizability_test.parquet \\
        --model_path ckpts/insynth-r1-8b --output outputs/insynth-r1-8b/synthesizability_test.jsonl
    # API models (e.g., DeepSeek-R1); synthesizability then falls back to the P-vote fraction of N samples
    python -m insynth_r1.eval.generate --backend api --base_url <endpoint serving DeepSeek-R1> \\
        --model deepseek-ai/DeepSeek-R1 --api_key_env API_KEY --data ... --output ...
"""

import argparse
import math

from insynth_r1.data.io import read_records, write_jsonl
from insynth_r1.reward.parsing import THINK_CLOSE, THINK_OPEN
from insynth_r1.reward.rubric import SYNTHESIZABILITY_SOURCE

LABEL_TOP_LOGPROBS = 20


def _to_list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _messages(row: dict) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in _to_list(row["prompt"])]


def label_probability(token_ids: list[int], position_logprobs: list[dict]) -> float | None:
    """p(P) / (p(P) + p(U)) at the first generated P/U token after the thinking block.

    ``position_logprobs[i]`` maps candidate token ids to vLLM ``Logprob`` entries (``logprob``, ``decoded_token``)
    and contains the sampled token ``token_ids[i]``. Returns ``None`` if no P/U token follows the thinking block,
    e.g. when the response is truncated inside it.
    """
    positions = list(zip(token_ids or [], position_logprobs or [], strict=False))
    sampled = [(candidates.get(token_id), candidates) for token_id, candidates in positions]
    decoded = [(entry.decoded_token or "") if entry is not None else "" for entry, _ in sampled]
    start = 0
    for i, token in enumerate(decoded):
        if THINK_CLOSE in token:
            start = i + 1
    if start == 0 and any(THINK_OPEN in token for token in decoded):
        return None
    for (_, candidates), token in zip(sampled[start:], decoded[start:], strict=True):
        if token.strip() not in ("P", "U"):
            continue
        probs = {"P": 0.0, "U": 0.0}
        for candidate in candidates.values():
            label = (candidate.decoded_token or "").strip()
            if label in probs:
                probs[label] += math.exp(candidate.logprob)
        return probs["P"] / (probs["P"] + probs["U"])
    return None


def generate_vllm(rows: list[dict], args) -> list[dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    outputs: list[dict] = [{} for _ in rows]
    sampling = {"n": args.n, "temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}
    kwargs = {"enable_thinking": not args.disable_thinking}

    precursor = [i for i, r in enumerate(rows) if not r["data_source"].startswith(SYNTHESIZABILITY_SOURCE)]
    if precursor:
        params = SamplingParams(**sampling)
        results = llm.chat([_messages(rows[i]) for i in precursor], params, chat_template_kwargs=kwargs)
        for i, result in zip(precursor, results, strict=True):
            outputs[i] = {"responses": [c.text for c in result.outputs]}

    synthesizability = [i for i, r in enumerate(rows) if r["data_source"].startswith(SYNTHESIZABILITY_SOURCE)]
    if synthesizability:
        params = SamplingParams(**sampling, logprobs=LABEL_TOP_LOGPROBS)
        results = llm.chat([_messages(rows[i]) for i in synthesizability], params, chat_template_kwargs=kwargs)
        for i, result in zip(synthesizability, results, strict=True):
            outputs[i] = {
                "responses": [c.text for c in result.outputs],
                "p_positive": [label_probability(c.token_ids, c.logprobs) for c in result.outputs],
            }
    return outputs


def generate_api(rows: list[dict], args) -> list[dict]:
    import asyncio

    from insynth_r1.coldstart.client import ChatClient

    client = ChatClient(args.base_url, args.model, args.api_key_env, concurrency=args.concurrency)
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}
    conversations = [_messages(r) for r in rows for _ in range(args.n)]
    results = asyncio.run(client.chat_many(conversations, **sampling))
    texts = []
    for result in results:
        if result is None:
            texts.append("")
        elif result.reasoning_content and "</think>" not in result.content:
            # Reasoning APIs return thinking separately; keep the <think>/<answer> layout for parsing.
            texts.append(f"<think>\n{result.reasoning_content}\n</think>\n\n{result.content}")
        else:
            texts.append(result.content)
    return [{"responses": texts[i * args.n : (i + 1) * args.n]} for i in range(len(rows))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="evaluation parquet produced by insynth_r1.data.preprocess")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["vllm", "api"], default="vllm")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--disable_thinking", action="store_true", help="non-reasoning mode for hybrid models")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api_key_env", default=None)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=8192, help="8192 (reasoning) / 2048 (non-reasoning models)")
    parser.add_argument("--max_model_len", type=int, default=32768, help="context window (vLLM backend)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    rows = read_records(args.data)[: args.limit]
    outputs = generate_vllm(rows, args) if args.backend == "vllm" else generate_api(rows, args)
    records = []
    for row, out in zip(rows, outputs, strict=True):
        extra = row.get("extra_info") or {}
        records.append(
            {
                "index": extra.get("index"),
                "data_source": row["data_source"],
                "target": extra.get("target"),
                "ground_truth": row["reward_model"]["ground_truth"],
                **out,
            }
        )
    write_jsonl(records, args.output)
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
