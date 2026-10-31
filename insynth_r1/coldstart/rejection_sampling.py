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
"""Complexity-utility-based rejection sampling for Stage-2 Rejective Fine-Tuning (paper Algorithm 1).

    for x in D:                                              # training reactions of the precursor selection data
        y ~ pi_delta(. | x)                                  # DeepSeek-R1, one trajectory per reaction
        C~_x, U~_x = evaluate(y)                             # DeepSeek-V3.1, 1-5 scales, normalized (s - 1) / 4
        S_x = alpha * C~_x + (1 - alpha) * U~_x               # alpha = 0.5
    C_top = TopK(C)                                          # top K = 50% by S_x
    D_R   = {(x, y) in C_top | y contains a correct final answer}   # exact precursor-set match
    D_R'  = HumanReview(D_R)                                 # kept if both chemistry experts accept
"""

import json
import re

from insynth_r1.data import prompts
from insynth_r1.reward.parsing import THINK_CLOSE, THINK_OPEN, extract_precursors, parse_response
from insynth_r1.reward.rubric import precursor_set_metrics

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def normalize_rating(score: float) -> float:
    """Map a 1-5 rating to [0, 1]."""
    return (float(score) - 1.0) / 4.0


def unified_score(complexity: float, utility: float, alpha: float = 0.5) -> float:
    """S_x = alpha * C~_x + (1 - alpha) * U~_x on normalized ratings."""
    return alpha * normalize_rating(complexity) + (1.0 - alpha) * normalize_rating(utility)


def parse_evaluation(text: str | None) -> tuple[int, int] | None:
    """Parse the evaluator's JSON verdict into integer ``(complexity, utility)`` ratings in [1, 5]."""
    if not text:
        return None
    match = _JSON_OBJECT.search(text)
    if not match:
        return None
    try:
        verdict = json.loads(match.group(0))
        complexity, utility = int(verdict["complexity"]), int(verdict["utility"])
    except (ValueError, KeyError, TypeError):
        return None
    if not (1 <= complexity <= 5 and 1 <= utility <= 5):
        return None
    return complexity, utility


def assemble_trajectory(content: str, reasoning_content: str | None = None, source: str = "content") -> str:
    """Build the candidate CoT trajectory ``<think>...</think><answer>...</answer>`` from a generator response.

    ``source="content"`` uses the structured CoT the rejection-sampling prompt asks for in the message content and
    falls back to the API's ``reasoning_content`` when the content has no thinking block. ``source="reasoning"``
    always uses ``reasoning_content`` (the generator's native thinking) as the thinking process.
    """
    content = content or ""
    has_think = THINK_OPEN in content and THINK_CLOSE in content
    if source == "content" and has_think:
        return content.strip()
    if reasoning_content:
        answer = content.rsplit(THINK_CLOSE, 1)[-1] if THINK_CLOSE in content else content
        return f"{THINK_OPEN}\n{reasoning_content.strip()}\n{THINK_CLOSE}\n\n{answer.strip()}"
    return content.strip()


def is_correct(trajectory: str | None, ground_truth) -> bool:
    """Answer verifier (Algorithm 1, line 11): the parsed precursor set exactly matches the reference set."""
    if not trajectory:
        return False
    predicted = extract_precursors(trajectory)
    return bool(precursor_set_metrics(predicted, ground_truth)["exact_match"])


def select_top_k(pool: list[dict], top_k: float) -> list[dict]:
    """Top-K by unified score; ``top_k`` in (0, 1] is a fraction of the scored pool, otherwise an absolute count."""
    scored = [r for r in pool if r.get("score") is not None]
    num = round(top_k * len(scored)) if 0 < top_k <= 1 else int(top_k)
    return sorted(scored, key=lambda r: (-r["score"], r["index"]))[:num]


def complexity_utility_selection(pool: list[dict], top_k: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Algorithm 1, lines 8-14. Returns ``(C_top, D_R)``."""
    top = select_top_k(pool, top_k)
    return top, [r for r in top if is_correct(r["trajectory"], r["precursors"])]


def human_review(records: list[dict], decisions: dict[int, list[bool]], min_accept: int = 2) -> list[dict]:
    """Algorithm 1, line 15: keep trajectories accepted by at least ``min_accept`` reviewers."""
    return [r for r in records if sum(decisions.get(r["index"], [])) >= min_accept]


def to_sft_record(record: dict, think_suffix: str = prompts.THINK_SUFFIX) -> dict | None:
    """Stage-2 SFT sample: the RL / evaluation prompt -> ``<think>...</think>\\n\\n<answer>one line</answer>``."""
    parsed = parse_response(record["trajectory"])
    precursors = extract_precursors(record["trajectory"])
    if not parsed.think or not precursors:
        return None
    answer = prompts.reaction_line(record["target"], precursors)
    response = f"{THINK_OPEN}\n{parsed.think}\n{THINK_CLOSE}\n\n<answer>{answer}</answer>"
    return {
        "messages": [
            {"role": "user", "content": prompts.precursor_reasoning_prompt(record["target"], think_suffix)},
            {"role": "assistant", "content": response},
        ],
        "index": record["index"],
        "target": record["target"],
        "score": record.get("score"),
    }
