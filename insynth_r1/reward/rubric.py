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
"""Synthesis-aware rubric judge (paper Sec. 3.3.1).

The total reward of a trajectory y is (Eq. 6)::

    r_total = R_accuracy + R_format + w_step * R_step + w_length * R_length

* R_accuracy: F1 score between the predicted and ground-truth precursor sets.
* R_format:   1 if the ``<think>...</think><answer>...</answer>`` template is satisfied, else 0.
* R_step:     key-step reward in {1, 0, -1} (Eq. 4, see :mod:`insynth_r1.reward.key_step`).
* R_length:   max(-1, min(0, -(|y| - L_soft) / (L_hard - L_soft))) (Eq. 5), only active from
              ``length_start_step`` on, i.e. in the later steps of RL training.

Use as a verl custom reward function::

    reward.custom_reward_function.path=pkg://insynth_r1.reward.rubric
    reward.custom_reward_function.name=compute_score
"""

import logging
from functools import lru_cache
from typing import Any

import numpy as np

from insynth_r1.chem import formula_key, is_valid_formula
from insynth_r1.reward.key_step import StepResult, llm_steps, rule_based_steps, split_models
from insynth_r1.reward.parsing import extract_precursors, parse_response

logger = logging.getLogger(__name__)

# RL covers precursor selection only; synthesizability prediction is trained in Stage-1 and evaluated in reasoning
# mode with its Stage-1 template (insynth_r1.eval).
PRECURSOR_SOURCE = "insynth/precursor"
SYNTHESIZABILITY_SOURCE = "insynth/synthesizability"

# Every call returns the same keys: verl stacks reward_extra_info across samples by key.
EXTRA_KEYS = (
    "acc",
    "exact_match",
    "precision",
    "recall",
    "format",
    "step",
    "step_analyze",
    "step_identify",
    "step_justify",
    "length",
    "length_active",
    "validity",
    "parsed",
    "step_judge_index",
)

_warned_missing_length = False


def _as_set(item: Any) -> list[str]:
    if isinstance(item, str):
        if "<-" in item or "<=" in item:
            item = item.replace("<=", "<-").split("<-", 1)[1]
        return [p.strip() for p in item.split("+") if p.strip()]
    if isinstance(item, np.ndarray):
        item = item.tolist()
    return [str(p).strip() for p in item if str(p).strip()]


def as_reference_sets(ground_truth: Any) -> list[list[str]]:
    """Normalize the ground truth into one or more acceptable precursor sets.

    Accepted forms: ``"A + B"`` / ``"T <- A + B"``; several reactions separated by newlines or ``" | "``; a flat list of
    precursors (one set); a list of lists or of reaction strings (several sets). A target with several reported
    reactions is scored against the best-matching one.
    """
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()
    if isinstance(ground_truth, str):
        parts = [p for p in ground_truth.replace(" | ", "\n").splitlines() if p.strip()]
    elif ground_truth and all(isinstance(g, list | tuple | np.ndarray) for g in ground_truth):
        parts = list(ground_truth)
    elif ground_truth and all(isinstance(g, str) and ("<-" in g or "<=" in g) for g in ground_truth):
        parts = list(ground_truth)
    else:
        parts = [ground_truth]
    return [s for s in (_as_set(p) for p in parts) if s]


def as_precursor_list(ground_truth: Any) -> list[str]:
    """First acceptable precursor set (kept for single-reference callers)."""
    sets = as_reference_sets(ground_truth)
    return sets[0] if sets else []


def _single_set_metrics(predicted: list[str] | None, reference: list[str]) -> dict[str, float]:
    ref = {formula_key(p) for p in reference}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}
    pred = {formula_key(p) for p in predicted}
    tp = len(pred & ref)
    precision = tp / len(pred)
    recall = tp / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "exact_match": float(pred == ref)}


def precursor_set_metrics(predicted: list[str] | None, reference: Any) -> dict[str, float]:
    """Per-reaction precision / recall / F1 / exact match on order-independent, normalized precursor sets.

    With several acceptable reference sets, the metrics of the best-matching set are returned (exact match first,
    then F1), i.e. predicting any one of the reported reactions counts as correct.
    """
    references = as_reference_sets(reference)
    if not references:
        return _single_set_metrics(predicted, [])
    candidates = [_single_set_metrics(predicted, ref) for ref in references]
    return max(candidates, key=lambda m: (m["exact_match"], m["f1"]))


@lru_cache(maxsize=8)
def load_inventory(path: str) -> frozenset[str]:
    """Load the precursor inventory (one formula per line, or a JSON list) as normalized keys."""
    import json

    with open(path, encoding="utf-8") as f:
        text = f.read()
    items = json.loads(text) if text.lstrip().startswith("[") else text.splitlines()
    return frozenset(formula_key(item) for item in items if str(item).strip())


def precursor_validity(predicted: list[str] | None, inventory_path: str | None = None) -> float:
    """Fraction of predicted precursors in the precursor inventory (paper's Validity).

    Without an inventory, the fraction of chemically parseable formulas is returned as a proxy.
    """
    if not predicted:
        return 0.0
    if inventory_path:
        inventory = load_inventory(inventory_path)
        return sum(formula_key(p) in inventory for p in predicted) / len(predicted)
    return sum(is_valid_formula(p) for p in predicted) / len(predicted)


def length_reward(response_length: int, length_soft: int, length_hard: int) -> float:
    """Paper Eq. 5."""
    assert length_hard > length_soft, f"{length_hard=} must be larger than {length_soft=}"
    return max(-1.0, min(0.0, -(response_length - length_soft) / (length_hard - length_soft)))


def _length_component(
    extra_info: dict, length_soft: int, length_hard: int, length_start_step: int
) -> tuple[float, bool]:
    global _warned_missing_length
    if length_start_step < 0:
        return 0.0, False
    if "length_active" in extra_info:
        active = bool(extra_info["length_active"])
    else:
        step = extra_info.get("global_steps")
        active = step is not None and int(step) >= length_start_step
    if not active:
        return 0.0, False
    response_length = extra_info.get("response_length")
    if response_length is None:
        if not _warned_missing_length:
            logger.warning(
                "extra_info has no 'response_length'; R_length is disabled. "
                "Use insynth_r1.reward.manager.InSynthRewardManager as the reward manager."
            )
            _warned_missing_length = True
        return 0.0, False
    return length_reward(int(response_length), length_soft, length_hard), True


def _key_steps(
    think: str | None, target: str, predicted: list[str] | None, answer: str | None, step_judge: str, judge: dict
) -> tuple[StepResult, int]:
    """Key steps and the judge that produced them: index into the judge models, or -1 for the rule-based judge."""
    if step_judge == "llm" and think:
        result = llm_steps(think, target, answer, **judge)
        if result is not None:
            return result
    return rule_based_steps(think or "", target, predicted), -1


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    w_step: float = 1.0,
    w_length: float = 1.0,
    length_soft: int = 1024,
    length_hard: int = 2048,
    length_start_step: int = 0,
    step_judge: str = "rule",
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    judge_fallback_models=None,
    judge_api_key_env: str = "JUDGE_API_KEY",
    judge_timeout: float = 300.0,
    allow_missing_think_open: bool = False,
    inventory_path: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Synthesis-aware rubric reward for one precursor-selection response.

    Args:
        data_source: ``insynth/precursor``.
        solution_str: decoded response.
        ground_truth: reference precursor set(s), see :func:`as_reference_sets`; several sets = any one is correct.
        extra_info: dataset ``extra_info``; ``target`` is required. ``response_length`` and ``global_steps`` are
            injected by :class:`insynth_r1.reward.manager.InSynthRewardManager`.
        w_step, w_length: reward weights in Eq. 6 (paper: both 1.0).
        length_soft, length_hard: L_soft / L_hard in Eq. 5.
        length_start_step: first global step at which R_length is active; negative disables it.
        step_judge: ``rule`` or ``llm``. The LLM judge calls ``judge_model`` and, if it fails, each of
            ``judge_fallback_models`` in order (defaults: env ``JUDGE_MODEL`` / ``JUDGE_FALLBACK_MODELS``) at
            ``judge_base_url`` (env ``JUDGE_API_BASE``), with the API key from env ``judge_api_key_env``. If every judge
            fails, the rule-based judge is used. ``step_judge_index`` in the output records which judge answered
            (0 = primary, -1 = rule-based). ``judge_timeout`` is the per-request timeout in seconds.
        inventory_path: precursor inventory used for the Validity metric.
    """
    extra_info = dict(extra_info or {})
    target = str(extra_info.get("target", ""))
    out = dict.fromkeys(EXTRA_KEYS, 0.0)

    r_length, length_active = _length_component(extra_info, length_soft, length_hard, length_start_step)
    out["length"], out["length_active"] = r_length, float(length_active)

    if data_source.startswith(SYNTHESIZABILITY_SOURCE):
        raise ValueError("RL covers precursor selection only; synthesizability prediction is a Stage-1 task")

    parsed = parse_response(solution_str, allow_missing_think_open)
    predicted = extract_precursors(solution_str, allow_missing_think_open)
    metrics = precursor_set_metrics(predicted, ground_truth)
    r_accuracy = metrics["f1"]
    r_format = float(parsed.format_ok)
    judge_models = split_models(judge_model) + split_models(judge_fallback_models)
    judge = {
        "base_url": judge_base_url,
        "models": judge_models or None,
        "api_key_env": judge_api_key_env,
        "timeout": judge_timeout,
    }
    steps, judge_index = _key_steps(parsed.think, target, predicted, parsed.answer, step_judge, judge)
    r_step = steps.reward
    out.update(
        acc=r_accuracy,
        exact_match=metrics["exact_match"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        step_analyze=float(steps.analyze),
        step_identify=float(steps.identify),
        step_justify=float(steps.justify),
        step_judge_index=float(judge_index),
        validity=precursor_validity(predicted, inventory_path),
        parsed=float(predicted is not None),
        pred=" + ".join(sorted(formula_key(p) for p in predicted)) if predicted else "",
    )

    out["format"], out["step"] = r_format, r_step
    out["score"] = r_accuracy + r_format + w_step * r_step + w_length * r_length
    return out
