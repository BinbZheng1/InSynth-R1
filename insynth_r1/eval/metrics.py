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
"""Evaluation metrics (paper Sec. 4.2.2 and Appendix B).

Precursor selection: N independent responses per test instance (N = 5).

* Accuracy@1, Precision, Recall, F1 and Validity are instance-level averages. Each of the N runs gives one value per
  metric; the reported number is mean ± std over the N runs.
* Accuracy@3 is estimated by bootstrap: three of the N correctness indicators are resampled with replacement and
  their maximum is taken (E[Acc@3] ≈ 1 - (1 - p̂)^3); the dataset mean is computed for every bootstrap round and
  reported as mean ± std over rounds.

Synthesizability prediction (PU learning, Stage-1 prompt in reasoning mode): recall on known positives,
precision corrected with the class prior α = 0.088 (Eq. B.2), and the decision threshold recalibrated by prior
matching, P(f(x) > τ' | x ∈ U) ≈ α. The score f(x) is the label probability p(P) / (p(P) + p(U)) of the answer
after the thinking block, averaged over the N responses.
"""

from dataclasses import dataclass

import numpy as np

from insynth_r1.reward.parsing import extract_label, extract_precursors
from insynth_r1.reward.rubric import precursor_set_metrics, precursor_validity

PU_CLASS_PRIOR = 0.088


@dataclass
class MeanStd:
    mean: float
    std: float

    def __str__(self) -> str:
        return f"{100 * self.mean:.2f}±{100 * self.std:.2f}"


def _mean_std(values) -> MeanStd:
    values = np.asarray(values, dtype=np.float64)
    return MeanStd(float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0)


def score_precursor_responses(
    responses: list[str], ground_truth, inventory_path: str | None = None, allow_missing_think_open: bool = False
) -> list[dict[str, float]]:
    """Per-response exact match / precision / recall / F1 / validity for one test instance."""
    out = []
    for response in responses:
        predicted = extract_precursors(response, allow_missing_think_open)
        metrics = precursor_set_metrics(predicted, ground_truth)
        metrics["validity"] = precursor_validity(predicted, inventory_path)
        out.append(metrics)
    return out


def precursor_selection_metrics(
    per_instance: list[list[dict[str, float]]], n_bootstrap: int = 1000, k: int = 3, seed: int = 0
) -> dict[str, MeanStd]:
    """Aggregate per-response scores of shape [num_instances][N] into the reported metrics."""
    assert per_instance, "no instances"
    n = min(len(responses) for responses in per_instance)
    assert n > 0, "every instance needs at least one response"

    def matrix(key: str) -> np.ndarray:
        return np.array([[r[key] for r in responses[:n]] for responses in per_instance], dtype=np.float64)

    exact = matrix("exact_match")
    results = {
        "accuracy@1": _mean_std(exact.mean(axis=0)),
        "precision": _mean_std(matrix("precision").mean(axis=0)),
        "recall": _mean_std(matrix("recall").mean(axis=0)),
        "f1": _mean_std(matrix("f1").mean(axis=0)),
        "validity": _mean_std(matrix("validity").mean(axis=0)),
    }
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n, size=(n_bootstrap, exact.shape[0], k))
    best_of_k = exact[np.arange(exact.shape[0])[None, :, None], picks].max(axis=2)
    results[f"accuracy@{k}"] = _mean_std(best_of_k.mean(axis=1))
    results[f"accuracy@{k}_analytic"] = MeanStd(float((1 - (1 - exact.mean(axis=1)) ** k).mean()), 0.0)
    return results


def synthesizability_score(record: dict, allow_missing_think_open: bool = False) -> float:
    """Synthesizability score f(x) of one generation record: the mean per-response score over the N responses.

    A response scores its label probability (``p_positive``, written by :mod:`insynth_r1.eval.generate` for local
    models) when available, otherwise 1 if its parsed label is ``P`` and 0 if not, e.g. for API models without
    log-probabilities, which gives the fraction of responses predicting ``P``.
    """
    responses = record.get("responses") or []
    probabilities = record.get("p_positive")
    if probabilities is None or isinstance(probabilities, int | float):
        probabilities = [probabilities] * len(responses)
    scores = [
        float(p) if p is not None else float(extract_label(r, allow_missing_think_open) == "P")
        for r, p in zip(responses, probabilities, strict=True)
    ]
    return float(np.mean(scores)) if scores else 0.0


def prior_matching_threshold(unlabeled_scores, class_prior: float = PU_CLASS_PRIOR) -> float:
    """τ' such that the fraction of unlabeled samples with f(x) > τ' is as close as possible to the prior."""
    scores = np.sort(np.asarray(unlabeled_scores, dtype=np.float64))
    candidates = np.unique(np.concatenate([scores, [scores[0] - 1e-9]]))
    positive_rate = (len(scores) - np.searchsorted(scores, candidates, side="right")) / len(scores)
    return float(candidates[np.argmin(np.abs(positive_rate - class_prior))])


def pu_metrics(
    positive_scores, unlabeled_scores, class_prior: float = PU_CLASS_PRIOR, threshold: float | None = None
) -> dict[str, float]:
    """PU recall / corrected precision / F1 (Eq. B.1-B.3) at the prior-matched (or given) threshold."""
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    unlabeled_scores = np.asarray(unlabeled_scores, dtype=np.float64)
    tau = prior_matching_threshold(unlabeled_scores, class_prior) if threshold is None else threshold
    recall = float((positive_scores > tau).mean())
    predicted_positive_rate = float((unlabeled_scores > tau).mean())
    precision = min(1.0, recall * class_prior / predicted_positive_rate) if predicted_positive_rate > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "threshold": tau,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "unlabeled_positive_rate": predicted_positive_rate,
    }
