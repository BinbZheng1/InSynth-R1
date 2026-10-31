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
import json

import numpy as np
import pandas as pd
import pytest

from insynth_r1.coldstart import rejection_sampling as rs
from insynth_r1.data import preprocess, prompts
from insynth_r1.data.split import deduplicate, random_split
from insynth_r1.eval.metrics import (
    precursor_selection_metrics,
    prior_matching_threshold,
    pu_metrics,
    synthesizability_score,
)
from insynth_r1.reward.parsing import parse_response

TRAJ = (
    "<think>\nLiSm(PO3)4 contains Li, Sm, P and O. Candidates: Li2CO3, Sm2O3, NH4H2PO4. "
    "Final choice is justified by practicality.\n</think>\n\n"
    "<answer>\n```\nLiSm(PO3)4 <- Li2CO3 + Sm2O3 + NH4H2PO4\n```\n</answer>"
)
GT = ["Li2CO3", "Sm2O3", "NH4H2PO4"]


def _reactions(n=100):
    return [{"target": f"Li{i + 1}SmO{i + 3}", "precursors": ["Li2CO3", "Sm2O3"]} for i in range(n)]


class TestPrompts:
    def test_templates_format(self):
        assert prompts.precursor_reasoning_prompt("LiSm(PO3)4").endswith("Target compound: LiSm(PO3)4 /think")
        assert "Target compound: LiSm(PO3)4" in prompts.rejection_sampling_prompt("LiSm(PO3)4")
        text = prompts.complexity_utility_prompt("LiSm(PO3)4", "TRAJ")
        assert "TRAJ" in text and '"complexity": <1-5>' in text
        assert prompts.precursor_stage1_prompt("LiSm(PO3)4").endswith("Provide 1 synthesis plan for target: LiSm(PO3)4")
        assert "Ce4In21Pd10" in prompts.synthesizability_stage1_prompt("Ce4In21Pd10")
        assert '"analyze": true|false' in prompts.key_step_judge_prompt("T", "think", None)

    def test_no_unfilled_placeholders(self):
        # Templates whose rendered text contains a literal JSON output schema are covered by the test above.
        rendered = [
            prompts.precursor_reasoning_prompt("T"),
            prompts.rejection_sampling_prompt("T"),
            prompts.precursor_stage1_prompt("T"),
            prompts.synthesizability_stage1_prompt("T"),
        ]
        for text in rendered:
            assert "{" not in text and "}" not in text


class TestData:
    def test_dedup_keeps_distinct_precursor_sets(self):
        records = [
            {"target": "LiCoO2", "precursors": ["Li2CO3", "Co3O4"]},
            {"target": "LiCoO2", "precursors": ["Co3O4", "CLi2O3"]},
            {"target": "LiCoO2", "precursors": ["LiOH", "Co3O4"]},
        ]
        unique, removed = deduplicate(records)
        assert removed == 1 and len(unique) == 2

    def test_random_split_is_seeded(self):
        records = _reactions()
        assert random_split(records, 0.2, 1) == random_split(records, 0.2, 1)
        train, test = random_split(records, 0.2, 1)
        assert len(test) == 20 and len(train) == 80

    def test_preprocess_end_to_end(self, tmp_path):
        raw = tmp_path / "reactions.jsonl"
        raw.write_text("\n".join(json.dumps(r) for r in _reactions()))
        synth = tmp_path / "synth.csv"
        compositions = [f"Ce{i + 1}In21Pd10" for i in range(20)]
        pd.DataFrame({"composition": compositions, "label": [1, 0] * 10}).to_csv(synth)
        out = tmp_path / "out"
        preprocess.main(["--reactions", str(raw), "--synthesizability", str(synth), "--out_dir", str(out)])

        rl = pd.read_parquet(out / "rl/precursor_train.parquet")
        row = rl.iloc[0]
        assert set(rl["data_source"]) == {"insynth/precursor"}
        assert row["prompt"][0]["content"] == prompts.precursor_reasoning_prompt(row["extra_info"]["target"])
        assert not (out / "rl/synthesizability_train.parquet").exists()
        sft = pd.read_parquet(out / "sft/stage1_train.parquet")
        assert set(sft["task"]) == {"precursor", "synthesizability"}
        coldstart = [json.loads(line) for line in (out / "coldstart/input.jsonl").read_text().splitlines()]
        assert len(coldstart) == len(rl) and all("precursors" in r for r in coldstart)
        synth_eval = pd.read_parquet(out / "eval/synthesizability_test.parquet")
        content = synth_eval.iloc[0]["prompt"][0]["content"]
        assert content.startswith("Role: Expert inorganic chemist.") and content.endswith(prompts.THINK_SUFFIX)
        assert set(synth_eval["reward_model"].map(lambda r: r["ground_truth"])) <= {"P", "U"}

    def test_preprocess_held_out_reactions(self, tmp_path):
        raw = tmp_path / "reactions.jsonl"
        raw.write_text("\n".join(json.dumps(r) for r in _reactions()))
        synth = tmp_path / "synth.csv"
        pd.DataFrame({"composition": ["Ce4In21Pd10", "CeIn3"], "label": [1, 0]}).to_csv(synth)
        held_out = tmp_path / "held_out.json"
        records = [
            {"reaction": "Li1SmO3 <- Sm2O3 + Li2CO3", "doi": "a"},  # also in the public data
            {"reaction": "NaYGeO4 <- Na2CO3 + Y2O3 + GeO2", "doi": "b"},
            {"reaction": "NaYGeO4 <- Na2CO3 + GeO2 + Y2O3", "doi": "c"},  # duplicate
            {"reaction": "Bi0.5Sb1.5Te3 <- Bi + Sb + Te", "doi": "d"},
        ]
        held_out.write_text(json.dumps(records))
        out = tmp_path / "out"
        preprocess.main(
            ["--reactions", str(raw), "--synthesizability", str(synth), "--eval_reactions", str(held_out)]
            + ["--out_dir", str(out)]
        )

        stats = json.loads((out / "split_stats.json").read_text())["inhouse"]
        assert stats["duplicates_removed"] == 1 and stats["overlap_with_public_removed"] == 1
        rows = pd.read_parquet(out / "eval/inhouse.parquet")
        assert [r["target"] for r in rows["extra_info"]] == ["NaYGeO4", "Bi0.5Sb1.5Te3"]
        assert rows.iloc[0]["reward_model"]["ground_truth"] == "Na2CO3 + Y2O3 + GeO2"


class TestColdStart:
    def test_unified_score(self):
        assert rs.unified_score(5, 5) == 1.0 and rs.unified_score(1, 1) == 0.0
        assert rs.unified_score(5, 1, alpha=0.5) == 0.5
        assert rs.unified_score(3, 5, alpha=0.25) == pytest.approx(0.25 * 0.5 + 0.75 * 1.0)

    def test_parse_evaluation(self):
        text = 'ok {"complexity_rationale": "a", "complexity": 4, "utility_rationale": "b", "utility": 5}'
        assert rs.parse_evaluation(text) == (4, 5)
        assert rs.parse_evaluation('{"complexity": 7, "utility": 2}') is None
        assert rs.parse_evaluation("no json") is None

    def test_algorithm1_selection(self):
        wrong = TRAJ.replace("NH4H2PO4\n", "H3PO4\n")
        base = {"target": "LiSm(PO3)4", "precursors": GT}
        pool = [
            {**base, "index": 0, "trajectory": TRAJ, "score": 0.9},
            {**base, "index": 1, "trajectory": wrong, "score": 0.8},
            {**base, "index": 2, "trajectory": TRAJ, "score": 0.2},
            {**base, "index": 3, "trajectory": TRAJ, "score": None},
        ]
        top, verified = rs.complexity_utility_selection(pool, top_k=0.67)
        assert [r["index"] for r in top] == [0, 1] and [r["index"] for r in verified] == [0]
        assert [r["index"] for r in rs.human_review(pool, {0: [True, True, False], 2: [True, False]})] == [0]

    def test_sft_record_matches_rl_format(self):
        record = rs.to_sft_record({"index": 0, "target": "LiSm(PO3)4", "trajectory": TRAJ})
        user, assistant = record["messages"]
        assert user["content"] == prompts.precursor_reasoning_prompt("LiSm(PO3)4")
        assert parse_response(assistant["content"]).format_ok

    def test_assemble_trajectory_from_reasoning_content(self):
        traj = rs.assemble_trajectory("<answer>T <- A + B</answer>", "native thinking")
        assert parse_response(traj).format_ok and "native thinking" in traj
        assert rs.assemble_trajectory(TRAJ, "other", source="content") == TRAJ.strip()


class TestMetrics:
    def test_precursor_metrics_and_bootstrap(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, size=200)
        constant = {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "validity": 1.0}
        per_instance = [[{"exact_match": float(rng.uniform() < pi), **constant} for _ in range(5)] for pi in p]
        metrics = precursor_selection_metrics(per_instance, n_bootstrap=2000)
        assert metrics["f1"].mean == pytest.approx(2 / 3) and metrics["f1"].std == 0.0
        assert metrics["accuracy@3"].mean == pytest.approx(metrics["accuracy@3_analytic"].mean, abs=0.01)
        assert metrics["accuracy@3"].mean >= metrics["accuracy@1"].mean

    def test_synthesizability_score(self):
        assert synthesizability_score({"p_positive": 0.7, "responses": ["U"]}) == 0.7
        assert synthesizability_score({"p_positive": None, "responses": ["P", "U", "P", "P"]}) == 0.75
        # Per-response label probabilities; a response without one counts by its parsed label
        record = {"p_positive": [0.9, None, 0.3], "responses": ["P", "<think>\nx\n</think>\n\nP", "U"]}
        assert synthesizability_score(record) == pytest.approx((0.9 + 1.0 + 0.3) / 3)

    def test_label_probability_after_thinking(self):
        from types import SimpleNamespace

        from insynth_r1.eval.generate import label_probability

        def position(sampled, candidates):
            entries = {i: SimpleNamespace(decoded_token=t, logprob=np.log(p)) for i, (t, p) in enumerate(candidates)}
            return next(i for i, (t, _) in enumerate(candidates) if t == sampled), entries

        # "<think> P is ... </think> \n\n P": the "P" inside the thinking block is ignored
        steps = [
            position("<think>", [("<think>", 1.0)]),
            position("P", [("P", 0.5), ("U", 0.5)]),
            position("</think>", [("</think>", 1.0)]),
            position("\n\n", [("\n\n", 0.9), ("P", 0.1)]),
            position("P", [("P", 0.6), ("U", 0.2), ("The", 0.2)]),
        ]
        token_ids, logprobs = zip(*steps, strict=True)
        assert label_probability(list(token_ids), list(logprobs)) == pytest.approx(0.75)
        # Truncated inside the thinking block
        assert label_probability(list(token_ids[:2]), list(logprobs[:2])) is None
        # Direct answer (thinking disabled in the chat template)
        assert label_probability([1], [position(" U", [("P", 0.2), (" U", 0.8)])[1]]) == pytest.approx(0.2)

    def test_pu_prior_matching(self):
        rng = np.random.default_rng(0)
        unlabeled = rng.uniform(0, 1, size=10000)
        positives = rng.uniform(0.5, 1, size=1000)
        tau = prior_matching_threshold(unlabeled, 0.088)
        assert (unlabeled > tau).mean() == pytest.approx(0.088, abs=1e-3)
        m = pu_metrics(positives, unlabeled, 0.088)
        assert m["recall"] == pytest.approx((positives > m["threshold"]).mean())
        assert m["precision"] == pytest.approx(min(1.0, m["recall"] * 0.088 / m["unlabeled_positive_rate"]))
