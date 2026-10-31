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
import math

import pytest

from insynth_r1.chem import elements_of, formula_key, is_valid_formula, parse_formula
from insynth_r1.reward.key_step import StepResult, extract_formulas, parse_step_verdict, rule_based_steps
from insynth_r1.reward.parsing import extract_label, extract_precursors, parse_reaction_line, parse_response
from insynth_r1.reward.rubric import compute_score, length_reward, precursor_set_metrics

GOOD = (
    "<think>\nThe target LiSm(PO3)4 contains the elements Li, Sm, P and O; lithium is +1 and samarium is +3.\n"
    "Candidate precursors: Li2CO3 or LiOH supply lithium, Sm2O3 supplies samarium, NH4H2PO4 or (NH4)2HPO4 "
    "provide phosphorus.\nFinal recommendation: Li2CO3, Sm2O3 and NH4H2PO4 are therefore selected because they "
    "are stable and common.\n</think>\n\n<answer>LiSm(PO3)4 <- Li2CO3 + Sm2O3 + NH4H2PO4</answer>"
)
GT = ["Li2CO3", "Sm2O3", "NH4H2PO4"]
INFO = {"target": "LiSm(PO3)4"}


class TestFormula:
    def test_parse_basic_and_brackets(self):
        assert dict(parse_formula("LiSm(PO3)4")) == {"Li": 1, "Sm": 1, "P": 4, "O": 12}
        assert dict(parse_formula("(NH4)2HPO4")) == {"N": 2, "H": 9, "P": 1, "O": 4}

    def test_decimal_stoichiometry_is_not_a_hydrate(self):
        assert dict(parse_formula("Tb1.8Ba2.1Ti1.9O11")) == {"Tb": 1.8, "Ba": 2.1, "Ti": 1.9, "O": 11}

    def test_hydrates(self):
        expected = {"Mg": 1, "S": 1, "O": 11, "H": 14}
        assert dict(parse_formula("MgSO4·7H2O")) == expected
        assert dict(parse_formula("MgSO4.7H2O")) == expected
        assert formula_key("MgSO4·7H2O") != formula_key("MgSO4")

    def test_order_independent_key(self):
        assert formula_key("Li2CO3") == formula_key("CLi2O3")
        assert formula_key("Li₂CO₃") == formula_key("Li2CO3")
        assert formula_key("Sm2O3 ") == formula_key("Sm2O3")

    def test_invalid(self):
        assert not is_valid_formula("Xx2O3")
        assert not is_valid_formula("Li2(CO3")
        assert not is_valid_formula("lithium carbonate")
        assert formula_key("Xx2O3").startswith("raw:")
        assert elements_of("Ba3CaZrWO9") == {"Ba", "Ca", "Zr", "W", "O"}


class TestParsing:
    def test_good_format(self):
        parsed = parse_response(GOOD)
        assert parsed.format_ok, parsed.format_errors
        assert extract_precursors(GOOD) == GT

    @pytest.mark.parametrize(
        "text",
        [
            "<answer>X <- A + B</answer>",
            "<think>t</think> extra <answer>X <- A + B</answer>",
            "<think>t</think><answer>X <- A + B</answer> trailing",
            "<think>t</think><answer>X <- A + B\nY <- C + D</answer>",
            "<think>t</think><answer>no reaction</answer>",
            "<think></think><answer>X <- A + B</answer>",
            "<think>a</think><think>b</think><answer>X <- A + B</answer>",
        ],
    )
    def test_bad_format(self, text):
        assert not parse_response(text).format_ok

    def test_missing_think_open(self):
        text = "reasoning</think>\n<answer>X <- A + B</answer>"
        assert not parse_response(text).format_ok
        assert parse_response(text, allow_missing_think_open=True).format_ok

    def test_fallback_without_answer_tags(self):
        text = "<think>reasoning</think>\nFinal: LiSm(PO3)4 <- Li2CO3 + Sm2O3 + NH4H2PO4"
        assert extract_precursors(text) == GT

    def test_reaction_line_variants(self):
        assert parse_reaction_line("`T <= A + B`") == ("T", ["A", "B"])
        assert parse_reaction_line("A + B -> T") == ("T", ["A", "B"])
        assert parse_reaction_line("[T] <- [A] + [B]") == ("T", ["A", "B"])
        assert parse_reaction_line("A and B") == (None, [])

    def test_label_after_thinking(self):
        assert extract_label("P") == "P"
        assert extract_label("<think>\n\n</think>\n\nU") == "U"
        assert extract_label("<think>\nCe and Pd form stable intermetallics.\n</think>\n\nP") == "P"
        assert extract_label("<think>\nCe-In-Pd ...\n</think>\n\n<answer>U</answer>") == "U"
        assert extract_label("maybe") is None


class TestKeyStep:
    def test_complete_steps(self):
        steps = rule_based_steps(parse_response(GOOD).think, "LiSm(PO3)4", GT)
        assert (steps.analyze, steps.identify, steps.justify) == (True, True, True)
        assert steps.reward == 1.0

    def test_partial_and_none(self):
        think = "Candidate precursors Li2CO3 and Sm2O3 and NH4H2PO4 can be used."
        steps = rule_based_steps(think, "LiSm(PO3)4", GT)
        assert not steps.analyze and steps.identify and steps.reward == 0.0
        assert rule_based_steps("hmm.", "LiSm(PO3)4", GT).reward == -1.0
        assert rule_based_steps("", "LiSm(PO3)4", GT).reward == -1.0

    def test_justify_requires_answer_precursors_in_reasoning(self):
        think = parse_response(GOOD).think
        assert not rule_based_steps(think, "LiSm(PO3)4", ["Li2CO3", "Sm2O3", "H3PO4"]).justify

    def test_eq4_mapping(self):
        assert StepResult(True, True, True).reward == 1.0
        assert StepResult(True, False, True).reward == 0.0
        assert StepResult(False, False, False).reward == -1.0

    def test_parse_llm_verdict(self):
        text = 'Reasoning...\n{"analyze": true, "identify": "false", "justify": true}'
        assert parse_step_verdict(text) == StepResult(True, False, True)
        reasoning = '<think>Draft: {"analyze": false} ... the trace compares candidates.</think>\n\n'
        assert parse_step_verdict(reasoning + '{"analyze": true, "identify": true, "justify": false}') == StepResult(
            True, True, False
        )
        for bad in ["no json", '{"analyze": true}', '{"analyze": "maybe", "identify": true, "justify": true}']:
            with pytest.raises((ValueError, KeyError)):
                parse_step_verdict(bad)

    def test_llm_judge_falls_back_to_rule(self, monkeypatch):
        monkeypatch.setattr("insynth_r1.reward.rubric.llm_steps", lambda *a, **k: None)
        out = compute_score("insynth/precursor", GOOD, GT, INFO, step_judge="llm", length_start_step=-1)
        assert out["step"] == 1.0 and out["step_judge_index"] == -1.0

    def test_llm_judge_tries_models_in_order(self, monkeypatch):
        import httpx

        calls = []

        def fake_post(url, json, headers, timeout):
            calls.append(json["model"])
            request = httpx.Request("POST", url)
            if json["model"] != "judge-c":
                return httpx.Response(503, request=request)
            content = '{"analyze": true, "identify": false, "justify": true}'
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]}, request=request)

        monkeypatch.setattr(httpx, "post", fake_post)
        out = compute_score(
            "insynth/precursor", GOOD, GT, INFO, step_judge="llm", judge_base_url="http://judge/v1",
            judge_model="judge-a", judge_fallback_models="judge-b, judge-c", length_start_step=-1,
        )
        assert calls == ["judge-a", "judge-a", "judge-b", "judge-b", "judge-c"]
        assert out["step"] == 0.0 and out["step_judge_index"] == 2.0

    def test_extract_formulas_skips_words(self):
        assert extract_formulas("In the end I use Li2CO3 and Sm2O3.") == ["Li2CO3", "Sm2O3"]


class TestRubric:
    def test_precursor_metrics(self):
        m = precursor_set_metrics(["Li2CO3", "Sm2O3", "H3PO4"], GT)
        assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
        assert m["f1"] == pytest.approx(2 / 3) and m["exact_match"] == 0.0
        assert precursor_set_metrics(["Sm2O3", "NH4H2PO4", "CLi2O3"], GT)["exact_match"] == 1.0
        assert precursor_set_metrics(None, GT)["f1"] == 0.0

    def test_multiple_references_any_one_is_correct(self):
        refs = "SbVO4 <- Sb2O3 + V2O3 + O2\nSbVO4 <- V2O5 + Sb2O3"
        assert precursor_set_metrics(["Sb2O3", "V2O5"], refs)["exact_match"] == 1.0
        nested = [["Sb2O3", "V2O3", "O2"], ["V2O5", "Sb2O3"]]
        assert precursor_set_metrics(["Sb2O3", "V2O3", "O2"], nested)["exact_match"] == 1.0
        partial = precursor_set_metrics(["Sb2O3", "V2O3"], refs)
        assert partial["exact_match"] == 0.0 and partial["f1"] == pytest.approx(0.8)
        assert precursor_set_metrics(["Sb2O3", "V2O5"], ["SbVO4 <- Sb2O3 + V2O3", "SbVO4 <- V2O5 + Sb2O3"])["f1"] == 1.0
        # A flat list is still one set
        assert precursor_set_metrics(["Li2CO3"], GT)["exact_match"] == 0.0

    def test_length_reward_eq5(self):
        assert length_reward(500, 1024, 4096) == 0.0
        assert length_reward(1024 + 1536, 1024, 4096) == pytest.approx(-0.5)
        assert length_reward(10000, 1024, 4096) == -1.0

    def test_total_reward_eq6(self):
        out = compute_score("insynth/precursor", GOOD, " + ".join(GT), INFO, length_start_step=-1)
        assert out["acc"] == 1.0 and out["format"] == 1.0 and out["step"] == 1.0 and out["length"] == 0.0
        assert out["score"] == pytest.approx(3.0)

    def test_length_activation_by_step(self):
        info = {**INFO, "response_length": 1536}
        early = compute_score("insynth/precursor", GOOD, GT, {**info, "global_steps": 10}, length_start_step=100)
        late = compute_score("insynth/precursor", GOOD, GT, {**info, "global_steps": 150}, length_start_step=100)
        assert early["length"] == 0.0 and early["length_active"] == 0.0
        assert late["length"] == pytest.approx(-0.5) and late["score"] == pytest.approx(2.5)

    def test_reward_weights(self):
        out = compute_score("insynth/precursor", GOOD, GT, INFO, w_step=0.5, length_start_step=-1)
        assert out["score"] == pytest.approx(2.5)

    def test_partial_answer_and_bad_format(self):
        text = "<think>Li2CO3 and Sm2O3</think><answer>LiSm(PO3)4 <- Li2CO3 + Sm2O3 + H3PO4</answer> extra"
        out = compute_score("insynth/precursor", text, GT, INFO, length_start_step=-1)
        assert out["acc"] == pytest.approx(2 / 3) and out["format"] == 0.0 and out["exact_match"] == 0.0

    def test_same_keys_for_every_response(self):
        good = compute_score("insynth/precursor", GOOD, GT, INFO)
        assert set(good) == set(compute_score("insynth/precursor", "", GT, INFO))

    def test_synthesizability_is_not_an_rl_task(self):
        with pytest.raises(ValueError):
            compute_score("insynth/synthesizability", "P", "P", {"target": "Ce4In21Pd10"})

    def test_score_is_finite_for_garbage(self):
        out = compute_score("insynth/precursor", "", GT, INFO)
        assert math.isfinite(out["score"]) and out["acc"] == 0.0 and out["step"] == -1.0
