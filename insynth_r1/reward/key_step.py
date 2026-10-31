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
"""Key-step reward R_step (paper Eq. 4).

R_step evaluates whether the thinking process contains the three key steps of the chemist-inspired
"Analyze-Identify-Justify" workflow:

    (i)   target compound analysis,
    (ii)  candidate precursor identification,
    (iii) final reaction justification,

and maps them to ``1`` (all three steps present), ``0`` (one or two steps present) and ``-1`` (no step present).

Two judges are provided:

* ``llm`` (paper setting): QwQ-32B at temperature 0 judges each step with the key-step evaluation template
  ``insynth_r1.data.prompts.KEY_STEP_JUDGE`` on an OpenAI-compatible endpoint (e.g. a vLLM server); falls back to
  ``rule`` on request or parsing failures.
* ``rule``: deterministic, chemistry-grounded checks on the thinking text, for offline runs without a judge server.
"""

import json
import logging
import os
import re
from dataclasses import dataclass

from insynth_r1.chem import elements_of, formula_key, is_valid_formula
from insynth_r1.data.prompts import key_step_judge_prompt

logger = logging.getLogger(__name__)

_NAMES = (
    "H hydrogen He helium Li lithium Be beryllium B boron C carbon N nitrogen O oxygen F fluorine Ne neon "
    "Na sodium Mg magnesium Al aluminium Si silicon P phosphorus S sulfur Cl chlorine Ar argon K potassium "
    "Ca calcium Sc scandium Ti titanium V vanadium Cr chromium Mn manganese Fe iron Co cobalt Ni nickel "
    "Cu copper Zn zinc Ga gallium Ge germanium As arsenic Se selenium Br bromine Kr krypton Rb rubidium "
    "Sr strontium Y yttrium Zr zirconium Nb niobium Mo molybdenum Tc technetium Ru ruthenium Rh rhodium "
    "Pd palladium Ag silver Cd cadmium In indium Sn tin Sb antimony Te tellurium I iodine Xe xenon "
    "Cs caesium Ba barium La lanthanum Ce cerium Pr praseodymium Nd neodymium Pm promethium Sm samarium "
    "Eu europium Gd gadolinium Tb terbium Dy dysprosium Ho holmium Er erbium Tm thulium Yb ytterbium "
    "Lu lutetium Hf hafnium Ta tantalum W tungsten Re rhenium Os osmium Ir iridium Pt platinum Au gold "
    "Hg mercury Tl thallium Pb lead Bi bismuth Po polonium At astatine Rn radon Fr francium Ra radium "
    "Ac actinium Th thorium Pa protactinium U uranium Np neptunium Pu plutonium Am americium"
).split()
ELEMENT_NAMES = dict(zip(_NAMES[0::2], _NAMES[1::2], strict=True))
# Stems also match common spellings and anion names (aluminum, sulphate, phosphate, nitrate, oxide, ...).
ELEMENT_NAMES.update(
    {"Al": "alumin", "S": "sul(?:f|ph)", "Cs": "ca?esium", "P": "phosph", "N": "nitr", "O": "ox(?:ygen|ide)"}
)

_ANALYZE_CUES = re.compile(
    r"element|composition|oxidation|valence|constituent|consist|contain|stoichiometr|charge|cation|anion", re.I
)
_IDENTIFY_CUES = re.compile(r"candidate|precursor|source|reagent|option|alternative|supply|provide", re.I)
_JUSTIFY_CUES = re.compile(
    r"final|recommend|therefore|thus|hence|select|choose|chose|justif|feasib|practical|because|conclu|best", re.I
)
# Formula-like tokens: an uppercase start followed by element symbols, digits, brackets and hydrate dots.
_FORMULA_TOKEN = re.compile(r"(?<![A-Za-z0-9])[A-Z(\[][A-Za-z0-9().\[\]·]*(?![A-Za-z0-9])")


@dataclass
class StepResult:
    analyze: bool
    identify: bool
    justify: bool

    @property
    def num_steps(self) -> int:
        return int(self.analyze) + int(self.identify) + int(self.justify)

    @property
    def reward(self) -> float:
        """Paper Eq. 4: 1 if all three steps are present, 0 if one or two are, -1 if none is."""
        if self.num_steps == 3:
            return 1.0
        if self.num_steps > 0:
            return 0.0
        return -1.0


def _mentions_element(text: str, symbol: str) -> bool:
    """A standalone element symbol (optionally with a charge: Li+, Sm3+, Fe(III), O²⁻) or the element name.

    Symbols that only occur inside formulas (Li2CO3, LiSm(PO3)4) do not count as analysis of the element.
    """
    charge = r"(?:\d*[+\-−]|\([IVX]+\)|[⁰¹²³⁴⁵⁶⁷⁸⁹]*[⁺⁻])?"
    if re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}{charge}(?![A-Za-z0-9])", text):
        return True
    name = ELEMENT_NAMES.get(symbol)
    return bool(name and re.search(rf"\b{name}", text, re.I))


def extract_formulas(text: str) -> list[str]:
    """Valid multi-element (or subscripted) formulas mentioned in ``text``, in order of appearance."""
    found = []
    for token in _FORMULA_TOKEN.findall(text):
        token = token.strip(".").strip("()[]") if token.count("(") != token.count(")") else token.strip(".")
        if not token or not is_valid_formula(token):
            continue
        # Single bare symbols ("In", "As", "I") are ordinary words far more often than formulas.
        if len(elements_of(token)) < 2 and not re.search(r"\d", token):
            continue
        found.append(token)
    return found


def rule_based_steps(
    think: str,
    target: str,
    predicted_precursors: list[str] | None,
    analyze_min_coverage: float = 0.75,
    identify_min_candidates: int = 2,
) -> StepResult:
    """Chemistry-grounded detection of the three key steps.

    * Analyze: an analysis cue and at least ``analyze_min_coverage`` of the target elements explicitly named
      (as standalone symbols or element names, not merely inside the target formula).
    * Identify: a precursor cue and at least ``identify_min_candidates`` distinct candidate formulas (other than
      the target) that supply at least one target element.
    * Justify: a justification cue and every precursor of the final answer discussed in the thinking, which ties
      the final reaction to the reasoning.
    """
    if not think:
        return StepResult(False, False, False)

    target_elements = elements_of(target)
    target_key = formula_key(target)

    analyze = False
    if target_elements and _ANALYZE_CUES.search(think):
        covered = sum(_mentions_element(think, element) for element in target_elements)
        analyze = covered / len(target_elements) >= analyze_min_coverage

    formulas = extract_formulas(think)
    candidate_keys = {
        formula_key(f) for f in formulas if formula_key(f) != target_key and elements_of(f) & target_elements
    }
    identify = bool(_IDENTIFY_CUES.search(think)) and len(candidate_keys) >= identify_min_candidates

    justify = False
    if predicted_precursors and _JUSTIFY_CUES.search(think):
        mentioned = {formula_key(f) for f in formulas}
        justify = all(formula_key(p) in mentioned for p in predicted_precursors)

    return StepResult(analyze, identify, justify)


JUDGE_API_BASE_ENV, JUDGE_MODEL_ENV, JUDGE_API_KEY_ENV = "JUDGE_API_BASE", "JUDGE_MODEL", "JUDGE_API_KEY"
JUDGE_FALLBACK_MODELS_ENV = "JUDGE_FALLBACK_MODELS"
STEP_KEYS = ("analyze", "identify", "justify")


def split_models(models) -> list[str]:
    """Model list from a comma-separated string or a list; empty entries are dropped."""
    if not models:
        return []
    if isinstance(models, str):
        models = models.split(",")
    return [str(m).strip() for m in models if str(m).strip()]


def judge_models_from_env() -> list[str]:
    """``JUDGE_MODEL`` followed by the comma-separated ``JUDGE_FALLBACK_MODELS``."""
    return split_models(os.getenv(JUDGE_MODEL_ENV)) + split_models(os.getenv(JUDGE_FALLBACK_MODELS_ENV))


def parse_step_verdict(content: str | None) -> StepResult:
    """Parse the judge's JSON verdict; raises ``ValueError`` if it is missing or malformed.

    The reasoning of a reasoning judge (QwQ-32B without a vLLM reasoning parser) ends with ``</think>`` and is skipped.
    """
    content = (content or "").rsplit("</think>", 1)[-1]
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in judge output")
    verdict = json.loads(match.group(0))
    values = []
    for key in STEP_KEYS:
        value = verdict[key]
        if isinstance(value, str):
            if value.strip().lower() not in ("true", "false"):
                raise ValueError(f"invalid value for {key}: {value!r}")
            value = value.strip().lower() == "true"
        values.append(bool(value))
    return StepResult(*values)


def llm_steps(
    think: str,
    target: str,
    answer: str | None,
    base_url: str | None = None,
    models=None,
    api_key_env: str = JUDGE_API_KEY_ENV,
    timeout: float = 300.0,
    max_retries: int = 1,
) -> tuple[StepResult, int] | None:
    """Judge the key steps with an OpenAI-compatible endpoint (``prompts.KEY_STEP_JUDGE``).

    ``models`` is the primary judge followed by fallback judges (list or comma-separated string); each is tried with
    ``max_retries`` retries before moving to the next. ``base_url`` / ``models`` default to the ``JUDGE_API_BASE`` /
    ``JUDGE_MODEL`` + ``JUDGE_FALLBACK_MODELS`` environment variables. The API key is read from the environment
    variable named by ``api_key_env``. ``timeout`` (seconds) covers
    the judge's reasoning, which can be long for QwQ-32B.

    Returns:
        ``(steps, index of the model that answered)``, or ``None`` if every model failed.
    """
    import httpx

    base_url = base_url or os.getenv(JUDGE_API_BASE_ENV)
    models = split_models(models) or judge_models_from_env()
    if not base_url or not models:
        raise ValueError(
            f"step_judge=llm needs judge_base_url/judge_model or the {JUDGE_API_BASE_ENV}/{JUDGE_MODEL_ENV} env vars"
        )
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv(api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/chat/completions"
    messages = [{"role": "user", "content": key_step_judge_prompt(target, think, answer)}]
    for index, model in enumerate(models):
        payload = {"model": model, "temperature": 0.0, "stream": False, "messages": messages}
        for attempt in range(max_retries + 1):
            try:
                response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
                response.raise_for_status()
                return parse_step_verdict(response.json()["choices"][0]["message"].get("content")), index
            except Exception as e:  # noqa: BLE001 - any judge failure moves on to the next judge
                if attempt == max_retries:
                    logger.warning("key-step judge %s failed (%s)", model, type(e).__name__)
    logger.warning("all key-step judges failed; falling back to the rule-based judge")
    return None


_SMOKE_TEST_THINK = (
    "The target LiSm(PO3)4 contains Li, Sm, P and O, with Li+ and Sm3+ and metaphosphate anions. Candidate "
    "precursors: Li2CO3 or LiOH for lithium, Sm2O3 for samarium, NH4H2PO4 or (NH4)2HPO4 for phosphorus. "
    "Li2CO3, Sm2O3 and NH4H2PO4 are selected because they are stable, inexpensive and common in solid-state synthesis."
)


def main(argv=None):
    """Smoke-test each candidate key-step judge: ``python -m insynth_r1.reward.key_step`` (uses JUDGE_* env vars)."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--models", default=None, help="comma-separated; default JUDGE_MODEL + JUDGE_FALLBACK_MODELS")
    parser.add_argument("--api_key_env", default=JUDGE_API_KEY_ENV)
    args = parser.parse_args(argv)

    target, answer = "LiSm(PO3)4", "LiSm(PO3)4 <- Li2CO3 + Sm2O3 + NH4H2PO4"
    predicted = ["Li2CO3", "Sm2O3", "NH4H2PO4"]
    # Expected R_step: 1 for the complete trace, 0 for the trace that only analyzes the target.
    cases = {
        "complete": (_SMOKE_TEST_THINK, 1.0),
        "analyze only": ("The target LiSm(PO3)4 contains Li, Sm, P and O.", 0.0),
    }
    for name, (think, expected) in cases.items():
        rule = rule_based_steps(think, target, predicted)
        print(f"[rule] {name}: {rule} R_step={rule.reward} (expected {expected})")

    usable = []
    for model in split_models(args.models) or judge_models_from_env():
        rewards, start = [], time.time()
        for think, _ in cases.values():
            result = llm_steps(think, target, answer, args.base_url, [model], args.api_key_env, max_retries=0)
            rewards.append(result[0].reward if result else None)
        expected = [e for _, e in cases.values()]
        status = "ok" if rewards == expected else ("failed" if None in rewards else "unexpected verdict")
        print(f"[{model}] R_step={rewards} (expected {expected}) {status} {time.time() - start:.1f}s")
        if status == "ok":
            usable.append(model)
    if usable:
        print(f"export {JUDGE_MODEL_ENV}={usable[0]!r} {JUDGE_FALLBACK_MODELS_ENV}={','.join(usable[1:])!r}")


if __name__ == "__main__":
    main()
