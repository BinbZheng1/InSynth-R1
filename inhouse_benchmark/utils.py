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
"""Shared helpers: LLM endpoint settings, prompt loading and the rule-based reaction check."""

import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROMPT_DIR = Path(__file__).resolve().parent / "prompt"
# Qwen3-plus (API model id "qwen-plus") behind every agent, served on an OpenAI-compatible endpoint
MODEL = os.getenv("LLM_MODEL", "qwen-plus")
BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")

_ELEMENT = re.compile(r"[A-Z][a-z]?")
# Element symbols (plus D for deuterium); placeholders such as Ln or RE are rejected
ELEMENTS = frozenset(
    "H D He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr "
    "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt "
    "Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc "
    "Lv Ts Og".split()
)
# Non-metals and the metalloids B, Si, As and Te
NON_METALS = frozenset("H D He B C N O F Ne Si P S Cl Ar As Se Br Kr Te I Xe At Rn".split())


@lru_cache(maxsize=1)
def get_client():
    from openai import OpenAI

    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def chat(system_prompt: str, user_content: str) -> str:
    completion = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return completion.choices[0].message.content


def load_prompt(name: str) -> str:
    """Read a prompt file relative to ``prompt/``."""
    try:
        return (PROMPT_DIR / name).read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Prompt file '{name}' not found.")
        return "Error: Prompt file not found."


def check_reaction_format(reaction: str) -> bool:
    """Deterministic screening of ``product <- reactant + reactant (+ ...)`` records.

    A single product without a variable ``x`` in its formula, at least two reactants, only element symbols, every
    reactant sharing at least one element with the product, every element of the product supplied by the reactants
    and every metal of the reactants present in the product.
    """
    if "<-" not in reaction:
        return False
    product, reactants = reaction.split("<-", 1)
    product = product.strip()
    reactants = [r.strip() for r in reactants.split("+")]

    if not re.match(r"^[A-Z][A-Za-wy-z0-9.]*$", product):
        return False
    if len(reactants) < 2:
        return False
    for r in reactants:
        if not re.match(r"^[A-Z][A-Za-z0-9]*$", r):
            return False

    product_elements = set(_ELEMENT.findall(product))
    reactant_elements = set()
    for r in reactants:
        elements = set(_ELEMENT.findall(r))
        if not elements & product_elements:
            return False
        reactant_elements |= elements
    if not (product_elements | reactant_elements) <= ELEMENTS:
        return False
    return product_elements <= reactant_elements and reactant_elements - NON_METALS <= product_elements


def extract_json_from_markdown(text: str) -> str:
    match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text
