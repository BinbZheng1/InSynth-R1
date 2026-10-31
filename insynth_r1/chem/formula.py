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
"""Chemical formula parsing and normalization.

Precursor sets are compared order-independently after normalizing every formula to its reduced
elemental composition, e.g. ``Li2CO3`` == ``CLi2O3`` and ``(NH4)H2PO4`` == ``NH4H2PO4``. Hydrated and
anhydrous reagents stay distinct (``MgSO4·7H2O`` != ``MgSO4``) because the water of crystallization is
part of the composition. Formulas that cannot be parsed fall back to a cleaned string key, so they only
match an identical string and never a valid formula.
"""

import re
from collections import defaultdict
from functools import lru_cache

ELEMENTS = frozenset(
    """
    H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
    Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm
    Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No
    Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og D
    """.split()
)

_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
# Hydrate / adduct separators. A plain "." is ambiguous with decimal stoichiometry (Tb1.8Ba2.1...), so it
# is only treated as a separator when it is followed by a water/ammonia adduct such as ".7H2O".
_HYDRATE_SEPARATORS = re.compile(r"\s*[·•∙⋅*]\s*|\.(?=\s*(?:\d+(?:\.\d+)?|[xny])?\s*(?:H2O|D2O|NH3)\b)")
# Lower-case parenthetical annotations: phases "(s)", "(aq)", polymorph names "(anatase)", etc.
_ANNOTATION = re.compile(r"\((?:s|l|g|aq|[a-z][a-z \-]{2,})\)")
_LEADING_COEFF = re.compile(r"^\s*(\d+(?:\.\d+)?|[xny])\s*(?=[A-Z(\[])")
_TOKEN = re.compile(r"[A-Z][a-z]?|\d+(?:\.\d+)?|[(\[{]|[)\]}]")
_OPEN, _CLOSE = "([{", ")]}"


class FormulaParseError(ValueError):
    """Raised when a string is not a valid chemical formula."""


def clean_formula(text: str) -> str:
    """Strip markup, whitespace, phase annotations and trailing punctuation from a formula string."""
    text = text.translate(_SUBSCRIPTS)
    text = re.sub(r"</?su[bp]>|[`*_$\\{}]{2,}|\\mathrm|\\text", "", text)
    text = text.replace("\u2009", " ").replace("\u00a0", " ")
    text = _ANNOTATION.sub("", text)
    return text.strip().strip(".,;:")


def _parse_group(tokens: list[str], pos: int, depth: int) -> tuple[dict[str, float], int]:
    counts: dict[str, float] = defaultdict(float)
    while pos < len(tokens):
        tok = tokens[pos]
        if tok in _CLOSE:
            if depth == 0:
                raise FormulaParseError(f"unbalanced bracket at token {pos}")
            return counts, pos
        if tok in _OPEN:
            inner, pos = _parse_group(tokens, pos + 1, depth + 1)
            if pos >= len(tokens) or tokens[pos] not in _CLOSE:
                raise FormulaParseError("missing closing bracket")
            pos += 1
            sub_counts = inner
        elif tok[0].isupper():
            if tok not in ELEMENTS:
                raise FormulaParseError(f"unknown element symbol {tok!r}")
            sub_counts = {tok: 1.0}
            pos += 1
        else:
            raise FormulaParseError(f"unexpected number {tok!r} at token {pos}")
        multiplier = 1.0
        if pos < len(tokens) and tokens[pos][0].isdigit():
            multiplier = float(tokens[pos])
            pos += 1
        for element, count in sub_counts.items():
            counts[element] += count * multiplier
    if depth != 0:
        raise FormulaParseError("missing closing bracket")
    return counts, pos


def _parse_part(part: str) -> dict[str, float]:
    multiplier = 1.0
    match = _LEADING_COEFF.match(part)
    if match:
        coeff = match.group(1)
        multiplier = 1.0 if coeff in "xny" else float(coeff)
        part = part[match.end() :]
    compact = re.sub(r"\s+", "", part)
    tokens = _TOKEN.findall(compact)
    if not tokens or "".join(tokens) != compact:
        raise FormulaParseError(f"invalid characters in {part!r}")
    counts, _ = _parse_group(tokens, 0, 0)
    return {element: count * multiplier for element, count in counts.items()}


@lru_cache(maxsize=65536)
def parse_formula(formula: str) -> tuple[tuple[str, float], ...]:
    """Parse a formula (with optional hydrate parts) into a sorted ``(element, count)`` tuple.

    A leading stoichiometric coefficient (``2Li2CO3``) is ignored for the first part, because it denotes
    the reaction coefficient rather than the identity of the reagent.
    """
    text = clean_formula(formula)
    if not text:
        raise FormulaParseError("empty formula")
    parts = [p for p in _HYDRATE_SEPARATORS.split(text) if p.strip()]
    if not parts:
        raise FormulaParseError("empty formula")
    parts[0] = _LEADING_COEFF.sub("", parts[0], count=1)
    total: dict[str, float] = defaultdict(float)
    for part in parts:
        for element, count in _parse_part(part).items():
            total[element] += count
    if not total:
        raise FormulaParseError("no elements found")
    return tuple(sorted((element, round(count, 4)) for element, count in total.items()))


def formula_key(formula: str) -> str:
    """Return a canonical key for order-independent precursor comparison."""
    try:
        composition = parse_formula(formula)
    except FormulaParseError:
        return "raw:" + re.sub(r"\s+", "", clean_formula(formula)).lower()
    return "".join(f"{element}{count:g}" for element, count in composition)


def is_valid_formula(formula: str) -> bool:
    try:
        parse_formula(formula)
    except FormulaParseError:
        return False
    return True


def elements_of(formula: str) -> frozenset[str]:
    """Constituent elements of a formula (empty set if it cannot be parsed)."""
    try:
        return frozenset(element for element, _ in parse_formula(formula))
    except FormulaParseError:
        return frozenset()
