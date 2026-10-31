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
"""Parsing of InSynth-R1 responses.

Message template (paper, Table 1)::

    <think> {thinking process} </think>
    <answer> {final reaction} </answer>

The final reaction is one line ``[Target] <- [Precursor] + [Precursor] + ...``. Following the evaluation
protocol (Appendix B.2), the prediction is taken from the ``<answer>`` block when available; otherwise the last
reaction line after ``</think>`` is used. Unparseable outputs yield ``None`` and count as wrong predictions.

Synthesizability prediction answers with a single ``P``/``U`` label after the thinking block (see
:func:`extract_label`).
"""

import re
from dataclasses import dataclass, field

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

_ARROW = re.compile(r"\s*(?:<--|<-|<=|←|⟵|=>|->|→)\s*")
_CODE_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_SYNTH_LABEL = re.compile(r"^\W*(P|U)\W*$")


@dataclass
class ParsedResponse:
    think: str | None = None
    answer: str | None = None
    format_ok: bool = False
    format_errors: list[str] = field(default_factory=list)


def _strip_answer(text: str) -> str:
    text = text.strip()
    text = _CODE_FENCE.sub("", text).strip()
    return text.strip("`").strip()


def parse_response(text: str, allow_missing_think_open: bool = False) -> ParsedResponse:
    """Split a response into thinking and answer, and check the required format.

    The format is satisfied iff the response is exactly ``<think>...</think><answer>...</answer>`` (whitespace
    allowed between blocks), each tag appears once, the thinking is non-empty and the answer is a single
    well-formed reaction line.

    Args:
        allow_missing_think_open: accept responses that start inside the thinking block, for chat templates that
            already put ``<think>`` into the generation prompt.
    """
    errors = []
    if allow_missing_think_open and THINK_OPEN not in text and THINK_CLOSE in text:
        text = THINK_OPEN + text

    for tag in (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE):
        if text.count(tag) != 1:
            errors.append(f"tag {tag} appears {text.count(tag)} times")

    think = None
    if THINK_CLOSE in text:
        head, _, _ = text.partition(THINK_CLOSE)
        think = head.split(THINK_OPEN, 1)[-1].strip()

    blocks = _BLOCK.findall(text)
    answer = _strip_answer(blocks[-1]) if blocks else None

    if not errors:
        i_to, i_tc = text.index(THINK_OPEN), text.index(THINK_CLOSE)
        i_ao, i_ac = text.index(ANSWER_OPEN), text.index(ANSWER_CLOSE)
        if not (i_to < i_tc < i_ao < i_ac):
            errors.append("tags out of order")
        else:
            if text[:i_to].strip():
                errors.append("text before <think>")
            if text[i_tc + len(THINK_CLOSE) : i_ao].strip():
                errors.append("text between </think> and <answer>")
            if text[i_ac + len(ANSWER_CLOSE) :].strip():
                errors.append("text after </answer>")
    if not think:
        errors.append("empty thinking")
    if not answer:
        errors.append("empty answer")
    elif len([line for line in answer.splitlines() if line.strip()]) != 1:
        errors.append("answer must be exactly one line")
    elif not parse_reaction_line(answer)[1]:
        errors.append("answer is not a reaction line")

    return ParsedResponse(think=think, answer=answer, format_ok=not errors, format_errors=errors)


def parse_reaction_line(line: str) -> tuple[str | None, list[str]]:
    """Parse ``Target <- P1 + P2`` into ``(target, [P1, P2])``. Returns ``(None, [])`` if no arrow is found."""
    line = _strip_answer(line)
    parts = _ARROW.split(line, maxsplit=1)
    if len(parts) != 2:
        return None, []
    left, right = parts
    # "A + B -> T" is accepted as well; the side with "+" is treated as the precursor side.
    if "+" in left and "+" not in right:
        left, right = right, left
    precursors = [p.strip().strip("[]").strip() for p in right.split("+")]
    return left.strip().strip("[]").strip() or None, [p for p in precursors if p]


def parse_label(text: str) -> str | None:
    match = _SYNTH_LABEL.match(_strip_answer(text))
    return match.group(1) if match else None


def extract_answer_text(text: str, allow_missing_think_open: bool = False) -> str | None:
    """Return the final-answer text: the last ``<answer>`` block, else the text after ``</think>``."""
    parsed = parse_response(text, allow_missing_think_open=allow_missing_think_open)
    if parsed.answer:
        return parsed.answer
    tail = text.rsplit(THINK_CLOSE, 1)[-1] if THINK_CLOSE in text else text
    return tail.strip() or None


def extract_precursors(text: str, allow_missing_think_open: bool = False) -> list[str] | None:
    """Predicted precursor list, or ``None`` if no reaction line can be parsed."""
    answer = extract_answer_text(text, allow_missing_think_open)
    if not answer:
        return None
    lines = [line for line in answer.splitlines() if line.strip()]
    for line in lines if ANSWER_OPEN in text else reversed(lines):
        _, precursors = parse_reaction_line(line)
        if precursors:
            return precursors
    return None


def extract_label(text: str, allow_missing_think_open: bool = False) -> str | None:
    """Synthesizability label (``P``/``U``): the last answer line after the thinking block, ``None`` if unparseable."""
    answer = extract_answer_text(text, allow_missing_think_open)
    if not answer:
        return None
    lines = [line for line in answer.splitlines() if line.strip()]
    return parse_label(lines[-1]) if lines else None
