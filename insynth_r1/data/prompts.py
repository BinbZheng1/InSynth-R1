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
# ruff: noqa: E501
"""Prompt templates.

Synthesizability prediction is trained in Stage-1 (domain fine-tuning) only; Stage-2 and RL are precursor selection
only. Synthesizability prediction is evaluated in reasoning mode with its Stage-1 template (``/think`` appended).

Templates from the paper appendix ("Prompt Templates"):

* ``SYNTHESIZABILITY_STAGE1``, ``PRECURSOR_STAGE1``: Stage-1 domain fine-tuning (direct answers).
* ``PRECURSOR_REJECTIVE_FINE_TUNING``: Stage-2 rejective fine-tuning; also used for RL and final evaluation. The
  target is appended as ``Target compound: {target}``.
* ``PRECURSOR_REJECTION_SAMPLING``: Stage-2 trajectory generation (target appended in the same way).
* ``PRECURSOR_COMPLEXITY_UTILITY``: Stage-2 complexity-utility evaluation.
* ``KEY_STEP_JUDGE``: RCPO key-step evaluation, i.e. the LLM judge for the key-step reward R_step.

RL and evaluation prompts end with ``/think`` (paper Table 1).
"""

OUTPUT_FORMAT = "[Target] <- [Precursor] + [Precursor] + [Precursor]"
THINK_SUFFIX = " /think"

# =====================================================================================================================
# Paper appendix: Stage-1
# =====================================================================================================================

SYNTHESIZABILITY_STAGE1 = """Role: Expert inorganic chemist.

Task: Classify synthesizability of the composition {composition}.

Assumptions: No structure, no external lookup. Use chemical intuition for solid-state/intermetallic synthesis.

Label rules:
- P if composition is chemically reasonable and likely forms an intermetallic phase (even if requiring specific conditions).
- U if composition is chemically implausible or likely decomposes / phase-separates with no clear stabilizing rationale.

Output constraint: return exactly one token, either P or U."""

PRECURSOR_STAGE1 = """You are an expert solid-state chemist planning realistic laboratory syntheses for inorganic materials. You will be given a target compound and must output exactly one plausible precursor set that could be used to synthesize the target.

Task: Select precursor reagents (typically solid powders; aqueous acids/salts are allowed) that collectively provide all elements in the target composition.

Core rules: The chosen precursors must jointly contain every element present in the target.

Minimum count: Use two or more precursors.

Open-system allowance: Because synthesis is assumed to occur in an open system where gases can escape, it is acceptable for precursors to contain extra non-metal elements that are not present in the final product, specifically: H, C, N, O, F, Cl, Br (e.g., carbonates, nitrates, ammonium salts, hydrates, halides are allowed).

Practicality preference: Prefer commercially common, stable, and widely used inorganic reagents (e.g., alkali carbonates/oxides, rare-earth oxides, ammonium phosphates, simple phosphates/oxides). Avoid exotic organometallics unless clearly necessary.

No quantities or procedures: Do not provide stoichiometric coefficients, balances, temperatures, atmospheres, time, grinding/annealing steps, or any explanation.

Output constraints: Output must be one line only.

Output must match exactly this format: {output_format}

Use standard chemical formulas for precursors, including arrows and plus signs. Now perform the task:

Provide 1 synthesis plan for target: {target}"""

# =====================================================================================================================
# Paper appendix: Stage-2 (precursor selection)
# =====================================================================================================================

PRECURSOR_REJECTIVE_FINE_TUNING = """You are an expert solid-state chemist planning realistic laboratory syntheses for inorganic materials. You will be given a target compound and must output exactly one plausible precursor set that could be used to synthesize the target in a conventional open-system solid-state reaction.

Task: Select precursor reagents (typically solid powders; aqueous acids/salts are allowed) that collectively provide all elements in the target composition.

Core rules:
1. Element coverage: The selected precursors must jointly contain every element present in the target.
2. Minimum count: Use two or more precursors.
3. Open-system allowance: Because gases can escape, it is acceptable for precursors to contain extra non-metal elements not present in the final product, specifically H, C, N, O, F, Cl, Br (e.g., carbonates, nitrates, ammonium salts, hydrates, halides are allowed).
4. Practicality preference: Prefer commercially common, stable, widely used inorganic reagents (e.g., alkali carbonates/oxides, rare-earth oxides, ammonium phosphates). Avoid exotic organometallics unless clearly necessary.

No stoichiometry or procedures: Do not provide coefficients, balancing, temperatures, atmospheres, time, or any procedural description in the final answer.

Reasoning requirements: Your reasoning process should follow these steps and explain each step in detail:
1. Analyze the target compound: identify the required elements, assign plausible oxidation states and check charge balance, and note properties relevant to precursor selection (e.g., structure type, dopants, volatile or redox-sensitive elements).
2. Identify candidate precursors: propose more than one candidate precursor set that ensures element coverage, and compare them in terms of stability, decomposition behavior, reactivity, cost, and practicality.
3. Recommend the final precursor set: justify why it is preferred over the alternatives and confirm that every element of the target is supplied.
4. You must conduct all reasoning inside <think> ... </think>. Then provide only the final synthesis line inside <answer> ... </answer> with no additional explanation.

Output constraints: Output must be one recommendation only. The <answer> must contain exactly one line in the following format: {output_format}

Example:
<think>Thinking Process</think>
<answer>{output_format}</answer>

Target compound: {target}"""

PRECURSOR_REJECTION_SAMPLING = """You are an expert solid-state chemist planning a material synthesis. Your goal is to identify suitable synthetic precursor reagents for a given target compound and provide a detailed reasoning process (chain of thought) to justify your choice. Your task is to choose appropriate precursor reagents and clearly explain your reasoning process.

Reasoning Steps:
1. Analyze the target compound: Determine the elemental composition of the target compound. Record any key chemical or physical properties that may affect precursor selection (e.g., crystal structure, oxidation state, stability, reactivity).
2. Identify candidate precursors: Propose multiple sets of candidate precursor reagents (each set contains at least two precursors) that together provide all the elements required for the target compound. Make sure each precursor is a realistic and commonly used chemical in solid-state synthesis. Consider that non-metallic elements (H, C, N, O, F, Cl, Br) in the precursors may escape as gases during the reaction.
3. Recommend final reaction formula and justify your selection: Choose the most suitable combination of candidate precursors for synthesis. For each recommended reaction formula, provide the following format: {output_format}. If multiple recommendations are needed, make sure each is different and separated by line breaks.

Justify your selection based on the following factors:
1. Reaction feasibility: chemical compatibility, thermodynamic favorability, and the likelihood of forming the target compound.
2. Economic cost: Availability and cost of precursors.
3. Practical considerations: Safety, ease of operation, and common use in solid-state synthesis.

Output Format: Provide Chain of Thought (CoT) reasoning in a clear Markdown format with the following sections:
1. Elemental composition and properties: Analyze the elements of the target compound and their related properties.
2. Candidate precursors: List possible precursor combinations and briefly describe them.
3. Final recommendation and justification: The selected reaction formula and explain its feasibility, cost, and practicality in detail.
Provide the final reaction formula in the specified format and wrap it in a code block. Split the entire response into two parts, wrapped in <XML> tags:

<think>Thinking Process</think>
<answer>{output_format}</answer>

Target compound: {target}"""

PRECURSOR_COMPLEXITY_UTILITY = """You are an expert solid-state chemist acting as an impartial evaluator of reasoning trajectories for inorganic precursor selection. You will be given a target compound and a candidate chain-of-thought (CoT) trajectory produced by another model. Evaluate the trajectory on two independent dimensions, complexity and utility, using the definitions and rating scales below. The reference precursor set is not provided, and the correctness of the final answer is verified separately; judge the chemical reasoning itself rather than guessing whether the answer matches the literature.

Dimension 1: Complexity. The depth and breadth of chemically meaningful reasoning, i.e., how many non-trivial chemical considerations are explicitly analyzed and integrated to reach the precursor choice. Relevant considerations include elemental composition and oxidation states, structural or stability properties of the target, alternative candidate precursor sets, comparison and elimination of candidates, decomposition and volatility behavior, and reactivity or redox compatibility. Complexity reflects reasoning substance, not length: repetition, circular self-verification, and irrelevant digressions do not increase complexity.
- 1 (Trivial): states the answer directly or only lists the elements, with no chemical consideration.
- 2 (Shallow): a single linear path that maps each element to a common precursor without justification or alternatives.
- 3 (Moderate): analyzes composition and some relevant properties and mentions at least one alternative, but the comparison is superficial.
- 4 (Substantial): compares multiple candidate sets on chemically grounded criteria (e.g., oxidation-state matching, decomposition products, volatility, hygroscopicity) with explicit elimination.
- 5 (Expert-level): integrates multiple interacting factors (e.g., redox control, compensation for volatile loss, intermediate phase formation, relative precursor reactivity) and handles element-specific subtleties in a coherent multi-step deliberation.

Dimension 2: Utility. The value of the trajectory as a demonstration of how to select precursors, i.e., whether its chemical statements are correct, its steps are logically connected and actually support the final recommendation, it satisfies the task constraints (all target elements covered, at least two precursors, extra elements limited to H, C, N, O, F, Cl, Br), and the recommended reagents are practical (commercially common, stable, widely used) with feasibility, cost, and practicality justified.
- 1 (Harmful): contains major chemical errors, violates the task constraints, or the final answer contradicts the reasoning.
- 2 (Limited): contains notable factual errors or unsupported leaps, and the final answer is only weakly connected to the reasoning.
- 3 (Acceptable): largely correct, with minor inaccuracies or loosely supported steps; constraints are satisfied.
- 4 (High): chemically correct, each step supports the final selection, constraints are satisfied, and practical factors are addressed.
- 5 (Exemplary): fully correct and concise, every step contributes to the decision, and the reasoning pattern is transferable to other targets.

Evaluation rules:
1. Rate the two dimensions independently. A long but flawed trajectory may have high complexity and low utility, and a short but sound one may have low complexity and high utility.
2. Do not reward length, formatting, or confident tone.
3. Base your judgment only on the given content and established chemical knowledge.
4. Write the rationale before each score, and give integer scores only.

Input:

Target compound: {target}

Candidate trajectory: {trajectory}

Output format: Return only a JSON object in the following format, with no additional text:

{{"complexity_rationale": "<1-3 sentences>", "complexity": <1-5>, "utility_rationale": "<1-3 sentences>", "utility": <1-5>}}"""

# =====================================================================================================================
# Paper appendix: RCPO key-step evaluation (LLM judge for R_step, Eq. 4; used when reward_kwargs.step_judge=llm)
# =====================================================================================================================

KEY_STEP_JUDGE = """You are an expert solid-state chemist acting as an impartial auditor of reasoning trajectories for inorganic precursor selection. You will be given a target compound, the thinking process of a model, and its final reaction. Decide independently whether the thinking process contains each of the three key steps of the "Analyze-Identify-Justify" workflow below. Judge only whether each step is explicitly carried out; do not judge whether the final reaction matches the literature.

Key steps:
1. analyze (Target compound analysis): the thinking explicitly analyzes the target compound, e.g., its constituent elements, oxidation states, or properties relevant to precursor selection. Merely restating the formula does not count.
2. identify (Candidate precursor identification): the thinking explicitly proposes candidate precursors that together supply the elements of the target, considering element coverage or practicality.
3. justify (Final reaction justification): the thinking explicitly explains why the final precursor set is selected (e.g., feasibility, stability, cost, or practicality), and the final reaction is consistent with the reasoning.

Evaluation rules:
1. Judge the three steps independently.
2. Do not reward length, formatting, or confident tone.
3. Base your judgment only on the given content.

Input:

Target compound: {target}

Thinking process: {think}

Final reaction: {answer}

Output format: Return only a JSON object in the following format, with no additional text:

{{"analyze": true|false, "identify": true|false, "justify": true|false}}"""


def precursor_stage1_prompt(target: str) -> str:
    return PRECURSOR_STAGE1.format(target=target, output_format=OUTPUT_FORMAT)


def synthesizability_stage1_prompt(composition: str) -> str:
    return SYNTHESIZABILITY_STAGE1.format(composition=composition)


def synthesizability_reasoning_prompt(composition: str, think_suffix: str = THINK_SUFFIX) -> str:
    """Synthesizability evaluation prompt: the Stage-1 template in reasoning mode (Table 1)."""
    return synthesizability_stage1_prompt(composition) + think_suffix


def precursor_reasoning_prompt(target: str, think_suffix: str = THINK_SUFFIX) -> str:
    """Rejective fine-tuning / RL / evaluation prompt (Table 1: ``{synthesis requirement} /think``)."""
    return PRECURSOR_REJECTIVE_FINE_TUNING.format(target=target, output_format=OUTPUT_FORMAT) + think_suffix


def rejection_sampling_prompt(target: str) -> str:
    return PRECURSOR_REJECTION_SAMPLING.format(target=target, output_format=OUTPUT_FORMAT)


def complexity_utility_prompt(target: str, trajectory: str) -> str:
    return PRECURSOR_COMPLEXITY_UTILITY.format(target=target, trajectory=trajectory)


def key_step_judge_prompt(target: str, think: str, answer: str | None) -> str:
    return KEY_STEP_JUDGE.format(target=target, think=think, answer=answer or "(none)")


def reaction_line(target: str, precursors: list[str]) -> str:
    return f"{target} <- " + " + ".join(precursors)
