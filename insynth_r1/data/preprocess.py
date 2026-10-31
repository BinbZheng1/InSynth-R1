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
"""Build MSFT / RCPO / evaluation datasets from raw reaction and synthesizability records.

Synthesizability prediction is trained in Stage-1 only (domain knowledge); Stage-2 cold start and RCPO use
precursor selection only. The final model is evaluated on both tasks.

Raw inputs (jsonl / json / csv / parquet):

* reactions: ``target`` and ``precursors`` (list, JSON list string, or ``"A + B + C"``), or a ``reaction`` string
  ``"target <- A + B + C"``, i.e. the 12k reactions of Kononova et al. after the SynthGPT filtering.
* synthesizability: ``composition`` and ``label`` (``P``/``U``, ``1``/``0`` or ``true``/``false``), i.e. the 41k
  synthesized and 352k hypothetical materials.
* eval reactions (optional): the temporally held-out in-house reactions (no split), same format as the reactions,
  e.g. ``inhouse_benchmark/data_sample.json``. After formula normalization and deduplication, records whose
  target-precursor pair also occurs in the public training or test set are removed.

Example::

    python -m insynth_r1.data.preprocess --reactions raw/reactions.jsonl \
        --synthesizability raw/synthesizability.csv --eval_reactions inhouse_benchmark/data_sample.json \
        --out_dir data/insynth

Outputs under ``out_dir``::

    sft/stage1_{train,test}.parquet              Stage-1 domain fine-tuning (both tasks, direct answers)
    coldstart/input.jsonl                        Stage-2 trajectory generation inputs (training reactions)
    rl/precursor_{train,test}.parquet            RCPO training / validation prompts (verl RL format)
    eval/precursor_test.parquet                  precursor selection test set (same as rl/precursor_test)
    eval/synthesizability_test.parquet           synthesizability test set (Stage-1 prompt, reasoning mode)
    eval/<eval_name>.parquet                     temporally held-out precursor selection set
    split_stats.json                             dataset sizes and dedup counts
"""

import argparse
import os

from insynth_r1.chem import formula_key
from insynth_r1.data import prompts
from insynth_r1.data.io import as_list, read_records, write_json, write_jsonl, write_parquet
from insynth_r1.data.split import deduplicate, random_split, reaction_key
from insynth_r1.reward.parsing import parse_reaction_line
from insynth_r1.reward.rubric import PRECURSOR_SOURCE, SYNTHESIZABILITY_SOURCE

_POSITIVE = {"p", "1", "true", "positive", "synthesized", "yes"}
_NEGATIVE = {"u", "0", "false", "unlabeled", "hypothetical", "no"}


def load_reactions(path: str) -> list[dict]:
    records = []
    for row in read_records(path):
        if "target" in row:
            target, precursors = str(row["target"]).strip(), as_list(row["precursors"])
        else:
            target, precursors = parse_reaction_line(str(row["reaction"]))
            target = (target or "").strip()
        if target and precursors:
            records.append({"target": target, "precursors": precursors})
    return records


def load_synthesizability(path: str) -> list[dict]:
    records = []
    for row in read_records(path):
        composition = str(row["composition"]).strip()
        label = str(row["label"]).strip().lower()
        if label not in _POSITIVE | _NEGATIVE:
            raise ValueError(f"Unknown synthesizability label {row['label']!r} for {composition}")
        if composition:
            records.append({"composition": composition, "label": "P" if label in _POSITIVE else "U"})
    records, _ = deduplicate(records, key_fn=lambda r: formula_key(r["composition"]))
    return records


def precursor_rl_rows(records: list[dict], split: str, think_suffix: str) -> list[dict]:
    """verl RL rows with the rejective fine-tuning prompt (used for RL, validation and evaluation)."""
    return [
        {
            "data_source": PRECURSOR_SOURCE,
            "prompt": [{"role": "user", "content": prompts.precursor_reasoning_prompt(r["target"], think_suffix)}],
            "ability": "chemistry",
            "reward_model": {"style": "rule", "ground_truth": " + ".join(r["precursors"])},
            "extra_info": {"split": split, "index": i, "target": r["target"]},
        }
        for i, r in enumerate(records)
    ]


def synthesizability_eval_rows(records: list[dict], split: str, think_suffix: str) -> list[dict]:
    """Evaluation rows with the Stage-1 synthesizability prompt in reasoning mode."""
    return [
        {
            "data_source": SYNTHESIZABILITY_SOURCE,
            "prompt": [
                {"role": "user", "content": prompts.synthesizability_reasoning_prompt(r["composition"], think_suffix)}
            ],
            "ability": "chemistry",
            "reward_model": {"style": "rule", "ground_truth": r["label"]},
            "extra_info": {"split": split, "index": i, "target": r["composition"]},
        }
        for i, r in enumerate(records)
    ]


def stage1_records(reactions: list[dict], synthesizability: list[dict]) -> list[dict]:
    out = []
    for r in reactions:
        user = prompts.precursor_stage1_prompt(r["target"])
        assistant = prompts.reaction_line(r["target"], r["precursors"])
        out.append({"messages": _chat(user, assistant), "task": "precursor"})
    for s in synthesizability:
        user = prompts.synthesizability_stage1_prompt(s["composition"])
        out.append({"messages": _chat(user, s["label"]), "task": "synthesizability"})
    return out


def _chat(user: str, assistant: str) -> list[dict]:
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reactions", required=True)
    parser.add_argument("--synthesizability", required=True)
    parser.add_argument("--eval_reactions", default=None, help="temporally held-out reactions (no split)")
    parser.add_argument("--eval_name", default="inhouse")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--think_suffix", default=prompts.THINK_SUFFIX, help="appended to RL / eval prompts (Table 1)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args(argv)
    out = args.out_dir

    reactions, num_dup = deduplicate(load_reactions(args.reactions))
    train, test = random_split(reactions, args.test_ratio, args.seed)
    synth_train, synth_test = random_split(load_synthesizability(args.synthesizability), args.test_ratio, args.seed)
    stats = {
        "seed": args.seed,
        "precursor": {"duplicates_removed": num_dup, "num_train": len(train), "num_test": len(test)},
        "synthesizability": {
            "num_train": len(synth_train),
            "num_test": len(synth_test),
            "train_positive": sum(s["label"] == "P" for s in synth_train),
            "test_positive": sum(s["label"] == "P" for s in synth_test),
        },
    }

    # Stage-1: both tasks
    write_parquet(stage1_records(train, synth_train), os.path.join(out, "sft/stage1_train.parquet"))
    write_parquet(stage1_records(test, synth_test), os.path.join(out, "sft/stage1_test.parquet"))
    # Stage-2 and RCPO: precursor selection
    write_jsonl([{"index": i, **r} for i, r in enumerate(train)], os.path.join(out, "coldstart/input.jsonl"))
    write_parquet(precursor_rl_rows(train, "train", args.think_suffix), os.path.join(out, "rl/precursor_train.parquet"))
    test_rows = precursor_rl_rows(test, "test", args.think_suffix)
    write_parquet(test_rows, os.path.join(out, "rl/precursor_test.parquet"))
    # Evaluation
    write_parquet(test_rows, os.path.join(out, "eval/precursor_test.parquet"))
    synth_rows = synthesizability_eval_rows(synth_test, "test", args.think_suffix)
    write_parquet(synth_rows, os.path.join(out, "eval/synthesizability_test.parquet"))
    if args.eval_reactions:
        eval_records, eval_dup = deduplicate(load_reactions(args.eval_reactions))
        public_reactions = {reaction_key(r) for r in reactions}
        held_out = [r for r in eval_records if reaction_key(r) not in public_reactions]
        public_keys = {formula_key(r["target"]) for r in reactions}
        stats[args.eval_name] = {
            "num_records": len(held_out),
            "duplicates_removed": eval_dup,
            "overlap_with_public_removed": len(eval_records) - len(held_out),
            "targets_in_public_data": sum(formula_key(r["target"]) in public_keys for r in held_out),
        }
        rows = precursor_rl_rows(held_out, args.eval_name, args.think_suffix)
        write_parquet(rows, os.path.join(out, f"eval/{args.eval_name}.parquet"))

    write_json(stats, os.path.join(out, "split_stats.json"))
    print(stats)


if __name__ == "__main__":
    main()
