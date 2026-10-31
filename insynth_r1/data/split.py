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
"""Train/test split of the public datasets (paper Sec. 4.2.1).

Reaction records (``target`` + ``precursors``) are deduplicated at the reaction level, so that no identical reaction
appears in both subsets, and then split 80/20 at random with a fixed seed.
"""

import random

from insynth_r1.chem import formula_key


def reaction_key(record: dict) -> str:
    return formula_key(record["target"]) + "|" + "+".join(sorted({formula_key(p) for p in record["precursors"]}))


def deduplicate(records: list[dict], key_fn=reaction_key) -> tuple[list[dict], int]:
    """Drop repeated records (normalized target-precursor-set pairs by default); returns ``(unique, num_removed)``.

    Distinct precursor sets of the same target are different reactions and are kept.
    """
    seen, unique = set(), []
    for record in records:
        key = key_fn(record)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique, len(records) - len(unique)


def random_split(records: list[dict], test_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    records = list(records)
    random.Random(seed).shuffle(records)
    num_test = round(len(records) * test_ratio)
    return records[num_test:], records[:num_test]
