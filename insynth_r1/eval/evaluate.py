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
"""Compute the paper's metrics from generations written by :mod:`insynth_r1.eval.generate`.

    python -m insynth_r1.eval.evaluate --generations outputs/insynth-r1-8b/precursor_test.jsonl \
        --inventory data/precursor_inventory.txt --output outputs/insynth-r1-8b/precursor_test_metrics.json
"""

import argparse
from collections import defaultdict

from insynth_r1.data.io import read_records, write_json
from insynth_r1.eval.metrics import (
    PU_CLASS_PRIOR,
    precursor_selection_metrics,
    pu_metrics,
    score_precursor_responses,
    synthesizability_score,
)
from insynth_r1.reward.rubric import SYNTHESIZABILITY_SOURCE


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--inventory", default=None, help="precursor inventory for Validity (one formula per line)")
    parser.add_argument("--class_prior", type=float, default=PU_CLASS_PRIOR)
    parser.add_argument("--allow_missing_think_open", action="store_true")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    if not args.inventory:
        print("WARNING: no --inventory given; Validity is the fraction of parseable formulas")

    by_source = defaultdict(list)
    for record in read_records(args.generations):
        by_source[record["data_source"]].append(record)

    results = {}
    for source, records in by_source.items():
        if source.startswith(SYNTHESIZABILITY_SOURCE):
            positives, unlabeled = [], []
            for r in records:
                score = synthesizability_score(r, args.allow_missing_think_open)
                (positives if str(r["ground_truth"]).upper() == "P" else unlabeled).append(score)
            results[source] = pu_metrics(positives, unlabeled, args.class_prior)
            f1, tau = results[source]["f1"], results[source]["threshold"]
            print(f"[{source}] calibrated F1={100 * f1:.2f} (tau'={tau:.3f})")
        else:
            per_instance = [
                score_precursor_responses(
                    r["responses"], r["ground_truth"], args.inventory, args.allow_missing_think_open
                )
                for r in records
            ]
            metrics = precursor_selection_metrics(per_instance, n_bootstrap=args.n_bootstrap)
            results[source] = {k: {"mean": v.mean, "std": v.std} for k, v in metrics.items()}
            row = " | ".join(f"{k}={v}" for k, v in metrics.items() if not k.endswith("_analytic"))
            print(f"[{source}] n={len(records)} | {row}")
    if args.output:
        write_json(results, args.output)


if __name__ == "__main__":
    main()
