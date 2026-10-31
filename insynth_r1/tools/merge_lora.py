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
"""Fold a LoRA adapter into its base model and save a plain HF checkpoint.

``verl.model_merger`` writes the base weights to ``<dir>`` and the adapter to ``<dir>/lora_adapter``; MSFT stages
and RL start from merged weights, so the adapter is merged here::

    python -m insynth_r1.tools.merge_lora --base_dir ckpts/stage1_hf --output_dir ckpts/stage1_merged
"""

import argparse
import os


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base_dir", required=True, help="output of verl.model_merger (contains lora_adapter/)")
    parser.add_argument("--adapter_dir", default=None, help="defaults to <base_dir>/lora_adapter")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = args.adapter_dir or os.path.join(args.base_dir, "lora_adapter")
    model = AutoModelForCausalLM.from_pretrained(args.base_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.save_pretrained(args.output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_dir, trust_remote_code=True).save_pretrained(args.output_dir)
    print(f"merged {adapter_dir} into {args.base_dir} -> {args.output_dir}")


if __name__ == "__main__":
    main()
