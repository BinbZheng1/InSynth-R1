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
"""Single-turn SFT dataset for MSFT (Stage-1 and Stage-2).

verl's ``MultiTurnSFTDataset`` applies the chat template to every message separately. Qwen3's template only keeps
``<think>...</think>`` for an assistant turn that follows a user query, so the per-message rendering drops the
reasoning of Stage-2 trajectories. This dataset renders the whole conversation once and supervises exactly the
tokens the policy generates at inference time:

    prompt  = template(messages[:-1], add_generation_prompt=True)
    target  = template(messages)[len(prompt):]       # loss_mask = 1 on these tokens

Use with ``data.custom_cls.path=pkg://insynth_r1.data.sft_dataset data.custom_cls.name=SingleTurnSFTDataset``.
"""

import torch

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class SingleTurnSFTDataset(MultiTurnSFTDataset):
    def _render(self, messages: list[dict], enable_thinking) -> tuple[list[int], list[int]]:
        kwargs = dict(self.apply_chat_template_kwargs)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        assert messages[-1]["role"] == "assistant", "the last message must be the assistant response"
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=False, **kwargs
        )
        full_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False, **kwargs)
        if not full_text.startswith(prompt_text):
            raise ValueError("chat template output of the full conversation does not start with the prompt")
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(full_text[len(prompt_text) :], add_special_tokens=False)["input_ids"]
        return prompt_ids, response_ids

    def __getitem__(self, item):
        messages = self._build_messages(self.dataframe.iloc[item].to_dict())
        enable_thinking = self.enable_thinking_default
        if self.enable_thinking is not None:
            enable_thinking = self.enable_thinking[item]
        # verl's SFT config default is the string "none".
        if isinstance(enable_thinking, str) and enable_thinking.strip().lower() in ("", "none", "null"):
            enable_thinking = None
        prompt_ids, response_ids = self._render(messages, None if enable_thinking is None else bool(enable_thinking))

        input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
        loss_mask = torch.tensor([0] * len(prompt_ids) + [1] * len(response_ids), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        length = input_ids.shape[0]

        if length > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"{length=} is larger than {self.max_length=}")
            keep = slice(-self.max_length, None) if self.truncation == "left" else slice(None, self.max_length)
            input_ids, loss_mask, attention_mask = input_ids[keep], loss_mask[keep], attention_mask[keep]
            length = self.max_length
        position_ids = torch.arange(length, dtype=torch.long)

        if self.pad_mode == DatasetPadMode.NO_PADDING:
            return {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": loss_mask}

        pad = self.max_length - length
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        return {
            "input_ids": torch.cat([input_ids, input_ids.new_full((pad,), pad_token_id)]),
            "attention_mask": torch.cat([attention_mask, attention_mask.new_zeros(pad)]),
            "position_ids": torch.cat([position_ids, position_ids.new_zeros(pad)]),
            "loss_mask": torch.cat([loss_mask, loss_mask.new_zeros(pad)]),
        }
