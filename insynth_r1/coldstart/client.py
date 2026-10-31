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
"""Minimal async client for OpenAI-compatible chat completion endpoints (DeepSeek, vLLM, SGLang, ...).

The API key is read from an environment variable.
"""

import asyncio
import logging
import os
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    content: str
    reasoning_content: str | None
    finish_reason: str | None


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        concurrency: int = 32,
        timeout: float = 600.0,
        max_retries: int = 5,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        api_key = os.getenv(api_key_env) if api_key_env else None
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.max_retries = max_retries

    async def chat(self, client: httpx.AsyncClient, messages: list[dict], **sampling) -> ChatResult | None:
        payload = {"model": self.model, "messages": messages, **{k: v for k, v in sampling.items() if v is not None}}
        async with self.semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
                    response.raise_for_status()
                    choice = response.json()["choices"][0]
                    message = choice["message"]
                    return ChatResult(
                        content=message.get("content") or "",
                        reasoning_content=message.get("reasoning_content"),
                        finish_reason=choice.get("finish_reason"),
                    )
                except (httpx.HTTPError, KeyError, ValueError) as e:
                    if attempt == self.max_retries:
                        logger.warning("request failed after %d retries: %s", attempt, type(e).__name__)
                        return None
                    await asyncio.sleep(min(60.0, 2.0**attempt))
        return None

    async def chat_many(self, conversations: list[list[dict]], **sampling) -> list[ChatResult | None]:
        async with httpx.AsyncClient() as client:
            return await asyncio.gather(*(self.chat(client, messages, **sampling) for messages in conversations))
