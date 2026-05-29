"""Qwen client — talks to Alibaba Cloud Model Studio via its OpenAI-compatible API.

Model Studio exposes an OpenAI-compatible endpoint, so we use the standard
`openai` SDK and just point base_url at Alibaba's gateway.

International (Singapore) endpoint:
    https://dashscope-intl.aliyuncs.com/compatible-mode/v1
China (Beijing) endpoint:
    https://dashscope.aliyuncs.com/compatible-mode/v1

Set your hackathon API key in the env var DASHSCOPE_API_KEY before running.
Everything is overridable by env so no endpoint/model is hard-locked.
"""
from __future__ import annotations

import os


DEFAULT_BASE_URL = os.environ.get(
    "QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")  # qwen-max / qwen-plus / qwen-turbo / qwen3-max
DEFAULT_EMBED_MODEL = os.environ.get("QWEN_EMBED_MODEL", "text-embedding-v3")


class QwenClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, embed_model: str | None = None):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.base_url = base_url or DEFAULT_BASE_URL
        self.model = model or DEFAULT_MODEL
        self.embed_model = embed_model or DEFAULT_EMBED_MODEL
        self._client = None

    def _ensure(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "No API key. Set DASHSCOPE_API_KEY (get free hackathon credits "
                    "at Alibaba Cloud Model Studio, Singapore region)."
                )
            from openai import OpenAI  # lazy import so the engine works without the dep
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def chat(self, messages: list[dict], temperature: float = 0.7,
             max_tokens: int = 1024) -> str:
        """messages: [{'role':'system'|'user'|'assistant','content':...}, ...]"""
        client = self._ensure()
        resp = client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def complete(self, prompt: str, system: str | None = None, **kw) -> str:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, **kw)

    def embed(self, texts) -> list[list[float]]:
        """Return embedding vectors for text(s), via Qwen's embedding model."""
        client = self._ensure()
        if isinstance(texts, str):
            texts = [texts]
        resp = client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
