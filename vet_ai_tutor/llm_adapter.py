"""Adapters for the LLM client layer."""

from __future__ import annotations

from typing import Any


class BaseAdapter:
    def __init__(self, llm: Any):
        self.llm = llm

    def invoke(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI-compatible Chat models."""

    pass


class OllamaAdapter(BaseAdapter):
    """Adapter for local Ollama models."""

    pass
