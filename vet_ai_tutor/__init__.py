"""Vet AI Tutor package entry points."""

from .config import *
from .llm_adapter import OpenAIAdapter, OllamaAdapter

__all__ = ["OpenAIAdapter", "OllamaAdapter"]
