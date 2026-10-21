from __future__ import annotations

from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_MODEL, OLLAMA_MODEL, RAW_DIR, VECTOR_PATH
from .vectorstore import build_vectorstore, has_vector_index, load_vectorstore

try:
    from langchain_ollama import OllamaLLM
except Exception:  # pragma: no cover
    OllamaLLM = None  # type: ignore


def get_llm():
    """Return the configured LLM client."""
    if OPENAI_API_KEY and ChatOpenAI is not None:
        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.2)

    if OllamaLLM is not None:
        return OllamaLLM(model=OLLAMA_MODEL, temperature=0.2)

    raise RuntimeError("No supported LLM backend is configured. Set OPENAI_API_KEY or install Ollama.")


def get_embeddings():
    """Return the configured embedding model."""
    if OPENAI_API_KEY and OpenAIEmbeddings is not None:
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_or_build_vectorstore() -> FAISS:
    """Create the vectorstore if missing; otherwise load the saved one."""
    embeddings = get_embeddings()
    if has_vector_index(VECTOR_PATH):
        return load_vectorstore(VECTOR_PATH, embeddings)

    return build_vectorstore(RAW_DIR, VECTOR_PATH, embeddings)
