from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import RAW_DIR, VECTOR_PATH
from .loaders import load_documents_from_directory


def has_vector_index(vector_path: str | Path = VECTOR_PATH) -> bool:
    vector_path = Path(vector_path)
    if not vector_path.exists():
        return False
    files = list(vector_path.iterdir())
    return any(
        p.name.endswith(".faiss") or p.name.endswith(".pkl") or p.name.endswith(".index")
        for p in files
    )


def build_vectorstore(
    raw_dir: str | Path = RAW_DIR,
    vector_path: str | Path = VECTOR_PATH,
    embeddings: Any = None,
) -> FAISS:
    """Load docs and build a FAISS index from them."""
    docs = load_documents_from_directory(raw_dir)
    if not docs:
        raise ValueError(f"No valid documents were found in '{raw_dir}'.")

    if embeddings is None:
        raise ValueError("Embeddings are required to build the vectorstore.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vector_path = Path(vector_path)
    vector_path.mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(vector_path))
    return db


def load_vectorstore(
    vector_path: str | Path = VECTOR_PATH,
    embeddings: Any = None,
) -> FAISS:
    """Load an existing FAISS index from disk."""
    if not has_vector_index(vector_path):
        raise FileNotFoundError(f"Vector index not found at '{vector_path}'.")

    if embeddings is None:
        raise ValueError("Embeddings are required to load the vectorstore.")

    return FAISS.load_local(str(vector_path), embeddings, allow_dangerous_deserialization=True)
