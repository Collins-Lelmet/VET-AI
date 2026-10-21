from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, UnstructuredPowerPointLoader
from langchain_core.documents import Document


def load_documents_from_directory(raw_dir: str | Path) -> List[Document]:
    """Load PDFs, DOCX, and PPTX files from a directory and return LangChain documents."""
    raw_dir = Path(raw_dir)
    documents: List[Document] = []

    if not raw_dir.exists():
        raise FileNotFoundError(f"Directory not found: {raw_dir}")

    for root, _, files in __import__("os").walk(raw_dir):
        for file in sorted(files):
            path = Path(root) / file
            try:
                if file.lower().endswith(".pdf"):
                    documents.extend(PyPDFLoader(str(path)).load())
                elif file.lower().endswith(".docx"):
                    documents.extend(Docx2txtLoader(str(path)).load())
                elif file.lower().endswith(".pptx"):
                    documents.extend(UnstructuredPowerPointLoader(str(path)).load())
            except Exception:
                continue

    return documents
