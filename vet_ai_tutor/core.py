"""Core tutor logic for retrieval-augmented veterinary study assistance."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, List, Tuple

try:
    from langchain_community.vectorstores import FAISS
except ImportError:  # pragma: no cover
    from langchain.vectorstores import FAISS

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # pragma: no cover
    OpenAIEmbeddings = None  # type: ignore

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover
    from langchain_huggingface import HuggingFaceEmbeddings

from .config import EMBEDDING_MODEL, OPENAI_API_KEY, VECTOR_PATH


class Tutor:
    """Simple retrieval-augmented tutoring class."""

    def __init__(
        self,
        llm: Any = None,
        vector_path: str | os.PathLike[str] | None = None,
        embeddings: Any = None,
    ):
        self.llm = llm
        self.vector_path = str(Path(vector_path) if vector_path else VECTOR_PATH)
        self.embeddings = embeddings or self._default_embeddings()
        self.db = self._load_vectorstore()

    def _default_embeddings(self):
        if OPENAI_API_KEY and OpenAIEmbeddings is not None:
            return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def _load_vectorstore(self):
        if not os.path.exists(self.vector_path):
            raise FileNotFoundError(
                f"Vectorstore not found at '{self.vector_path}'. "
                "Run the build step first or upload the knowledge base."
            )

        return FAISS.load_local(self.vector_path, self.embeddings, allow_dangerous_deserialization=True)

    def _format_context(self, docs: Iterable[Any]) -> str:
        return "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)

    def _invoke_llm(self, prompt: str) -> str:
        if self.llm is None:
            raise ValueError("A language model must be provided.")

        if hasattr(self.llm, "invoke"):
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)

        if hasattr(self.llm, "predict"):
            return str(self.llm.predict(prompt))

        return str(self.llm(prompt))

    def ask(self, question: str, k: int = 3) -> Tuple[str, List[Any]]:
        docs = self.db.similarity_search(question, k=k)
        context = self._format_context(docs)
        system_prompt = (
            "You are a helpful veterinary tutor. Answer using only the context provided below. "
            "If the answer is not in the context, say that you do not have enough information and "
            "ask the student to provide more detail."
        )
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{question}"
        answer = self._invoke_llm(prompt)
        return answer, docs

    def reset_history(self):
        """Compatibility method for older CLI scripts."""
        return None


def build_tutor(llm: Any = None, vector_path: str | os.PathLike[str] | None = None, embeddings: Any = None) -> Tutor:
    return Tutor(llm=llm, vector_path=vector_path, embeddings=embeddings)
