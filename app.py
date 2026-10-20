"""Streamlit front-end for the Vet AI Tutor app."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore

try:
    from langchain_ollama import OllamaLLM
except Exception:  # pragma: no cover
    OllamaLLM = None  # type: ignore

from vet_ai_tutor.config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OLLAMA_MODEL,
    RAW_DIR,
    VECTOR_PATH,
)
from vet_ai_tutor.core import Tutor


@st.cache_resource
def get_llm():
    """Return an LLM provider. OpenAI is preferred when configured."""
    if OPENAI_API_KEY and ChatOpenAI is not None:
        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.2)

    if OllamaLLM is not None:
        return OllamaLLM(model=OLLAMA_MODEL, temperature=0.2)

    raise RuntimeError(
        "No supported LLM backend was found. Set OPENAI_API_KEY or install a local Ollama model."
    )


@st.cache_resource
def get_embeddings():
    if OPENAI_API_KEY and OpenAIEmbeddings is not None:
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to initialize embeddings: {exc}")


@st.cache_resource
def load_or_build_vectorstore():
    if VECTOR_PATH.exists() and any(VECTOR_PATH.iterdir()):
        embeddings = get_embeddings()
        return FAISS.load_local(str(VECTOR_PATH), embeddings, allow_dangerous_deserialization=True)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"No data directory was found at '{RAW_DIR}'. Add PDFs, DOCX, or PPTX files there first."
        )

    docs = []
    for root, _, files in os.walk(RAW_DIR):
        for file in files:
            path = Path(root) / file
            try:
                if file.lower().endswith(".pdf"):
                    docs.extend(PyPDFLoader(str(path)).load())
                elif file.lower().endswith(".docx"):
                    docs.extend(Docx2txtLoader(str(path)).load())
                elif file.lower().endswith(".pptx"):
                    docs.extend(UnstructuredPowerPointLoader(str(path)).load())
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Skipping {file}: {exc}")

    if not docs:
        raise ValueError(f"No valid documents were found in '{RAW_DIR}'.")

    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_documents(docs)
    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    VECTOR_PATH.mkdir(parents=True, exist_ok=True)
    db.save_local(str(VECTOR_PATH))
    return db


@st.cache_resource
def build_tutor():
    llm = get_llm()
    load_or_build_vectorstore()
    return Tutor(llm=llm, vector_path=str(VECTOR_PATH))


def main():
    st.set_page_config(page_title="Vet AI Tutor", page_icon="🐄", layout="centered")
    st.title("🐄 Vet AI Tutor")
    st.caption("A cloud-ready veterinary tutor built for student learning and scalable deployment.")

    if "history" not in st.session_state:
        st.session_state.history = []

    try:
        tutor = build_tutor()
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.markdown("Add study notes to `data/raw_notes`, or configure `OPENAI_API_KEY` for the cloud deployment flow.")
        st.stop()

    question = st.chat_input("Ask a veterinary question")
    if question:
        with st.spinner("Thinking..."):
            answer, docs = tutor.ask(question, k=3)

        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "content": answer})

    for item in st.session_state.history:
        if item["role"] == "user":
            st.markdown(f"**You:** {item['content']}")
        else:
            st.markdown(f"**Tutor:** {item['content']}")

    if st.button("Clear chat"):
        st.session_state.history = []


if __name__ == "__main__":
    main()
