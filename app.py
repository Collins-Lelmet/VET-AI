"""Streamlit front-end for the Vet AI Tutor app."""

from __future__ import annotations

import streamlit as st

from vet_ai_tutor.app_service import get_or_build_vectorstore, get_llm
from vet_ai_tutor.config import VECTOR_PATH
from vet_ai_tutor.core import Tutor


@st.cache_resource
def build_tutor():
    llm = get_llm()
    get_or_build_vectorstore()
    return Tutor(llm=llm, vector_path=str(VECTOR_PATH))


def main():
    st.set_page_config(page_title="Vet AI Tutor", page_icon="🐄", layout="centered")
    st.title("🐄 Vet AI Tutor")
    st.caption("A cloud-ready veterinary tutor built for student learning and scalable deployment.")

    if "history" not in st.session_state:
        st.session_state.history = []

    try:
        tutor = build_tutor()
    except Exception as exc:  # pragma: no cover
        st.error(str(exc))
        st.markdown("Add study files to `data/raw_notes` or configure the API key in Streamlit secrets.")
        st.stop()

    question = st.chat_input("Ask a veterinary question")
    if question:
        with st.spinner("Thinking..."):
            answer, _ = tutor.ask(question, k=3)

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
