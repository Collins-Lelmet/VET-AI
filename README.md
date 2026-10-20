# Vet AI Tutor

This project is a cloud-ready veterinary tutoring app that uses retrieval-augmented generation (RAG) to answer student questions from course notes and study material.

## Overview

- Local-first prototype
- Cloud-ready architecture for future deployment
- Uses OpenAI or Ollama for the LLM layer
- Uses FAISS for vector search
- Loads PDFs, DOCX, and PPTX study materials

## Project structure

- `app.py` – Streamlit app entry point
- `vet_ai_tutor/` – core Python package
- `data/raw_notes/` – place PDFs, DOCX, or PPTX files here
- `data/vectorstore/` – saved FAISS vector index

## Quick start

1. Create a virtual environment
   ```bash
   python -m venv .venv
   . .venv/bin/activate
   ```

2. Install dependencies
   ```bash
   python -m pip install -r requirements.txt
   ```

3. Add your API key if using OpenAI
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

4. Add study files to `data/raw_notes/`

5. Run the app
   ```bash
   streamlit run app.py
   ```

## Cloud deployment note

This version is designed to be migrated into a scalable online deployment using:
- Streamlit Cloud or Render
- OpenAI/Azure OpenAI
- Pinecone or Qdrant for vector storage
- Supabase or PostgreSQL for users and session data

## Environment variables

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `DATA_DIR`
- `RAW_DIR`
- `VECTOR_PATH`
- `EMBEDDING_MODEL`
- `LLM_PROVIDER`
- `OLLAMA_MODEL`
