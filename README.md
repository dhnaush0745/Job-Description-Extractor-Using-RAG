# JD Q&A Assistant (Mini RAG App)

This project is a small Retrieval-Augmented Generation (RAG) application that answers questions about a set of documents such as job descriptions, PDFs, or markdown files.

It demonstrates an end-to-end LLM workflow: document ingestion → embeddings → vector database → retrieval → LLM-based answer generation.

## Features

- Ingests `.txt`, `.md`, and `.pdf` documents
- Splits documents into chunks and generates embeddings
- Stores embeddings in a local ChromaDB vector database
- Retrieves top-k relevant chunks for a user query
- Generates context-aware answers using a transformer-based LLM
- Exposes a FastAPI `/ask` endpoint
- Includes basic unit tests

## Tech Stack

- Python, FastAPI
- sentence-transformers (embeddings)
- ChromaDB (vector store)
- Hugging Face Transformers (LLM)
- Pytest

## How to Run

```bash
pip install -r requirements.txt

# Add documents
# Put JD / PDFs in data/docs/

# Build vector index
python -m app.ingest

# Start API
uvicorn app.main:app --reload
