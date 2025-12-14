from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

from app.config import DOCS_PATH, INDEX_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(folder: Path) -> List[str]:
    texts = []

    for file in folder.iterdir():
        if file.suffix in {".txt", ".md"}:
            texts.append(file.read_text(encoding="utf-8"))

        elif file.suffix == ".pdf":
            reader = PdfReader(str(file))
            pdf_text = "\n".join(
                page.extract_text()
                for page in reader.pages
                if page.extract_text()
            )
            if pdf_text.strip():
                texts.append(pdf_text)

    return texts


def chunk_text(text: str) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def ingest():
    print("🔹 Loading documents...")
    documents = load_documents(DOCS_PATH)

    print("🔹 Chunking text...")
    chunks = []
    for doc in documents:
        chunks.extend(chunk_text(doc))

    print(f"🔹 Total chunks: {len(chunks)}")

    print("🔹 Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print("🔹 Creating vector database...")
    client = chromadb.Client(
        Settings(
            persist_directory=str(INDEX_PATH),
            anonymized_telemetry=False
        )
    )

    collection = client.get_or_create_collection(name="jd_docs")

    embeddings = embedder.encode(
        chunks,
        show_progress_bar=True,
        batch_size=32
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "jd_docs"} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=[e.tolist() for e in embeddings],
        ids=ids,
        metadatas=metadatas
    )

    print(f"✅ Ingestion complete. Stored {collection.count()} chunks.")


if __name__ == "__main__":
    ingest()
