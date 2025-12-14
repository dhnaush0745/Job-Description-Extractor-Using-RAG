import os

# ✅ MUST be set before importing transformers / sentence_transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer

from app.config import INDEX_PATH, EMBEDDING_MODEL, LLM_MODEL, TOP_K


class RAGPipeline:
    def __init__(self):
        # 🔹 Embedding model
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # 🔹 Chroma client (persistent)
        self.client = chromadb.Client(
            Settings(
                persist_directory=str(INDEX_PATH),
                anonymized_telemetry=False
            )
        )

        # ✅ SAFE: create if missing
        self.collection = self.client.get_or_create_collection(
            name="jd_docs"
        )

        # 🔹 LLM (CPU-safe)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

        self.llm = pipeline(
            task="text2text-generation",
            model=LLM_MODEL,
            tokenizer=tokenizer,
            max_length=256,
            device=-1  # CPU
        )

    def retrieve(self, question: str):
        query_emb = self.embedder.encode(question).tolist()

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=TOP_K
        )

        return results["documents"][0]

    def generate_answer(self, question: str):
        contexts = self.retrieve(question)

        context_text = "\n".join(contexts)

        prompt = f"""You are an assistant that answers questions using ONLY the provided context.

Context:
{context_text}

Question:
{question}

Answer clearly and concisely.
"""

        result = self.llm(prompt)
        answer = result[0]["generated_text"]

        return answer, contexts
