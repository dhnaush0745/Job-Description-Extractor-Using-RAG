from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / "data" / "docs"
INDEX_PATH = BASE_DIR / "data" / "index"

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"  # small, fast, good for QA

# RAG settings
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 3
