from fastapi import FastAPI
from pydantic import BaseModel

from app.rag_pipeline import RAGPipeline

app = FastAPI(title="JD Q&A Assistant")

rag = RAGPipeline()


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QuestionRequest):
    answer, sources = rag.generate_answer(request.question)

    return {
        "question": request.question,
        "answer": answer,
        "sources": sources
    }
