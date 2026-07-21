# src/rag/api.py
from fastapi import FastAPI
from pydantic import BaseModel

from .chain import ask, build_chain
from .vectorstore import get_retriever, get_vector_store

app = FastAPI(title="RAG Pipeline")

# Build once at startup, not per request: opening the store and
# constructing the chain are expensive and should be reused.
_store = get_vector_store()
_retriever = get_retriever(_store)
_chain = build_chain(_retriever)


class Query(BaseModel):
    question: str


@app.post("/query")
def query(q: Query):
    return {"answer": ask(_chain, q.question)}


@app.get("/health")
def health():
    return {"status": "ok"}