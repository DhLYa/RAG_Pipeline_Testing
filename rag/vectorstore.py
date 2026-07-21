import hashlib
 
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_voyageai import VoyageAIEmbeddings
 
from .config import EMBEDDING_MODEL, CHROMA_DIR, RETRIEVAL_K

def get_embedding_model() -> VoyageAIEmbeddings:
    embedding_model = VoyageAIEmbeddings(model=EMBEDDING_MODEL)
    return embedding_model

def get_vector_store() -> Chroma:
    vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=get_embedding_model()
)
    return vector_store

def chunk_id(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    h = hashlib.sha256(f"{source}-{page}-{doc.page_content}".encode()).hexdigest()
    return h

def upsert_chunks(store: Chroma, chunks: list[Document]) -> int:
    ids = [chunk_id(c) for c in chunks]
    existing = set(store.get()["ids"])
    new_chunks = [c for c, i in zip(chunks, ids) if i not in existing]
    new_ids = [i for i in ids if i not in existing]

    if new_chunks:
        store.add_documents(documents=new_chunks, ids=new_ids)
        print(f"Added {len(new_chunks)} new chunks.")
    else:
        print("No new chunks to add.")
    return

def get_retriever(store: Chroma, k: int = RETRIEVAL_K):
    retriever = store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": RETRIEVAL_K}
)
    return retriever