def get_embedding_model() -> VoyageAIEmbeddings:
    return

def get_vector_store() -> Chroma:
    return

def chunk_id(doc: Document) -> str:
    return

def upsert_chunks(store: Chroma, chunks: list[Document]) -> int:
    return

def get_retriever(store: Chroma, k: int = RETRIEVAL_K):
    return
