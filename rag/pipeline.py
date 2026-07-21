from langchain_chroma import Chroma

from .chain import ask, build_chain
from .chunking import chunk_documents
from .ingest import load_documents
from .vectorstore import get_retriever, get_vector_store, upsert_chunks


def ingest() -> Chroma:
    documents = load_documents()
    print(f"Loaded {len(documents)} pages.")

    chunks = chunk_documents(documents)
    print(f"split documents in {len(chunks)} chunks.")

    store = get_vector_store()
    added = upsert_chunks(store, chunks)
    print(f"Added {added} new chunks." if added else "No new chunks to add.")

    return store

def main():
    store = ingest()
    retriever = get_retriever(store)
    chain = build_chain(retriever)

    print("\nRAG ready, Ask a question. (empty line to quit).")
    while True:
        try:
            question = input("\n>").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        print(ask(chain, question))

if __name__ == "__main__":
    main()