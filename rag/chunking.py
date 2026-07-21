from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

def get_splitter() -> TextSplitter:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
    return splitter

def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = get_splitter()
    return splitter.split_documents(documents)


