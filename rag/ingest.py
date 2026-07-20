from collections import defaultdict
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from .config import DOCS_DIR

def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    loader = PyPDFDirectoryLoader(str(docs_dir))
    return loader.load()