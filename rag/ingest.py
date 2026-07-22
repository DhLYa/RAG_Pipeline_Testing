from collections import defaultdict
from pathlib import Path
import json

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from .config import DOCS_DIR, PAGES_FILE

def load_documents(pages_file: Path = PAGES_FILE) -> list[Document]:
    with open(pages_file, "r", encoding="utf-8") as f:
        pages = json.load(f)

    return [
        Document(page_content=text, metadata={"source": title})
        for title, text in pages.items()
    ]