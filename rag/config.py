from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGES_FILE = Path("documents", "wikipedia_pages.json")
DOCS_DIR = Path("documents")
CHROMA_DIR = Path("chroma_db")
CHUNK_CACHE_DIR = ""

EMBEDDING_MODEL = "voyage-4-lite"
LLM_MODEL = "gemini-3.6-flash"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_CHARS = 80

RETRIEVAL_K = 5