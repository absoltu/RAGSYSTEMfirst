from pathlib import Path

BASE_DIR = Path(__file__).parent

CHROMA_DB_PATH = BASE_DIR / "chroma_db"

# LM Studio
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

# embedding model

# LM Studio model name
EMBEDDING_MODEL = "text-embedding-qwen3-embedding-0.6b"

# HuggingFace tokenizer
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# chat model
CHAT_MODEL = "qwen3.5-4b"

# chunk defaults
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_OVERLAP = 200

USE_CUDA=True

ENABLE_OCR=False
ENABLE_TABLE_STRUCTURE=True

PDF_BATCH_PAGES=20

# Search configuration
SEARCH_TYPE = "def"  # "def" for default search, "mmr" for MMR search
MMR_LAMBDA = 0.5  # Relevance-diversity balance for MMR (0-1)
CONTEXT_TOP_N = 5  # Number of chunks to include in context

# Chat configuration
TIMEOUT_REQUEST=320 #Время получения ответа
RAG_ENABLED = True  # Enable or disable RAG context in chat
RAG_COLLECTION_NAME = "digital_processing_of_signals_chtype_token_chsize_1024_overlap_200_outtype_markdown"
CHAT_TEMPERATURE = 0.7
CHAT_MAX_TOKENS = 1000
if RAG_ENABLED:
    SYSTEM_PROMPT = "You are a helpful assistant that answers questions based on the provided context. Use only the information from the context to answer the question. If the context doesn't contain enough information to answer the question, say so. Be concise but informative."
else:
    SYSTEM_PROMPT= "You are a helpful assistant"