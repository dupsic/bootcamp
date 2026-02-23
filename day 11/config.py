import os

# --- LLM Provider ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")

# --- Embedding ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434")

# --- Retrieval ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
COLLECTION_NAME = "bootcamp_docs"

# --- Generation ---
SYSTEM_PROMPT = """You are a helpful assistant answering questions based on provided context.
Use ONLY the context below to answer. If the answer is not in the context, say "I cannot find this information in the provided documents."

Context:
{context}
"""
MAX_TOKENS = 1024
TEMPERATURE = 0.3
