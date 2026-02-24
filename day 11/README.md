Local LLM: Powered by `qwen3:8b` via Ollama.
Vector Storage: Uses `ChromaDB` for high-performance document retrieval.
Embeddings: Utilizes `nomic-embed-text` for semantic understanding.
Structured Output: Provides answers along with source metadata and performance metrics.


Environment Setup
   python3 -m venv .venv
   source .venv/bin/activate
   pip install langchain langchain-community langchain-ollama langchain-chroma chromadb pypdf requests

Launch:
	python3 pipeline.py
Benchmarking
	python3 benchmark.py
