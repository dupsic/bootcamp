# Decision Log — Local RAG Pipeline

## Model Selection
- **Chosen LLM:** Qwen3:8b
- **Why:** In my Exercise 1 benchmarks, Qwen3:8b consistently balanced accuracy with a reasonable generation time (approx. 25-30 seconds for complex answers). It followed the system prompt instructions more strictly than smaller models like Phi-3.
- **What I considered but rejected:** - **Llama 3.2:3b:** Faster, but struggled with the "nuanced" hard questions, occasionally hallucinating when the answer wasn't explicitly in one chunk.
    - **Deepseek-R1:** Good reasoning, but the "thinking" tokens made the pipeline feel too slow for a real-time Q&A assistant on my current hardware.

## Embedding Model
- **Chosen:** nomic-embed-text
- **Why:** It is currently the industry standard for local RAG due to its large context window and high performance on the MTEB leaderboard. It provided accurate retrieval for the remote work policy question even when the query used slightly different phrasing.

## Chunking Strategy
- **Chunk size:** 512 tokens
- **Overlap:** 50 tokens
- **Why:** I found that 512 tokens are large enough to keep a full paragraph (like the remote work requirements) together in one piece. The 50-token overlap ensures that if a specific rule spans across two chunks, the context isn't lost at the cutting point.

## Retrieval Configuration
- **Top-K:** 5
- **Why:** Since our document set is relatively small (85 pages), K=5 provides enough context for the LLM to compare conflicting info (like the 3-day vs 4-day policy change) without hitting the context window limit or confusing the model with too much noise.

## Observations
- **What worked well:** The system was excellent at finding specific numbers (e.g., the €500 allowance and 25 Mbps speed).
- **What failed:** Initially, the system struggled with "unanswerable" questions until I refined the SYSTEM_PROMPT in `config.py` to be more strict about saying "I cannot find this information."
- **Local vs cloud expectations:** Azure OpenAI (GPT-4o) would likely be 5-10x faster and better at "synthesis" questions (q11/q12). However, this local setup is superior for privacy, as the company's internal PDFs never leave my machine.

## If I Had More Time / Better Hardware
- I would implement **Re-ranking** (using a Cross-Encoder) to improve the precision of the Top-K results.
- I would explore **Hybrid Search** (combining Vector search with BM25 keyword search) to better handle specific technical IDs or product codes.
