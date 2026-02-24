import time
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import config

def retrieve_chunks(query):
    start_time = time.time()


    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.EMBEDDING_BASE_URL
    )

    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name=config.COLLECTION_NAME
    )

    results = db.similarity_search_with_relevance_scores(query, k=config.TOP_K)
    elapsed_time_ms = int((time.time() - start_time) * 1000)

    retrieved_chunks = []
    for doc, score in results:
        retrieved_chunks.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 0),
            "relevance_score": round(float(score), 4)
        })

    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_time_ms": elapsed_time_ms
    }

if __name__ == "__main__":
    test_query = "What is the company policy on remote work?"
    print(f"Testing retrieval for: {test_query}")
    data = retrieve_chunks(test_query)

    for i, chunk in enumerate(data["retrieved_chunks"]):
        print(f"\n--- Chunk {i+1} (Score: {chunk['relevance_score']}) ---")
        print(f"Source: {chunk['source']} (Page {chunk['page']})")
        print(f"Text: {chunk['content'][:100]}...")
