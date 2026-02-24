from retrieve import retrieve_chunks
from generate import generate_answer
import json

def answer_question(query: str) -> dict:

    retrieval_result = retrieve_chunks(query)
    generation_result = generate_answer(
        query=query,
        chunks=retrieval_result["retrieved_chunks"]
    )

    return {
        **generation_result,
        "retrieval": retrieval_result,
    }

if __name__ == "__main__":
    user_query = input("Ask a question: ")
    final_output = answer_question(user_query)

    print("\n--- AI ANSWER ---")
    print(final_output["answer"])
    print("\n--- METADATA ---")
    print(f"Model: {final_output['model']}")
    print(f"Gen Time: {final_output['generation_time_ms']}ms")
    print(f"Chunks Found: {len(final_output['retrieval']['retrieved_chunks'])}")
