# generate.py
import requests
import time
import config
import json

def generate_answer(query, chunks):
    start_time = time.time()

    context_used = "\n\n".join([c['content'] for c in chunks])

    prompt = config.SYSTEM_PROMPT.format(context=context_used)

    base_url = config.LLM_BASE_URL.replace("/v1", "") 
    url = f"{base_url}/api/chat"

    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        "stream": False,
        "options": {
            "temperature": config.TEMPERATURE,
            "num_predict": config.MAX_TOKENS
        }
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        generation_time_ms = int((time.time() - start_time) * 1000)
        return {
            "query": query,
            "answer": result['message']['content'],
            "context_used": context_used,
            "model": config.LLM_MODEL,
            "generation_time_ms": generation_time_ms,
            "tokens_generated": result.get('eval_count', 0)
        }
    except Exception as e:
        return {
            "query": query,
            "answer": f"Error during generation: {str(e)}",
            "context_used": context_used,
            "model": config.LLM_MODEL,
            "generation_time_ms": 0,
            "tokens_generated": 0
        }
