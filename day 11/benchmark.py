# benchmark.py
import json
import time
from openai import OpenAI

MODELS = ["qwen3:8b", "llama3.2:3b"]  
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"

PROMPTS = [
    {"category": "factual", "prompt": "What causes tides on Earth? Answer in 2-3 sentences."},
    {"category": "reasoning", "prompt": "If a train travels 120km in 1.5 hours, and then 80km in 1 hour, what is the average speed for the entire journey?"},
    {"category": "summarization", "prompt": "Summarize this in one sentence: The internet is a global system of interconnected computer networks that use the Internet protocol suite to link devices worldwide."},
    {"category": "structured", "prompt": 'List 3 European capitals. Respond ONLY with valid JSON: [{"city": "name", "country": "name"}]'},
    {"category": "code", "prompt": "Write a Python function to calculate the area of a circle given its radius."},
    {"category": "instruction", "prompt": "Write a short poem about a cat. Do NOT use the word 'meow' or 'fur'."}
]

def run_benchmark(model: str, prompt_data: dict) -> dict:
    prompt_text = prompt_data["prompt"]
    category = prompt_data["category"]

    print(f"Testing {model} on {category}...")

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.7,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    usage = response.usage

    return {
        "model": model,
        "category": category,
        "prompt": prompt_text,
        "response": content,
        "time_seconds": round(elapsed, 2),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "tokens_per_second": round(usage.completion_tokens / elapsed, 1) if usage and usage.completion_tokens else None,
        "subjective_quality_1_5": None
    }

if __name__ == "__main__":
    results = []

    for model in MODELS:
        print(f"\n--- Starting benchmark for model: {model} ---")
        for prompt_data in PROMPTS:
            try:
                result = run_benchmark(model, prompt_data)
                results.append(result)
            except Exception as e:
                print(f"Error running {model}: {e}")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nsaved to benchmark_results.json")
