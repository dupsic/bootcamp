import os
import json

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models import AzureOpenAIModel


# ---------------------------
# Azure Judge Configuration
# ---------------------------
AZURE_API_KEY = "FETY3IuwNRGrjakHIAasKPMcEjIEiWCIVFke6APDw5zIbb5qCMBDJQQJ99BLACfhMk5XJ3w3AAAAACOGPChD"
AZURE_ENDPOINT = "https://ds-ai-internship.openai.azure.com/"
AZURE_API_VERSION = "2025-01-01-preview"
DEPLOYMENT_NAME = "gpt-4.1-mini"

azure_judge = AzureOpenAIModel(
    model="gpt-4.1-mini",
    deployment_name=DEPLOYMENT_NAME,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)

# ---------------------------
# Load Pipeline Outputs
# ---------------------------

with open("pipeline_outputs_local.json") as f:
    results = json.load(f)

test_cases = []

for r in results[:5]:
    tc = LLMTestCase(
        input=r["question"],
        actual_output=r["actual_answer"],
        expected_output=r["expected_answer"],
        retrieval_context=r["retrieved_context"],
    )
    test_cases.append(tc)

# ---------------------------
# Define Metrics (IMPORTANT: pass model=azure_judge)
# ---------------------------

metrics = [
    FaithfulnessMetric(
        threshold=0.7,
        include_reason=True,
        model=azure_judge,
	async_mode=False,
    ),
    AnswerRelevancyMetric(
        threshold=0.7,
        include_reason=True,
        model=azure_judge,
	async_mode=False,
    ),
    ContextualRelevancyMetric(
        threshold=0.7,
        include_reason=True,
        model=azure_judge,
	async_mode=False,
    ),
    ContextualRecallMetric(
        threshold=0.7,
        include_reason=True,
        model=azure_judge,
	async_mode=False,
    ),
    ContextualPrecisionMetric(
        threshold=0.7,
        include_reason=True,
        model=azure_judge,
	async_mode=False,
    ),
]

# ---------------------------
# Run Evaluation
# ---------------------------

print(f"Running evaluation on {len(test_cases)} test cases...")

eval_results = evaluate(test_cases, metrics)

# ---------------------------
# Save Results
# ---------------------------

with open("eval_results_local.json", "w") as f:
    json.dump(eval_results.model_dump(), f, indent=2)

print("Evaluation complete.")
print("Results saved to eval_results_local.json")
