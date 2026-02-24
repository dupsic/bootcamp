from deepeval.models import AzureOpenAIModel

azure_judge = AzureOpenAIModel(
    model="gpt-4.1-mini",
    deployment_name="gpt-4.1-mini",
    azure_endpoint="https://ds-ai-internship.openai.azure.com/",
    api_key="FETY3IuwNRGrjakHIAasKPMcEjIEiWCIVFke6APDw5zIbb5qCMBDJQQJ99BLACfhMk5XJ3w3AAAAACOGPChD",
    api_version="2025-01-01-preview",
)
