from functools import lru_cache

from langchain_openai import AzureChatOpenAI

from cypher_agent.config import settings


@lru_cache(maxsize=1)
def get_model() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=settings.azure_deployment,
        azure_endpoint=settings.azure_endpoint,
        api_key=settings.azure_api_key,
        api_version=settings.azure_api_version,
        temperature=1,
    )
