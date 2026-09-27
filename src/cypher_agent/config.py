import os
import warnings
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# src/cypher_agent/config.py -> parents[0]=cypher_agent, [1]=src, [2]=repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(raw: str) -> Path:
    """Relative paths are resolved against PROJECT_ROOT, not the current working
    directory, so behaviour doesn't depend on where the process/notebook was launched from."""
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    azure_deployment: str
    azure_endpoint: str
    azure_api_key: str
    azure_api_version: str
    tools_path: Path
    schema_path: Path
    max_step_attempts: int = 5


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")  # explicit path: independent of cwd too
    warnings.filterwarnings(
        "ignore", message="Pydantic serializer warnings", category=UserWarning, module="pydantic"
    )
    return Settings(
        neo4j_uri=os.environ["NEO4J_URI"],
        neo4j_username=os.environ["NEO4J_USERNAME"],
        neo4j_password=os.environ["NEO4J_PASSWORD"],
        neo4j_database=os.getenv("NEO4J_DATABASE", "spine"),
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        tools_path=_resolve(os.getenv("CYPHER_AGENT_TOOLS_PATH", "data/tool_definitions.json")),
        schema_path=_resolve(os.getenv("CYPHER_AGENT_SCHEMA_PATH", "data/schema.md")),
        max_step_attempts=int(os.getenv("CYPHER_AGENT_MAX_STEP_ATTEMPTS", "5")),
    )


settings = load_settings()