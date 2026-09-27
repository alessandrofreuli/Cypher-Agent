import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, LiteralString, cast

from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field, create_model

from cypher_agent.config import settings
from cypher_agent.db import get_driver

_TYPE_MAP: dict[str, type] = {"string": str, "integer": int, "number": float, "boolean": bool}


def load_tool_definitions(path: Path | None = None) -> list[dict[str, Any]]:
    with open(path or settings.tools_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_db_schema(path: Path | None = None) -> str:
    with open(path or settings.schema_path, "r", encoding="utf-8") as f:
        return f.read()


def make_tool(defn: dict[str, Any]) -> StructuredTool:
    name: str = defn["name"]
    description: str = defn["description"]
    query: LiteralString = cast(LiteralString, defn["query"])
    param_specs: dict[str, Any] = defn.get("parameters", {})

    fields = {
        param_name: (
            Annotated[
                _TYPE_MAP.get(spec.get("type", "string"), str) | None,
                Field(description=spec.get("description", "")),
            ],
            None,
        )
        for param_name, spec in param_specs.items()
    }
    args_schema: type[BaseModel] | None = create_model(f"{name}_Args", **fields) if fields else None  # type: ignore[call-overload]

    def _impl(**kwargs: Any) -> str:
        try:
            with get_driver().session(database=settings.neo4j_database) as session:
                rows = session.run(query, kwargs).data()
            # coerce Neo4j types (datetime, Node, etc.) to plain Python
            return json.dumps(rows, default=str)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}", "tool": name})

    _impl.__name__ = name
    _impl.__doc__ = description
    return tool(_impl, args_schema=args_schema) if args_schema else tool(_impl)  # type: ignore[return-value]


def build_tools_summary(defns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for d in defns:
        lines.append(f"- {d['name']}: {d['description']}")
        params = d.get("parameters", {})
        if params:
            lines.append("  Parameters:")
            for param_name, spec in params.items():
                lines.append(f"    - {param_name} ({spec.get('type', 'string')}): {spec.get('description', '')}")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def get_toolbox() -> tuple[list[StructuredTool], dict[str, StructuredTool], str]:
    """Returns (tools, tools_by_name, tools_summary)."""
    defns = load_tool_definitions()
    tools = [make_tool(d) for d in defns]
    return tools, {t.name: t for t in tools}, build_tools_summary(defns)
