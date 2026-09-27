# Cypher Agent

Plan-and-execute agent over a Neo4j knowledge graph, built with LangGraph + Azure OpenAI.

Two nested graphs:

- **plan level** (`plan_graph.py`): `plan_node → agent_node → replan_node`, looping until a
  response is produced or the plan needs regenerating.
- **step level** (`step_graph.py`): `strategy_node → tool_execution_node → critic_node`,
  retrying a single step up to `CYPHER_AGENT_MAX_STEP_ATTEMPTS` times under critic supervision.

Tools are generated at runtime from `data/tool_definitions.json` (each entry = name,
description, Cypher query, parameters) and executed against Neo4j; `data/schema.md` is
injected into the planner prompt.

## Project structure

```
src/cypher_agent/
  config.py      env/settings
  db.py          Neo4j driver (lazy, cached)
  llm.py         Azure OpenAI model (lazy, cached)
  tools.py       JSON → StructuredTool generation + tool summary
  schemas.py     Pydantic models + graph state TypedDicts
  prompts.py     all prompts
  chains.py      planner / replanner / critic runnables
  step_graph.py  step-level graph
  plan_graph.py  plan-level graph
  printing.py    console pretty-printing
  runner.py      question() entry point
  cli.py         argparse CLI
data/            tool_definitions.json, schema.md
notebooks/       interactive exploration + graph rendering
```

## Setup

```bash
cp .env.example .env         # fill in Neo4j + Azure credentials
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run

```bash
cypher-agent "Which persons have offences recorded in 2023?"
```

or from Python:

```python
from cypher_agent.runner import question
await question("...")
```
