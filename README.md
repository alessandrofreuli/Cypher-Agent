# Cypher Agent

Autonomous **plan-and-execute** agent that answers natural-language questions over a **Neo4j** knowledge graph, built with **LangGraph** and **Azure OpenAI**.

Instead of letting the LLM write free-form Cypher, the agent uses a curated **toolbox of parameterised, read-only queries** (declared in JSON). It breaks multi-hop questions into steps, verifies each intermediate result, and either returns a grounded answer or honestly says it cannot answer.

## Features

- **Two nested LangGraph loops**: a plan level (`plan → agent → replan`) and a step level (`strategy → tools → critic`).
- **Critic-supervised steps**: each step is checked against the raw data and retried up to `CYPHER_AGENT_MAX_STEP_ATTEMPTS` times.
- **Declarative tools**: add a capability by adding a JSON entry, not code.
- **Schema-aware planning**: graph schema and tool summary are injected into the planner prompt.
- **Feasibility gate**: unanswerable questions end with an explicit "I cannot answer this question."
- **Anti-hallucination guardrails**: id-level matching of records, no back-filling from unrelated nodes, answers must contain the actual retrieved data.
- **Dual memory**: a prose `knowledge` summary for planning, plus raw `artifacts` per step.
- **Fully autonomous**: never asks the user for clarification.
- **CLI or async Python library**, with readable console tracing.

## How it works

### Plan level (`plan_graph.py`)

```
START → plan_node → agent_node → replan_node ─┬─→ agent_node  (steps remain)
                                              ├─→ plan_node   (plan was wrong)
                                              └─→ END         (final response)
```

| Node | Responsibility |
| --- | --- |
| `plan_node` | Builds an ordered list of steps after a feasibility check. On replanning, also uses accumulated knowledge and feedback. |
| `agent_node` | Runs the first step through the step-level graph and stores prose result and raw artifact. |
| `replan_node` | Picks exactly one outcome: final `response`, remaining `steps`, or `replan_feedback`. Maintains the `knowledge` summary. |

### Step level (`step_graph.py`)

```
START → strategy_node ─┬─→ tool_execution_node → critic_node ─┬─→ END            (accepted)
                       └─→ critic_node                        └─→ strategy_node  (retry)
```

| Node | Responsibility |
| --- | --- |
| `strategy_node` | Chooses Neo4j tool calls, or answers directly from existing artifacts when only filtering or aggregation is needed. |
| `tool_execution_node` | Runs the tool calls; errors are returned as JSON, never raised. |
| `critic_node` | Re-checks the output against the raw artifacts, including id-level cross-checks. Accepts or asks for a retry. |

### State

| Field | Purpose |
| --- | --- |
| `input` | The user's question. |
| `plan` | Remaining steps. |
| `past_steps` | `(action, prose result)` for each completed step. |
| `knowledge` | Curated summary kept by the replanner. |
| `artifacts` | Raw output per step. |
| `response` | Final answer. |
| `replan_feedback` | Why the previous plan failed. |

## Project structure

```
cypher-agent/
├── src/cypher_agent/
│   ├── config.py         # env vars → frozen Settings
│   ├── db.py             # Neo4j driver
│   ├── llm.py            # Azure OpenAI chat model
│   ├── tools.py          # JSON → StructuredTool, tool summary, schema loader
│   ├── schemas.py        # Pydantic models and graph state
│   ├── prompts.py        # strategy, critic, planner, replanner prompts
│   ├── chains.py         # planner / replanner / critic runnables
│   ├── step_graph.py
│   ├── plan_graph.py
│   ├── printing.py       # console tracing
│   ├── runner.py         # question() async entry point
│   └── cli.py            # `cypher-agent` command
├── data/
│   ├── tool_definitions.json
│   └── schema.md         # not in the repo, see below
├── notebooks/
│   └── explore.ipynb
├── .env.example
├── pyproject.toml
└── README.md
```

## Installation

Requires **Python ≥ 3.11**, a reachable **Neo4j** instance (5.x driver), and an **Azure OpenAI** chat deployment with tool calling and structured output.

```bash
git clone https://github.com/alessandrofreuli/Cypher-Agent.git
cd Cypher-Agent

python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

pip install -e ".[dev]"

cp .env.example .env             # then fill in your credentials
```

## Usage

Command line:

```bash
cypher-agent "Find the person with the phone number '43832726839', then find where they live"
```

Python (in a notebook, just `await question("...")`):

```python
import asyncio
from cypher_agent.runner import question

asyncio.run(question("Find the person with the phone number '43832726839', then find where they live"))
```

## Adapting to your graph

No Python changes are needed: replace `data/tool_definitions.json` and `data/schema.md`.

**Tools**: each entry has a name, a description the LLM uses to choose it, a parameterised Cypher query and its parameters:

```json
{
  "name": "get_person_details_by_id",
  "description": "Use this tool when the unique identifier of the person is known. Returns a single result since the id is unique.",
  "query": "MATCH (p:Person {id: $id}) RETURN p",
  "parameters": {
    "id": { "type": "string", "description": "Unique identifier of the person." }
  }
}
```

Good descriptions say *when* to use the tool, which identifier it expects, and which sibling tool to prefer in adjacent cases.

**Schema**: `schema.md` lists node labels (properties, types, required/optional) and relationship types, and is injected verbatim into the planner prompt.

> ⚠️ `data/schema.md` is in `.gitignore`, so a fresh clone does not include it. Generate your own, for example by exporting the schema with `apoc.meta.schema()` or `db.schema.visualization()` and formatting it as Markdown.
