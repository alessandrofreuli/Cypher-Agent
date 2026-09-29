# Cypher Agent

> Autonomous **plan-and-execute** agent that answers natural-language questions over a **Neo4j** knowledge graph, built with **LangGraph** and **Azure OpenAI**.

Instead of letting an LLM write free-form Cypher, Cypher Agent gives it a curated **toolbox of parameterised, read-only queries** (defined in JSON) and wraps it in two nested LangGraph loops: one that **plans and replans** at the question level, and one that **executes and critiques** each individual step. The result is an agent that decomposes multi-hop questions ("find the person with this phone number, then tell me where they live"), verifies its own intermediate results, and either returns a grounded answer or explicitly says it cannot answer.

---

## Table of contents

- [Key features](#key-features)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [The toolbox: `data/tool_definitions.json`](#the-toolbox-datatool_definitionsjson)
- [The graph schema: `data/schema.md`](#the-graph-schema-dataschemamd)
- [Included tools](#included-tools)
- [Design notes](#design-notes)
- [Development](#development)
- [Security notes](#security-notes)
- [Limitations and roadmap](#limitations-and-roadmap)

---

## Key features

- **Two-level plan-and-execute architecture** – a plan-level graph (`plan → agent → replan`) drives a step-level graph (`strategy → tool execution → critic`).
- **Critic-supervised steps** – every step is verified against the raw data before it is accepted, with automatic retries (up to `CYPHER_AGENT_MAX_STEP_ATTEMPTS`).
- **Declarative tools** – Neo4j queries are declared in `data/tool_definitions.json` and turned into typed LangChain `StructuredTool`s at runtime. Adding a capability means adding a JSON entry, not code.
- **Schema-aware planning** – the graph schema (`data/schema.md`) and a summary of the available tools are injected into the planner prompt.
- **Feasibility gate** – if the question cannot be answered with the available schema and tools, the planner emits a single `give_up` step and the agent replies with an honest "I cannot answer this question."
- **Anti-hallucination guardrails** – prompts enforce id-level matching of sub-records to parent entities, forbid back-filling answers from unrelated nodes, and require the final answer to contain the actual retrieved data (not just counts).
- **Dual memory** – a curated prose `knowledge` summary for planning, plus raw, unsummarised `artifacts` per step so later steps can filter and aggregate on exact data.
- **Fully autonomous** – no human-in-the-loop: prompts explicitly forbid asking for clarification.
- **Readable console tracing** – every node's output is pretty-printed as it streams.
- **Usable as a CLI or as a Python library** (async).

---

## How it works

The agent is composed of two LangGraph state machines.

### Plan level (`plan_graph.py`)

```
START → plan_node → agent_node → replan_node ─┬─→ agent_node   (steps remain)
                                              ├─→ plan_node    (plan was fundamentally wrong)
                                              └─→ END          (final response produced)
```

| Node | Responsibility |
| --- | --- |
| `plan_node` | Builds an ordered list of structured steps (`action`, `description`, `reason`) using the DB schema and the tool summary. Runs the **feasibility gate** first. On replanning, it also receives the accumulated knowledge and the replanner's feedback. |
| `agent_node` | Takes the first step of the plan and runs it through the step-level graph. Stores the prose result in `past_steps` and the raw output in `artifacts`. A `give_up` step is not executed. |
| `replan_node` | Evaluates the last step and picks **exactly one** outcome: `response` (final answer, or an explicit "I cannot answer this question."), `steps` (remaining plan), or `replan_feedback` (regenerate the plan from scratch). It also maintains the curated `knowledge` summary. |

### Step level (`step_graph.py`)

```
START → strategy_node ─┬─→ tool_execution_node → critic_node ─┬─→ END              (verdict has a result)
                       └─→ critic_node                        └─→ strategy_node    (verdict has feedback → retry)
```

| Node | Responsibility |
| --- | --- |
| `strategy_node` | Decides how to progress on the step: call one or more Neo4j tools, or – for pure filtering/matching/aggregation over data already in the artifacts – answer directly, exhaustively and with concrete ids/values. |
| `tool_execution_node` | Executes the requested tool calls against Neo4j and accumulates the raw output as the step artifact. Unknown tools and exceptions are returned as JSON errors, never raised. |
| `critic_node` | Independently re-checks the output against the raw artifacts available *before* the step (including id-level cross-checks for misattribution). Returns either a prose `result` (accept) or `feedback` (retry). After the maximum number of attempts, the step is closed as failed. |

### State

| Field | Purpose |
| --- | --- |
| `input` | The user's question. |
| `plan` | Remaining steps to execute. |
| `past_steps` | `(action, prose result)` for every completed step. |
| `knowledge` | Curated prose summary maintained by the replanner. |
| `artifacts` | Raw, unsummarised output per step action (merged with a non-destructive reducer). |
| `response` | Final human-readable answer. |
| `replan_feedback` | Why the previous plan failed, if a full replan is needed. |

---

## Project structure

```
cypher-agent/
├── src/cypher_agent/
│   ├── __init__.py       # exports build_plan_graph, build_step_graph
│   ├── config.py         # environment variables → frozen Settings dataclass
│   ├── db.py             # Neo4j driver (lazy, cached, connectivity-checked)
│   ├── llm.py            # Azure OpenAI chat model (lazy, cached)
│   ├── tools.py          # JSON → StructuredTool generation + tool summary + schema loader
│   ├── schemas.py        # Pydantic models (Plan, Act, StepVerdict) and graph state TypedDicts
│   ├── prompts.py        # strategy, critic, planner and replanner prompts
│   ├── chains.py         # planner / replanner / critic runnables
│   ├── step_graph.py     # step-level graph
│   ├── plan_graph.py     # plan-level graph
│   ├── printing.py       # console pretty-printing of streamed updates
│   ├── runner.py         # question() async entry point
│   └── cli.py            # argparse CLI (`cypher-agent`)
├── data/
│   ├── tool_definitions.json   # declarative Neo4j tools
│   └── schema.md               # graph schema injected into the planner prompt (see notes)
├── notebooks/
│   └── explore.ipynb           # interactive exploration and graph rendering
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Requirements

- **Python** ≥ 3.11
- A reachable **Neo4j** instance (5.x driver) with the target database loaded
- An **Azure OpenAI** deployment of a chat model that supports **tool calling** and **structured output**

Main dependencies: `langgraph`, `langchain-core`, `langchain-openai`, `neo4j`, `pydantic`, `python-dotenv`.

---

## Installation

```bash
git clone https://github.com/alessandrofreuli/Cypher-Agent.git
cd Cypher-Agent

python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

pip install -e ".[dev]"          # core + ruff and pyright
# optional, for notebook graph rendering:
pip install -e ".[notebook]"     # ipython, pygraphviz
```

---

## Configuration

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
| --- | :---: | --- | --- |
| `NEO4J_URI` | ✅ | – | Bolt/Neo4j URI, e.g. `neo4j://localhost:7687` |
| `NEO4J_USERNAME` | ✅ | – | Neo4j user |
| `NEO4J_PASSWORD` | ✅ | – | Neo4j password |
| `NEO4J_DATABASE` | | `spine` | Database name |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | ✅ | – | Name of the Azure OpenAI deployment |
| `AZURE_OPENAI_ENDPOINT` | ✅ | – | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | ✅ | – | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | ✅ | – | Azure OpenAI API version |
| `CYPHER_AGENT_TOOLS_PATH` | | `data/tool_definitions.json` | Path to the tool definitions |
| `CYPHER_AGENT_SCHEMA_PATH` | | `data/schema.md` | Path to the schema description |
| `CYPHER_AGENT_MAX_STEP_ATTEMPTS` | | `5` | Max strategy attempts per step before the critic gives up |

Relative paths are resolved against the **repository root**, not the current working directory, so the agent behaves the same from a shell, an IDE or a notebook.

---

## Usage

### Command line

```bash
cypher-agent "Which persons have offences recorded in 2023?"
```

Options:

| Flag | Default | Description |
| --- | --- | --- |
| `--recursion-limit` | `30` | Maximum number of LangGraph super-steps at plan level. If exhausted, the agent reports that it cannot answer. |

The console shows the plan, each executed step (strategy, tool calls, critic verdict), the replanner's decisions and finally the answer.

### Python

```python
import asyncio
from cypher_agent.runner import question

asyncio.run(question("Find the person with the phone number '43832726839', then find where they live"))
```

In a notebook, simply `await question("...")`.

### Building the graphs directly

```python
from cypher_agent import build_plan_graph, build_step_graph

plan_app = build_plan_graph()   # compiled LangGraph
step_app = build_step_graph()
```

`notebooks/explore.ipynb` shows how to render both graphs as Mermaid diagrams and how to run example questions.

---

## The toolbox: `data/tool_definitions.json`

Each tool is a JSON object with a name, a description the LLM uses to choose it, a **parameterised Cypher query**, and its parameters:

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

> The query above is illustrative – see the file for the real ones.

At startup, `tools.py`:

1. builds a Pydantic args model per tool (supported types: `string`, `integer`, `number`, `boolean`; all parameters optional);
2. wraps the query in a LangChain `StructuredTool` that runs it in a Neo4j session on the configured database;
3. serialises rows to JSON (Neo4j types such as datetimes and nodes are coerced to strings);
4. returns errors as JSON (`{"error": ..., "tool": ...}`) so the critic can react instead of crashing the run;
5. generates a plain-text **tools summary** injected into the planner prompt.

Because the LLM only supplies *parameters*, never Cypher text, queries stay reviewable, predictable and safe from prompt-driven query injection.

**Tips for writing good tool descriptions:** say *when* to use the tool, what identifier it expects, and which sibling tool to prefer for adjacent cases. The planner relies on these hints heavily.

---

## The graph schema: `data/schema.md`

`schema.md` documents node labels (with property names, types and `required`/`optional` flags) and relationship types of the target graph. It is loaded verbatim into the planner prompt so plans reference entities and relationships that actually exist.

The included schema targets a `spine` database containing multi-label nodes across several domains, for example:

- **`POL`** – police records: `Arrest`, `Offence`, `Location`, `ChargeRelease`, `Person`
- **`HMPS`** – custody records: `CustodyStatus`, `CorrectionalFacility`, `JudicialSentence`, `MarkScarTattoo`, `Person`
- **`DFEX`** – digital-forensics extractions: `DeviceExtraction`, `Call`, `Contact`, `InstantMessage`, `Email`, `File`, `Journey`, `LocationEvent`, `CellTower`, `SIMCard`, `WirelessNetwork`, …
- **`ELR`** – electoral-roll-style records: `Person`, `Location`, `PhoneNumber`, `EmailAddress`

To adapt the agent to **your own graph**, replace `schema.md` with your schema and `tool_definitions.json` with your tools – no Python changes are required.

> ⚠️ `data/schema.md` is listed in `.gitignore`, so it is **not** part of a fresh clone. Generate or provide your own before running (for instance by exporting your Neo4j schema, e.g. with `apoc.meta.schema()` or `db.schema.visualization()`, and formatting it as Markdown).

---

## Included tools

The bundled `tool_definitions.json` ships 16 read-only tools:

| Domain | Tools |
| --- | --- |
| **Person lookup** | `get_person_details_by_exact_name`, `get_person_details_by_unsure_name`, `get_person_details_by_id` |
| **Addresses & phones** | `get_person_addresses_by_id`, `get_person_by_location_components`, `get_phone_number_by_person_id`, `get_person_id_by_phone_number` |
| **Offences & arrests** | `get_offence_by_person_id`, `get_arrests_by_person_id`, `get_person_by_arrest_id`, `get_offence_by_arrest_id`, `get_arrests_by_offence_id` |
| **Distinguishing marks** | `get_marks_scars_tattoos_by_person_id`, `get_persons_by_mark_scar_tattoo_filters` |
| **Custody** | `get_person_id_by_custody_type`, `get_custody_details_by_person_id` |

---

## Design notes

- **Plan / step separation.** The planner reasons about *what* to do; the step executor decides *how* (which tool, which inputs, or pure reasoning) and is held accountable by a critic.
- **Raw data never gets lost.** Summaries are lossy; the `artifacts` store keeps the exact output of every step so later filtering, joining and final reporting use real values.
- **Explicit failure is a feature.** Both the planner (`give_up`) and the replanner ("I cannot answer this question.") are instructed to fail honestly rather than fabricate an answer from adjacent data.
- **Bounded execution.** A per-step attempt cap and a plan-level recursion limit guarantee termination.
- **Flat structured-output schemas.** `Act` avoids `oneOf`, which Azure OpenAI structured output does not support.
- **Lazy, cached singletons.** The Neo4j driver, LLM, toolbox and chains are created on first use (`lru_cache`), so imports stay cheap.

---

## Development

```bash
ruff check .          # lint (line length 120)
ruff format .         # format
pyright               # type-check
```

Adding a new capability:

1. Add a tool entry to `data/tool_definitions.json` (name, description, query, parameters).
2. If new entities or relationships are involved, update `data/schema.md`.
3. Try it with `cypher-agent "..."` and inspect the trace; refine the tool description if the planner picks the wrong tool.

---

## Security notes

- Never commit `.env`; it is already in `.gitignore`. Rotate any credential that has been shared or pushed by mistake.
- Use a **read-only Neo4j user** for the agent. The bundled tools only read, but a least-privilege account is a cheap safety net.
- The graph data may include **sensitive personal information** (names, addresses, phone numbers, criminal and custody records, device extractions). Make sure that using it with an external LLM provider complies with your legal, contractual and data-protection obligations (e.g. GDPR) and that your Azure OpenAI deployment is configured accordingly.
- LLM output can be wrong. Treat answers as decision *support* and verify critical facts against source records.

---

## Limitations and roadmap

- Answers are limited to what the predefined tools can retrieve; new question types need new tools.
- Tool parameters are all optional and loosely typed (`string`, `integer`, `number`, `boolean`).
- Model temperature is fixed in `llm.py` and not yet configurable through the environment.
- No automated test suite or CI yet.
- Ideas: tool-level unit tests against a fixture graph, configurable model parameters, result caching, a web/API front-end, and observability via LangSmith or OpenTelemetry.

---

## License

No license file is currently included. Add one (e.g. MIT or Apache-2.0) before distributing or accepting contributions.