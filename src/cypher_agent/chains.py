from functools import lru_cache
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from cypher_agent.llm import get_model
from cypher_agent.prompts import (
    CRITIC_SYSTEM_PROMPT,
    PLANNER_SYSTEM_TEMPLATE,
    REPLANNER_PROMPT_TEMPLATE,
)
from cypher_agent.schemas import Act, Plan, StepVerdict
from cypher_agent.tools import get_toolbox, load_db_schema


@lru_cache(maxsize=1)
def get_model_with_tools() -> Any:
    tools, _, _ = get_toolbox()
    return get_model().bind_tools(tools)  # type: ignore[arg-type]


@lru_cache(maxsize=1)
def get_planner() -> Any:
    _, _, tools_summary = get_toolbox()
    system = PLANNER_SYSTEM_TEMPLATE.format(db_schema=load_db_schema(), tools_summary=tools_summary)
    prompt = ChatPromptTemplate.from_messages([("system", system), ("user", "{messages}")])
    return prompt | get_model_with_tools().with_structured_output(Plan)


@lru_cache(maxsize=1)
def get_replanner() -> Any:
    prompt = ChatPromptTemplate.from_template(REPLANNER_PROMPT_TEMPLATE)
    return prompt | get_model_with_tools().with_structured_output(Act)


@lru_cache(maxsize=1)
def get_critic() -> Any:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CRITIC_SYSTEM_PROMPT),
            (
                "user",
                "Task: {task}\nKnowledge: {knowledge}\n"
                "Raw artifacts available before this step:\n{artifacts}\n"
                "Messages so far: {messages}",
            ),
        ]
    )
    return prompt | get_model().with_structured_output(StepVerdict)
