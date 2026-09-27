"""Entry point for running a question through the plan-level graph."""

from functools import lru_cache

from langgraph.errors import GraphRecursionError

from cypher_agent.plan_graph import build_plan_graph
from cypher_agent.printing import indent, print_node
from cypher_agent.schemas import PlanExecute


@lru_cache(maxsize=1)
def get_app():
    return build_plan_graph()


def initial_state(q: str) -> PlanExecute:
    return {
        "input": q,
        "plan": [],
        "past_steps": [],
        "knowledge": "",
        "artifacts": {},
        "response": None,
        "replan_feedback": None,
    }


async def question(q: str, recursion_limit: int = 30) -> None:
    print(f"Question: {q}")
    try:
        async for chunk in get_app().astream(
            initial_state(q), config={"recursion_limit": recursion_limit}
        ):
            for node, updates in chunk.items():
                if node != "__end__":
                    print_node(node, updates)
    except GraphRecursionError:
        print("\n✗ Cannot answer :")
        print(
            indent(
                "I cannot answer this question. Attempt budget exhausted without reaching a complete answer.",
                4,
            )
        )
    print(f"\n{'═' * 64}")
