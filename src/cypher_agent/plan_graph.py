"""Plan-level graph: plan → agent (runs the step graph) → replan."""

from functools import lru_cache
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from cypher_agent.chains import get_planner, get_replanner
from cypher_agent.printing import print_header, print_step_node
from cypher_agent.schemas import Act, PlanExecute
from cypher_agent.step_graph import MAX_STEP_ATTEMPTS, build_step_graph


@lru_cache(maxsize=1)
def _step_app():
    return build_step_graph()


def plan_node(state: PlanExecute) -> dict[str, Any]:
    messages = [("user", state["input"])]
    context_parts: list[str] = []
    if state.get("knowledge"):
        context_parts.append(f"Accumulated knowledge so far:\n{state['knowledge']}")
    if state.get("replan_feedback"):
        context_parts.append(f"Replanner feedback on why the previous plan failed:\n{state['replan_feedback']}")
    if context_parts:
        context_parts.append(
            "Generate a different plan that addresses this feedback and builds on the existing knowledge."
        )
        messages.append(("system", "\n".join(context_parts)))
    plan = get_planner().invoke({"messages": messages})
    return {"plan": plan.steps, "replan_feedback": None}


def agent_node(state: PlanExecute) -> dict[str, Any]:
    step = state["plan"][0]

    print_header("AGENT NODE")

    if step.action.strip().lower() == "give_up":
        print(f"\n  ▶ Skipping execution: [{step.action}] (give_up)")
        return {"past_steps": [(step.action, step.description)]}

    plan_str = "\n".join(f"{i + 1}. [{s.action}] {s.description}" for i, s in enumerate(state["plan"]))
    task = (
        f"Full plan:\n{plan_str}\n"
        f"Execute step 1 only:\n"
        f"  Action: {step.action}\n"
        f"  What to do: {step.description}\n"
        f"  Why: {step.reason}"
    )

    print(f"\n  ▶ Executing step: [{step.action}]")

    step_result: dict[str, Any] = {"result": None, "step_artifact": None}
    for chunk in _step_app().stream(
        {
            "task": task,
            "knowledge": state.get("knowledge", ""),
            "artifacts": state.get("artifacts", {}),
            "messages": [],
            "feedback": None,
            "result": None,
            "step_artifact": None,
            "attempts": 0,
        },
        config={"recursion_limit": 3 * MAX_STEP_ATTEMPTS + 2},
    ):
        for node, updates in chunk.items():
            if node == "__end__":
                continue
            print_step_node(node, updates)
            if updates.get("result") is not None:
                step_result["result"] = updates["result"]
            if updates.get("step_artifact") is not None:
                step_result["step_artifact"] = updates["step_artifact"]

    return {
        "past_steps": [(step.action, step_result["result"])],
        "artifacts": {step.action: step_result.get("step_artifact")},
    }


def replan_node(state: PlanExecute) -> dict[str, Any]:
    if state["past_steps"] and state["past_steps"][-1][0].strip().lower() == "give_up":
        reason = state["past_steps"][-1][1]
        return {"response": f"I cannot answer this question. {reason}"}
    last_action = state["past_steps"][-1][0] if state.get("past_steps") else None
    last_step_artifact = state.get("artifacts", {}).get(last_action) if last_action else None
    replanner_input = {
        **state,
        "last_step_artifact": last_step_artifact if last_step_artifact is not None else "(none)",
    }
    output: Act = get_replanner().invoke(replanner_input)
    result: dict[str, Any] = {"knowledge": output.knowledge}
    if output.response:
        result["response"] = output.response
    elif output.replan_feedback:
        result["replan_feedback"] = output.replan_feedback
    else:
        result["plan"] = output.steps or []
        result["replan_feedback"] = None
    return result


def should_end(state: PlanExecute) -> Literal["agent_node", "plan_node", "__end__"]:
    if state.get("response"):
        return END
    if state.get("replan_feedback"):
        return "plan_node"
    return "agent_node"


def build_plan_graph():
    workflow = StateGraph(PlanExecute)
    workflow.add_node("plan_node", plan_node)
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("replan_node", replan_node)
    workflow.add_edge(START, "plan_node")
    workflow.add_edge("plan_node", "agent_node")
    workflow.add_edge("agent_node", "replan_node")
    workflow.add_conditional_edges("replan_node", should_end, ["agent_node", "plan_node", END])
    return workflow.compile()  # type: ignore[return-value]
