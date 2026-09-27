"""Step-level graph: Strategy → Tool execution → Critic."""

import json
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from cypher_agent.chains import get_critic, get_model_with_tools
from cypher_agent.config import settings
from cypher_agent.prompts import STRATEGY_SYSTEM_PROMPT
from cypher_agent.schemas import StepState, StepVerdict
from cypher_agent.tools import get_toolbox

MAX_STEP_ATTEMPTS = settings.max_step_attempts


def strategy_node(state: StepState) -> dict[str, Any]:
    context = f"Task: {state['task']}\nCurated knowledge (prose): {state['knowledge']}"
    if state.get("artifacts"):
        context += (
            "\nRaw artifacts from previous steps (this is the actual data — use it for any filtering, "
            "matching, or aggregation; do not rely only on the prose knowledge above):\n"
            f"{json.dumps(state['artifacts'], default=str)}"
        )
    messages: list[Any] = [("system", STRATEGY_SYSTEM_PROMPT), ("user", context), *state["messages"]]
    if state.get("feedback"):
        messages.append(
            (
                "user",
                f"Critic feedback on your previous attempt above: {state['feedback']}\n"
                "Correct this specific issue and produce a new, corrected result.",
            )
        )
    ai_msg = cast(AIMessage, get_model_with_tools().invoke(messages))
    update: dict[str, Any] = {
        "messages": [ai_msg],
        "feedback": None,
        "attempts": state.get("attempts", 0) + 1,
    }
    if not ai_msg.tool_calls:
        update["step_artifact"] = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
    return update


def strategy_route(state: StepState) -> Literal["tool_execution_node", "critic_node"]:
    last = state["messages"][-1]
    return "tool_execution_node" if getattr(last, "tool_calls", None) else "critic_node"


def tool_execution_node(state: StepState) -> dict[str, Any]:
    _, tools_by_name, _ = get_toolbox()
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    results: list[ToolMessage] = []
    raw_parts: list[str] = []
    for call in tool_calls:
        tool_fn = tools_by_name.get(call["name"])
        if tool_fn is None:
            content = json.dumps({"error": f"Unknown tool: {call['name']}"})
        else:
            try:
                content = tool_fn.invoke(call["args"])
            except Exception as e:
                content = json.dumps({"error": f"{type(e).__name__}: {e}", "tool": call["name"]})
        results.append(ToolMessage(content=str(content), tool_call_id=call["id"], name=call["name"]))
        raw_parts.append(f"[{call['name']}({call['args']})] -> {content}")
    previous = state.get("step_artifact") or ""
    new_chunk = "\n".join(raw_parts)
    combined = f"{previous}\n{new_chunk}" if previous else new_chunk
    return {"messages": results, "step_artifact": combined}


def critic_node(state: StepState) -> dict[str, Any]:
    if state.get("attempts", 0) >= MAX_STEP_ATTEMPTS:
        return {
            "result": (
                f"Step failed after {MAX_STEP_ATTEMPTS} attempts. "
                f"Last feedback: {state.get('feedback') or 'none'}"
            ),
            "step_artifact": state.get("step_artifact"),
        }
    verdict: StepVerdict = get_critic().invoke(
        {
            "task": state["task"],
            "knowledge": state["knowledge"],
            "artifacts": json.dumps(state.get("artifacts", {}), default=str),
            "messages": state["messages"],
        }
    )
    if verdict.result:
        return {"result": verdict.result}
    return {
        "feedback": verdict.feedback
        or "Retry with a different tool, different inputs, or go through the raw artifacts more thoroughly."
    }


def step_should_end(state: StepState) -> Literal["strategy_node", "__end__"]:
    return END if state.get("result") else "strategy_node"


def build_step_graph():
    workflow = StateGraph(StepState)
    workflow.add_node("strategy_node", strategy_node)
    workflow.add_node("tool_execution_node", tool_execution_node)
    workflow.add_node("critic_node", critic_node)
    workflow.add_edge(START, "strategy_node")
    workflow.add_conditional_edges("strategy_node", strategy_route, ["tool_execution_node", "critic_node"])
    workflow.add_edge("tool_execution_node", "critic_node")
    workflow.add_conditional_edges("critic_node", step_should_end, ["strategy_node", END])
    return workflow.compile()  # type: ignore[return-value]
