"""Pretty-printing helpers for streamed graph updates."""

from langchain_core.messages import AIMessage, ToolMessage

_BAR = "─" * 62


def indent(text: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in str(text).splitlines())


def fmt_steps(steps: list) -> str:
    lines: list[str] = []
    for i, s in enumerate(steps, 1):
        lines.append(f"  {i}. [{s.action}]")
        lines.append(f"     What : {s.description}")
        lines.append(f"     Why  : {s.reason}")
    return "\n".join(lines)


def print_header(label: str) -> None:
    print(f"\n┌{_BAR}┐")
    print(f"│  {label:<60}│")
    print(f"└{_BAR}┘")


def fmt_tool_calls(tool_calls: list[dict]) -> str:
    return "\n".join(f"  → {call['name']}({call['args']})" for call in tool_calls)


def print_step_node(node: str, updates: dict) -> None:
    """Pretty-print updates coming from the step-level graph (strategy/tool/critic)."""
    label = f"  {node.replace('_', ' ')}"
    print(f"\n  ┄┄ {label} " + "┄" * max(0, 50 - len(label)))

    for m in updates.get("messages") or []:
        if isinstance(m, AIMessage):
            if getattr(m, "tool_calls", None):
                print("  Tool calls requested:")
                print(fmt_tool_calls(m.tool_calls))
            elif m.content:
                print("  Reasoning output (candidate step_artifact):")
                print(indent(m.content, 4))
        elif isinstance(m, ToolMessage):
            print(f"  Tool result [{m.name}]:")
            print(indent(m.content, 4))

    if updates.get("feedback"):
        print("  ⚠ Critic feedback (retry needed):")
        print(indent(updates["feedback"], 4))

    if updates.get("result"):
        print("  ✓ Step result accepted:")
        print(indent(updates["result"], 4))

    if "attempts" in updates:
        print(f"  Attempt #{updates['attempts']}")


def print_node(node: str, updates: dict) -> None:
    if node != "agent_node":
        print_header(node.replace("_", " ").upper())
    if updates.get("plan"):
        steps = updates["plan"]
        noun = "step" if len(steps) == 1 else "steps"
        print(f"\nPlan ({len(steps)} {noun}):")
        print(fmt_steps(steps))
    if updates.get("past_steps"):
        for action, result in updates["past_steps"]:
            print(f"\nExecuted : {action}")
            print(indent(result, 4))
    if updates.get("artifacts"):
        for action, artifact in updates["artifacts"].items():
            size = len(str(artifact)) if artifact is not None else 0
            print(f"\nArtifact stored ({action}): {size} chars raw data")
    if updates.get("knowledge"):
        print("\nKnowledge :")
        print(indent(updates["knowledge"], 4))
    if updates.get("replan_feedback"):
        print("\n⚠ Replan feedback :")
        print(indent(updates["replan_feedback"], 4))
    if updates.get("response"):
        gave_up = updates["response"].startswith("I cannot answer")
        print(f"\n{'✗ Cannot answer' if gave_up else '✓ Final answer'} :")
        print(indent(updates["response"], 4))
