import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


def merge_artifacts(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Reducer: later steps add/overwrite entries, nothing is ever silently dropped."""
    merged = dict(existing)
    merged.update(new)
    return merged


class PlanStep(BaseModel):
    action: str = Field(description="Short label for the step (e.g. 'get_person_details').")
    description: str = Field(description="Exactly what to do and which tool to call with which inputs.")
    reason: str = Field(description="Why this step is needed and what it contributes to answering the objective.")


class Plan(BaseModel):
    steps: list[PlanStep] = Field(description="Ordered list of structured steps to execute.")


class PlanExecute(TypedDict):
    input: str
    plan: list[PlanStep]
    past_steps: Annotated[list[tuple], operator.add]  # type: ignore[misc]
    knowledge: str  # curated prose summary, maintained by the replanner
    artifacts: Annotated[dict[str, Any], merge_artifacts]  # raw, unsummarized output per step action
    response: str | None
    replan_feedback: str | None


# Flat schema avoids oneOf, which Azure OpenAI structured output does not support
class Act(BaseModel):
    knowledge: str = Field(
        description=(
            "Updated knowledge base after evaluating this step. "
            "Preserve all useful facts from previous knowledge and incorporate new findings. "
            "This is a curated PROSE summary for planning purposes only — raw data lives in artifacts, not here."
        )
    )
    response: str | None = Field(
        default=None,
        description=(
            "Final answer derived from the knowledge base AND the raw artifact of the last step. Set only when "
            "the objective is fully answered. Must include the actual concrete data requested (e.g. the full list "
            "of matching records with their ids/dates/fields), not only counts or a prose summary. Write it as "
            "clear natural-language prose for a human reader (sentences / bullet points / a simple list per "
            "record) — NEVER as raw JSON, a Python dict/list repr, or any other machine-formatted structure."
        ),
    )
    steps: list[PlanStep] | None = Field(
        default=None,
        description="Remaining steps if the current plan is progressing correctly. List only steps not yet executed.",
    )
    replan_feedback: str | None = Field(
        default=None,
        description=(
            "Critical feedback for the planner if the plan is fundamentally flawed and needs to be "
            "regenerated from scratch. Explain what failed and what a better plan should do instead."
        ),
    )


class StepState(TypedDict):
    task: str
    knowledge: str  # curated prose summary (read-only context)
    artifacts: dict[str, Any]  # raw artifacts from previous steps (read-only context)
    messages: Annotated[list[BaseMessage], add_messages]  # AIMessage(tool_calls) + ToolMessage(results) history
    feedback: str | None  # Critic's feedback to Strategy, if a retry is needed
    result: str | None  # prose result for this step (goes into past_steps/knowledge)
    step_artifact: str | None  # raw output of this step (goes into artifacts, untouched)
    attempts: int  # number of Strategy attempts made so far


class StepVerdict(BaseModel):
    result: str | None = Field(
        default=None,
        description=(
            "Prose summary answering the task for this step, if the available data already answers it. "
            "Set only this OR feedback, never both."
        ),
    )
    feedback: str | None = Field(
        default=None,
        description=(
            "Explanation of what went wrong (wrong tool, wrong input, empty/irrelevant result, or the reasoning "
            "step did not actually process the raw artifacts) and what the strategy should try instead. "
            "Set only if a retry is needed."
        ),
    )
