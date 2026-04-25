"""
llm/workflow/research_planner/tool/generate_discovery_deliverable.py
=====================================================================
Generates the Phase 1 Discovery deliverable.

InjectedState reads: thread_id, field, sub_field,
                     research_intent, ingested_sources

Inner LLM call produces DiscoveryDeliverable via structured output.
Markdown renderer converts it to display string.

Used by: main_agent (bound tool)
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from core.llm_factory import get_deliverable_generator_llm
from core.logging import get_logger, timer
from llm.llm_schema.deliverables import DiscoveryDeliverable
from llm.workflow.research_planner.graph_state import ResearchPlannerState
from utils.markdown_renderer import render_discovery_markdown

logger = get_logger(__name__)


@tool
async def generate_discovery_deliverable(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[ResearchPlannerState, InjectedState()],
) -> Command:
    """
    Generate the Discovery phase deliverable.

    Call this when you have gathered sufficient information about:
    - The researcher's field and sub-field
    - Their research intent/question
    - Target output type, venue, and constraints
    - An overview of the uploaded corpus

    Do NOT call this if discovery is already in accepted_deliverables
    unless the user wants to refine it.
    """
    with timer(
        "generate_discovery_deliverable",
        logger,
        extra={"thread_id": state.thread_id}
    ):
        llm = get_deliverable_generator_llm()
        structured_llm = llm.with_structured_output(DiscoveryDeliverable)

        # Build context from state
        sources_summary = _summarize_sources(state)

        prompt = f"""Generate a Discovery deliverable for this research session.

Research Context:
- Field: {state.field or 'To be determined'}
- Sub-field: {state.sub_field or 'To be determined'}
- Research intent: {state.research_intent or 'To be determined'}
- Target output: {state.target_output or 'Not specified'}
- Deadline: {state.deadline or 'Not specified'}

Corpus:
{sources_summary}

Conversation history is available above. Use it to populate all fields accurately.
Generate a comprehensive Discovery deliverable based on everything gathered."""

        deliverable: DiscoveryDeliverable = await structured_llm.ainvoke(prompt)
        markdown = render_discovery_markdown(deliverable)

    logger.info(
        "Discovery deliverable generated",
        extra={"thread_id": state.thread_id}
    )

    return Command(
        update={
            "messages": [ToolMessage(
                content="Discovery deliverable generated successfully.",
                tool_call_id=tool_call_id,
            )],
            "discovery_deliverable": deliverable,
            "deliverables_markdown": {
                **state.deliverables_markdown,
                "discovery": markdown,
            },
            "suggested_next_phase": "clustering",
        }
    )


def _summarize_sources(state: ResearchPlannerState) -> str:
    if not state.ingested_sources:
        return "No sources ingested yet."
    lines = []
    for src in state.ingested_sources:
        lines.append(
            f"- {src.source_type.value}: {src.origin} "
            f"({src.chunk_count} chunks, langs: {', '.join(src.languages)})"
        )
    return "\n".join(lines)