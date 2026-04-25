"""
llm/workflow/research_planner/tool/generate_clustering_deliverable.py
======================================================================
Generates the Phase 2 Clustering deliverable.

Prerequisite: discovery_deliverable must exist in state.
InjectedState reads: thread_id, discovery_deliverable

Used by: main_agent (bound tool)
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from core.llm_factory import get_deliverable_generator_llm
from core.logging import get_logger, timer
from llm.llm_schema.deliverables import ClusteringDeliverable
from llm.workflow.research_planner.graph_state import ResearchPlannerState
from utils.markdown_renderer import render_clustering_markdown

logger = get_logger(__name__)


@tool
async def generate_clustering_deliverable(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[ResearchPlannerState, InjectedState()],
) -> Command:
    """
    Generate the Clustering phase deliverable.

    Call this when you have proposed a taxonomy of the corpus and
    the user has confirmed or corrected the cluster structure.

    Requires: Discovery deliverable must be accepted first.
    """
    # Prerequisite check
    if state.discovery_deliverable is None:
        return Command(update={"messages": [ToolMessage(
            content="Cannot generate clustering deliverable: discovery deliverable not found.",
            tool_call_id=tool_call_id,
        )]})

    if "discovery" not in state.accepted_deliverables:
        return Command(update={"messages": [ToolMessage(
            content="Cannot generate clustering deliverable: discovery phase not accepted yet.",
            tool_call_id=tool_call_id,
        )]})

    with timer(
        "generate_clustering_deliverable",
        logger,
        extra={"thread_id": state.thread_id}
    ):
        llm = get_deliverable_generator_llm()
        structured_llm = llm.with_structured_output(ClusteringDeliverable)

        discovery = state.discovery_deliverable

        prompt = f"""Generate a Clustering deliverable for this research corpus.

Discovery Context:
- Field: {discovery.field_summary}
- Research Intent: {discovery.research_intent}
- Corpus: {discovery.corpus_overview.total_sources} sources
- Dominant themes: {', '.join(discovery.corpus_overview.dominant_themes)}

Ingested sources:
{_format_sources(state)}

Based on the conversation history and corpus content, identify 3-7 thematic
clusters, their relationships, and any orphan sources.
Generate a comprehensive Clustering deliverable."""

        deliverable: ClusteringDeliverable = await structured_llm.ainvoke(prompt)
        markdown = render_clustering_markdown(deliverable)

    logger.info(
        "Clustering deliverable generated",
        extra={"thread_id": state.thread_id}
    )

    return Command(
        update={
            "messages": [ToolMessage(
                content="Clustering deliverable generated successfully.",
                tool_call_id=tool_call_id,
            )],
            "clustering_deliverable": deliverable,
            "deliverables_markdown": {
                **state.deliverables_markdown,
                "clustering": markdown,
            },
            "suggested_next_phase": "gap_analysis",
        }
    )


def _format_sources(state: ResearchPlannerState) -> str:
    if not state.ingested_sources:
        return "No sources available."
    return "\n".join(
        f"- {s.origin} ({s.source_type.value})"
        for s in state.ingested_sources
    )