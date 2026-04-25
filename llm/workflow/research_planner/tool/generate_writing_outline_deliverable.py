"""
llm/workflow/research_planner/tool/generate_writing_outline_deliverable.py
===========================================================================
Generates the Phase 4 Writing Outline deliverable.

This IS the final research roadmap.
Prerequisite: gap_analysis_deliverable must exist in state.
InjectedState reads: thread_id, discovery_deliverable,
                     clustering_deliverable, gap_analysis_deliverable

Used by: main_agent (bound tool)
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from core.llm_factory import get_deliverable_generator_llm
from core.logging import get_logger, timer
from llm.llm_schema.deliverables import WritingOutlineDeliverable
from llm.workflow.research_planner.graph_state import ResearchPlannerState
from utils.markdown_renderer import render_writing_outline_markdown

logger = get_logger(__name__)


@tool
async def generate_writing_outline_deliverable(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[ResearchPlannerState, InjectedState()],
) -> Command:
    """
    Generate the Writing Outline deliverable — the final research roadmap.

    Call this when:
    - Gap analysis is accepted
    - A specific gap has been chosen
    - The user is ready to plan their paper structure

    Requires: Gap Analysis deliverable must be accepted first.
    """
    # Prerequisite check
    if state.gap_analysis_deliverable is None:
        return Command(update={"messages": [ToolMessage(
            content="Cannot generate writing outline: gap analysis deliverable not found.",
            tool_call_id=tool_call_id,
        )]})

    if "gap_analysis" not in state.accepted_deliverables:
        return Command(update={"messages": [ToolMessage(
            content="Cannot generate writing outline: gap analysis phase not accepted yet.",
            tool_call_id=tool_call_id,
        )]})

    with timer(
        "generate_writing_outline_deliverable",
        logger,
        extra={"thread_id": state.thread_id}
    ):
        llm = get_deliverable_generator_llm()
        structured_llm = llm.with_structured_output(WritingOutlineDeliverable)

        discovery = state.discovery_deliverable
        clustering = state.clustering_deliverable
        gap = state.gap_analysis_deliverable

        chosen_gap_text = "Not yet selected"
        if gap.chosen_gap:
            chosen_gap_text = (
                f"{gap.chosen_gap.title}: {gap.chosen_gap.description}"
            )

        cluster_themes = ", ".join(
            c.label for c in clustering.clusters
        )

        sources_for_citation = "\n".join(
            f"- {s.origin}"
            for s in state.ingested_sources[:20]
        )

        prompt = f"""Generate a Writing Outline deliverable for this research paper.

Research Foundation:
- Field: {discovery.field_summary}
- Research Intent: {discovery.research_intent}
- Target Output: {discovery.target_output}
- Citation Style: {discovery.constraints.citation_style or 'APA'}

Corpus Themes: {cluster_themes}

Chosen Research Gap: {chosen_gap_text}
Rationale: {gap.rationale or 'Not specified'}

Available Sources for Citation:
{sources_for_citation}

Generate:
1. Three title options (descriptive to provocative)
2. 150-250 word abstract draft
3. Complete section outline with paragraph intents and citations
4. Use the specified citation style

This is the final research roadmap — make it comprehensive and actionable."""

        deliverable: WritingOutlineDeliverable = await structured_llm.ainvoke(
            prompt
        )
        markdown = render_writing_outline_markdown(deliverable)

    logger.info(
        "Writing outline deliverable generated",
        extra={"thread_id": state.thread_id}
    )

    return Command(
        update={
            "messages": [ToolMessage(
                content="Writing outline deliverable generated successfully.",
                tool_call_id=tool_call_id,
            )],
            "writing_outline_deliverable": deliverable,
            "deliverables_markdown": {
                **state.deliverables_markdown,
                "writing_outline": markdown,
            },
            "suggested_next_phase": None,
        }
    )
