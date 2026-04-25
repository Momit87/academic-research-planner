"""
llm/workflow/research_planner/tool/generate_gap_analysis_deliverable.py
========================================================================
Generates the Phase 3 Gap Analysis deliverable.

Prerequisite: clustering_deliverable must exist in state.
InjectedState reads: thread_id, discovery_deliverable,
                     clustering_deliverable

Used by: main_agent (bound tool)
"""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from core.llm_factory import get_deliverable_generator_llm
from core.logging import get_logger, timer
from llm.llm_schema.deliverables import GapAnalysisDeliverable
from llm.workflow.research_planner.graph_state import ResearchPlannerState
from utils.markdown_renderer import render_gap_analysis_markdown

logger = get_logger(__name__)


@tool
async def generate_gap_analysis_deliverable(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[ResearchPlannerState, InjectedState()],
) -> Command:
    """
    Generate the Gap Analysis phase deliverable.

    Call this when you have identified research gaps from the cluster
    structure and the user has indicated which gap to pursue.

    Requires: Clustering deliverable must be accepted first.
    """
    # Prerequisite check
    if state.clustering_deliverable is None:
        return Command(update={"messages": [ToolMessage(
            content="Cannot generate gap analysis: clustering deliverable not found.",
            tool_call_id=tool_call_id,
        )]})

    if "clustering" not in state.accepted_deliverables:
        return Command(update={"messages": [ToolMessage(
            content="Cannot generate gap analysis: clustering phase not accepted yet.",
            tool_call_id=tool_call_id,
        )]})

    with timer(
        "generate_gap_analysis_deliverable",
        logger,
        extra={"thread_id": state.thread_id}
    ):
        llm = get_deliverable_generator_llm()
        structured_llm = llm.with_structured_output(GapAnalysisDeliverable)

        discovery = state.discovery_deliverable
        clustering = state.clustering_deliverable

        cluster_summary = "\n".join(
            f"- {c.label}: {c.description}"
            for c in clustering.clusters
        )

        relationships = "\n".join(
            f"- {r.cluster_a} {r.relationship_type} {r.cluster_b}: {r.description}"
            for r in clustering.cross_cluster_relationships
        )

        prompt = f"""Generate a Gap Analysis deliverable for this research.

Research Intent: {discovery.research_intent}

Clusters:
{cluster_summary}

Cross-cluster relationships:
{relationships or 'None identified'}

Orphan sources: {', '.join(clustering.orphan_sources) or 'None'}

Based on the cluster structure and conversation history:
1. Identify 3-6 specific research gaps
2. Classify each gap (empirical/theoretical/methodological/application/replication)
3. Rate feasibility and novelty
4. Summarize related work
5. If the user has indicated a preferred gap, set chosen_gap accordingly

Generate a comprehensive Gap Analysis deliverable."""

        deliverable: GapAnalysisDeliverable = await structured_llm.ainvoke(prompt)
        markdown = render_gap_analysis_markdown(deliverable)

    logger.info(
        "Gap analysis deliverable generated",
        extra={"thread_id": state.thread_id}
    )

    return Command(
        update={
            "messages": [ToolMessage(
                content="Gap analysis deliverable generated successfully.",
                tool_call_id=tool_call_id,
            )],
            "gap_analysis_deliverable": deliverable,
            "deliverables_markdown": {
                **state.deliverables_markdown,
                "gap_analysis": markdown,
            },
            "suggested_next_phase": "writing_outline",
        }
    )