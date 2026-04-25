"""
llm/workflow/research_planner/node/should_summarize.py
=======================================================
Deterministic conditional node that decides whether to run
the summarize_agent before the main_agent.

This is NOT an LLM call. It is pure Python logic that reads
approx_prompt_tokens from state and returns a routing string.

Routing:
    "summarize"  → summarize_agent node
    "main"       → main_agent node (directly)

Decision threshold: settings.summarize_token_threshold (default 12000)

Why this is a separate node (not middleware or inline logic):
    - Makes the compression boundary visible in the graph
    - State mutation (token count update) happens at a clear point
    - Easy to test in isolation
    See DECISIONS.md D-003.
"""

from core.config import get_settings
from core.logging import get_logger
from llm.workflow.research_planner.graph_state import ResearchPlannerState

logger = get_logger(__name__)

# Routing return values — must match edge labels in graph.py
ROUTE_SUMMARIZE = "summarize"
ROUTE_MAIN = "main"


def should_summarize(state: ResearchPlannerState) -> str:
    """
    Conditional edge function for the LangGraph router.

    Reads approx_prompt_tokens from state.
    Returns routing string consumed by the graph's conditional edge.

    Args:
        state: current ResearchPlannerState

    Returns:
        "summarize" if tokens exceed threshold, "main" otherwise
    """
    settings = get_settings()
    threshold = settings.summarize_token_threshold
    current_tokens = state.approx_prompt_tokens

    if current_tokens > threshold:
        logger.info(
            "Routing to summarize_agent",
            extra={
                "thread_id": state.thread_id,
                "current_tokens": current_tokens,
                "threshold": threshold,
            }
        )
        return ROUTE_SUMMARIZE

    logger.debug(
        "Routing to main_agent",
        extra={
            "thread_id": state.thread_id,
            "current_tokens": current_tokens,
            "threshold": threshold,
        }
    )
    return ROUTE_MAIN