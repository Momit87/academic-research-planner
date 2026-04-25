"""
llm/workflow/research_planner/graph.py
=======================================
StateGraph assembly for the Research Planner workflow.

Graph structure:
    START
      │
      ▼
    [should_summarize] ──── "summarize" ────► [summarize_agent]
      │                                              │
      │ "main"                                       │
      ▼                                              ▼
    [main_agent] ◄─────────────────────────────────┘
      │
      ├── has tool_calls ──► [tools] ──► [main_agent]  (loop, max N=3)
      │
      └── no tool_calls ──► END

Two agents only: main_agent + summarize_agent.
All tools bound to main_agent simultaneously.
See DECISIONS.md D-003.

Note on imports:
    main_agent and summarize_agent nodes are imported lazily
    to avoid circular imports at module load time.
    They are fully initialized when build_graph() is called.
"""

from functools import lru_cache

from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from core.logging import get_logger
from llm.workflow.research_planner.graph_state import ResearchPlannerState
from llm.workflow.research_planner.node.should_summarize import (
    ROUTE_MAIN,
    ROUTE_SUMMARIZE,
    should_summarize,
)

logger = get_logger(__name__)

# Node name constants — avoids magic strings scattered across files
NODE_SUMMARIZE_AGENT = "summarize_agent"
NODE_MAIN_AGENT = "main_agent"
NODE_TOOLS = "tools"


def _route_after_main_agent(state: ResearchPlannerState) -> str:
    """
    Conditional edge after main_agent.

    If the last message contains tool_calls → route to tools node.
    If no tool_calls OR tool_call_rounds >= MAX → route to END.

    This implements the soft cap (DECISIONS.md D-006):
    After MAX_TOOL_ROUNDS, force END even if tool_calls present.
    The finalization mode prompt injection handles the LLM side.
    """
    from core.config import get_settings
    settings = get_settings()

    # Check tool call round limit
    if state.tool_call_rounds >= settings.max_tool_rounds:
        logger.info(
            "Tool round limit reached — forcing END",
            extra={
                "thread_id": state.thread_id,
                "tool_call_rounds": state.tool_call_rounds,
                "max": settings.max_tool_rounds,
            }
        )
        return END

    # Check if last message has tool calls
    if not state.messages:
        return END

    last_message = state.messages[-1]
    has_tool_calls = (
        hasattr(last_message, "tool_calls") and
        bool(last_message.tool_calls)
    )

    if has_tool_calls:
        return NODE_TOOLS

    return END


def build_graph(checkpointer=None) -> CompiledStateGraph:
    """
    Build and compile the Research Planner StateGraph.

    Imports agent nodes lazily to avoid circular imports.
    Called once at application startup (or per-request in tests).

    Args:
        checkpointer: LangGraph checkpointer instance.
                      Pass AsyncShallowPostgresSaver in production.
                      Pass None for testing without persistence.

    Returns:
        CompiledStateGraph ready for .ainvoke() and .astream()
    """
    # Lazy imports — agents import tools which import state
    # All imports here to break any circular dependency chains
    from langchain_core.tools import BaseTool
    from langgraph.prebuilt import ToolNode

    from llm.workflow.research_planner.node.main_agent import main_agent_node
    from llm.workflow.research_planner.node.summarize_agent import summarize_agent_node
    from llm.workflow.research_planner.tool.query_corpus import query_corpus
    from llm.workflow.research_planner.tool.firecrawl_search import firecrawl_search
    from llm.workflow.research_planner.tool.generate_discovery_deliverable import (
        generate_discovery_deliverable,
    )
    from llm.workflow.research_planner.tool.generate_clustering_deliverable import (
        generate_clustering_deliverable,
    )
    from llm.workflow.research_planner.tool.generate_gap_analysis_deliverable import (
        generate_gap_analysis_deliverable,
    )
    from llm.workflow.research_planner.tool.generate_writing_outline_deliverable import (
        generate_writing_outline_deliverable,
    )

    # All tools bound to main_agent simultaneously
    all_tools: list[BaseTool] = [
        query_corpus,
        firecrawl_search,
        generate_discovery_deliverable,
        generate_clustering_deliverable,
        generate_gap_analysis_deliverable,
        generate_writing_outline_deliverable,
    ]

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------
    graph = StateGraph(ResearchPlannerState)

    # Add nodes (should_summarize is a pure router — no node needed)
    graph.add_node(NODE_SUMMARIZE_AGENT, summarize_agent_node)
    graph.add_node(NODE_MAIN_AGENT, main_agent_node)
    graph.add_node(NODE_TOOLS, ToolNode(all_tools))

    # ------------------------------------------------------------------
    # Add edges
    # ------------------------------------------------------------------

    # Entry point: route directly from START via should_summarize
    graph.add_conditional_edges(
        START,
        should_summarize,
        {
            ROUTE_SUMMARIZE: NODE_SUMMARIZE_AGENT,
            ROUTE_MAIN: NODE_MAIN_AGENT,
        }
    )

    # After summarize_agent → always go to main_agent
    graph.add_edge(NODE_SUMMARIZE_AGENT, NODE_MAIN_AGENT)

    # After main_agent → tools (if tool_calls) OR END
    graph.add_conditional_edges(
        NODE_MAIN_AGENT,
        _route_after_main_agent,
        {
            NODE_TOOLS: NODE_TOOLS,
            END: END,
        }
    )

    # After tools → back to main_agent (the loop)
    graph.add_edge(NODE_TOOLS, NODE_MAIN_AGENT)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=[],   # no human-in-the-loop interrupts (yet)
        interrupt_after=[],
    )

    logger.info(
        "Research Planner graph compiled",
        extra={"checkpointer": type(checkpointer).__name__}
    )

    return compiled


async def get_graph(checkpointer=None) -> CompiledStateGraph:
    """
    Async wrapper for graph construction.

    Used in FastAPI lifespan and request handlers.
    The checkpointer is set up externally and passed in.

    Args:
        checkpointer: initialized AsyncShallowPostgresSaver

    Returns:
        CompiledStateGraph
    """
    return build_graph(checkpointer=checkpointer)