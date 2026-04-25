"""
llm/workflow/research_planner/tool/query_corpus.py
====================================================
RAG retrieval tool — searches the user's private Qdrant corpus.

InjectedState: reads thread_id from graph state automatically.
The LLM only sees the `query` parameter.

Used by: main_agent (bound tool)
"""

from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from core.logging import get_logger, timer
from llm.workflow.research_planner.graph_state import ResearchPlannerState
from service.embedder import Embedder
from service.qdrant import QdrantService
from utils.filtering_logic import apply_default_filters

logger = get_logger(__name__)

_embedder = Embedder()
_qdrant = QdrantService()


@tool
async def query_corpus(
    query: str,
    state: Annotated[ResearchPlannerState, InjectedState()],
) -> str:
    """
    Search the user's uploaded research corpus for relevant content.

    Use this tool to ground your responses in the user's actual uploaded
    material. Call this before asking questions the corpus might answer.

    Args:
        query: natural language search query

    Returns:
        Formatted string of relevant chunks with source citations
    """
    thread_id = state.thread_id

    with timer("query_corpus", logger, extra={"thread_id": thread_id}):
        # Embed the query using search_query input_type
        query_vector = await _embedder.embed_query(query)

        # Search Qdrant
        chunks = await _qdrant.search(
            thread_id=thread_id,
            query_vector=query_vector,
        )

        # Apply post-retrieval filters
        chunks = apply_default_filters(chunks)

    if not chunks:
        return (
            "No relevant content found in the corpus for this query. "
            "Consider using firecrawl_search to search the web instead."
        )

    # Format results for the LLM
    lines = [f"Found {len(chunks)} relevant chunks:\n"]

    for i, chunk in enumerate(chunks, 1):
        source_info = f"{chunk.origin}"
        if chunk.page:
            source_info += f" (page {chunk.page})"
        if chunk.lang:
            source_info += f" [{chunk.lang}]"

        modality_tag = "📷 Image" if chunk.modality.value == "image" else "📄 Text"

        lines.append(
            f"[{i}] {modality_tag} | Score: {chunk.score:.3f} | {source_info}"
        )
        if chunk.text:
            # Truncate long chunks for the prompt
            text_preview = chunk.text[:500]
            if len(chunk.text) > 500:
                text_preview += "..."
            lines.append(f"    {text_preview}")
        lines.append("")

    return "\n".join(lines)