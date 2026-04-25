"""
llm/workflow/research_planner/tool/firecrawl_search.py
=======================================================
Live web search tool using Firecrawl.

No InjectedState needed — the LLM passes the query directly.
Used when corpus is insufficient or recency matters.

Used by: main_agent (bound tool)
"""

from langchain_core.tools import tool

from core.logging import get_logger, timer
from service.firecrawl import FirecrawlService

logger = get_logger(__name__)

_firecrawl = FirecrawlService()


@tool
async def firecrawl_search(query: str) -> str:
    """
    Search the web for current research information.

    Use this when:
    - The user's corpus doesn't contain relevant information
    - You need recent papers, news, or developments
    - You want to verify facts beyond the uploaded material

    Args:
        query: search query string

    Returns:
        Formatted string of top web search results
    """
    with timer("firecrawl_search_tool", logger, extra={"query": query}):
        results = await _firecrawl.search(query, k=5)

    if not results:
        return "No web search results found for this query."

    lines = [f"Web search results for '{query}':\n"]

    for i, result in enumerate(results, 1):
        lines.append(f"[{i}] {result.title}")
        lines.append(f"    URL: {result.url}")
        if result.snippet:
            lines.append(f"    {result.snippet}")
        lines.append("")

    return "\n".join(lines)