"""
service/firecrawl.py
=====================
Firecrawl service wrapper: scrape() and search().

Two entry points:
    scrape(url)         — used at ingestion time by url_ingestor
    search(query, k=5)  — used as a tool during chat for live web search

Firecrawl returns clean markdown, stripping ads/navigation/boilerplate.
Much better than raw HTML for academic content extraction.
"""

from dataclasses import dataclass
from typing import Optional

from firecrawl import FirecrawlApp
from langsmith import traceable

from core.config import get_settings
from core.logging import get_logger, timer

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Return types
# ------------------------------------------------------------------

@dataclass
class ScrapedDocument:
    """Result of a single URL scrape."""
    url: str
    markdown: Optional[str]
    title: Optional[str]
    description: Optional[str]


@dataclass
class SearchResult:
    """A single result from Firecrawl web search."""
    title: str
    url: str
    snippet: str


# ------------------------------------------------------------------
# Service
# ------------------------------------------------------------------

class FirecrawlService:
    """
    Thin wrapper around the Firecrawl Python SDK.

    scrape()  — called by url_ingestor at onboarding
    search()  — called by firecrawl_search tool during chat
    """

    def __init__(self):
        settings = get_settings()
        self._app = FirecrawlApp(api_key=settings.firecrawl_api_key)

    @traceable(name="firecrawl_scrape", run_type="retriever")
    async def scrape(self, url: str) -> ScrapedDocument:
        """
        Scrape a single URL and return clean markdown.

        Args:
            url: URL to scrape

        Returns:
            ScrapedDocument with markdown content and metadata
        """
        with timer("firecrawl_scrape", logger, extra={"url": url}):
            try:
                # Firecrawl SDK is synchronous — run in thread pool
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._app.scrape_url(
                        url,
                        params={"formats": ["markdown"]}
                    )
                )

                return ScrapedDocument(
                    url=url,
                    markdown=result.get("markdown"),
                    title=result.get("metadata", {}).get("title"),
                    description=result.get("metadata", {}).get("description"),
                )

            except Exception as e:
                logger.warning(
                    "Firecrawl scrape failed",
                    extra={"url": url, "error": str(e)}
                )
                return ScrapedDocument(
                    url=url,
                    markdown=None,
                    title=None,
                    description=None,
                )

    @traceable(name="firecrawl_search", run_type="retriever")
    async def search(
        self,
        query: str,
        k: int = 5
    ) -> list[SearchResult]:
        """
        Search the web via Firecrawl and return top-k results.

        Called by the firecrawl_search tool during chat when
        the user's private corpus is insufficient or when
        recency matters.

        Args:
            query: search query string
            k: number of results to return (default 5)

        Returns:
            list of SearchResult with title, url, snippet
        """
        with timer("firecrawl_search", logger,
                   extra={"query": query, "k": k}):
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: self._app.search(query, limit=k)
                )

                search_results = []
                for item in results.get("data", []):
                    search_results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("description", ""),
                    ))

                logger.info(
                    "Firecrawl search complete",
                    extra={
                        "query": query,
                        "results_returned": len(search_results)
                    }
                )
                return search_results

            except Exception as e:
                logger.warning(
                    "Firecrawl search failed",
                    extra={"query": query, "error": str(e)}
                )
                return []
            