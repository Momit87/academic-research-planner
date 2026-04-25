"""
service/ingestion/url_ingestor.py
===================================
Ingests URLs (web pages, arXiv, journals, blog posts) into RawDocuments.

Uses Firecrawl's scrape() endpoint to extract clean markdown from URLs.
Handles PDF-hosted-at-URL specially — detects and routes to pdf_ingestor.

Used by:
    service/ingestion pipeline during onboarding
"""

from langsmith import traceable

from core.logging import get_logger, timer
from llm.llm_schema.state_models import SourceType
from service.ingestion.chunker import RawDocument

logger = get_logger(__name__)


class UrlIngestor:
    """
    Extracts text content from URLs using Firecrawl scrape.

    Firecrawl returns clean markdown, stripping navigation,
    ads, and boilerplate. Much better than raw HTML parsing.
    """

    def __init__(self):
        # Import here to avoid circular imports at module load
        from service.firecrawl import FirecrawlService
        self._firecrawl = FirecrawlService()

    @traceable(name="url_ingestor", run_type="retriever")
    async def ingest(self, urls: list[str]) -> list[RawDocument]:
        """
        Scrape a list of URLs and return RawDocuments.

        Args:
            urls: list of URL strings to scrape

        Returns:
            list of RawDocument objects (failed URLs are skipped with warning)
        """
        documents: list[RawDocument] = []

        for url in urls:
            with timer("url_scrape", logger, extra={"url": url}):
                try:
                    doc = await self._scrape_url(url)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.warning(
                        "URL ingestion failed — skipping",
                        extra={"url": url, "error": str(e)}
                    )

        logger.info(
            "URL ingestion complete",
            extra={
                "requested": len(urls),
                "succeeded": len(documents),
                "failed": len(urls) - len(documents),
            }
        )
        return documents

    async def _scrape_url(self, url: str) -> RawDocument | None:
        """
        Scrape a single URL.

        PDF-hosted-at-URL detection:
        If Firecrawl returns very little text but the URL ends in .pdf,
        the caller should route to pdf_ingestor instead.
        We handle this by checking content length and URL suffix.
        """
        # Check if URL points directly to a PDF
        if url.lower().endswith(".pdf"):
            logger.info(
                "PDF URL detected — downloading as binary",
                extra={"url": url}
            )
            return await self._ingest_pdf_url(url)

        # Scrape as web page
        result = await self._firecrawl.scrape(url)

        if not result or not result.markdown:
            logger.warning(
                "Firecrawl returned empty content",
                extra={"url": url}
            )
            return None

        return RawDocument(
            source_type=SourceType.URL,
            origin=url,
            title=result.title,
            text=result.markdown,
            modality=__import__(
                "llm.llm_schema.state_models", fromlist=["Modality"]
            ).Modality.TEXT,
        )

    async def _ingest_pdf_url(self, url: str) -> RawDocument | None:
        """
        Download a PDF from a URL and return as RawDocument
        with raw bytes for the PDF ingestor to process.

        Returns a RawDocument with source_type=PDF so the
        downstream pipeline routes it correctly.
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()

            # Return with PDF source type — chunker will handle it
            from service.ingestion.pdf_ingestor import PdfIngestor
            ingestor = PdfIngestor()
            docs = await ingestor.ingest_bytes(response.content, origin=url)
            return docs[0] if docs else None

        except Exception as e:
            logger.warning(
                "PDF URL download failed",
                extra={"url": url, "error": str(e)}
            )
            return None