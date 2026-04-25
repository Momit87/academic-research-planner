"""
service/ingestion/pdf_ingestor.py
===================================
Extracts text from PDF files page by page using PyMuPDF (fitz).

Each page becomes one RawDocument. The chunker then splits
long pages into token-sized chunks with overlap.

Handles:
    - Direct file upload (base64 decoded bytes)
    - PDF bytes from URL ingestor
    - Preserves page numbers in metadata

Used by:
    Onboarding pipeline for PDF file uploads
    URL ingestor for PDF-hosted-at-URL
"""

import base64

import fitz  # PyMuPDF

from langsmith import traceable

from core.logging import get_logger, timer
from llm.llm_schema.state_models import Modality, SourceType
from service.ingestion.chunker import RawDocument

logger = get_logger(__name__)


class PdfIngestor:
    """
    Extracts text from PDF files using PyMuPDF.

    PyMuPDF (fitz) is fast, handles complex layouts,
    and preserves page structure better than pdfplumber
    for academic papers with multi-column layouts.
    """

    @traceable(name="pdf_ingestor", run_type="retriever")
    async def ingest(
        self,
        files: list[dict]  # [{"filename": str, "base64_content": str, "mime_type": str}]
    ) -> list[RawDocument]:
        """
        Ingest a list of base64-encoded PDF files.

        Args:
            files: list of FileUpload dicts with base64_content

        Returns:
            list of RawDocument objects, one per page per PDF
        """
        documents: list[RawDocument] = []

        for file in files:
            filename = file.get("filename", "unknown.pdf")
            with timer("pdf_extract", logger, extra={"pdf_file": filename}):
                try:
                    raw_bytes = base64.b64decode(file["base64_content"])
                    docs = await self.ingest_bytes(raw_bytes, origin=filename)
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(
                        "PDF ingestion failed — skipping",
                        extra={"pdf_file": filename, "error": str(e)}
                    )

        logger.info(
            "PDF ingestion complete",
            extra={
                "files_processed": len(files),
                "pages_extracted": len(documents),
            }
        )
        return documents

    async def ingest_bytes(
        self,
        pdf_bytes: bytes,
        origin: str
    ) -> list[RawDocument]:
        """
        Extract text from raw PDF bytes.

        Args:
            pdf_bytes: raw PDF file content
            origin: source identifier (filename or URL)

        Returns:
            list of RawDocument, one per page
        """
        documents: list[RawDocument] = []

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            # Extract document title from metadata
            title = doc.metadata.get("title") or None

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                # Fallback: try blocks extraction if plain text is empty
                if not text or not text.strip():
                    blocks = page.get_text("blocks")
                    text = "\n".join(
                        b[4] for b in blocks if len(b) > 4 and isinstance(b[4], str)
                    )

                if not text or not text.strip():
                    logger.debug(
                        "PDF page has no extractable text — skipping",
                        extra={"origin": origin, "page": page_num + 1},
                    )
                    continue

                documents.append(RawDocument(
                    source_type=SourceType.PDF,
                    origin=origin,
                    title=title,
                    text=text,
                    modality=Modality.TEXT,
                    page=page_num + 1,  # 1-indexed page numbers
                ))

            doc.close()

            if not documents:
                logger.warning(
                    "PDF opened but no text extracted — may be image-based or empty",
                    extra={"origin": origin, "pages": len(doc)},
                )

        except Exception as e:
            logger.error(
                "PDF parsing failed",
                extra={"origin": origin, "error": str(e)},
                exc_info=True,
            )

        return documents