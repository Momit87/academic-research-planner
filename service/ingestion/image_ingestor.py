"""
service/ingestion/image_ingestor.py
=====================================
Ingests images (PNG/JPEG) as RawDocuments for native embedding.

Strategy:
    PRIMARY: Pass image bytes directly to Cohere Embed v4.
             Cohere supports native image embedding in the same
             vector space as text — no captioning needed.

    FALLBACK: If image bytes are corrupted or unsupported format,
              log warning and skip. Vision captioning fallback
              (image_describer.yml) is reserved for future use
              when the embedding model cannot handle images natively.

Used by:
    Onboarding pipeline for image file uploads (figures, whiteboards, etc.)
"""

import base64

from langsmith import traceable

from core.logging import get_logger, timer
from llm.llm_schema.state_models import Modality, SourceType
from service.ingestion.chunker import RawDocument

logger = get_logger(__name__)

# Supported image MIME types
SUPPORTED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}


class ImageIngestor:
    """
    Converts image uploads into RawDocument objects for embedding.

    Images are kept as raw bytes — no text extraction.
    Cohere Embed v4 embeds images directly into the same
    vector space as text chunks, enabling text queries to
    retrieve relevant figures.
    """

    @traceable(name="image_ingestor", run_type="retriever")
    async def ingest(
        self,
        files: list[dict]  # [{"filename": str, "base64_content": str, "mime_type": str}]
    ) -> list[RawDocument]:
        """
        Ingest a list of base64-encoded image files.

        Args:
            files: list of FileUpload dicts with base64_content

        Returns:
            list of RawDocument objects, one per image
        """
        documents: list[RawDocument] = []

        for file in files:
            filename = file.get("filename", "unknown_image")
            mime_type = file.get("mime_type", "image/jpeg")

            with timer("image_ingest", logger, extra={"filename": filename}):
                try:
                    doc = await self._ingest_image(file, mime_type)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.warning(
                        "Image ingestion failed — skipping",
                        extra={"filename": filename, "error": str(e)}
                    )

        logger.info(
            "Image ingestion complete",
            extra={
                "requested": len(files),
                "succeeded": len(documents),
            }
        )
        return documents

    async def _ingest_image(
        self,
        file: dict,
        mime_type: str
    ) -> RawDocument | None:
        """
        Process a single image file.

        Validates MIME type and decodes base64 content.
        Returns None for unsupported formats.
        """
        filename = file.get("filename", "unknown_image")

        # Validate MIME type
        if mime_type not in SUPPORTED_MIME_TYPES:
            logger.warning(
                "Unsupported image format — skipping",
                extra={"filename": filename, "mime_type": mime_type}
            )
            return None

        # Decode base64 content
        try:
            image_bytes = base64.b64decode(file["base64_content"])
        except Exception as e:
            logger.warning(
                "Base64 decode failed",
                extra={"filename": filename, "error": str(e)}
            )
            return None

        # Validate image is not empty
        if not image_bytes:
            logger.warning(
                "Empty image file — skipping",
                extra={"filename": filename}
            )
            return None

        return RawDocument(
            source_type=SourceType.IMAGE,
            origin=filename,
            title=filename,
            text=None,               # No text — native image embedding
            image_bytes=image_bytes,
            modality=Modality.IMAGE,
            page=None,
            slide=None,
            lang=None,               # Images have no language
        )
    