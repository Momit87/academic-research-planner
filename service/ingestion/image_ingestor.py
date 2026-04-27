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

import groq as groq_sdk
from langsmith import traceable

from core.config import get_settings
from core.logging import get_logger, timer
from llm.llm_schema.state_models import Modality, SourceType
from service.ingestion.chunker import RawDocument

logger = get_logger(__name__)

CAPTION_PROMPT = (
    "Describe this image concisely for academic research context. "
    "Focus on any data, diagrams, charts, equations, tables, or key concepts shown. "
    "If it is a photograph, describe what is depicted. Be factual and precise."
)

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

    Images are embedded natively by Cohere Embed v4 AND captioned by
    a Groq vision LLM (llama-4-scout). The caption is stored as the
    chunk's text field so the agent can describe image content in chat.
    """

    def __init__(self):
        settings = get_settings()
        self._groq = groq_sdk.AsyncGroq(api_key=settings.groq_api_key)
        self._vision_model = settings.vision_model

    async def _caption_image(self, image_bytes: bytes, filename: str) -> str | None:
        """
        Generate a text caption for an image using the Groq vision LLM.

        Returns the caption string, or None if captioning fails.
        The caption is stored as the chunk's text field so the agent
        can describe and reason about image content in chat.
        """
        try:
            mime = self._detect_mime(image_bytes)
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            data_uri = f"data:{mime};base64,{b64}"

            response = await self._groq.chat.completions.create(
                model=self._vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": CAPTION_PROMPT},
                    ],
                }],
                max_tokens=512,
            )
            caption = response.choices[0].message.content
            logger.info(
                "Image captioned",
                extra={"file_name": filename, "caption_len": len(caption or "")}
            )
            return caption
        except Exception as e:
            logger.warning(
                "Image captioning failed — storing without caption",
                extra={"file_name": filename, "error": str(e)}
            )
            return None

    @staticmethod
    def _detect_mime(image_bytes: bytes) -> str:
        """Detect MIME type from magic bytes; defaults to image/jpeg."""
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if image_bytes[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"

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

            with timer("image_ingest", logger, extra={"file_name": filename}):
                try:
                    doc = await self._ingest_image(file, mime_type)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.warning(
                        "Image ingestion failed — skipping",
                        extra={"file_name": filename, "error": str(e)}
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
                extra={"file_name": filename, "mime_type": mime_type}
            )
            return None

        # Decode base64 content
        try:
            image_bytes = base64.b64decode(file["base64_content"])
        except Exception as e:
            logger.warning(
                "Base64 decode failed",
                extra={"file_name": filename, "error": str(e)}
            )
            return None

        # Validate image is not empty
        if not image_bytes:
            logger.warning(
                "Empty image file — skipping",
                extra={"file_name": filename}
            )
            return None

        caption = await self._caption_image(image_bytes, filename)

        return RawDocument(
            source_type=SourceType.IMAGE,
            origin=filename,
            title=filename,
            text=caption,            # Vision LLM caption for agent comprehension
            image_bytes=image_bytes,
            modality=Modality.IMAGE,
            page=None,
            slide=None,
            lang=None,               # Images have no language
        )
    