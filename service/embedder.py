"""
service/embedder.py
====================
Provider-agnostic embedding wrapper using Cohere Embed v4.

Cohere Embed v4 supports:
    - Multilingual text (100+ languages in one vector space)
    - Native image embedding (same space as text)
    - input_type parameter controls query vs document mode

Two call modes (required by Cohere):
    embed_documents() — for ingestion (input_type="search_document")
    embed_query()     — for retrieval (input_type="search_query")

Using the wrong input_type degrades retrieval quality significantly.

Abstraction:
    The Embedder class is the ONLY place Cohere is referenced.
    Swapping to Gemini multimodal embeddings = change this one file.
    See DECISIONS.md D-001.
"""

import base64
from typing import Union

import cohere
from langsmith import traceable

from core.config import get_settings
from core.logging import get_logger, timer
from llm.llm_schema.state_models import ChunkRecord, Modality

logger = get_logger(__name__)

# Cohere Embed v4 model name
COHERE_EMBED_MODEL = "embed-v4.0"

# Batch size limit for Cohere API
# Cohere allows max 96 items per batch for images, 2048 for text
TEXT_BATCH_SIZE = 96   # conservative — works for both text and mixed
IMAGE_BATCH_SIZE = 20  # images are larger, keep batches smaller


class Embedder:
    """
    Cohere Embed v4 wrapper for multilingual + multimodal embeddings.

    All text and image chunks are embedded into the same 1536-dimensional
    vector space, enabling cross-modal retrieval (text query → image result).
    """

    def __init__(self):
        settings = get_settings()
        self._client = cohere.AsyncClient(api_key=settings.cohere_api_key)
        self._vector_size = settings.qdrant_vector_size  # 1536 for Cohere embed-v4.0

    @traceable(name="cohere_embed_documents", run_type="embedding")
    async def embed_documents(
        self,
        chunks: list[ChunkRecord]
    ) -> list[list[float]]:
        """
        Embed a list of ChunkRecords for storage (ingestion mode).

        Uses input_type="search_document" — optimized for storage.
        Handles both text and image chunks in the same call sequence.

        Args:
            chunks: list of ChunkRecord objects to embed

        Returns:
            list of embedding vectors, same order as input chunks
            Each vector is a list of 1536 floats.
        """
        if not chunks:
            return []

        # Separate text and image chunks
        text_chunks = [(i, c) for i, c in enumerate(chunks)
                       if c.modality == Modality.TEXT]
        image_chunks = [(i, c) for i, c in enumerate(chunks)
                        if c.modality == Modality.IMAGE]

        # Initialize result array
        embeddings: list[list[float] | None] = [None] * len(chunks)

        # Embed text chunks
        if text_chunks:
            indices, records = zip(*text_chunks)
            texts = [r.text or "" for r in records]
            vecs = await self._embed_texts(texts, input_type="search_document")
            for idx, vec in zip(indices, vecs):
                embeddings[idx] = vec

        # Embed image chunks
        if image_chunks:
            indices, records = zip(*image_chunks)
            images = [r.image_bytes for r in records]
            vecs = await self._embed_images(images, input_type="search_document")
            for idx, vec in zip(indices, vecs):
                embeddings[idx] = vec

        # Verify no None embeddings
        result = [e for e in embeddings if e is not None]
        if len(result) != len(chunks):
            logger.warning(
                "Some chunks failed to embed",
                extra={
                    "expected": len(chunks),
                    "got": len(result)
                }
            )

        logger.info(
            "Document embedding complete",
            extra={
                "total_chunks": len(chunks),
                "text_chunks": len(text_chunks),
                "image_chunks": len(image_chunks),
            }
        )
        return result

    @traceable(name="cohere_embed_query", run_type="embedding")
    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for retrieval (query mode).

        Uses input_type="search_query" — optimized for retrieval.
        NEVER use embed_documents() for queries — it degrades results.

        Args:
            query: the search query string

        Returns:
            list of 1536 floats (the query embedding vector)
        """
        with timer("embed_query", logger, extra={"query_len": len(query)}):
            vecs = await self._embed_texts(
                [query],
                input_type="search_query"
            )
            return vecs[0]

    async def _embed_texts(
        self,
        texts: list[str],
        input_type: str
    ) -> list[list[float]]:
        """
        Embed text strings in batches.

        Args:
            texts: list of text strings
            input_type: "search_document" or "search_query"

        Returns:
            list of embedding vectors
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), TEXT_BATCH_SIZE):
            batch = texts[i: i + TEXT_BATCH_SIZE]

            response = await self._client.embed(
                texts=batch,
                model=COHERE_EMBED_MODEL,
                input_type=input_type,
                embedding_types=["float"],
            )
            all_embeddings.extend(response.embeddings.float)

        return all_embeddings

    async def _embed_images(
        self,
        images: list[bytes],
        input_type: str
    ) -> list[list[float]]:
        """
        Embed raw image bytes in batches using Cohere's multimodal API.

        Cohere Embed v4 requires images as data URIs:
        "data:<mime_type>;base64,<b64_data>"

        Args:
            images: list of raw image bytes
            input_type: "search_document" or "search_query"

        Returns:
            list of embedding vectors
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(images), IMAGE_BATCH_SIZE):
            batch = images[i: i + IMAGE_BATCH_SIZE]

            # Cohere requires data URIs, not bare base64
            data_uris = [
                f"data:{self._detect_mime(img)};base64,{base64.b64encode(img).decode('utf-8')}"
                for img in batch
            ]

            response = await self._client.embed(
                images=data_uris,
                model=COHERE_EMBED_MODEL,
                input_type=input_type,
                embedding_types=["float"],
            )
            all_embeddings.extend(response.embeddings.float)

        return all_embeddings

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