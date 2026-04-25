"""
service/ingestion/chunker.py
=============================
Converts RawDocument objects into ChunkRecord objects ready for embedding.

Design:
- Token-based chunking with overlap (not character-based)
- Source metadata preserved on every chunk
- Images pass through as single chunks (no splitting)
- Language detection per chunk via langdetect

Used by:
    All four ingestors feed their RawDocument output through this chunker.
    Output ChunkRecords go directly to service/embedder.py
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import tiktoken
from langdetect import detect, LangDetectException

from core.config import get_settings
from core.logging import get_logger
from llm.llm_schema.state_models import ChunkRecord, Modality, SourceType

logger = get_logger(__name__)


# ------------------------------------------------------------------
# RawDocument — output of every ingestor, input to chunker
# ------------------------------------------------------------------

@dataclass
class RawDocument:
    """
    Normalized document produced by source-specific ingestors.
    One RawDocument per page (PDF), per URL, per image, per doc section.

    The chunker converts these into ChunkRecord objects.
    """
    source_type: SourceType
    origin: str                          # URL or filename
    title: Optional[str] = None
    text: Optional[str] = None           # text content (None for images)
    image_bytes: Optional[bytes] = None  # raw bytes (None for text)
    modality: Modality = Modality.TEXT
    page: Optional[int] = None           # PDF page number
    slide: Optional[int] = None          # PPTX slide number
    lang: Optional[str] = None           # pre-detected language (optional)


# ------------------------------------------------------------------
# Chunker
# ------------------------------------------------------------------

class Chunker:
    """
    Splits RawDocument objects into ChunkRecord objects.

    Token-based chunking uses tiktoken (cl100k_base encoding).
    This encoder is model-agnostic and gives consistent token counts
    regardless of which LLM provider is used downstream.

    Image documents are passed through as single chunks — no splitting.
    """

    def __init__(self):
        self.settings = get_settings()
        self.chunk_size = self.settings.chunk_token_size      # 512
        self.chunk_overlap = self.settings.chunk_token_overlap # 64
        # cl100k_base is used by GPT-4, Claude, and Gemini tokenizers
        # Good enough approximation for all providers
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def chunk(self, documents: list[RawDocument]) -> list[ChunkRecord]:
        """
        Convert a list of RawDocuments into ChunkRecords.

        Args:
            documents: list of RawDocument objects from ingestors

        Returns:
            list of ChunkRecord objects ready for embedding
        """
        all_chunks: list[ChunkRecord] = []

        for doc in documents:
            if doc.modality == Modality.IMAGE:
                chunks = self._chunk_image(doc)
            else:
                chunks = self._chunk_text(doc)
            all_chunks.extend(chunks)

        logger.info(
            "Chunking complete",
            extra={
                "input_documents": len(documents),
                "output_chunks": len(all_chunks),
            }
        )
        return all_chunks

    def _chunk_text(self, doc: RawDocument) -> list[ChunkRecord]:
        """
        Split a text document into overlapping token-based chunks.

        Strategy:
        1. Encode full text to tokens
        2. Slide a window of chunk_size tokens with chunk_overlap step
        3. Decode each window back to text
        4. Detect language on first chunk (assume uniform per document)
        """
        if not doc.text or not doc.text.strip():
            logger.warning(
                "Empty text document skipped",
                extra={"origin": doc.origin}
            )
            return []

        source_id = self._make_source_id(doc.origin)
        tokens = self._encoder.encode(doc.text)
        step = self.chunk_size - self.chunk_overlap

        chunks: list[ChunkRecord] = []
        chunk_index = 0
        pos = 0

        while pos < len(tokens):
            window = tokens[pos: pos + self.chunk_size]
            chunk_text = self._encoder.decode(window)

            # Detect language — use doc.lang if pre-detected
            lang = doc.lang or self._detect_language(chunk_text)

            chunk_id = self._make_chunk_id(source_id, chunk_index)

            chunks.append(ChunkRecord(
                chunk_id=chunk_id,
                source_id=source_id,
                source_type=doc.source_type,
                origin=doc.origin,
                modality=Modality.TEXT,
                text=chunk_text,
                image_bytes=None,
                page=doc.page,
                slide=doc.slide,
                lang=lang,
                title=doc.title,
                chunk_index=chunk_index,
                ingested_at=datetime.utcnow(),
            ))

            chunk_index += 1
            pos += step

            # Stop if remaining tokens less than overlap
            if pos + self.chunk_overlap >= len(tokens):
                break

        return chunks

    def _chunk_image(self, doc: RawDocument) -> list[ChunkRecord]:
        """
        Images are not split — one image = one chunk.

        The embedding model (Cohere Embed v4) handles images natively.
        No text extraction needed here.
        """
        if not doc.image_bytes:
            logger.warning(
                "Empty image document skipped",
                extra={"origin": doc.origin}
            )
            return []

        source_id = self._make_source_id(doc.origin)
        chunk_id = self._make_chunk_id(source_id, 0)

        return [ChunkRecord(
            chunk_id=chunk_id,
            source_id=source_id,
            source_type=doc.source_type,
            origin=doc.origin,
            modality=Modality.IMAGE,
            text=None,
            image_bytes=doc.image_bytes,
            page=doc.page,
            slide=doc.slide,
            lang=None,        # images have no detected language
            title=doc.title,
            chunk_index=0,
            ingested_at=datetime.utcnow(),
        )]

    def _detect_language(self, text: str) -> Optional[str]:
        """
        Detect language of a text string.
        Returns ISO 639-1 code or None on failure.
        """
        try:
            # langdetect needs at least ~20 chars to be reliable
            if len(text.strip()) < 20:
                return None
            return detect(text)
        except LangDetectException:
            return None

    @staticmethod
    def _make_source_id(origin: str) -> str:
        """SHA-256 hash of the origin URL or filename."""
        return hashlib.sha256(origin.encode()).hexdigest()[:16]

    @staticmethod
    def _make_chunk_id(source_id: str, chunk_index: int) -> str:
        """Unique chunk ID: hash(source_id + index)."""
        raw = f"{source_id}_{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]