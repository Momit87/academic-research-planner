"""
service/qdrant.py
==================
Qdrant vector store operations: collection lifecycle, upsert, search.

Design:
    - One collection per thread: "thread_{thread_id}"
    - Cosine distance (required for Cohere embeddings)
    - Payload indexes on source_type, lang, modality, source_id
    - Async client throughout

Used by:
    - Onboarding pipeline: create_collection() + upsert_chunks()
    - query_corpus tool: search()
    - Thread cleanup: delete_collection()
"""

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from langsmith import traceable

from core.config import get_settings
from core.logging import get_logger, timer
from llm.llm_schema.state_models import ChunkRecord, RetrievedChunk, SourceType, Modality

logger = get_logger(__name__)


def _collection_name(thread_id: str) -> str:
    """Qdrant collection name for a given thread."""
    return f"thread_{thread_id}"


class QdrantService:
    """
    Manages Qdrant collections and vector operations.

    One instance per application — shared across all threads.
    Collection isolation is achieved via naming convention.
    """

    def __init__(self):
        settings = get_settings()
        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self._vector_size = settings.qdrant_vector_size
        self._top_k = settings.qdrant_top_k

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    @traceable(name="qdrant_create_collection", run_type="retriever")
    async def create_collection(self, thread_id: str) -> None:
        """
        Create a new Qdrant collection for a thread.

        Called once at onboarding. Idempotent — safe to call
        if collection already exists (recreates it cleanly).

        Args:
            thread_id: unique thread identifier
        """
        name = _collection_name(thread_id)

        with timer("qdrant_create_collection", logger,
                   extra={"thread_id": thread_id}):

            # Delete if exists (clean slate for new onboarding)
            exists = await self._client.collection_exists(name)
            if exists:
                await self._client.delete_collection(name)
                logger.info(
                    "Existing collection deleted for fresh onboarding",
                    extra={"thread_id": thread_id}
                )

            await self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

            # Create payload indexes for efficient filtered search
            await self._create_indexes(name)

        logger.info(
            "Qdrant collection created",
            extra={"thread_id": thread_id, "collection": name}
        )

    async def _create_indexes(self, collection_name: str) -> None:
        """Create payload indexes for common filter fields."""
        index_fields = [
            ("source_type", PayloadSchemaType.KEYWORD),
            ("lang", PayloadSchemaType.KEYWORD),
            ("modality", PayloadSchemaType.KEYWORD),
            ("source_id", PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in index_fields:
            await self._client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )

    async def delete_collection(self, thread_id: str) -> None:
        """
        Delete a thread's collection and all its vectors.

        Called when a thread ends or is explicitly deleted.
        """
        name = _collection_name(thread_id)
        exists = await self._client.collection_exists(name)
        if exists:
            await self._client.delete_collection(name)
            logger.info(
                "Qdrant collection deleted",
                extra={"thread_id": thread_id}
            )

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    @traceable(name="qdrant_upsert", run_type="retriever")
    async def upsert_chunks(
        self,
        thread_id: str,
        chunks: list[ChunkRecord],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store chunk embeddings and payloads in Qdrant.

        Args:
            thread_id: collection identifier
            chunks: ChunkRecord objects (metadata)
            embeddings: corresponding embedding vectors
                        (same order and length as chunks)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings "
                f"({len(embeddings)}) must have equal length"
            )

        name = _collection_name(thread_id)

        # Build Qdrant points
        points = []
        for chunk, vector in zip(chunks, embeddings):
            payload = {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "source_type": chunk.source_type.value,
                "origin": chunk.origin,
                "modality": chunk.modality.value,
                "text": chunk.text,
                "page": chunk.page,
                "slide": chunk.slide,
                "lang": chunk.lang,
                "title": chunk.title,
                "chunk_index": chunk.chunk_index,
            }
            # Use chunk_id as point ID (hash → deterministic int)
            point_id = int(chunk.chunk_id[:8], 16)

            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            ))

        with timer("qdrant_upsert", logger,
                   extra={"thread_id": thread_id, "points": len(points)}):
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i: i + batch_size]
                await self._client.upsert(
                    collection_name=name,
                    points=batch,
                )

        logger.info(
            "Qdrant upsert complete",
            extra={"thread_id": thread_id, "chunks_stored": len(chunks)}
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @traceable(name="qdrant_search", run_type="retriever")
    async def search(
        self,
        thread_id: str,
        query_vector: list[float],
        top_k: int | None = None,
        filter_modality: str | None = None,
        filter_lang: str | None = None,
        filter_source_type: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Search for similar chunks in a thread's collection.

        Args:
            thread_id: collection to search
            query_vector: embedded query vector (1536 dims)
            top_k: number of results (defaults to settings.qdrant_top_k)
            filter_modality: optional filter "text" or "image"
            filter_lang: optional ISO 639-1 language filter
            filter_source_type: optional source type filter

        Returns:
            list of RetrievedChunk sorted by similarity score (desc)
        """
        name = _collection_name(thread_id)
        k = top_k or self._top_k

        # Build optional filter
        must_conditions = []
        if filter_modality:
            must_conditions.append(
                FieldCondition(
                    key="modality",
                    match=MatchValue(value=filter_modality)
                )
            )
        if filter_lang:
            must_conditions.append(
                FieldCondition(
                    key="lang",
                    match=MatchValue(value=filter_lang)
                )
            )
        if filter_source_type:
            must_conditions.append(
                FieldCondition(
                    key="source_type",
                    match=MatchValue(value=filter_source_type)
                )
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        with timer("qdrant_search", logger,
                   extra={"thread_id": thread_id, "top_k": k}):
            response = await self._client.query_points(
                collection_name=name,
                query=query_vector,
                limit=k,
                query_filter=query_filter,
                with_payload=True,
            )

        chunks = []
        for hit in response.points:
            payload = hit.payload or {}
            chunks.append(RetrievedChunk(
                chunk_id=payload.get("chunk_id", ""),
                source_type=SourceType(
                    payload.get("source_type", "url")
                ),
                origin=payload.get("origin", ""),
                modality=Modality(
                    payload.get("modality", "text")
                ),
                text=payload.get("text"),
                page=payload.get("page"),
                lang=payload.get("lang"),
                score=hit.score,
            ))

        return chunks