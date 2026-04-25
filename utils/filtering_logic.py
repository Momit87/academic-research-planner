"""
utils/filtering_logic.py
==========================
Post-retrieval metadata filtering for RetrievedChunk lists.

Applied after Qdrant search to further narrow results
based on relevance score, modality, language, or source type.

Used by:
    llm/workflow/research_planner/tool/query_corpus.py
"""

from llm.llm_schema.state_models import Modality, RetrievedChunk, SourceType

# Minimum cosine similarity score to include in results.
# Cohere Embed v4 cosine similarities for relevant content typically range 0.15-0.40.
# A threshold of 0.15 filters out truly unrelated content while keeping relevant chunks.
DEFAULT_MIN_SCORE = 0.15


def filter_by_score(
    chunks: list[RetrievedChunk],
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[RetrievedChunk]:
    """Remove chunks below the minimum relevance score."""
    return [c for c in chunks if c.score >= min_score]


def filter_by_modality(
    chunks: list[RetrievedChunk],
    modality: Modality,
) -> list[RetrievedChunk]:
    """Keep only chunks of a specific modality."""
    return [c for c in chunks if c.modality == modality]


def filter_by_language(
    chunks: list[RetrievedChunk],
    lang: str,
) -> list[RetrievedChunk]:
    """Keep only chunks in a specific language."""
    return [c for c in chunks if c.lang == lang]


def apply_default_filters(
    chunks: list[RetrievedChunk],
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[RetrievedChunk]:
    """
    Apply standard post-retrieval filtering pipeline.

    Filters applied in order:
        1. Score threshold — remove irrelevant results
        2. Deduplication — remove chunks from same source with near-identical scores

    Args:
        chunks: raw Qdrant results
        min_score: minimum cosine similarity (default 0.5)

    Returns:
        filtered list, sorted by score descending
    """
    # 1. Score filter
    filtered = filter_by_score(chunks, min_score)

    # 2. Sort by score descending
    filtered.sort(key=lambda c: c.score, reverse=True)

    # 3. Light deduplication — keep max 3 chunks per source
    source_counts: dict[str, int] = {}
    deduplicated = []
    for chunk in filtered:
        count = source_counts.get(chunk.origin, 0)
        if count < 3:
            deduplicated.append(chunk)
            source_counts[chunk.origin] = count + 1

    return deduplicated