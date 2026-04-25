"""
llm/llm_schema/state_models.py
===============================
Supporting Pydantic models and enums for ResearchPlannerState.

These types are imported by:
    - llm/workflow/research_planner/graph_state.py  (state definition)
    - api/schema/research_planner.py                (request/response models)
    - llm/workflow/research_planner/tool/*          (tool signatures)
    - service/ingestion/*                           (ingestion metadata)

Design rules:
    - No LLM imports here — pure data models only
    - Every field has a description for LLM structured output clarity
    - Optional fields default to None — never use empty string as sentinel
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Phase Enum
# ------------------------------------------------------------------

class PhaseName(str, Enum):
    """
    The four soft phases of the research planning journey.

    Inherits from str so it serializes cleanly to JSON
    and can be compared directly with string values.

    Order matters for prerequisite checking:
        DISCOVERY → CLUSTERING → GAP_ANALYSIS → WRITING_OUTLINE
    """
    DISCOVERY = "discovery"
    CLUSTERING = "clustering"
    GAP_ANALYSIS = "gap_analysis"
    WRITING_OUTLINE = "writing_outline"


# Ordered list used for phase progression logic
PHASE_ORDER: list[PhaseName] = [
    PhaseName.DISCOVERY,
    PhaseName.CLUSTERING,
    PhaseName.GAP_ANALYSIS,
    PhaseName.WRITING_OUTLINE,
]


def get_next_phase(current: PhaseName) -> Optional[PhaseName]:
    """
    Return the next phase in the sequence, or None if already at the last.

    Args:
        current: the current PhaseName

    Returns:
        The next PhaseName, or None if current is WRITING_OUTLINE
    """
    idx = PHASE_ORDER.index(current)
    if idx + 1 < len(PHASE_ORDER):
        return PHASE_ORDER[idx + 1]
    return None


def get_remaining_phases(accepted: list[str]) -> list[PhaseName]:
    """
    Return phases not yet in accepted_deliverables.

    Args:
        accepted: list of accepted phase name strings

    Returns:
        List of PhaseName values not yet accepted
    """
    return [p for p in PHASE_ORDER if p.value not in accepted]


# ------------------------------------------------------------------
# Ingestion Metadata
# ------------------------------------------------------------------

class SourceType(str, Enum):
    """Type of source material ingested."""
    URL = "url"
    PDF = "pdf"
    IMAGE = "image"
    DOCX = "docx"
    PPTX = "pptx"


class Modality(str, Enum):
    """
    Embedding modality of a chunk.
    Both text and image chunks live in the same Qdrant collection.
    """
    TEXT = "text"
    IMAGE = "image"


class IngestedSourceMeta(BaseModel):
    """
    Metadata record for a single ingested source.
    Stored in ResearchPlannerState.ingested_sources.

    One record per source (URL, PDF file, image, doc) —
    not per chunk. Chunk count is aggregated here.
    """
    source_id: str = Field(
        description="SHA-256 hash of the source origin URL or filename"
    )
    source_type: SourceType = Field(
        description="Type of the source: url, pdf, image, docx, pptx"
    )
    origin: str = Field(
        description="URL or original filename of the source"
    )
    title: Optional[str] = Field(
        default=None,
        description="Inferred or extracted title of the source"
    )
    chunk_count: int = Field(
        description="Number of chunks produced from this source"
    )
    languages: list[str] = Field(
        default_factory=list,
        description="ISO 639-1 language codes detected in this source"
    )
    modalities: list[Modality] = Field(
        default_factory=list,
        description="Embedding modalities present: text, image, or both"
    )
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when this source was ingested"
    )


class IngestionSummary(BaseModel):
    """
    High-level summary of what was ingested during onboarding.
    Returned in OnboardingResponse.
    """
    total_sources: int = Field(description="Total number of sources processed")
    total_chunks: int = Field(description="Total number of chunks embedded")
    sources_by_type: dict[str, int] = Field(
        default_factory=dict,
        description="Count of sources per type: {url: 2, pdf: 3, ...}"
    )
    languages_detected: list[str] = Field(
        default_factory=list,
        description="All unique language codes detected across sources"
    )
    failed_sources: list[str] = Field(
        default_factory=list,
        description="Origins that failed ingestion — logged but not fatal"
    )


# ------------------------------------------------------------------
# Research Profile — seeded at onboarding, refined during Discovery
# ------------------------------------------------------------------

class ResearchProfile(BaseModel):
    """
    Best-effort inference of the researcher's domain and intent.

    Seeded by the onboarding profiling LLM pass over sampled chunks.
    Refined interactively by the main_agent during Discovery phase.

    IMPORTANT: This is a seed, not ground truth.
    The main_agent always confirms and refines these values with the user.
    """
    field: Optional[str] = Field(
        default=None,
        description="Broad academic field, e.g. 'Computer Science', 'Biology'"
    )
    sub_field: Optional[str] = Field(
        default=None,
        description="Specific sub-field, e.g. 'Natural Language Processing'"
    )
    research_intent: Optional[str] = Field(
        default=None,
        description="Draft research question or objective inferred from corpus"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Profiling LLM confidence score (0.0 to 1.0)"
    )


# ------------------------------------------------------------------
# Chunk Record — produced by ingestion, stored in Qdrant payload
# ------------------------------------------------------------------

class ChunkRecord(BaseModel):
    """
    A single embeddable unit produced by the chunker.

    The vector is NOT stored here — it lives in Qdrant.
    This object represents the payload stored alongside each vector.

    Used by:
        - service/embedder.py      (embedding input)
        - service/qdrant.py        (payload storage)
        - llm/workflow/tool/query_corpus.py  (retrieval output)
    """
    chunk_id: str = Field(
        description="Unique ID: SHA-256(source_id + chunk_index)"
    )
    source_id: str = Field(
        description="SHA-256 hash of the origin URL or filename"
    )
    source_type: SourceType = Field(
        description="Type of the original source"
    )
    origin: str = Field(
        description="URL or filename this chunk came from"
    )
    modality: Modality = Field(
        description="Whether this chunk is text or image"
    )
    text: Optional[str] = Field(
        default=None,
        description="Text content — present for text chunks, None for image chunks"
    )
    image_bytes: Optional[bytes] = Field(
        default=None,
        description="Raw image bytes — present for image chunks, None for text"
    )
    page: Optional[int] = Field(
        default=None,
        description="Page number for PDF sources"
    )
    slide: Optional[int] = Field(
        default=None,
        description="Slide number for PPTX sources"
    )
    lang: Optional[str] = Field(
        default=None,
        description="ISO 639-1 language code detected for this chunk"
    )
    title: Optional[str] = Field(
        default=None,
        description="Title of the source document if extractable"
    )
    chunk_index: int = Field(
        description="Position of this chunk within its source (0-indexed)"
    )
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of ingestion"
    )

    class Config:
        # Allow bytes fields for image_bytes
        arbitrary_types_allowed = True


# ------------------------------------------------------------------
# Retrieved Chunk — returned by query_corpus tool
# ------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """
    A chunk returned from Qdrant retrieval.
    Includes the relevance score alongside the payload.

    Returned by the query_corpus tool as part of its response.
    The main_agent uses these to ground its responses.
    """
    chunk_id: str
    source_type: SourceType
    origin: str
    modality: Modality
    text: Optional[str] = None
    page: Optional[int] = None
    lang: Optional[str] = None
    score: float = Field(
        description="Cosine similarity score from Qdrant (0.0 to 1.0)"
    )


# ------------------------------------------------------------------
# Phase Completion Checklist
# ------------------------------------------------------------------

class PhaseChecklist(BaseModel):
    """
    Tracks what information has been gathered for a phase
    and what is still missing.

    Injected into the main_agent's system prompt so it knows
    exactly what to ask about next.
    """
    phase: PhaseName
    gathered: list[str] = Field(
        default_factory=list,
        description="Fields/topics successfully collected for this phase"
    )
    missing: list[str] = Field(
        default_factory=list,
        description="Fields/topics still needed before generating deliverable"
    )

    @property
    def is_complete(self) -> bool:
        """True if all required fields have been gathered."""
        return len(self.missing) == 0


# ------------------------------------------------------------------
# Summarization Output
# ------------------------------------------------------------------

class SummarizationOutput(BaseModel):
    """
    Structured output from the summarize_agent.

    The summarize_agent compresses message history into a single
    summary string and reports the new token count.
    Used by the graph to update messages and approx_prompt_tokens.
    """
    summary: str = Field(
        description=(
            "Compressed conversation summary preserving all facts, "
            "decisions, and gathered research information. "
            "Strips pleasantries and filler."
        )
    )
    total_tokens_after: int = Field(
        description="Approximate token count of the summary message"
    )