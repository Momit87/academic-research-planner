"""
api/schema/research_planner.py
================================
FastAPI request and response models for all research planner endpoints.

These are the API contract — what clients send and receive.
They are deliberately separate from the internal LLM schemas
to allow the API surface to evolve independently of internal state.

Endpoints covered:
    POST /research-planner/onboarding
    POST /research-planner/chat
    POST /research-planner/chat/stream  (optional)
    GET  /research-planner/deliverables/{thread_id}
"""

from typing import Optional

from pydantic import BaseModel, Field, HttpUrl

from llm.llm_schema.deliverables import (
    ClusteringDeliverable,
    DiscoveryDeliverable,
    GapAnalysisDeliverable,
    WritingOutlineDeliverable,
)
from llm.llm_schema.state_models import (
    IngestionSummary,
    PhaseName,
    ResearchProfile,
)


# ==================================================================
# Shared sub-models
# ==================================================================

class FileUpload(BaseModel):
    """
    A file uploaded as base64 payload.
    Used for PDFs, images, DOCX, and PPTX in OnboardingRequest.
    """
    filename: str = Field(description="Original filename with extension")
    base64_content: str = Field(description="Base64-encoded file content")
    mime_type: str = Field(
        description="MIME type: application/pdf, image/png, image/jpeg, etc."
    )


# ==================================================================
# Onboarding
# ==================================================================

class OnboardingRequest(BaseModel):
    """
    POST /research-planner/onboarding

    Accepts heterogeneous research material for ingestion.
    All fields are optional — but at least one source is required
    (enforced at the router level, not schema level).

    A new thread_id is generated server-side and returned in the response.
    """
    urls: list[HttpUrl] = Field(
        default_factory=list,
        description="URLs to scrape: arXiv, journal pages, blogs, PDF URLs"
    )
    pdfs: list[FileUpload] = Field(
        default_factory=list,
        description="PDF files as base64 payloads"
    )
    images: list[FileUpload] = Field(
        default_factory=list,
        description="Image files (PNG/JPEG) as base64 payloads"
    )
    docs: list[FileUpload] = Field(
        default_factory=list,
        description="DOCX or PPTX files as base64 payloads"
    )
    field_hint: Optional[str] = Field(
        default=None,
        description=(
            "Optional hint about the research field. "
            "Helps the profiling LLM — not required."
        )
    )


class OnboardingResponse(BaseModel):
    """
    Response from POST /research-planner/onboarding.

    Returns the thread_id the client must include in all subsequent
    chat requests. The profile is a best-effort seed from the
    profiling LLM pass — refined during Discovery phase.
    """
    thread_id: str = Field(
        description="UUID identifying this research session. Store this."
    )
    profile: ResearchProfile = Field(
        description="Best-effort research profile inferred from corpus"
    )
    ingestion_summary: IngestionSummary = Field(
        description="What was ingested: source counts, languages, failures"
    )
    phase_hint: PhaseName = Field(
        default=PhaseName.DISCOVERY,
        description="Suggested starting phase — always DISCOVERY"
    )


# ==================================================================
# Chat
# ==================================================================

class ChatRequest(BaseModel):
    """
    POST /research-planner/chat

    Sent on every conversation turn.

    is_deliverable_accepted:
        Set to True when the user accepts the deliverable shown
        in the PREVIOUS turn's response. The API layer handles
        this before invoking LangGraph — tools never see this flag
        (see DECISIONS.md D-002).

    current_phase:
        The phase the client believes is active. Used by the API
        layer for acceptance logic. The graph may update this.
    """
    thread_id: str = Field(description="Thread ID from OnboardingResponse")
    user_message: str = Field(description="The user's message text")
    current_phase: PhaseName = Field(
        description="Current active phase from the client's perspective"
    )
    is_deliverable_accepted: bool = Field(
        default=False,
        description=(
            "True if the user is accepting the deliverable shown "
            "in the previous response. Handled by API layer before "
            "graph execution. See DECISIONS.md D-002."
        )
    )


class ChatResponse(BaseModel):
    """
    Response from POST /research-planner/chat.

    The response field contains the agent's prose message.
    deliverables_markdown contains rendered markdown for any
    deliverable generated or updated this turn.
    """
    response: str = Field(
        description="The main_agent's prose response to show in chat"
    )
    current_phase: PhaseName = Field(
        description="Active phase after this turn"
    )
    suggested_next_phase: Optional[PhaseName] = Field(
        default=None,
        description="Phase the agent recommends moving to next"
    )
    accepted_deliverables: list[str] = Field(
        default_factory=list,
        description="List of phase names with accepted deliverables"
    )
    deliverables_markdown: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Rendered markdown per accepted phase: "
            "{'discovery': '## Discovery...', 'clustering': '...'}"
        )
    )


# ==================================================================
# Deliverables fetch
# ==================================================================

class DeliverablesResponse(BaseModel):
    """
    Response from GET /research-planner/deliverables/{thread_id}.

    Returns all current deliverables — both structured Pydantic objects
    and pre-rendered markdown strings.
    Ungenerated deliverables are None.
    """
    thread_id: str

    # Structured Pydantic objects
    discovery: Optional[DiscoveryDeliverable] = None
    clustering: Optional[ClusteringDeliverable] = None
    gap_analysis: Optional[GapAnalysisDeliverable] = None
    writing_outline: Optional[WritingOutlineDeliverable] = None

    # Pre-rendered markdown (deterministic, no LLM)
    discovery_markdown: Optional[str] = None
    clustering_markdown: Optional[str] = None
    gap_analysis_markdown: Optional[str] = None
    writing_outline_markdown: Optional[str] = None

    # Phase state
    current_phase: PhaseName = PhaseName.DISCOVERY
    accepted_deliverables: list[str] = Field(default_factory=list)