"""
api/routers/research_planner.py
================================
FastAPI router for all Research Planner endpoints.

Endpoints:
    POST /research-planner/onboarding
    POST /research-planner/chat
    GET  /research-planner/deliverables/{thread_id}

Acceptance logic (DECISIONS.md D-002):
    is_deliverable_accepted is handled HERE before LangGraph runs.
    Tools never manage accepted_deliverables.
"""

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage

from api.schema.research_planner import (
    ChatRequest,
    ChatResponse,
    DeliverablesResponse,
    OnboardingRequest,
    OnboardingResponse,
)
from core.config import get_settings
from core.logging import get_logger, timer
from llm.llm_schema.state_models import (
    IngestionSummary,
    IngestedSourceMeta,
    PhaseName,
    ResearchProfile,
    SourceType,
    Modality,
    get_next_phase,
)
from llm.workflow.research_planner.graph import build_graph
from llm.workflow.research_planner.graph_state import ResearchPlannerState
from service.embedder import Embedder
from service.firecrawl import FirecrawlService
from service.ingestion.chunker import Chunker
from service.ingestion.doc_ingestor import DocIngestor
from service.ingestion.image_ingestor import ImageIngestor
from service.ingestion.pdf_ingestor import PdfIngestor
from service.ingestion.url_ingestor import UrlIngestor
from service.qdrant import QdrantService
from utils.token_checker import count_messages_tokens

logger = get_logger(__name__)
router = APIRouter(tags=["research-planner"])
settings = get_settings()

# Service singletons
_embedder = Embedder()
_qdrant = QdrantService()
_chunker = Chunker()
_url_ingestor = UrlIngestor()
_pdf_ingestor = PdfIngestor()
_image_ingestor = ImageIngestor()
_doc_ingestor = DocIngestor()


# ------------------------------------------------------------------
# POST /onboarding
# ------------------------------------------------------------------

@router.post("/onboarding", response_model=OnboardingResponse)
async def onboarding(
    request: OnboardingRequest,
    app_request: Request,
) -> OnboardingResponse:
    """
    One-shot ingestion endpoint.

    Accepts heterogeneous research material, ingests and embeds it,
    runs a profiling LLM pass to seed state, and returns a thread_id.
    """
    thread_id = str(uuid.uuid4())

    with timer("onboarding", logger, extra={"thread_id": thread_id}):

        # ------------------------------------------------------------------
        # Step 1 — Extract raw documents from all sources
        # ------------------------------------------------------------------
        raw_docs = []
        failed_sources = []

        # URLs
        if request.urls:
            url_strings = [str(u) for u in request.urls]
            try:
                docs = await _url_ingestor.ingest(url_strings)
                raw_docs.extend(docs)
            except Exception as e:
                logger.warning("URL ingestion failed", extra={"error": str(e)})
                failed_sources.extend(url_strings)

        # PDFs
        if request.pdfs:
            pdf_dicts = [p.model_dump() for p in request.pdfs]
            try:
                docs = await _pdf_ingestor.ingest(pdf_dicts)
                raw_docs.extend(docs)
                origins_with_docs = {d.origin for d in docs}
                failed_sources.extend(
                    p.filename for p in request.pdfs
                    if p.filename not in origins_with_docs
                )
            except Exception as e:
                logger.warning("PDF ingestion failed", extra={"error": str(e)})
                failed_sources.extend([p.filename for p in request.pdfs])

        # Images
        if request.images:
            image_dicts = [i.model_dump() for i in request.images]
            try:
                docs = await _image_ingestor.ingest(image_dicts)
                raw_docs.extend(docs)
                origins_with_docs = {d.origin for d in docs}
                failed_sources.extend(
                    i.filename for i in request.images
                    if i.filename not in origins_with_docs
                )
            except Exception as e:
                logger.warning("Image ingestion failed", extra={"error": str(e)})
                failed_sources.extend([i.filename for i in request.images])

        # Docs (DOCX/PPTX)
        if request.docs:
            doc_dicts = [d.model_dump() for d in request.docs]
            try:
                docs = await _doc_ingestor.ingest(doc_dicts)
                raw_docs.extend(docs)
                origins_with_docs = {d.origin for d in docs}
                failed_sources.extend(
                    d.filename for d in request.docs
                    if d.filename not in origins_with_docs
                )
            except Exception as e:
                logger.warning("Doc ingestion failed", extra={"error": str(e)})
                failed_sources.extend([d.filename for d in request.docs])

        if not raw_docs:
            raise HTTPException(
                status_code=422,
                detail="No content could be extracted from provided sources."
            )

        # ------------------------------------------------------------------
        # Step 2 — Chunk
        # ------------------------------------------------------------------
        chunks = _chunker.chunk(raw_docs)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="Chunking produced no output. Check source content."
            )

        # ------------------------------------------------------------------
        # Step 3 — Embed
        # ------------------------------------------------------------------
        embeddings = await _embedder.embed_documents(chunks)

        # ------------------------------------------------------------------
        # Step 4 — Store in Qdrant
        # ------------------------------------------------------------------
        await _qdrant.create_collection(thread_id)
        await _qdrant.upsert_chunks(thread_id, chunks, embeddings)

        # ------------------------------------------------------------------
        # Step 5 — Build ingestion metadata
        # ------------------------------------------------------------------
        ingested_sources = _build_ingested_sources(raw_docs, chunks)
        ingestion_summary = _build_ingestion_summary(
            ingested_sources, failed_sources
        )

        # ------------------------------------------------------------------
        # Step 6 — Profiling LLM pass
        # ------------------------------------------------------------------
        profile = await _run_profiling_pass(chunks, request.field_hint)

        # ------------------------------------------------------------------
        # Step 7 — Seed LangGraph state so chat has corpus context
        # ------------------------------------------------------------------
        checkpointer = app_request.app.state.checkpointer
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        await graph.aupdate_state(
            config,
            {
                "thread_id": thread_id,
                "ingested_sources": ingested_sources,
                "field": profile.field,
                "sub_field": profile.sub_field,
                "research_intent": profile.research_intent,
                "current_phase": PhaseName.DISCOVERY,
            },
            as_node="main_agent",
        )

    logger.info(
        "Onboarding complete",
        extra={
            "thread_id": thread_id,
            "chunks": len(chunks),
            "sources": len(ingested_sources),
        }
    )

    return OnboardingResponse(
        thread_id=thread_id,
        profile=profile,
        ingestion_summary=ingestion_summary,
        phase_hint=PhaseName.DISCOVERY,
    )


# ------------------------------------------------------------------
# POST /chat
# ------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, app_request: Request) -> ChatResponse:
    """
    Single chat turn endpoint.

    Handles deliverable acceptance before LangGraph execution.
    Returns full ChatResponse with agent response and phase state.
    """
    thread_id = request.thread_id
    checkpointer = app_request.app.state.checkpointer

    with timer("chat", logger, extra={"thread_id": thread_id}):

        # ------------------------------------------------------------------
        # Step 1 — Load current state from Postgres
        # ------------------------------------------------------------------
        graph = build_graph(checkpointer=checkpointer)
        config = {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "thread_id": thread_id,
                "current_phase": request.current_phase.value,
            }
        }

        # Get current state snapshot
        state_snapshot = await graph.aget_state(config)
        current_state = state_snapshot.values if state_snapshot.values else {}

        # ------------------------------------------------------------------
        # Step 2 — Handle acceptance BEFORE graph execution (D-002)
        # ------------------------------------------------------------------
        accepted_deliverables = list(
            current_state.get("accepted_deliverables", [])
        )
        current_phase = current_state.get("current_phase", PhaseName.DISCOVERY)

        if request.is_deliverable_accepted:
            phase_value = request.current_phase.value
            if phase_value not in accepted_deliverables:
                accepted_deliverables.append(phase_value)
                logger.info(
                    "Deliverable accepted",
                    extra={
                        "thread_id": thread_id,
                        "phase": phase_value,
                    }
                )

            # Advance current phase
            next_phase = get_next_phase(request.current_phase)
            if next_phase:
                current_phase = next_phase

            # Update state with acceptance before graph runs
            await graph.aupdate_state(
                config,
                {
                    "accepted_deliverables": accepted_deliverables,
                    "current_phase": current_phase,
                    "is_deliverable_accepted": True,
                },
                as_node="main_agent",
            )

        # ------------------------------------------------------------------
        # Step 3 — Reset tool_call_rounds for this turn
        # ------------------------------------------------------------------
        await graph.aupdate_state(
            config,
            {"tool_call_rounds": 0},
            as_node="main_agent",
        )

        # ------------------------------------------------------------------
        # Step 4 — Invoke LangGraph
        # ------------------------------------------------------------------
        graph_input = {
            "thread_id": thread_id,
            "messages": [HumanMessage(content=request.user_message)],
            "user_message": request.user_message,
            "current_phase": current_phase,
        }

        final_state = await graph.ainvoke(graph_input, config=config)

    # ------------------------------------------------------------------
    # Step 5 — Build response
    # ------------------------------------------------------------------
    response_text = final_state.get("ai_last_message", "")
    if not response_text:
        # Fallback: extract from last AI message
        messages = final_state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                if not getattr(msg, "tool_calls", None):
                    response_text = msg.content
                    break

    return ChatResponse(
        response=response_text,
        current_phase=final_state.get("current_phase", current_phase),
        suggested_next_phase=final_state.get("suggested_next_phase"),
        accepted_deliverables=final_state.get("accepted_deliverables", []),
        deliverables_markdown=final_state.get("deliverables_markdown", {}),
    )


# ------------------------------------------------------------------
# GET /deliverables/{thread_id}
# ------------------------------------------------------------------

@router.get(
    "/deliverables/{thread_id}",
    response_model=DeliverablesResponse
)
async def get_deliverables(
    thread_id: str,
    app_request: Request
) -> DeliverablesResponse:
    """
    Return all current deliverables for a thread.
    Both structured Pydantic objects and pre-rendered markdown.
    """
    checkpointer = app_request.app.state.checkpointer
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    state_snapshot = await graph.aget_state(config)

    if not state_snapshot or not state_snapshot.values:
        raise HTTPException(
            status_code=404,
            detail=f"Thread {thread_id} not found."
        )

    state = state_snapshot.values
    md = state.get("deliverables_markdown", {})

    return DeliverablesResponse(
        thread_id=thread_id,
        discovery=state.get("discovery_deliverable"),
        clustering=state.get("clustering_deliverable"),
        gap_analysis=state.get("gap_analysis_deliverable"),
        writing_outline=state.get("writing_outline_deliverable"),
        discovery_markdown=md.get("discovery"),
        clustering_markdown=md.get("clustering"),
        gap_analysis_markdown=md.get("gap_analysis"),
        writing_outline_markdown=md.get("writing_outline"),
        current_phase=state.get("current_phase", PhaseName.DISCOVERY),
        accepted_deliverables=state.get("accepted_deliverables", []),
    )


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def _build_ingested_sources(raw_docs, chunks) -> list[IngestedSourceMeta]:
    """Build IngestedSourceMeta list from raw docs and chunks."""
    from collections import defaultdict
    import hashlib

    source_data: dict[str, dict] = defaultdict(lambda: {
        "source_type": None,
        "chunk_count": 0,
        "languages": set(),
        "modalities": set(),
        "title": None,
    })

    for chunk in chunks:
        origin = chunk.origin
        source_data[origin]["source_type"] = chunk.source_type
        source_data[origin]["chunk_count"] += 1
        source_data[origin]["modalities"].add(chunk.modality)
        if chunk.lang:
            source_data[origin]["languages"].add(chunk.lang)
        if chunk.title:
            source_data[origin]["title"] = chunk.title

    result = []
    for origin, data in source_data.items():
        source_id = hashlib.sha256(origin.encode()).hexdigest()[:16]
        result.append(IngestedSourceMeta(
            source_id=source_id,
            source_type=data["source_type"],
            origin=origin,
            title=data["title"],
            chunk_count=data["chunk_count"],
            languages=list(data["languages"]),
            modalities=list(data["modalities"]),
        ))
    return result


def _build_ingestion_summary(
    ingested_sources: list[IngestedSourceMeta],
    failed_sources: list[str],
) -> IngestionSummary:
    """Build IngestionSummary from ingested sources."""
    from collections import Counter

    total_chunks = sum(s.chunk_count for s in ingested_sources)
    source_types = Counter(s.source_type.value for s in ingested_sources)
    all_languages = set()
    for s in ingested_sources:
        all_languages.update(s.languages)

    return IngestionSummary(
        total_sources=len(ingested_sources),
        total_chunks=total_chunks,
        sources_by_type=dict(source_types),
        languages_detected=list(all_languages),
        failed_sources=failed_sources,
    )


async def _run_profiling_pass(chunks, field_hint=None) -> ResearchProfile:
    """
    Run a lightweight LLM pass over sampled chunks to infer research profile.
    Returns a best-effort ResearchProfile seed (not ground truth).
    """
    from core.llm_factory import get_profiling_llm
    from llm.llm_schema.state_models import ResearchProfile
    import random

    try:
        # Sample representative text chunks
        text_chunks = [c for c in chunks if c.text]
        sample_size = min(settings.profiling_sample_chunks, len(text_chunks))
        sampled = random.sample(text_chunks, sample_size) if text_chunks else []

        if not sampled:
            return ResearchProfile()

        sample_text = "\n\n---\n\n".join(
            c.text[:300] for c in sampled
        )

        hint_text = f"\nField hint from user: {field_hint}" if field_hint else ""

        prompt = f"""Analyze these research document excerpts and infer:
1. The broad academic field (e.g., Computer Science, Biology, Economics)
2. The specific sub-field (e.g., Natural Language Processing, Genomics)
3. A draft research intent or question

Document excerpts:{hint_text}

{sample_text}

Return a ResearchProfile with your best-effort inference.
Set confidence between 0.0 and 1.0 based on how clear the evidence is."""

        llm = get_profiling_llm()
        structured_llm = llm.with_structured_output(ResearchProfile)
        profile: ResearchProfile = await structured_llm.ainvoke(prompt)
        return profile

    except Exception as e:
        logger.warning(
            "Profiling pass failed — returning empty profile",
            extra={"error": str(e)}
        )
        return ResearchProfile()