# Academic Research Planner

An AI-powered research coach that ingests heterogeneous research material and guides researchers through a structured four-phase workflow: **Discovery → Clustering → Gap Analysis → Writing Outline**.

Built with FastAPI, LangGraph, and a multimodal RAG pipeline backed by Qdrant and Cohere Embed v4. Every conversation turn is statefully persisted in PostgreSQL. All LLM calls have automatic fallback chains. Every deliverable is a typed Pydantic object rendered to markdown deterministically — no LLM involvement in rendering.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Tech Stack](#tech-stack)
3. [Feature Overview](#feature-overview)
4. [Project File Structure](#project-file-structure)
5. [API Endpoints](#api-endpoints)
6. [LangGraph Workflow](#langgraph-workflow)
   - [Graph Topology](#graph-topology)
   - [Nodes](#nodes)
   - [Tools](#tools)
   - [Graph State](#graph-state)
7. [Ingestion Pipeline](#ingestion-pipeline)
   - [Ingestors](#ingestors)
   - [Chunker](#chunker)
   - [Embedder](#embedder)
   - [Qdrant Vector Store](#qdrant-vector-store)
8. [LLM Factory and Model Configuration](#llm-factory-and-model-configuration)
9. [Data Models](#data-models)
   - [Deliverable Schemas](#deliverable-schemas)
10. [Utilities](#utilities)
11. [Configuration Reference](#configuration-reference)
12. [Setup and Running](#setup-and-running)
13. [End-to-End Request Flow](#end-to-end-request-flow)
14. [Troubleshooting](#troubleshooting)

---

## System Architecture

```
                         ┌──────────────────────────────────────────────────┐
                         │              FastAPI Application                 │
                         │                                                  │
  HTTP Client ──────────►│  POST /onboarding   POST /chat  GET /deliverables│
                         │                                                  │
                         └─────────┼────────────────┼───────────────┼───────┘
                                   │                │               │
                    ┌──────────────▼───┐    ┌───────▼──────┐        │
                    │Ingestion Pipeline│    │ LangGraph    │        │
                    │                  │    │ StateGraph   │        │
                    │ URL → Firecrawl  │    │              │        │
                    │ PDF → PyMuPDF    │    │ main_agent   │◄───────┘
                    │ DOCX → python-   │    │     │tools   │
                    │        docx      │    │     │        │
                    │                  │    │ summarize_   │
                    │                  │    │ agent        │
                    │ Image → Groq     │    │              │
                    │        Vision    │    └───────┬──────┘
                    │                  │            │
                    │ Chunker (tiktoken│            │ LLM calls
                    │ 512 tok, 64 ovlp)│     ┌──────▼───────────────────┐
                    └────────┬──────────┘    │ LLM Providers            │
                             │               │                          │
                    ┌────────▼──────────┐    │  Groq (primary)          │
                    │ Cohere Embed v4   │    │  ├─ llama-4-scout        │
                    │ Multimodal        │    │  └─ llama-3.1-8b-instant │
                    │ 1536-dim vectors  │    │                          │
                    └────────┬──────────┘    │  Gemini (fallback)       │
                             │               │  └─ gemini-2.0-flash     │
                    ┌────────▼──────────┐    └──────────────────────────┘
                    │ Qdrant            │
                    │ Per-thread        │     ┌──────────────────────────┐
                    │ collections       │     │ PostgreSQL               │
                    │ Cosine distance   │     │ LangGraph checkpointer   │
                    └───────────────────┘     │ Per-turn state snapshot  │
                                              └──────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| API | FastAPI 0.115, Uvicorn | HTTP server and request routing |
| Agent framework | LangGraph 0.2 | Stateful multi-turn agent workflow |
| LLM (primary) | Groq — llama-4-scout-17b-16e-instruct | Main agent tool calling and chat |
| LLM (summarize) | Groq — llama-3.1-8b-instant | Fast conversation compression |
| LLM (deliverables) | Groq — llama-3.3-70b-versatile | Structured output generation |
| LLM (fallback) | Google Gemini 2.0 Flash | Automatic fallback on Groq failures |
| LLM (vision) | Groq — llama-4-scout-17b-16e-instruct | Image captioning at ingestion |
| Embeddings | Cohere Embed v4 | Multilingual + multimodal 1536-d vectors |
| Vector store | Qdrant | Per-session RAG collections |
| State persistence | PostgreSQL + LangGraph AsyncShallowPostgresSaver | Cross-turn conversation state |
| PDF extraction | PyMuPDF (fitz) | Page-by-page text extraction |
| DOCX | python-docx| Office document text extraction |
| Web scraping | Firecrawl | Clean markdown extraction from URLs |
| Tokenization | tiktoken (cl100k_base) | Token-accurate chunking |
| Language detection | langdetect | Per-chunk ISO 639-1 language tagging |
| Observability | LangSmith | Full LLM call tracing |
| Infrastructure | Docker Compose | Qdrant + Postgres local setup |

---

## Feature Overview

### Multimodal Ingestion
- **PDFs** — page-by-page extraction via PyMuPDF; preserves page numbers in metadata
- **URLs** — clean markdown via Firecrawl; auto-detects PDF-at-URL and routes to PyMuPDF
- **DOCX** — paragraph extraction via python-docx; reads document title from core properties
- **Images (PNG/JPEG/WebP)** — dual-track: Groq vision LLM captions for agent comprehension + Cohere native image embedding for semantic retrieval; both live in the same 1536-d vector space as text

### RAG Pipeline
- Token-based chunking with configurable size (default 512 tokens) and overlap (default 64 tokens) using tiktoken cl100k_base
- Per-chunk language detection via langdetect
- Cohere Embed v4 multilingual + multimodal embeddings: text and images in the same vector space so text queries can surface image chunks
- Per-session Qdrant collections named `thread_{uuid}` — sessions are fully isolated
- Payload indexes on `source_type`, `lang`, `modality`, `source_id` for efficient filtered retrieval
- Post-retrieval filtering: minimum cosine score (0.15), deduplication (max 3 chunks per source)

### Stateful Multi-Turn Agent
- LangGraph `StateGraph` persisted to PostgreSQL via `AsyncShallowPostgresSaver`
- Each chat turn loads state by `thread_id`, runs the graph, and saves the updated state
- Conversation history accumulates using LangGraph's `add_messages` reducer (append-only, never replaced)
- Tool call loop with configurable hard cap (`max_tool_rounds`, default 3); finalization mode injected into prompt when cap is reached
- Automatic conversation summarization when token count exceeds `summarize_token_threshold` (default 12,000)

### Four-Phase Research Workflow
Each phase produces a typed, validated Pydantic deliverable rendered to markdown deterministically. The user explicitly accepts each phase before advancing.

| Phase | Deliverable | Key Content |
|---|---|---|
| Discovery | `DiscoveryDeliverable` | Field, research intent, corpus overview, venue constraints |
| Clustering | `ClusteringDeliverable` | 3-7 thematic clusters, cross-cluster relationships, orphan sources |
| Gap Analysis | `GapAnalysisDeliverable` | 3-6 research gaps with type/feasibility/novelty ratings, chosen gap |
| Writing Outline | `WritingOutlineDeliverable` | 3 title options, abstract draft, full section outline with per-paragraph intents and citations |

### LLM Provider Fallback
Every LLM role has a primary + fallback chain built via LangChain `with_fallbacks()`. If the primary model fails (rate limit, outage), the fallback triggers automatically with no code change needed. Tools must be bound to each model individually before the fallback chain is assembled — this is handled in `build_main_agent_chain_with_tools()`.

### Observability
- LangSmith tracing on all LLM calls, embeddings, and retrieval operations via `@traceable` decorators
- Structured JSON logging on every significant operation with a `timer()` context manager that logs elapsed time
- All log entries include `thread_id` for per-session filtering

---

## Project File Structure

```
academic-research-planner/
│
├── main.py                          # FastAPI app factory + lifespan manager
├── docker-compose.yml               # Qdrant + PostgreSQL local infrastructure
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variable template
│
├── api/
│   ├── routers/
│   │   └── research_planner.py      # All HTTP endpoints + onboarding pipeline orchestration
│   └── schema/
│       └── research_planner.py      # Pydantic request/response models for the API layer
│
├── core/
│   ├── config.py                    # Typed settings via pydantic-settings; all env vars live here
│   ├── llm_factory.py               # LLM instantiation with fallback chains; ONLY place models are built
│   └── logging.py                   # Structured JSON logger + timer() context manager
│
├── llm/
│   ├── llm_schema/
│   │   ├── state_models.py          # Shared enums and Pydantic models (ChunkRecord, RetrievedChunk, etc.)
│   │   └── deliverables.py          # Four deliverable schemas (DiscoveryDeliverable, etc.)
│   ├── prompt/
│   │   ├── main_agent.yml           # Socratic coach system prompt (YAML sections assembled at runtime)
│   │   └── summarize_agent.yml      # Compression agent system prompt
│   └── workflow/
│       └── research_planner/
│           ├── graph.py             # StateGraph builder: nodes, edges, compilation
│           ├── graph_state.py       # ResearchPlannerState TypedDict — all fields with reducers
│           ├── node/
│           │   ├── main_agent.py    # Main agent node: builds prompt, calls LLM, tracks token count
│           │   ├── summarize_agent.py  # Summarize node: compresses history, replaces messages
│           │   └── should_summarize.py # Pure conditional: routes to summarize or main based on token count
│           └── tool/
│               ├── query_corpus.py                      # RAG retrieval tool (InjectedState for thread_id)
│               ├── firecrawl_search.py                  # Live web search tool
│               ├── generate_discovery_deliverable.py    # Phase 1 generator (returns Command with state update)
│               ├── generate_clustering_deliverable.py   # Phase 2 generator (requires discovery accepted)
│               ├── generate_gap_analysis_deliverable.py # Phase 3 generator (requires clustering accepted)
│               └── generate_writing_outline_deliverable.py # Phase 4 generator (the final roadmap)
│
├── service/
│   ├── embedder.py                  # Cohere Embed v4 wrapper: embed_documents() + embed_query()
│   ├── qdrant.py                    # Qdrant collection lifecycle, upsert, search
│   ├── firecrawl.py                 # Firecrawl SDK wrapper: scrape() + search()
│   └── ingestion/
│       ├── chunker.py               # Chunker + RawDocument dataclass
│       ├── pdf_ingestor.py          # PyMuPDF page-by-page extraction
│       ├── url_ingestor.py          # URL scraping + PDF-at-URL detection
│       ├── image_ingestor.py        # Image captioning (Groq vision) + validation
│       └── doc_ingestor.py          # DOCX paragraph extraction slide extraction
│
├── utils/
│   ├── markdown_renderer.py         # Deterministic Pydantic → markdown (no LLM)
│   ├── filtering_logic.py           # Post-retrieval chunk filtering (score, dedup)
│   └── token_checker.py             # tiktoken-based message token counting
│
```

### File-by-file descriptions

**`main.py`**
FastAPI application factory. Configures CORS (permissive in development, restricted in production). Sets LangSmith env vars from settings before any LangChain import. Uses `asynccontextmanager` lifespan (replaces deprecated `@app.on_event`) to initialize the PostgreSQL checkpointer and warm up all LLM caches at startup. Registers a global exception handler that returns structured JSON for all 500 errors.

**`api/routers/research_planner.py`**
The core orchestration file. Contains all three endpoint handlers plus helper functions. `onboarding` runs the full 7-step ingestion pipeline (extract → chunk → embed → store → summarize → profile → seed state). `chat` handles deliverable acceptance before invoking LangGraph (DECISIONS D-002: acceptance is managed by the router, never by tools). `get_deliverables` reads state from Postgres and returns all deliverables with pre-rendered markdown.

**`api/schema/research_planner.py`**
Pydantic models for HTTP request/response bodies: `OnboardingRequest`, `OnboardingResponse`, `ChatRequest`, `ChatResponse`, `DeliverablesResponse`, `FileUpload`. Separate from state models to avoid coupling the API contract to internal state.

**`core/config.py`**
Single `Settings` class via `pydantic-settings`. All env vars are typed and validated at startup. `lru_cache` ensures `.env` is read exactly once. Includes validators for `database_url` (must use `asyncpg` driver), `qdrant_vector_size` (must be positive), and `chunk_token_overlap` (must be less than `chunk_token_size`). Call `get_settings.cache_clear()` in tests to reload with patched env vars.

**`core/llm_factory.py`**
The only file where LLM instances are created. Auto-detects provider from model name prefix (`gemini*` → Google, everything else → Groq). Provides four factory functions (`get_main_agent_llm`, `get_summarize_llm`, `get_deliverable_generator_llm`, `get_profiling_llm`) each returning a fallback chain. `build_main_agent_chain_with_tools()` binds tools to each model individually before assembling the fallback chain — required because `bind_tools()` doesn't propagate through `with_fallbacks()`. Temperature is always 0.

**`core/logging.py`**
Structured JSON logger using Python's `logging` module. Provides `get_logger(__name__)` for per-module loggers and `timer()`, a context manager that logs elapsed time in milliseconds with arbitrary extra fields.

**`llm/llm_schema/state_models.py`**
Pure data models with no LLM imports. Defines `PhaseName` enum, `SourceType`, `Modality`, `ChunkRecord`, `RetrievedChunk`, `IngestedSourceMeta`, `IngestionSummary`, `ResearchProfile`, `PhaseChecklist`, and `SummarizationOutput`. Also exports `get_next_phase()` and `get_remaining_phases()` helpers.

**`llm/llm_schema/deliverables.py`**
Four deliverable Pydantic schemas with all fields documented for LLM structured output clarity. `DiscoveryDeliverable`, `ClusteringDeliverable` (with `Cluster` and `Relationship` nested models), `GapAnalysisDeliverable` (with `Gap` nested model), `WritingOutlineDeliverable` (with `OutlineSection` and `CitationEntry` nested models).

**`llm/workflow/research_planner/graph.py`**
Assembles the LangGraph `StateGraph`. Defines routing logic in `_route_after_main_agent()` which checks `tool_call_rounds` against `max_tool_rounds` and the presence of `tool_calls` on the last message. `build_graph()` imports all nodes and tools lazily to break circular imports. Returns a `CompiledStateGraph` ready for `.ainvoke()`.

**`llm/workflow/research_planner/graph_state.py`**
`ResearchPlannerState` Pydantic model — the complete in-memory state for one research session. The `messages` field uses `Annotated[list[AnyMessage], add_messages]` for append-only behavior. All other fields use last-write-wins semantics. Contains: identity fields, user profile, ingestion record, conversation history, phase management, four deliverable slots, pre-rendered markdown dict, and token count.

---

## API Endpoints

### `GET /health`

Liveness check. No auth required.

```bash
curl http://localhost:8000/health
# → {"status": "ok", "env": "development"}
```

---

### `POST /research-planner/onboarding`

Ingests research material, embeds it into a per-session Qdrant collection, runs an LLM profiling pass to infer the research domain, and seeds the LangGraph state. Returns a `thread_id` for all subsequent calls.

**Request body:**
```json
{
  "urls": ["https://arxiv.org/pdf/1706.03762"],
  "pdfs": [
    {"filename": "paper.pdf", "base64_content": "<base64>", "mime_type": "application/pdf"}
  ],
  "images": [
    {"filename": "figure1.png", "base64_content": "<base64>", "mime_type": "image/png"}
  ],
  "docs": [
    {"filename": "notes.docx", "base64_content": "<base64>", "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
  ],
  "field_hint": "deep learning, transformers"
}
```

All source arrays are optional. Provide at least one source. `field_hint` is a free-text hint to improve profiling accuracy.

**Response (200):**
```json
{
  "thread_id": "7fb84082-5aae-4081-9171-b73f7a8b77a1",
  "profile": {
    "field": "Computer Science",
    "sub_field": "Natural Language Processing",
    "research_intent": "Improving sequence learning with self-attention mechanisms",
    "confidence": null
  },
  "ingestion_summary": {
    "total_sources": 2,
    "total_chunks": 47,
    "sources_by_type": {"url": 1, "image": 1},
    "languages_detected": ["en", "de"],
    "failed_sources": []
  },
  "phase_hint": "discovery"
}
```

**Error (422):** Returned if all sources fail ingestion or chunking produces no output.

---

### `POST /research-planner/chat`

Single conversational turn with the research agent. Handles deliverable acceptance and phase advancement before invoking LangGraph.

**Request body:**
```json
{
  "thread_id": "7fb84082-5aae-4081-9171-b73f7a8b77a1",
  "user_message": "What are the main architectural innovations in my corpus?",
  "current_phase": "discovery",
  "is_deliverable_accepted": false
}
```

`current_phase` must be one of: `"discovery"`, `"clustering"`, `"gap_analysis"`, `"writing_outline"`.

Set `is_deliverable_accepted: true` to accept the current phase's deliverable and advance. This is processed by the router before LangGraph runs — the agent is never responsible for phase transitions.

**Response (200):**
```json
{
  "response": "=== RESPONSE ===\nThe paper introduces...\n\n=== PHASE HINT ===\n...",
  "current_phase": "discovery",
  "suggested_next_phase": "clustering",
  "accepted_deliverables": [],
  "deliverables_markdown": {
    "discovery": "## 📚 Discovery\n\n### Field\n..."
  }
}
```

`deliverables_markdown` is populated when a deliverable has been generated. `suggested_next_phase` is set by the deliverable generator tools.

---

### `GET /research-planner/deliverables/{thread_id}`

Returns all generated deliverables for a session — both structured Pydantic objects and pre-rendered markdown strings.

**Response (200):**
```json
{
  "thread_id": "7fb84082...",
  "current_phase": "clustering",
  "accepted_deliverables": ["discovery"],
  "discovery": {
    "field_summary": "...",
    "research_intent": "...",
    "corpus_overview": {...},
    "constraints": {...},
    "target_output": "..."
  },
  "clustering": null,
  "gap_analysis": null,
  "writing_outline": null,
  "discovery_markdown": "## 📚 Discovery\n...",
  "clustering_markdown": null,
  "gap_analysis_markdown": null,
  "writing_outline_markdown": null
}
```

**Error (404):** Thread not found (onboarding not run, or Postgres was restarted).

---

## LangGraph Workflow

### Graph Topology

```
START
  │
  ▼
[should_summarize] ── "summarize" ──► [summarize_agent]
  │                                          │
  │ "main"                                   │
  ▼                                          ▼
[main_agent] ◄─────────────────────────────┘
  │
  ├── has tool_calls AND rounds < max ──► [tools] ──► [main_agent]  (loop)
  │
  └── no tool_calls OR rounds >= max ──► END
```

The graph loops between `main_agent` and `tools` until either the LLM stops emitting tool calls or `tool_call_rounds` reaches `max_tool_rounds` (default: 3). At that point, a finalization instruction is injected into the system prompt and the graph routes to `END`.

### Nodes

#### `should_summarize` (conditional router, no LLM)

**File:** `llm/workflow/research_planner/node/should_summarize.py`

Pure Python function. Reads `state.approx_prompt_tokens` and compares against `settings.summarize_token_threshold` (default 12,000). Returns the string `"summarize"` or `"main"`. Not an LLM node — zero latency.

- **Input state fields:** `approx_prompt_tokens`, `thread_id`
- **Returns:** routing string `"summarize"` | `"main"`

#### `summarize_agent`

**File:** `llm/workflow/research_planner/node/summarize_agent.py`

Triggered when conversation history grows large. Reads the full message history, calls the summarize LLM (`llama-3.1-8b-instant`) with structured output to produce a `SummarizationOutput`, then replaces all existing messages with a single `SystemMessage` containing the compressed summary. Uses `RemoveMessage` to explicitly delete existing messages before appending the summary — necessary because the `add_messages` reducer only appends by default.

- **LLM:** `get_summarize_llm()` (Groq llama-3.1-8b-instant → Gemini fallback)
- **Structured output:** `SummarizationOutput` (fields: `summary`, `total_tokens_after`)
- **Input state fields:** `messages`, `approx_prompt_tokens`, `thread_id`
- **Output state updates:** `messages` (replaced), `approx_prompt_tokens` (updated to compressed count)
- **Prompt:** assembled from `llm/prompt/summarize_agent.yml` sections: `role`, `compression_principles`, `hard_constraints`, `few_shot_example`, `output_format`

#### `main_agent`

**File:** `llm/workflow/research_planner/node/main_agent.py`

The Socratic research coach. Builds a dynamic system prompt from YAML sections with `{{variable}}` placeholders substituted from state. Calls the main agent LLM with all 6 tools bound. Tracks token count and tool call rounds.

- **LLM:** `build_main_agent_chain_with_tools(tools)` (Groq llama-4-scout → Gemini fallback, tools bound individually to each model)
- **Input state fields:** `messages`, `current_phase`, `accepted_deliverables`, `field`, `sub_field`, `research_intent`, `approx_prompt_tokens`, `tool_call_rounds`, `phase_completion_checklist`
- **Output state updates:** `messages` (appended AI response), `approx_prompt_tokens`, `tool_call_rounds` (incremented if tool calls emitted), `ai_last_message` (convenience field for API layer)
- **Prompt assembly** (from `llm/prompt/main_agent.yml`):
  1. `role` — Socratic coach persona and responsibilities
  2. `reasoning_protocol` — chain-of-thought and tool usage rules
  3. `phase_awareness` — current phase, accepted phases, remaining phases injected
  4. `available_tools` — what each tool does and when to use it
  5. `phase_guidance` — phase-specific instructions (selected by current phase)
  6. `guardrails` — what the agent must never do
  7. `negative_examples` — bad response examples to avoid
  8. `output_format` — structured response format
  9. Finalization notice (appended only when `tool_call_rounds >= max_tool_rounds`)
- **State substitutions:** `{{current_phase}}`, `{{accepted_deliverables}}`, `{{remaining_phases}}`, `{{phase_checklist_gathered}}`, `{{field}}`, `{{sub_field}}`, `{{research_intent}}`, `{{tool_call_rounds}}`, `{{max_tool_rounds}}`
- **Tools bound:** all 6 (see Tools section below)

#### `tools` (LangGraph ToolNode)

**File:** `llm/workflow/research_planner/graph.py` (configured inline)

LangGraph's built-in `ToolNode` wrapping all 6 tools. Receives the `tool_calls` from the last `main_agent` response, invokes the corresponding tool functions, and appends `ToolMessage` results to the message history. For tools that return `Command` objects (the four deliverable generators), the `Command.update` dict is merged directly into state, allowing tools to write fields like `discovery_deliverable` without going through the node return.

---

### Tools

All tools are `@tool`-decorated async functions bound to `main_agent`. Four of them use `InjectedState` to read graph state transparently (the LLM only sees the user-visible parameters). Four of them return `Command` objects to write deliverable data directly into state.

#### `query_corpus`

**File:** `llm/workflow/research_planner/tool/query_corpus.py`

RAG retrieval over the user's private Qdrant collection. The LLM provides a natural language `query`; the tool handles embedding and retrieval internally.

- **LLM-visible parameters:** `query: str`
- **InjectedState reads:** `thread_id`
- **Steps:**
  1. Embed `query` via `Embedder.embed_query()` with `input_type="search_query"`
  2. Search `QdrantService.search()` for top-k similar chunks
  3. Apply `apply_default_filters()` (score threshold 0.15, max 3 per source)
  4. Format results as numbered list with source, page, language, modality icon, and text preview
- **Returns:** Formatted string of retrieved chunks, or fallback message suggesting `firecrawl_search`

#### `firecrawl_search`

**File:** `llm/workflow/research_planner/tool/firecrawl_search.py`

Live web search via Firecrawl. Used when the corpus is insufficient or recency matters.

- **LLM-visible parameters:** `query: str`
- **InjectedState:** none
- **Steps:** calls `FirecrawlService.search(query, k=5)` and formats as numbered list with title, URL, and snippet
- **Returns:** Formatted string of search results, or fallback message

#### `generate_discovery_deliverable`

**File:** `llm/workflow/research_planner/tool/generate_discovery_deliverable.py`

Generates the Phase 1 deliverable. Takes no LLM-visible parameters — reads all context from `InjectedState`.

- **LLM-visible parameters:** none
- **InjectedState reads:** `thread_id`, `field`, `sub_field`, `research_intent`, `target_output`, `deadline`, `ingested_sources`, `deliverables_markdown`
- **Steps:**
  1. Builds a context prompt from state fields
  2. Calls `get_deliverable_generator_llm().with_structured_output(DiscoveryDeliverable)`
  3. Renders markdown via `render_discovery_markdown()`
- **Returns:** `Command` updating state with `discovery_deliverable`, `deliverables_markdown`, `suggested_next_phase: "clustering"`
- **Prerequisite:** none

#### `generate_clustering_deliverable`

**File:** `llm/workflow/research_planner/tool/generate_clustering_deliverable.py`

Generates the Phase 2 deliverable. Enforces that discovery is accepted before proceeding.

- **LLM-visible parameters:** none
- **InjectedState reads:** `thread_id`, `discovery_deliverable`, `accepted_deliverables`, `ingested_sources`, `deliverables_markdown`
- **Steps:** same pattern as discovery; uses discovery context in prompt; calls `.with_structured_output(ClusteringDeliverable)`
- **Returns:** `Command` updating state with `clustering_deliverable`, `deliverables_markdown`, `suggested_next_phase: "gap_analysis"`
- **Prerequisite:** `discovery_deliverable` not None AND `"discovery"` in `accepted_deliverables`

#### `generate_gap_analysis_deliverable`

**File:** `llm/workflow/research_planner/tool/generate_gap_analysis_deliverable.py`

Generates the Phase 3 deliverable. Uses cluster structure to identify research gaps.

- **LLM-visible parameters:** none
- **InjectedState reads:** `thread_id`, `discovery_deliverable`, `clustering_deliverable`, `accepted_deliverables`, `deliverables_markdown`
- **Steps:** formats cluster summaries and relationships into prompt; calls `.with_structured_output(GapAnalysisDeliverable)`
- **Returns:** `Command` updating state with `gap_analysis_deliverable`, `deliverables_markdown`, `suggested_next_phase: "writing_outline"`
- **Prerequisite:** `clustering_deliverable` not None AND `"clustering"` in `accepted_deliverables`

#### `generate_writing_outline_deliverable`

**File:** `llm/workflow/research_planner/tool/generate_writing_outline_deliverable.py`

Generates the Phase 4 (final) deliverable — the complete paper writing roadmap.

- **LLM-visible parameters:** none
- **InjectedState reads:** `thread_id`, `discovery_deliverable`, `clustering_deliverable`, `gap_analysis_deliverable`, `ingested_sources`, `deliverables_markdown`
- **Steps:** uses chosen gap, cluster themes, citation style, and source list in prompt; calls `.with_structured_output(WritingOutlineDeliverable)`
- **Returns:** `Command` updating state with `writing_outline_deliverable`, `deliverables_markdown`, `suggested_next_phase: null`
- **Prerequisite:** `gap_analysis_deliverable` not None AND `"gap_analysis"` in `accepted_deliverables`

---

### Graph State

**File:** `llm/workflow/research_planner/graph_state.py`

`ResearchPlannerState` is the single source of truth for the entire workflow. Every node reads from and writes to it. Persisted to Postgres on every turn via `AsyncShallowPostgresSaver`.

| Field | Type | Reducer | Description |
|---|---|---|---|
| `thread_id` | `str` | last-write | UUID for this research session |
| `is_onboarding` | `bool` | last-write | True during initial onboarding run |
| `field` | `Optional[str]` | last-write | Broad academic field |
| `sub_field` | `Optional[str]` | last-write | Specific sub-field |
| `research_intent` | `Optional[str]` | last-write | Draft research question |
| `target_output` | `Optional[str]` | last-write | Target paper type |
| `deadline` | `Optional[str]` | last-write | Submission deadline |
| `ingested_sources` | `list[IngestedSourceMeta]` | last-write | Metadata for all ingested sources |
| `messages` | `list[AnyMessage]` | **add_messages** | Full conversation history (append-only) |
| `user_message` | `Optional[str]` | last-write | Current turn's user message |
| `ai_last_message` | `Optional[str]` | last-write | Last AI prose response for API layer |
| `current_phase` | `PhaseName` | last-write | Active research phase |
| `suggested_next_phase` | `Optional[PhaseName]` | last-write | Agent's recommended next phase |
| `is_deliverable_accepted` | `bool` | last-write | Set by router before graph run |
| `phase_completion_checklist` | `dict[str, list[str]]` | last-write | Per-phase gathered fields |
| `tool_call_rounds` | `int` | last-write | Resets to 0 at start of each turn |
| `discovery_deliverable` | `Optional[DiscoveryDeliverable]` | last-write | Phase 1 output |
| `clustering_deliverable` | `Optional[ClusteringDeliverable]` | last-write | Phase 2 output |
| `gap_analysis_deliverable` | `Optional[GapAnalysisDeliverable]` | last-write | Phase 3 output |
| `writing_outline_deliverable` | `Optional[WritingOutlineDeliverable]` | last-write | Phase 4 output |
| `deliverables_markdown` | `dict[str, str]` | last-write | Pre-rendered markdown per phase |
| `accepted_deliverables` | `list[str]` | last-write | Phase names with accepted deliverables |
| `approx_prompt_tokens` | `int` | last-write | Token count for summarization gate |

---

## Ingestion Pipeline

The onboarding endpoint runs a 7-step pipeline synchronously before returning.

```
Step 1: Extract raw documents from all sources (parallel by type)
Step 2: Chunk all RawDocuments into ChunkRecords (tiktoken, 512t/64t overlap)
Step 3: Embed all ChunkRecords via Cohere Embed v4
Step 4: Create Qdrant collection + upsert vectors + payloads
Step 5: Build ingestion metadata (IngestedSourceMeta, IngestionSummary)
Step 6: Run profiling LLM pass over sampled chunks → ResearchProfile
Step 7: Seed LangGraph state via graph.aupdate_state()
```

### Ingestors

All ingestors output `list[RawDocument]` and are fed into the `Chunker`.

#### `PdfIngestor`

**File:** `service/ingestion/pdf_ingestor.py`

Uses `fitz` (PyMuPDF) to extract text page-by-page. Tries `get_text("text")` first; falls back to `get_text("blocks")` for complex layouts. Skips blank pages. Each page becomes one `RawDocument` with `page` number set. Reads document title from PDF metadata. Also exposes `ingest_bytes()` for use by `UrlIngestor` when a URL points directly to a PDF.

#### `UrlIngestor`

**File:** `service/ingestion/url_ingestor.py`

Calls `FirecrawlService.scrape()` to get clean markdown. Detects PDF-at-URL via `.endswith(".pdf")` and routes to `PdfIngestor.ingest_bytes()` via an `httpx` async download instead of scraping. Failed URLs are skipped with a warning — they do not abort the pipeline.

#### `ImageIngestor`

**File:** `service/ingestion/image_ingestor.py`

Dual-track ingestion:
1. **Captioning:** Calls Groq `meta-llama/llama-4-scout-17b-16e-instruct` vision API with the image as a base64 data URI. The academic-context caption is stored in `RawDocument.text` so the agent can describe the image in chat.
2. **Embedding:** Raw image bytes stored in `RawDocument.image_bytes` and passed to Cohere Embed v4 as a data URI for native multimodal embedding.

Validates MIME type against supported set (`image/png`, `image/jpeg`, `image/jpg`, `image/webp`). Detects actual MIME from magic bytes (`_detect_mime()`). Captioning failure is non-fatal — image is stored without caption.

#### `DocIngestor`

**File:** `service/ingestion/doc_ingestor.py`

Handles DOCX via file extension and MIME type detection.

- **DOCX:** `python-docx` extracts all non-empty paragraphs, joined into a single `RawDocument`. Reads title from `core_properties`.

### Chunker

**File:** `service/ingestion/chunker.py`

Converts `RawDocument` objects into `ChunkRecord` objects.

- **Text chunking:** tiktoken `cl100k_base` encoding. Sliding window of `chunk_token_size` (512) tokens with step `chunk_size - chunk_overlap` (448). Stops when remaining tokens ≤ overlap. Language detected on first chunk via `langdetect`.
- **Image passthrough:** Images are never split. One image = one `ChunkRecord`. Caption from `RawDocument.text` is passed through to `ChunkRecord.text` so it survives to Qdrant and is readable by the agent.
- `_make_source_id()`: SHA-256 of origin URL/filename, truncated to 16 hex chars
- `_make_chunk_id()`: SHA-256 of `source_id + chunk_index`, truncated to 16 hex chars

### Embedder

**File:** `service/embedder.py`

Cohere Embed v4 wrapper. Uses `input_type` parameter correctly:
- `embed_documents()` → `input_type="search_document"` for ingestion
- `embed_query()` → `input_type="search_query"` for retrieval (using the wrong type degrades retrieval quality significantly)

Text chunks are embedded in batches of 96. Image chunks are embedded in batches of 20 (images are larger). Images are sent as data URIs (`data:<mime>;base64,<b64>`). Returns `list[list[float]]` in same order as input chunks.

### Qdrant Vector Store

**File:** `service/qdrant.py`

- Collection name: `thread_{thread_id}` — one collection per research session
- Vector config: 1536 dimensions, cosine distance (required for Cohere embeddings)
- Payload indexes created on: `source_type`, `lang`, `modality`, `source_id` (keyword type)
- Point IDs: first 8 hex chars of `chunk_id` converted to int (deterministic)
- Upsert batched in groups of 100
- `search()` supports optional filters: `filter_modality`, `filter_lang`, `filter_source_type`
- `create_collection()` is idempotent — deletes and recreates if collection exists

---

## LLM Factory and Model Configuration

**File:** `core/llm_factory.py`

Four LLM roles, each with primary + fallback:

| Role | Primary (env var) | Fallback (env var) | Use |
|---|---|---|---|
| `main_agent` | `MAIN_AGENT_MODEL` | `MAIN_AGENT_FALLBACK_MODEL` | Socratic coach + tool calling |
| `summarize` | `SUMMARIZE_MODEL` | `SUMMARIZE_FALLBACK_MODEL` | Conversation compression |
| `deliverable_generator` | `DELIVERABLE_MODEL` | `DELIVERABLE_FALLBACK_MODEL` | Structured deliverable output |
| `profiling` | `PROFILING_MODEL` | `PROFILING_FALLBACK_MODEL` | One-shot onboarding domain inference |

Model names are auto-detected: `gemini*` or `models/*` → `ChatGoogleGenerativeAI`; everything else → `ChatGroq`. Temperature is always 0 (determinism).

**Critical:** for `main_agent`, use `build_main_agent_chain_with_tools(tools)` which binds tools to each model before `with_fallbacks()`. Binding after doesn't propagate correctly. This function is not cached (tools list isn't hashable) but is cheap to rebuild.

**Known model issue:** `llama-3.3-70b-versatile` on Groq has a broken tool-calling format — generates `<function=name{...}>` instead of proper JSON. Always use `meta-llama/llama-4-scout-17b-16e-instruct` for the main agent.

---

## Data Models

### Core Enums

**`PhaseName`** (`str` enum): `"discovery"`, `"clustering"`, `"gap_analysis"`, `"writing_outline"`. Inherits from `str` for clean JSON serialization.

**`SourceType`** (`str` enum): `"url"`, `"pdf"`, `"image"`, `"docx"`, 

**`Modality`** (`str` enum): `"text"`, `"image"`

### Ingestion Models

**`RawDocument`** (dataclass, `chunker.py`): Output of every ingestor, input to chunker. Fields: `source_type`, `origin`, `title`, `text`, `image_bytes`, `modality`, `page`, `slide`, `lang`.

**`ChunkRecord`** (Pydantic, `state_models.py`): Output of chunker. Stored as Qdrant payload. Fields: `chunk_id`, `source_id`, `source_type`, `origin`, `modality`, `text`, `image_bytes`, `page`, `slide`, `lang`, `title`, `chunk_index`, `ingested_at`.

**`RetrievedChunk`** (Pydantic, `state_models.py`): Returned by `query_corpus`. Includes `score: float` (cosine similarity). Does not carry `image_bytes`.

**`IngestedSourceMeta`** (Pydantic, `state_models.py`): One record per source (not per chunk). Stores: `source_id`, `source_type`, `origin`, `title`, `chunk_count`, `languages`, `modalities`, `ingested_at`.

**`IngestionSummary`** (Pydantic, `state_models.py`): Aggregate for `OnboardingResponse`. Fields: `total_sources`, `total_chunks`, `sources_by_type`, `languages_detected`, `failed_sources`.

**`ResearchProfile`** (Pydantic, `state_models.py`): Best-effort domain inference. Fields: `field`, `sub_field`, `research_intent`, `confidence`. Seeded at onboarding; refined by `main_agent`.

**`SummarizationOutput`** (Pydantic, `state_models.py`): Structured output from `summarize_agent`. Fields: `summary`, `total_tokens_after`.

### Deliverable Schemas

**`DiscoveryDeliverable`** (`deliverables.py`)
```
field_summary: str                   — 2-3 sentence field description
research_intent: str                 — specific research question/objective
corpus_overview: CorpusOverview      — total_sources, source_breakdown, languages,
                                       key_authors, date_range, dominant_themes
constraints: Constraints             — target_venue, page_limit, deadline, citation_style
target_output: str                   — e.g. "Conference paper (8 pages)"
```

**`ClusteringDeliverable`** (`deliverables.py`)
```
clusters: list[Cluster]              — 3-7 clusters, each with:
  cluster_id, label, description,     cluster_id, label, description
  source_ids, key_concepts,           source_ids (list), key_concepts (list)
  representative_source               representative_source
cross_cluster_relationships:         — list[Relationship]: cluster_a, cluster_b,
  list[Relationship]                  relationship_type (builds_on|contradicts|
                                       complements|overlaps|applies_to), description
orphan_sources: list[str]            — sources that don't fit any cluster
taxonomy_summary: str                — 2-3 sentence narrative of what clustering reveals
```

**`GapAnalysisDeliverable`** (`deliverables.py`)
```
gaps: list[Gap]                      — 3-6 gaps, each with:
                                       gap_id, title, description, evidence (list),
                                       gap_type (empirical|theoretical|methodological|
                                                 application|replication),
                                       feasibility (high|medium|low),
                                       novelty (high|medium|low)
chosen_gap: Optional[Gap]            — the gap the researcher chose to pursue
rationale: Optional[str]             — researcher's reasoning for gap choice
related_work_summary: str            — paragraph on existing work near the chosen gap
```

**`WritingOutlineDeliverable`** (`deliverables.py`)
```
title_options: list[str]             — 3 candidate paper titles
abstract_draft: str                  — 150-250 word abstract draft
sections: list[OutlineSection]       — 5-8 sections, each with:
                                       section_number, title,
                                       paragraph_intents (list[str]),
                                       citations (list[CitationEntry]:
                                         citation_key, origin, suggested_context),
                                       estimated_length
citation_style: Literal["APA","MLA","Chicago","IEEE"]
estimated_total_length: Optional[str]
```

---

## Utilities

### `utils/markdown_renderer.py`

Four pure functions — no LLM. `render_discovery_markdown()`, `render_clustering_markdown()`, `render_gap_analysis_markdown()`, `render_writing_outline_markdown()`. Plus `render_deliverable_markdown(phase, deliverable)` as a dispatcher. Called inside each deliverable generator tool after structured output is produced.

### `utils/filtering_logic.py`

Post-retrieval chunk filtering applied by `query_corpus`:
- `filter_by_score(chunks, min_score=0.15)` — removes chunks below cosine similarity threshold
- `filter_by_modality(chunks, modality)` — keeps only text or only image chunks
- `filter_by_language(chunks, lang)` — keeps only chunks in a specific language
- `apply_default_filters(chunks)` — runs score filter + sort descending + dedup (max 3 per source)

### `utils/token_checker.py`

`count_messages_tokens(messages)` — uses tiktoken `cl100k_base` to estimate prompt token count across a list of `AnyMessage` objects. Used by `main_agent_node` to update `approx_prompt_tokens` after each turn.

---

## Configuration Reference

All configuration is via environment variables. Copy `.env.example` to `.env`.

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | `development` or `production` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `GOOGLE_API_KEY` | required | Google AI (Gemini) API key |
| `GROQ_API_KEY` | required | Groq API key |
| `COHERE_API_KEY` | required | Cohere API key (Embed v4) |
| `FIRECRAWL_API_KEY` | required | Firecrawl API key |
| `DATABASE_URL` | required | `postgresql+asyncpg://user:pass@host:port/db` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | `""` | Qdrant API key (empty for local) |
| `QDRANT_VECTOR_SIZE` | `1536` | Cohere Embed v4 output dimension |
| `QDRANT_TOP_K` | `8` | Chunks returned per corpus query |
| `LANGCHAIN_TRACING_V2` | `true` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | `""` | LangSmith API key |
| `LANGCHAIN_PROJECT` | `academic-research-planner` | LangSmith project name |
| `MAIN_AGENT_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Main agent LLM (do NOT use llama-3.3-70b-versatile) |
| `MAIN_AGENT_FALLBACK_MODEL` | `gemini-2.0-flash` | Main agent fallback |
| `SUMMARIZE_MODEL` | `llama-3.1-8b-instant` | Summarization LLM |
| `SUMMARIZE_FALLBACK_MODEL` | `gemini-2.0-flash-lite` | Summarization fallback |
| `DELIVERABLE_MODEL` | `llama-3.3-70b-versatile` | Deliverable generator LLM |
| `DELIVERABLE_FALLBACK_MODEL` | `gemini-2.0-flash-lite` | Deliverable generator fallback |
| `PROFILING_MODEL` | `gemini-2.0-flash-lite` | Onboarding profiling LLM |
| `PROFILING_FALLBACK_MODEL` | `llama-3.1-8b-instant` | Profiling fallback |
| `VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Image captioning LLM (Groq) |
| `SUMMARIZE_TOKEN_THRESHOLD` | `12000` | Tokens above which summarization triggers |
| `MAX_TOOL_ROUNDS` | `3` | Max tool-call rounds per chat turn |
| `CHUNK_TOKEN_SIZE` | `512` | Max tokens per chunk |
| `CHUNK_TOKEN_OVERLAP` | `64` | Token overlap between adjacent chunks |
| `PROFILING_SAMPLE_CHUNKS` | `15` | Chunks sampled for onboarding profiling |

---

## Setup and Running

### 1. Clone and install

```bash
git clone <repo>
cd academic-research-planner

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in all required API keys
```

### 3. Start infrastructure

```bash
docker-compose up -d
# Qdrant → localhost:6333
# PostgreSQL → localhost:5433 (host port)
```

Verify:
```bash
curl http://localhost:6333/healthz   # → {"title": "qdrant - Ready"}
```

### 4. Run the server

```bash
uvicorn main:app --reload --port 8000
```

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 5. Run the end-to-end demo

```bash
bash demo.sh
```

This runs all 5 demo steps: health check, onboarding with the Attention Is All You Need paper, corpus query, discovery deliverable generation, phase acceptance, and deliverables retrieval.

---

## End-to-End Request Flow

### Onboarding (`POST /onboarding`)

```
HTTP request arrives
  │
  ├─ URL inputs → UrlIngestor.ingest()
  │     └─ Firecrawl.scrape() → markdown → RawDocument
  │         (if .pdf URL) → httpx download → PdfIngestor.ingest_bytes()
  │
  ├─ PDF inputs → PdfIngestor.ingest()
  │     └─ fitz.open() → page text → RawDocument (per page)
  │
  ├─ Image inputs → ImageIngestor.ingest()
  │     └─ Groq vision → caption → RawDocument (text=caption, image_bytes=bytes)
  │
  ├─ Doc inputs → DocIngestor.ingest()
  │     ├─ python-docx → paragraphs → RawDocument
  │     
  │
  ├─ Chunker.chunk(raw_docs)
  │     ├─ Text → tiktoken encode → sliding window → ChunkRecord list
  │     └─ Image → passthrough → single ChunkRecord (text=caption)
  │
  ├─ Embedder.embed_documents(chunks)
  │     ├─ Text batches → Cohere embed (search_document) → vectors
  │     └─ Image batches → Cohere embed (data URI, search_document) → vectors
  │
  ├─ QdrantService.create_collection(thread_id)
  ├─ QdrantService.upsert_chunks(thread_id, chunks, embeddings)
  │
  ├─ Profiling LLM pass → ResearchProfile
  │
  ├─ graph.aupdate_state(thread_id, initial_state)  → Postgres
  │
  └─ Return OnboardingResponse {thread_id, profile, ingestion_summary, phase_hint}
```

### Chat (`POST /chat`)

```
HTTP request arrives {thread_id, user_message, current_phase, is_deliverable_accepted}
  │
  ├─ [if is_deliverable_accepted]
  │     ├─ Append phase to accepted_deliverables
  │     ├─ Advance current_phase via get_next_phase()
  │     └─ graph.aupdate_state() with acceptance state
  │
  ├─ graph.aupdate_state() — reset tool_call_rounds = 0
  │
  ├─ graph.ainvoke({messages: [HumanMessage(user_message)], ...}, config)
  │     │
  │     ├─ should_summarize:
  │     │     approx_prompt_tokens > 12000? → summarize_agent → main_agent
  │     │     else → main_agent
  │     │
  │     ├─ main_agent:
  │     │     build_system_prompt(state) → YAML sections + substitutions
  │     │     LLM.ainvoke([SystemMessage(prompt)] + state.messages)
  │     │     → AIMessage with optional tool_calls
  │     │
  │     ├─ [if tool_calls AND rounds < max_tool_rounds]
  │     │     tools:
  │     │       query_corpus → embed query → Qdrant search → filtered chunks
  │     │       firecrawl_search → Firecrawl web search
  │     │       generate_*_deliverable → LLM structured output → Command(state_update)
  │     │     → back to main_agent
  │     │
  │     └─ [no tool_calls OR rounds >= max] → END
  │
  └─ Return ChatResponse {response, current_phase, suggested_next_phase,
                           accepted_deliverables, deliverables_markdown}
```

---

## Troubleshooting

**Server won't start**
- Check `.env` has all required API keys
- Postgres must be running on port 5433
- Qdrant must be running on port 6333
- `DATABASE_URL` must use `postgresql+asyncpg://` driver prefix

**`422 No content could be extracted from provided sources`**
- All sources failed ingestion
- For URLs: check if Firecrawl can access the URL
- For ArXiv: use `https://arxiv.org/pdf/XXXX` (PDF) not `https://arxiv.org/abs/XXXX` (abstract page only)

**Chat returns `500 BadRequestError: tool_use_failed` with `failed_generation: '<function=name{...}>'`**
- Wrong main agent model in `.env`
- `llama-3.3-70b-versatile` generates `<function=name{...}>` format instead of proper JSON tool calls
- Fix: set `MAIN_AGENT_MODEL=meta-llama/llama-4-scout-17b-16e-instruct`

**Chat returns empty response**
- Can be caused by the wrong model (see above)
- Can also occur if the agent hit the tool round cap with no final prose response — the `ai_last_message` field falls back to scanning `messages` for the last non-tool AIMessage

**Qdrant returns no results / score too low**
- Default minimum score is 0.15 (cosine similarity for Cohere Embed v4)
- Check Qdrant collection exists: `curl http://localhost:6333/collections`
- If collection is missing, re-run onboarding

**`404 Thread not found`**
- Postgres was restarted and lost checkpoint data
- Or `thread_id` from a different environment
- Re-run onboarding to create a new thread

**Gemini 429 quota errors**
- Gemini free tier: 50 requests/day per model
- Current config uses Groq as primary; Gemini is only triggered as fallback
- Quota resets at midnight Pacific time

**`InvalidUpdateError: Ambiguous update, specify as_node`**
- `graph.aupdate_state()` called without the `as_node` parameter
- Must pass `as_node="main_agent"` to all `aupdate_state()` calls in the router

**Image captioning failed but ingestion still succeeded**
- Captioning failure is non-fatal — image is stored without caption text
- The image bytes are still embedded by Cohere for semantic retrieval
- Agent will not be able to describe image content in chat without a caption

---

## License

MIT
