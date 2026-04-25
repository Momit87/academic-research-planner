# Academic Research Planner

An AI-powered research coach that ingests heterogeneous research material and guides researchers through a structured four-phase workflow: **Discovery → Clustering → Gap Analysis → Writing Outline**.

Built with FastAPI, LangGraph, and a multimodal RAG pipeline backed by Qdrant and Cohere Embed v4.

---

## Overview

Researchers upload papers, documents, images, and URLs at onboarding. The system extracts, chunks, and embeds all content into a per-session vector store. A LangGraph agent then conducts a multi-turn conversation, generating structured deliverables at each research phase. Every phase output can be accepted or revised before the agent advances.

```
User Material                       Agent Workflow
─────────────                       ──────────────
PDFs        ┐                       ┌─ Discovery Deliverable
DOCX/PPTX   ├─► Ingest ─► Embed ──►│  Clustering Deliverable
Images      │   (Cohere  (Qdrant)   │  Gap Analysis Deliverable
URLs        ┘    v4)                └─ Writing Outline Deliverable
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI 0.115, Uvicorn |
| Agent framework | LangGraph 0.2 |
| LLM providers | Groq (llama-3.3-70b, llama-3.1-8b) · Google Gemini 2.0 (fallback) |
| Embeddings | Cohere Embed v4 — multilingual + multimodal |
| Vector store | Qdrant — one collection per session |
| State persistence | PostgreSQL via LangGraph `AsyncShallowPostgresSaver` |
| PDF extraction | PyMuPDF (fitz) |
| DOCX / PPTX | python-docx · python-pptx |
| Web scraping | Firecrawl |
| Observability | LangSmith tracing |
| Infrastructure | Docker Compose (Qdrant + Postgres) |

---

## Prerequisites

- Python 3.12+
- Docker and Docker Compose
- API keys for: Groq, Google AI (Gemini), Cohere, Firecrawl, LangSmith

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/Momit87/academic-research-planner.git
cd academic-research-planner

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys (see [Configuration](#configuration) below).

### 3. Start infrastructure

```bash
docker-compose up -d
```

This starts:
- **Qdrant** on `localhost:6333` (vector store)
- **PostgreSQL** on `localhost:5433` (LangGraph checkpointer)

### 4. Run the server

```bash
uvicorn main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`.

---

## API Reference

### `POST /research-planner/onboarding`

Ingests research material, embeds it, runs a profiling pass, and returns a `thread_id` for subsequent chat turns.

**Request body**

```json
{
  "urls": ["https://example.com/paper"],
  "pdfs": [
    {
      "filename": "paper.pdf",
      "base64_content": "<base64-encoded PDF bytes>",
      "mime_type": "application/pdf"
    }
  ],
  "images": [
    {
      "filename": "figure1.png",
      "base64_content": "<base64-encoded image bytes>",
      "mime_type": "image/png"
    }
  ],
  "docs": [
    {
      "filename": "notes.docx",
      "base64_content": "<base64-encoded DOCX bytes>",
      "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }
  ],
  "field_hint": "Natural Language Processing"
}
```

All source fields are optional — provide at least one. `field_hint` is an optional plain-text hint to improve profiling accuracy.

**Response**

```json
{
  "thread_id": "uuid",
  "profile": {
    "field": "Computer Science",
    "sub_field": "Natural Language Processing",
    "research_intent": "...",
    "confidence": 0.85
  },
  "ingestion_summary": {
    "total_sources": 3,
    "total_chunks": 142,
    "sources_by_type": { "pdf": 1, "url": 1, "image": 1 },
    "languages_detected": ["en"],
    "failed_sources": []
  },
  "phase_hint": "discovery"
}
```

---

### `POST /research-planner/chat`

Single conversational turn with the research agent. Pass `is_deliverable_accepted: true` to accept the current phase output and advance.

**Request body**

```json
{
  "thread_id": "uuid",
  "user_message": "Can you expand on the third theme?",
  "current_phase": "discovery",
  "is_deliverable_accepted": false
}
```

**Response**

```json
{
  "response": "Agent reply...",
  "current_phase": "discovery",
  "suggested_next_phase": "clustering",
  "accepted_deliverables": [],
  "deliverables_markdown": {
    "discovery": "## Discovery\n..."
  }
}
```

---

### `GET /research-planner/deliverables/{thread_id}`

Returns all generated deliverables for a session in both structured and Markdown form.

**Response**

```json
{
  "thread_id": "uuid",
  "current_phase": "clustering",
  "accepted_deliverables": ["discovery"],
  "discovery": { ... },
  "clustering": { ... },
  "gap_analysis": null,
  "writing_outline": null,
  "discovery_markdown": "## Discovery\n...",
  "clustering_markdown": null,
  "gap_analysis_markdown": null,
  "writing_outline_markdown": null
}
```

---

### `GET /health`

Liveness check.

```json
{ "status": "ok", "env": "development" }
```

---

## Research Phases

| Phase | Description |
|---|---|
| **Discovery** | Identifies core themes, key papers, and the research landscape |
| **Clustering** | Groups literature into conceptual clusters |
| **Gap Analysis** | Identifies unexplored areas and open research questions |
| **Writing Outline** | Produces a structured outline for a literature review or paper |

Each phase produces a structured deliverable. The agent stays in the current phase until the user accepts the output via `is_deliverable_accepted: true`.

---

## Project Structure

```
academic-research-planner/
├── main.py                    # FastAPI app factory + lifespan
├── docker-compose.yml         # Qdrant + Postgres
├── requirements.txt
├── .env.example
│
├── api/
│   ├── routers/
│   │   └── research_planner.py   # All HTTP endpoints
│   └── schema/
│       └── research_planner.py   # Pydantic request/response models
│
├── core/
│   ├── config.py              # Typed settings via pydantic-settings
│   ├── llm_factory.py         # LLM instantiation (Groq + Gemini fallback)
│   └── logging.py             # Structured logging + timer context manager
│
├── llm/
│   ├── llm_schema/
│   │   ├── state_models.py    # All Pydantic state/data models
│   │   └── deliverables.py    # Deliverable schemas per phase
│   ├── prompt/                # YAML prompt templates
│   └── workflow/
│       └── research_planner/
│           ├── graph.py       # LangGraph graph builder
│           ├── graph_state.py # ResearchPlannerState TypedDict
│           ├── node/          # Agent, summarize, deliverable nodes
│           └── tool/          # query_corpus, firecrawl_search tools
│
├── service/
│   ├── embedder.py            # Cohere Embed v4 wrapper
│   ├── qdrant.py              # Qdrant collection + search operations
│   ├── firecrawl.py           # Firecrawl scrape + search
│   └── ingestion/
│       ├── chunker.py         # Token-aware chunker with overlap
│       ├── pdf_ingestor.py    # PyMuPDF PDF → RawDocument
│       ├── url_ingestor.py    # URL → RawDocument via Firecrawl
│       ├── image_ingestor.py  # Image → ChunkRecord (multimodal)
│       └── doc_ingestor.py    # DOCX/PPTX → RawDocument
│
├── model/                     # Shared data models
├── utils/                     # Token counting, helpers
└── tests/                     # Pytest test suite
```

---

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and set these values:

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI (Gemini) API key |
| `GROQ_API_KEY` | Groq API key |
| `COHERE_API_KEY` | Cohere API key (embeddings) |
| `FIRECRAWL_API_KEY` | Firecrawl API key (web scraping) |
| `LANGCHAIN_API_KEY` | LangSmith API key |
| `LANGCHAIN_PROJECT` | LangSmith project name |
| `LANGCHAIN_TRACING_V2` | `true` to enable LangSmith tracing |
| `DATABASE_URL` | PostgreSQL connection string |
| `QDRANT_URL` | Qdrant server URL |
| `MAIN_AGENT_MODEL` | Primary LLM for the main agent |
| `MAIN_AGENT_FALLBACK_MODEL` | Fallback LLM (Gemini) |
| `SUMMARIZE_TOKEN_THRESHOLD` | Token count that triggers summarization |
| `MAX_TOOL_ROUNDS` | Max tool calls per chat turn |
| `QDRANT_TOP_K` | Chunks returned per RAG query |
| `CHUNK_TOKEN_SIZE` | Tokens per chunk (default 512) |
| `CHUNK_TOKEN_OVERLAP` | Overlap between adjacent chunks |

---

## License

MIT
