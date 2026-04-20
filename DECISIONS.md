# DECISIONS.md — Academic Research Planner

> Architectural Decision Record (ADR).
> Every non-obvious choice is documented here with its rationale.
> When you're confused about why something is built a certain way — check here first.

---

## D-001 — Embedding Provider: Cohere Embed v4

**Decision:** Use Cohere Embed v4 as the embedding model.

**Requirements it satisfies:**
- Multilingual: 100+ languages in a single embedding space
- Multimodal: text and image inputs embedded into the same vector space
- Single collection per thread (no per-language splits needed)

**Vector dimension:** 1024 (set via `QDRANT_VECTOR_SIZE` in `.env`)

**Abstraction:** `service/embedder.py` exposes a thin `Embedder` protocol.
Swapping to Gemini multimodal embeddings requires changing one file.

**Documented in:** `service/embedder.py`

---

## D-002 — Deliverable Acceptance: API Layer, Not Tools

**What the PRD says:**
FR-6 implies deliverable generator tools push to `accepted_deliverables`
when `is_deliverable_accepted` is True.

**What we build:**
Acceptance is handled by the FastAPI router **before** LangGraph execution.
Deliverable generator tools only generate and preview.
They never touch `accepted_deliverables`.

**Why:**
Acceptance refers to a deliverable the user has already seen (previous turn).
Generation happens in the current turn. Collapsing both into one tool call
means accepting something the user hasn't reviewed yet — breaking the UX contract.

**Consequence:**
`is_deliverable_accepted` on `ChatRequest` is consumed entirely by the router.
It is stored in state for agent awareness but no tool reads it
to decide whether to push to `accepted_deliverables`.

**Documented in:** `api/routers/research_planner.py`

---

## D-003 — Two Agents Only (main_agent + summarize_agent)

**Decision:** No phase router node, no safety node, no formatter node.

**Why:**
- Phase routing is non-linear. A router checking `current_phase` becomes
  a state machine with ever-growing exception cases.
- All four phase agents are structurally identical — same tools, same retrieval,
  same question-asking loop. Separate agents = maintenance duplication.
- Safety is enforced more intelligently by the main agent's system prompt
  than by a binary safety classifier node.
- Deterministic markdown rendering is a pure function — no LLM node needed.

**Result:** Graph stays simple. Intelligence lives in the LLM where it belongs.

---

## D-004 — LLM Provider Strategy and Swappability

**Decision:** Centralize all LLM construction in `core/llm_factory.py`.
Model names live in `.env` only — never in logic files.

**Provider assignments:**

| Role | Primary | Fallback |
|---|---|---|
| main_agent | OpenAI (gpt-4o) | Anthropic → Gemini |
| summarize_agent | Groq (llama-3.1-8b-instant) | OpenAI mini |
| deliverable_generators | Groq (llama-3.3-70b-versatile) | OpenAI mini |
| profiling_pass | Gemini (gemini-2.0-flash) | OpenAI mini |

**Swapping a model:** Edit `.env`. No logic files change.

**Groq rationale:** Fast inference for summarization and deliverable generation
where speed matters more than raw capability. Groq is significantly faster
than OpenAI/Anthropic for these tasks.

---

## D-005 — Checkpointer: AsyncShallowPostgresSaver

**Decision:** Use `AsyncShallowPostgresSaver` from `langgraph-checkpoint-postgres`.

**Implication:** Only the latest checkpoint is stored per thread.
Full replay and rewind are not supported.

**Why acceptable:**
- The use case is resumption, not replay.
- A researcher returning to their session needs current state, not history.
- Full checkpoint history multiplies storage without product benefit here.

**Observability compensation:**
Structured logging with `thread_id` on every log line provides
sufficient debugging context without full checkpoint history.

---

## D-006 — Tool Loop: Soft Cap at N=3

**Decision:** Main agent tool-call rounds are capped at 3 per turn.
At round 3, finalization mode is triggered — agent cannot call more tools
and must produce a response with gathered context.

**Why not hard stop:**
Cutting off reasoning mid-flow produces incoherent responses.
Finalization mode produces the best possible response given what was gathered.

**Implementation:**
`tool_call_rounds: int = 0` field in `ResearchPlannerState`.
Reset at the start of each graph invocation.
The finalization mode instruction is injected into the system prompt
when `tool_call_rounds >= MAX_TOOL_ROUNDS`.

---

## D-007 — Onboarding Profiling Pass

**Decision:** After ingestion, run a lightweight LLM call over
`PROFILING_SAMPLE_CHUNKS` sampled chunks to infer:
- `field`
- `sub_field`
- `research_intent` (draft)

**This is a seed state, not ground truth.**
The main agent refines this interactively during the Discovery phase.

**Why not start empty:**
Starting with an empty state forces the agent to ask basic questions
that could be inferred from the corpus. Seeding from the corpus
makes the first conversation turn immediately substantive.

**Model used:** Gemini (good reading comprehension, cost-effective for one-shot).

---

## D-008 — LangSmith Tracing

**Decision:** Enable LangSmith tracing for all LangChain/LangGraph operations.
Non-LangChain code (ingestion, Qdrant, Cohere) decorated with `@traceable`.

**Configuration:** Three env vars in `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=academic-research-planner
```

**Metadata attached to every trace:**
- `thread_id` — filter all traces for one conversation
- `current_phase` — filter traces by research phase
- `node` — filter traces by graph node

**Disable in CI:** Set `LANGCHAIN_TRACING_V2=false` in test environment.

---

## D-009 — InjectedState Tools

**Decision:** Five tools use `InjectedState` to read from graph state
without the LLM passing those values as arguments.

| Tool | Injected fields |
|---|---|
| `query_corpus` | `thread_id` |
| `generate_discovery_deliverable` | `thread_id`, `field`, `sub_field`, `research_intent`, `ingested_sources` |
| `generate_clustering_deliverable` | `thread_id`, `discovery_deliverable` (prerequisite) |
| `generate_gap_analysis_deliverable` | `thread_id`, `discovery_deliverable`, `clustering_deliverable` (prerequisite) |
| `generate_writing_outline_deliverable` | `thread_id`, `discovery_deliverable`, `clustering_deliverable`, `gap_analysis_deliverable` (prerequisite) |

`firecrawl_search` uses no `InjectedState` — the LLM passes the query string directly.

---

_Last updated: Milestone 0 — Foundation_
_Add a new entry for every non-obvious decision made during implementation._

---

## D-010 — Python 3.12 Compatibility

**Python version:** 3.12.3

**Two compatibility adjustments from the original spec:**

### Language Detection
`fasttext-langdetect` requires C++ compilation that fails on Python 3.12.
Replaced with `langdetect>=1.0.9` — pure Python, no build step, works on 3.12.

Trade-off: `langdetect` is slightly slower and less accurate than fastText
for short strings, but entirely adequate for chunk-level language detection
where chunks are 512 tokens (plenty of text to detect from).

### FastAPI Lifespan
`@app.on_event("startup")` is deprecated in FastAPI 0.115+ and emits
deprecation warnings on Python 3.12. Replaced with the `lifespan`
context manager pattern — the modern, warning-free equivalent.

---

_Last updated: Milestone 0 — Foundation (Python 3.12.3)_