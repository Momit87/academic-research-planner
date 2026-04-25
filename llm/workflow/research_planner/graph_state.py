"""
llm/workflow/research_planner/graph_state.py
=============================================
ResearchPlannerState — the single source of truth for the LangGraph workflow.

Every node reads from and writes to this state.
Nothing is passed between nodes as function arguments — it all goes through here.

Key design points:
    - messages uses add_messages reducer (append, never replace)
    - All other fields use plain assignment (last write wins)
    - tool_call_rounds resets to 0 at the start of each graph invocation
    - accepted_deliverables is a list of strings (phase name values)

See DECISIONS.md D-006 for tool loop design.
"""

from typing import Annotated, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from llm.llm_schema.deliverables import (
    ClusteringDeliverable,
    DiscoveryDeliverable,
    GapAnalysisDeliverable,
    WritingOutlineDeliverable,
)
from llm.llm_schema.state_models import (
    IngestedSourceMeta,
    PhaseName,
)


class ResearchPlannerState(BaseModel):
    """
    Complete state for the Research Planner LangGraph workflow.

    Persisted to Postgres via AsyncShallowPostgresSaver on every turn.
    Loaded at the start of each graph invocation via thread_id.

    Field groups:
        IDENTITY        — thread identification
        USER PROFILE    — seeded at onboarding, refined during Discovery
        INGESTION       — record of what was ingested
        CONVERSATION    — message history + convenience fields
        PHASE MGMT      — phase tracking, tool loop counter
        DELIVERABLES    — four phase outputs + rendered markdown
        TOKEN TRACKING  — for summarization gate
    """

    # ------------------------------------------------------------------
    # IDENTITY
    # ------------------------------------------------------------------
    thread_id: str = Field(
        default="",
        description="UUID identifying this research session"
    )
    is_onboarding: bool = Field(
        default=False,
        description="True during the initial onboarding graph run"
    )

    # ------------------------------------------------------------------
    # USER PROFILE
    # Seeded by onboarding profiling pass, refined by main_agent
    # ------------------------------------------------------------------
    field: Optional[str] = Field(
        default=None,
        description="Broad academic field, e.g. 'Computer Science'"
    )
    sub_field: Optional[str] = Field(
        default=None,
        description="Specific sub-field, e.g. 'Natural Language Processing'"
    )
    research_intent: Optional[str] = Field(
        default=None,
        description="Draft research question inferred from corpus"
    )
    target_output: Optional[str] = Field(
        default=None,
        description="Target paper type, e.g. 'Conference paper (8 pages)'"
    )
    deadline: Optional[str] = Field(
        default=None,
        description="Submission deadline if specified"
    )

    # ------------------------------------------------------------------
    # INGESTION RECORD
    # ------------------------------------------------------------------
    ingested_sources: list[IngestedSourceMeta] = Field(
        default_factory=list,
        description="Metadata for each ingested source"
    )

    # ------------------------------------------------------------------
    # CONVERSATION
    # messages MUST use add_messages reducer — it appends, never replaces.
    # Plain assignment would wipe history on every node return.
    # ------------------------------------------------------------------
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list,
        description="Full conversation history including tool calls/results"
    )
    user_message: Optional[str] = Field(
        default=None,
        description="Current turn's user message (convenience field)"
    )
    ai_last_message: Optional[str] = Field(
        default=None,
        description="Last AI prose response (convenience field for API layer)"
    )

    # ------------------------------------------------------------------
    # PHASE MANAGEMENT
    # ------------------------------------------------------------------
    current_phase: PhaseName = Field(
        default=PhaseName.DISCOVERY,
        description="Active research phase"
    )
    suggested_next_phase: Optional[PhaseName] = Field(
        default=None,
        description="Phase the main_agent recommends moving to next"
    )
    is_deliverable_accepted: bool = Field(
        default=False,
        description=(
            "Set by API layer before graph execution when user accepts "
            "previous deliverable. Read by agent for awareness only. "
            "See DECISIONS.md D-002."
        )
    )
    phase_completion_checklist: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Per-phase tracking: what info has been gathered. "
            "e.g. {'discovery': ['field', 'sub_field'], ...}"
        )
    )

    # Tool loop counter — RESETS TO 0 AT START OF EACH GRAPH INVOCATION
    # Incremented by main_agent node each time it emits tool_calls.
    # When >= MAX_TOOL_ROUNDS, finalization mode is injected into prompt.
    tool_call_rounds: int = Field(
        default=0,
        description="Number of tool-call rounds used in current turn"
    )

    # ------------------------------------------------------------------
    # DELIVERABLES
    # Plain assignment — last write wins (refinement replaces, not appends)
    # ------------------------------------------------------------------
    discovery_deliverable: Optional[DiscoveryDeliverable] = Field(
        default=None,
        description="Phase 1 output — set when generated, replaced when refined"
    )
    clustering_deliverable: Optional[ClusteringDeliverable] = Field(
        default=None,
        description="Phase 2 output"
    )
    gap_analysis_deliverable: Optional[GapAnalysisDeliverable] = Field(
        default=None,
        description="Phase 3 output"
    )
    writing_outline_deliverable: Optional[WritingOutlineDeliverable] = Field(
        default=None,
        description="Phase 4 output — the final research roadmap"
    )
    deliverables_markdown: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Deterministically rendered markdown per phase. "
            "{'discovery': '## Discovery...', 'clustering': '...'}"
        )
    )
    accepted_deliverables: list[str] = Field(
        default_factory=list,
        description=(
            "Phase name values with user-accepted deliverables. "
            "e.g. ['discovery', 'clustering']"
        )
    )

    # ------------------------------------------------------------------
    # TOKEN TRACKING
    # ------------------------------------------------------------------
    approx_prompt_tokens: int = Field(
        default=0,
        description=(
            "Approximate token count of current message history. "
            "Updated by main_agent and summarize_agent nodes. "
            "Checked by should_summarize conditional."
        )
    )

    class Config:
        # Allow AnyMessage and other non-standard types
        arbitrary_types_allowed = True
        