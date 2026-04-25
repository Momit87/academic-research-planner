"""
llm/workflow/research_planner/node/main_agent.py
=================================================
Main agent node — the Socratic research coach.

Responsibilities:
- Builds system prompt with current phase state injected
- Calls LLM with all tools bound
- Updates approx_prompt_tokens and tool_call_rounds in state
- Injects finalization mode instruction when tool rounds hit limit
"""

import yaml
from pathlib import Path

from langchain_core.messages import AIMessage, SystemMessage

from core.config import get_settings
from core.llm_factory import build_main_agent_chain_with_tools
from core.logging import get_logger, timer
from llm.llm_schema.state_models import get_remaining_phases
from llm.workflow.research_planner.graph_state import ResearchPlannerState
from llm.workflow.research_planner.tool.query_corpus import query_corpus
from llm.workflow.research_planner.tool.firecrawl_search import firecrawl_search
from llm.workflow.research_planner.tool.generate_discovery_deliverable import (
    generate_discovery_deliverable,
)
from llm.workflow.research_planner.tool.generate_clustering_deliverable import (
    generate_clustering_deliverable,
)
from llm.workflow.research_planner.tool.generate_gap_analysis_deliverable import (
    generate_gap_analysis_deliverable,
)
from llm.workflow.research_planner.tool.generate_writing_outline_deliverable import (
    generate_writing_outline_deliverable,
)
from utils.token_checker import count_messages_tokens

logger = get_logger(__name__)

# Load prompt YAML once — path: llm/prompt/main_agent.yml
_PROMPT_PATH = Path(__file__).parent.parent.parent.parent / "prompt" / "main_agent.yml"
_PROMPT_YAML: dict = yaml.safe_load(_PROMPT_PATH.read_text())

# All tools for the main agent
_ALL_TOOLS = [
    query_corpus,
    firecrawl_search,
    generate_discovery_deliverable,
    generate_clustering_deliverable,
    generate_gap_analysis_deliverable,
    generate_writing_outline_deliverable,
]

# Build tools-bound fallback chain once at module load.
# Tools must be bound to each model BEFORE with_fallbacks() is called —
# binding after doesn't propagate correctly through RunnableWithFallbacks.
_LLM_WITH_TOOLS = build_main_agent_chain_with_tools(_ALL_TOOLS)


def _build_system_prompt(state: ResearchPlannerState) -> str:
    """Build the system prompt by assembling YAML sections with state injected."""
    settings = get_settings()

    remaining = get_remaining_phases(state.accepted_deliverables)
    remaining_str = ", ".join(p.value for p in remaining) or "none"

    checklist = state.phase_completion_checklist.get(state.current_phase.value, {})
    gathered = checklist if isinstance(checklist, list) else []

    # Select phase-specific guidance section
    phase_guidance = (
        _PROMPT_YAML.get("phase_guidance", {})
        .get(state.current_phase.value, "")
    )

    # Assemble sections in documented assembly order
    sections = [
        _PROMPT_YAML.get("role", ""),
        _PROMPT_YAML.get("reasoning_protocol", ""),
        _PROMPT_YAML.get("phase_awareness", ""),
        _PROMPT_YAML.get("available_tools", ""),
        phase_guidance,
        _PROMPT_YAML.get("guardrails", ""),
        _PROMPT_YAML.get("negative_examples", ""),
        _PROMPT_YAML.get("output_format", ""),
    ]

    # Add finalization notice when tool round cap is reached
    if state.tool_call_rounds >= settings.max_tool_rounds:
        sections.append(
            "⚠️ FINALIZATION MODE: You have reached the maximum tool call limit. "
            "Do NOT call any more tools. Synthesize all gathered information and "
            "produce your best response now."
        )

    prompt = "\n\n".join(s.strip() for s in sections if s and s.strip())

    # Substitute {{variable}} placeholders with actual state values
    replacements = {
        "{{current_phase}}": state.current_phase.value,
        "{{accepted_deliverables}}": str(state.accepted_deliverables or "none"),
        "{{remaining_phases}}": remaining_str,
        "{{phase_checklist_gathered}}": str(gathered or "none yet"),
        "{{phase_checklist_missing}}": "still gathering",
        "{{tool_call_rounds}}": str(state.tool_call_rounds),
        "{{max_tool_rounds}}": str(settings.max_tool_rounds),
        "{{field}}": state.field or "To be determined",
        "{{sub_field}}": state.sub_field or "To be determined",
        "{{research_intent}}": state.research_intent or "To be determined",
        "{{phase_hint}}": (
            f"Current phase: {state.current_phase.value}. "
            "Gather the required information to complete this phase and "
            "generate the corresponding deliverable."
        ),
    }
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)

    return prompt


async def main_agent_node(
    state: ResearchPlannerState,
) -> dict:
    """
    Main agent node — runs the Socratic research coach LLM.

    Returns state updates:
        messages: appended AI response
        approx_prompt_tokens: updated count
        tool_call_rounds: incremented if tool calls present
        ai_last_message: convenience field for API layer
    """
    settings = get_settings()

    with timer(
        "main_agent_node",
        logger,
        extra={
            "thread_id": state.thread_id,
            "phase": state.current_phase.value,
            "tool_rounds": state.tool_call_rounds,
        }
    ):
        # Build system prompt with injected state
        system_prompt = _build_system_prompt(state)

        # Build messages: system prompt + conversation history
        messages = [SystemMessage(content=system_prompt)] + list(state.messages)

        # Invoke LLM (tools already bound at module load via _LLM_WITH_TOOLS)
        response: AIMessage = await _LLM_WITH_TOOLS.ainvoke(messages)

    # Count tokens after response
    all_messages = list(state.messages) + [response]
    new_token_count = count_messages_tokens(all_messages)

    # Increment tool_call_rounds if this response has tool calls
    new_tool_rounds = state.tool_call_rounds
    if hasattr(response, "tool_calls") and response.tool_calls:
        new_tool_rounds += 1
        logger.info(
            "Tool calls emitted",
            extra={
                "thread_id": state.thread_id,
                "tools": [tc["name"] for tc in response.tool_calls],
                "round": new_tool_rounds,
            }
        )

    # Extract prose content for convenience field
    ai_last = response.content if isinstance(response.content, str) else ""

    return {
        "messages": [response],
        "approx_prompt_tokens": new_token_count,
        "tool_call_rounds": new_tool_rounds,
        "ai_last_message": ai_last,
    }
