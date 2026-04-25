"""
llm/workflow/research_planner/node/summarize_agent.py
======================================================
Summarize agent node — compresses conversation history when token
count exceeds the threshold.

Replaces the full message history with a single SystemMessage summary.
Updates approx_prompt_tokens to reflect the compressed state.
"""

import yaml
from pathlib import Path

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage

from core.llm_factory import get_summarize_llm
from core.logging import get_logger, timer
from llm.llm_schema.state_models import SummarizationOutput
from llm.workflow.research_planner.graph_state import ResearchPlannerState
from utils.token_checker import count_messages_tokens

logger = get_logger(__name__)

# Path: llm/prompt/summarize_agent.yml
_PROMPT_PATH = (
    Path(__file__).parent.parent.parent.parent / "prompt" / "summarize_agent.yml"
)
_PROMPT_YAML: dict = yaml.safe_load(_PROMPT_PATH.read_text())

# Assemble system prompt from YAML sections in assembly order
_SYSTEM_PROMPT: str = "\n\n".join(
    _PROMPT_YAML.get(section, "").strip()
    for section in [
        "role",
        "compression_principles",
        "hard_constraints",
        "few_shot_example",
        "output_format",
    ]
    if _PROMPT_YAML.get(section, "").strip()
)

# User message template — uses {{conversation_history}} placeholder
_USER_TEMPLATE: str = _PROMPT_YAML.get(
    "user",
    "Summarize the following conversation:\n\n{{conversation_history}}"
)


async def summarize_agent_node(state: ResearchPlannerState) -> dict:
    """
    Summarize agent node — compresses message history.

    Reads full conversation history from state.
    Produces a compact summary preserving all research facts.
    Replaces messages with single SystemMessage summary.
    Updates approx_prompt_tokens.

    Returns state updates:
        messages: replaced with [SystemMessage(summary)]
        approx_prompt_tokens: updated to compressed count
    """
    with timer(
        "summarize_agent_node",
        logger,
        extra={
            "thread_id": state.thread_id,
            "messages_before": len(state.messages),
            "tokens_before": state.approx_prompt_tokens,
        }
    ):
        llm = get_summarize_llm()
        structured_llm = llm.with_structured_output(SummarizationOutput)

        # Format conversation for summarization
        conversation_text = _format_conversation(state)

        user_prompt = _USER_TEMPLATE.replace(
            "{{conversation_history}}", conversation_text
        )

        result: SummarizationOutput = await structured_llm.ainvoke(
            _SYSTEM_PROMPT + "\n\n" + user_prompt
        )

    logger.info(
        "Conversation compressed",
        extra={
            "thread_id": state.thread_id,
            "tokens_before": state.approx_prompt_tokens,
            "tokens_after": result.total_tokens_after,
        }
    )

    # Replace all existing messages with the compressed summary.
    # add_messages reducer appends by default, so we must explicitly
    # remove existing messages before adding the summary.
    summary_message = SystemMessage(
        content=f"CONVERSATION SUMMARY:\n{result.summary}"
    )
    removes = [
        RemoveMessage(id=m.id)
        for m in state.messages
        if getattr(m, "id", None)
    ]

    return {
        "messages": removes + [summary_message],
        "approx_prompt_tokens": result.total_tokens_after,
    }


def _format_conversation(state: ResearchPlannerState) -> str:
    """Format message history as readable text for summarization."""
    lines = []
    for msg in state.messages:
        role = type(msg).__name__.replace("Message", "")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        lines.append(f"{role}: {content[:1000]}")  # truncate very long messages
    return "\n\n".join(lines)
