"""
utils/token_checker.py
========================
Approximate token counting for conversation history.

Used by:
    - should_summarize node: checks if history exceeds threshold
    - main_agent node: updates approx_prompt_tokens after each turn
    - summarize_agent node: reports tokens after compression

Uses tiktoken cl100k_base — good approximation across all providers.
"""

import tiktoken
from langchain_core.messages import AnyMessage

from core.logging import get_logger

logger = get_logger(__name__)

# Shared encoder instance — created once, reused
_encoder = tiktoken.get_encoding("cl100k_base")


def count_messages_tokens(messages: list[AnyMessage]) -> int:
    """
    Count approximate tokens across a list of LangChain messages.

    Each message contributes:
        - ~4 tokens for message overhead (role, separators)
        - tokens in the content string

    Args:
        messages: list of AnyMessage (HumanMessage, AIMessage, etc.)

    Returns:
        approximate total token count
    """
    total = 0
    for message in messages:
        # Message overhead
        total += 4

        # Content tokens
        content = message.content
        if isinstance(content, str):
            total += len(_encoder.encode(content))
        elif isinstance(content, list):
            # Multi-part content (text + tool calls)
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text", "")
                    if text:
                        total += len(_encoder.encode(str(text)))
                elif isinstance(part, str):
                    total += len(_encoder.encode(part))

    return total


def count_text_tokens(text: str) -> int:
    """
    Count tokens in a plain string.

    Args:
        text: any string

    Returns:
        approximate token count
    """
    return len(_encoder.encode(text))


def is_over_threshold(
    messages: list[AnyMessage],
    threshold: int
) -> bool:
    """
    Check if message history exceeds the summarization threshold.

    Args:
        messages: current conversation messages
        threshold: token limit (from settings.summarize_token_threshold)

    Returns:
        True if token count exceeds threshold
    """
    count = count_messages_tokens(messages)
    over = count > threshold

    if over:
        logger.info(
            "Token threshold exceeded — summarization triggered",
            extra={"token_count": count, "threshold": threshold}
        )

    return over
