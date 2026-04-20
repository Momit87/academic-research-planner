"""
core/llm_factory.py
===================
Centralized LLM construction with fallback chains.

This is the ONLY file where model names, providers, and temperature are set.
Every agent and tool imports from here — never instantiate LLMs elsewhere.

Swapping a model:
    1. Edit the model name in .env
    2. That's it. No logic files change.

Fallback behavior:
    If the primary provider fails (rate limit, outage, timeout),
    LangChain automatically tries the next in the fallback list.
    temperature=0 on all calls — required by NFR-2 (determinism).

Usage:
    from core.llm_factory import get_main_agent_llm

    llm = get_main_agent_llm()
    response = await llm.ainvoke(messages)
"""

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Internal builder helpers
# ------------------------------------------------------------------

def _openai(model: str, **kwargs) -> ChatOpenAI:
    """Build a ChatOpenAI instance with standard settings."""
    settings = get_settings()
    return ChatOpenAI(
        model=model,
        temperature=0,
        api_key=settings.openai_api_key,
        **kwargs,
    )


def _anthropic(model: str, **kwargs) -> ChatAnthropic:
    """Build a ChatAnthropic instance with standard settings."""
    settings = get_settings()
    return ChatAnthropic(
        model=model,
        temperature=0,
        api_key=settings.anthropic_api_key,
        **kwargs,
    )


def _google(model: str, **kwargs) -> ChatGoogleGenerativeAI:
    """Build a ChatGoogleGenerativeAI instance with standard settings."""
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=settings.google_api_key,
        **kwargs,
    )


def _groq(model: str, **kwargs) -> ChatGroq:
    """Build a ChatGroq instance with standard settings."""
    settings = get_settings()
    return ChatGroq(
        model=model,
        temperature=0,
        api_key=settings.groq_api_key,
        **kwargs,
    )


# ------------------------------------------------------------------
# Public factory functions
# Each returns a fully configured fallback chain.
# lru_cache ensures we build each chain only once per process.
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_main_agent_llm() -> BaseChatModel:
    """
    LLM for the main_agent node.

    Requirements:
    - Strong tool-calling reliability (agent loop depends on it)
    - Large context window (full conversation history passed every turn)
    - Best reasoning capability in the stack

    Chain: OpenAI (primary) → Anthropic (fallback 1) → Gemini (fallback 2)

    Returns:
        BaseChatModel: fallback chain ready for .ainvoke() or .bind_tools()
    """
    settings = get_settings()

    primary = _openai(settings.main_agent_model)
    fallback_1 = _anthropic(settings.main_agent_fallback_1_model)
    fallback_2 = _google(settings.main_agent_fallback_2_model)

    chain = primary.with_fallbacks([fallback_1, fallback_2])

    logger.info(
        "Main agent LLM chain built",
        extra={
            "primary": settings.main_agent_model,
            "fallback_1": settings.main_agent_fallback_1_model,
            "fallback_2": settings.main_agent_fallback_2_model,
        },
    )
    return chain


@lru_cache(maxsize=1)
def get_summarize_llm() -> BaseChatModel:
    """
    LLM for the summarize_agent node.

    Requirements:
    - Fast inference (triggered when conversation is already long)
    - Adequate instruction-following (compress without losing facts)
    - Low cost (runs frequently in long sessions)

    Chain: Groq/Llama (primary) → OpenAI mini (fallback)

    Returns:
        BaseChatModel: fallback chain for summarization
    """
    settings = get_settings()

    primary = _groq(settings.summarize_model)
    fallback = _openai(settings.summarize_fallback_model)

    chain = primary.with_fallbacks([fallback])

    logger.info(
        "Summarize LLM chain built",
        extra={
            "primary": settings.summarize_model,
            "fallback": settings.summarize_fallback_model,
        },
    )
    return chain


@lru_cache(maxsize=1)
def get_deliverable_generator_llm() -> BaseChatModel:
    """
    LLM for the four deliverable generator tools.

    Requirements:
    - Reliable structured output (Pydantic via with_structured_output)
    - Good instruction-following for complex schemas
    - Moderate capability (schema complexity is high)

    Chain: Groq/Llama-70b (primary) → OpenAI mini (fallback)

    Note: Each generator tool calls .with_structured_output(PydanticModel)
    on the returned chain — this factory provides the base chain only.

    Returns:
        BaseChatModel: base chain; callers apply with_structured_output()
    """
    settings = get_settings()

    primary = _groq(settings.deliverable_model)
    fallback = _openai(settings.deliverable_fallback_model)

    chain = primary.with_fallbacks([fallback])

    logger.info(
        "Deliverable generator LLM chain built",
        extra={
            "primary": settings.deliverable_model,
            "fallback": settings.deliverable_fallback_model,
        },
    )
    return chain


@lru_cache(maxsize=1)
def get_profiling_llm() -> BaseChatModel:
    """
    LLM for the onboarding profiling pass.

    Requirements:
    - Good reading comprehension (infers field from raw chunk samples)
    - Reliable structured output (produces ResearchProfile)
    - Runs once per thread — cost is less critical than accuracy

    Chain: Gemini (primary) → OpenAI mini (fallback)

    Returns:
        BaseChatModel: base chain; caller applies with_structured_output()
    """
    settings = get_settings()

    primary = _google(settings.profiling_model)
    fallback = _openai(settings.profiling_fallback_model)

    chain = primary.with_fallbacks([fallback])

    logger.info(
        "Profiling LLM chain built",
        extra={
            "primary": settings.profiling_model,
            "fallback": settings.profiling_fallback_model,
        },
    )
    return chain


# ------------------------------------------------------------------
# Convenience: get any LLM by role name (useful in tests)
# ------------------------------------------------------------------

_FACTORY_MAP = {
    "main_agent": get_main_agent_llm,
    "summarize": get_summarize_llm,
    "deliverable_generator": get_deliverable_generator_llm,
    "profiling": get_profiling_llm,
}


def get_llm_by_role(role: str) -> BaseChatModel:
    """
    Return the LLM chain for a named role.

    Args:
        role: one of "main_agent", "summarize",
              "deliverable_generator", "profiling"

    Returns:
        BaseChatModel: the corresponding fallback chain

    Raises:
        ValueError: if role is not recognized
    """
    factory = _FACTORY_MAP.get(role)
    if factory is None:
        raise ValueError(
            f"Unknown LLM role: '{role}'. "
            f"Valid roles: {list(_FACTORY_MAP.keys())}"
        )
    return factory()


def clear_llm_cache() -> None:
    """
    Clear all cached LLM instances.

    Use in tests when you need to rebuild chains with
    different settings (after patching env vars).
    """
    get_main_agent_llm.cache_clear()
    get_summarize_llm.cache_clear()
    get_deliverable_generator_llm.cache_clear()
    get_profiling_llm.cache_clear()
    logger.debug("LLM factory cache cleared")