"""
core/config.py
==============
Typed application settings loaded from environment variables via Pydantic Settings.

Usage:
    from core.config import get_settings
    settings = get_settings()
    print(settings.google_api_key)

All environment variables are documented in .env.example.
Never call os.getenv() directly — always go through this module.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central settings object. Loaded once at startup via get_settings().
    Every field maps 1:1 to a variable in .env / .env.example.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,       # GOOGLE_API_KEY and google_api_key both work
        extra="ignore",             # ignore unknown env vars, don't raise
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_env: Literal["development", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ------------------------------------------------------------------
    # LLM Provider API Keys
    # Gemini + Groq only (free tier access)
    # ------------------------------------------------------------------
    google_api_key: str = Field(..., description="Google Gemini API key")
    groq_api_key: str = Field(..., description="Groq API key")

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    cohere_api_key: str = Field(..., description="Cohere API key for Embed v4")

    # ------------------------------------------------------------------
    # Vector Store — Qdrant
    # ------------------------------------------------------------------
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant instance URL"
    )
    qdrant_api_key: str = Field(
        default="",
        description="Qdrant API key — empty string for local dev"
    )
    qdrant_vector_size: int = Field(
        default=1536,
        description="Cohere embed-v4.0 output dimension (1536)"
    )
    qdrant_top_k: int = Field(
        default=8,
        description="Number of chunks returned per corpus query"
    )

    # ------------------------------------------------------------------
    # Postgres — LangGraph Checkpointer
    # ------------------------------------------------------------------
    database_url: str = Field(
        ...,
        description="Async Postgres connection string for LangGraph checkpointer"
    )

    # ------------------------------------------------------------------
    # Web Scraping + Search
    # ------------------------------------------------------------------
    firecrawl_api_key: str = Field(..., description="Firecrawl API key")

    # ------------------------------------------------------------------
    # Observability — LangSmith
    # ------------------------------------------------------------------
    langchain_tracing_v2: bool = Field(
        default=True,
        description="Enable LangSmith tracing"
    )
    langchain_api_key: str = Field(
        default="",
        description="LangSmith API key — empty disables tracing"
    )
    langchain_project: str = Field(
        default="academic-research-planner",
        description="LangSmith project name for trace grouping"
    )

    # ------------------------------------------------------------------
    # LLM Model Selection
    # These are the ONLY places model strings live.
    # Swap a model by editing .env — never touch logic files.
    # Providers: Gemini (reasoning) + Groq (speed)
    # ------------------------------------------------------------------

    # Main agent — strongest reasoning + tool-calling reliability
    # Gemini primary, Groq fallback
    main_agent_model: str = Field(default="gemini-2.0-flash")
    main_agent_fallback_model: str = Field(default="llama-3.3-70b-versatile")

    # Summarize agent — speed over raw capability
    # Groq primary, Gemini fallback
    summarize_model: str = Field(default="llama-3.1-8b-instant")
    summarize_fallback_model: str = Field(default="gemini-2.0-flash-lite")

    # Deliverable generators — structured output, moderate capability
    # Groq primary, Gemini fallback
    deliverable_model: str = Field(default="llama-3.3-70b-versatile")
    deliverable_fallback_model: str = Field(default="gemini-2.0-flash-lite")

    # Profiling LLM — runs once at onboarding to seed state
    # Gemini primary, Groq fallback
    profiling_model: str = Field(default="gemini-2.0-flash-lite")
    profiling_fallback_model: str = Field(default="llama-3.1-8b-instant")

    # ------------------------------------------------------------------
    # Runtime Thresholds
    # ------------------------------------------------------------------
    summarize_token_threshold: int = Field(
        default=12000,
        description="Token count above which summarize_agent is triggered"
    )
    max_tool_rounds: int = Field(
        default=3,
        description="Soft cap on tool-call rounds per main_agent turn"
    )

    # ------------------------------------------------------------------
    # Ingestion — Chunking
    # ------------------------------------------------------------------
    chunk_token_size: int = Field(
        default=512,
        description="Maximum tokens per chunk"
    )
    chunk_token_overlap: int = Field(
        default=64,
        description="Token overlap between consecutive chunks"
    )

    # ------------------------------------------------------------------
    # Onboarding — Profiling Pass
    # ------------------------------------------------------------------
    profiling_sample_chunks: int = Field(
        default=15,
        description="How many chunks to sample for the onboarding LLM profiling pass"
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """
        Ensure database URL uses asyncpg driver.
        LangGraph's AsyncShallowPostgresSaver requires an async connection.
        """
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must use the asyncpg driver. "
                "Format: postgresql+asyncpg://user:pass@host:port/dbname"
            )
        return v

    @field_validator("qdrant_vector_size")
    @classmethod
    def validate_vector_size(cls, v: int) -> int:
        """
        Cohere Embed v4 outputs 1536-dimensional vectors.
        Changing this without re-embedding all collections will break retrieval.
        """
        if v <= 0:
            raise ValueError("qdrant_vector_size must be a positive integer")
        return v

    @field_validator("chunk_token_overlap")
    @classmethod
    def validate_overlap_less_than_chunk(cls, v: int, info) -> int:
        """Overlap must be smaller than chunk size."""
        chunk_size = info.data.get("chunk_token_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_token_overlap ({v}) must be less than "
                f"chunk_token_size ({chunk_size})"
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Uses lru_cache so the .env file is read exactly once
    at first call, then cached for the lifetime of the process.
    In tests, call get_settings.cache_clear() to reload settings
    with different env vars.

    Returns:
        Settings: validated, fully populated settings object

    Raises:
        ValidationError: if required env vars are missing or invalid
    """
    return Settings()
