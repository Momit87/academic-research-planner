"""
main.py
=======
FastAPI application entry point.

Startup sequence:
1. Load settings (validates all env vars — fails fast if misconfigured)
2. Configure logging
3. Register routers
4. Lifespan: initialize Postgres checkpointer on startup

Run locally:
    uvicorn main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import get_settings
from core.logging import get_logger, setup_logging

# Initialize logging before anything else
setup_logging()
logger = get_logger(__name__)

# Load and validate settings at import time
# If any required env var is missing, this raises immediately
settings = get_settings()

# Set LangSmith env vars from settings
# LangChain reads these from os.environ — must be set before any LC import
os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project


# ------------------------------------------------------------------
# Lifespan — replaces deprecated @app.on_event (FastAPI 0.115+)
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    from langgraph.checkpoint.postgres.aio import AsyncShallowPostgresSaver
    from core.llm_factory import get_main_agent_llm, get_summarize_llm, get_deliverable_generator_llm

    # Convert SQLAlchemy-style URL to psycopg-compatible URL
    # langgraph-checkpoint-postgres v3+ uses psycopg which needs postgresql:// not postgresql+asyncpg://
    postgres_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")

    async with AsyncShallowPostgresSaver.from_conn_string(
        postgres_url
    ) as checkpointer:
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        get_main_agent_llm()
        get_summarize_llm()
        get_deliverable_generator_llm()
        logger.info("Application started")
        yield
    logger.info("Application shutting down")


# ------------------------------------------------------------------
# Application factory
# ------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Routers are registered here as milestones complete.
    Currently: no routers registered (Milestone 0).
    """
    app = FastAPI(
        title="Academic Research Planner",
        description=(
            "An agentic, RAG-grounded conversational coach that ingests "
            "research material and walks researchers through Discovery → "
            "Clustering → Gap Analysis → Writing Outline."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,          # replaces @app.on_event
    )

    # CORS — permissive in development, restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Global error handler — structured JSON errors for all 500s
    # ------------------------------------------------------------------
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled exception",
            extra={"path": str(request.url), "error": str(exc)},
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "type": type(exc).__name__,
                "details": str(exc),
            },
        )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    from api.routers.research_planner import router as planner_router
    app.include_router(planner_router, prefix="/research-planner")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    @app.get("/health", tags=["system"])
    async def health() -> dict:
        """Basic liveness check."""
        return {"status": "ok", "env": settings.app_env}

    return app


app = create_app()