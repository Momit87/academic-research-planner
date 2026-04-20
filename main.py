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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    """
    Manages application startup and shutdown.
    Code before yield runs on startup.
    Code after yield runs on shutdown.
    """
    # --- Startup ---
    logger.info(
        "Application starting",
        extra={
            "env": settings.app_env,
            "langsmith_tracing": settings.langchain_tracing_v2,
            "langsmith_project": settings.langchain_project,
            "python_version": "3.12",
        },
    )
    # Postgres checkpointer setup added in Milestone 9
    # LLM factory warm-up added in Milestone 8

    yield  # Application runs here

    # --- Shutdown ---
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
    # Routers (uncomment as milestones complete)
    # ------------------------------------------------------------------
    # from api.routers.research_planner import router as planner_router
    # app.include_router(planner_router, prefix="/research-planner")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    @app.get("/health", tags=["system"])
    async def health() -> dict:
        """Basic liveness check."""
        return {"status": "ok", "env": settings.app_env}

    return app


app = create_app()