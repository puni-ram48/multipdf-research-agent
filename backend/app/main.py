from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import logger
from app.api.v1.router import api_router
from app.db.database import init_db

# Import services to pre-warm models at server startup
from app.services.embedding import embedding_service
from app.services.pdf_parser import pdf_parser_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    logger.info(f"Starting up {settings.PROJECT_NAME} backend...")

    # 1. Initialize PostgreSQL Database Schemas
    try:
        init_db()
        logger.info("✅ PostgreSQL database schemas initialized successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Database connection deferred: {e}")

    # 2. Pre-warm models into RAM
    logger.info("Pre-warming models into RAM to eliminate ingestion cold-starts...")
    embedding_service.get_embeddings()
    _ = pdf_parser_service.digital_converter
    logger.info("✅ All core models pre-warmed successfully!")

    yield  # Application processes HTTP requests

    # --- SHUTDOWN LOGIC ---
    logger.info(f"Shutting down {settings.PROJECT_NAME} backend...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Setup (allows Streamlit frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "app": settings.PROJECT_NAME}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)