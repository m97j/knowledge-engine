# main.py

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware

from api.v1 import search, system
from core.config import settings
from core.exceptions import setup_exception_handlers
from core.logger import setup_logger
from models.embedder import TextEmbedder
from models.reranker import TextReranker
from scripts.setup_db import download_knowledge_base
from services.search_service import HybridSearchService
from storage.qdrant_client import QdrantStorage
from storage.sqlite_client import SQLiteStorage

logger = setup_logger("knowledge_engine")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan function to manage startup and shutdown events.
    On startup, it initializes all necessary components (DB connections, models, services) and injects them into the app state.
    On shutdown, it ensures that all resources are properly cleaned up (e.g., closing DB connections).
     - This approach centralizes all initialization logic in one place, making it easier to manage dependencies and handle errors during startup.
     - If any critical error occurs during startup, it logs the error and prevents the server from starting in an unstable state.
    """
    logger.info("🚀 Starting Knowledge Engine API...")

    qdrant_client = None
    sqlite_client = None

    try:
        # 0. Prepare dependency data (DB) (Download if unavailable, skip if available)
        logger.info("Checking and preparing Knowledge Base data...")
        download_knowledge_base()

        # 1. Infrastructure Connection (Database)
        qdrant_client = QdrantStorage(path=settings.QDRANT_PATH, collection_name=settings.QDRANT_COLLECTION)
        sqlite_client = SQLiteStorage(db_path=settings.SQLITE_PATH)
        
        # 2. Load AI Model (Singleton)
        embedder = TextEmbedder(model_name=settings.EMBEDDER_NAME, use_fp16=True)
        reranker = TextReranker(model_name=settings.RERANKER_NAME)
        
        # 3. Business Service Orchestration (Instantiate the HybridSearchService with all dependencies)
        search_service = HybridSearchService(
            qdrant=qdrant_client,
            sqlite=sqlite_client,
            embedder=embedder,
            reranker=reranker
        )
        
        # 4. Injecting services into FastAPI app state for global accessibility in routers
        app.state.search_service = search_service
        logger.info("✅ All services and models initialized successfully.")
        
        yield  # --- From this point, the server starts receiving traffic ---
        
    except Exception as e:
        logger.critical(f"❌ Application failed to start: {e}", exc_info=True)
        raise e
        
    finally:
        logger.info("🛑 Shutting down. Cleaning up resources...")
        # Safe termination of DB connections, etc.
        if qdrant_client is not None: qdrant_client.close()
        if sqlite_client is not None: sqlite_client.close()
        logger.info("Resources cleaned up.")

# ---------------------------
# FastAPI Instance Creation
# ---------------------------
app = FastAPI(
    title="Hybrid RAG Knowledge Engine API",
    description="Qdrant and BGE-M3-based high-performance hybrid search engine API",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS, JS, etc.) if needed (e.g., for demo pages)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------
# Router Registration
# ---------------------------
app.include_router(system.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/api/v1/search/demo")

# -----------------------------------
# Register global exception handlers
# -----------------------------------
setup_exception_handlers(app)