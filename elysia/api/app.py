from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Core
from elysia.api.core.config import settings
from elysia.api.core.logging import logger

# Routes
from elysia.api.routes import (
    query,
    collections,
    tree,
    processor,
    utils
)

# Middleware
from elysia.api.middleware.error_handlers import register_error_handlers

# Services
from elysia.api.services.tree import TreeManager

# Create FastAPI app instance
app = FastAPI(
    title="Elysia API",
    version=settings.VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register error handlers
register_error_handlers(app)

# Initialize global services
tree_manager = TreeManager(collection_names=settings.COLLECTION_NAMES)

# Include routers
app.include_router(utils.router, prefix="/api", tags=["utilities"])
app.include_router(collections.router, prefix="/api", tags=["collections"])
app.include_router(tree.router, prefix="/api", tags=["tree"])
app.include_router(query.router, prefix="/ws", tags=["websockets"])
app.include_router(processor.router, prefix="/ws", tags=["websockets"])

# Health check endpoint (kept in main app.py due to its simplicity)
@app.get("/api/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {"status": "healthy"}