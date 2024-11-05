import os
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError


from backend.tree.tree import Tree
from backend.util.logging import backend_print
from backend.api.types import ProcessData

# App variable declaration
version = "0.1.0"
logger = logging.getLogger("uvicorn")


# app declaration
app = FastAPI(title = "Elysia API", version = version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables, to be changed to user-based later
global tree
tree = Tree(verbosity=1)

# Request validation exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logging.error(f"{request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint.
    Returns a 200 status code with a "healthy" status if the API is running.
    """
    logger.info("Health check requested")
    return JSONResponse(content={"status": "healthy"}, status_code=200)

# Process endpoint
@app.post("/api/process")
async def process(data: ProcessData):
    logger.info(f"Processing user prompt: {data.user_prompt}")
    return tree.process(data.user_prompt)