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
from backend.api.types import ProcessData, GetCollectionData
from backend.globals.weaviate_client import client

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

@app.get("/api/collections")
async def collections():

    collection_names = list(tree.decision_nodes["collection"].options.keys())
    out = []
    for collection_name in collection_names:
        collection = client.collections.get(collection_name)
        out.append({
            "name": collection_name,
            "total": len(collection),
            "vectorizer": collection.config.get().vectorizer
        })

    return JSONResponse(content={
        "collections": out,
        "error": ""
    }, status_code=200)

@app.post("/api/get_collection")
async def get_collection(data: GetCollectionData):
    
    # get collection from client
    collection = client.collections.get(data.collection_name)

    # get collection properties
    config = collection.config.get()
    properties = config.properties

    # obtain paginated results from collection
    items = []
    for i, item in enumerate(collection.iterator()):
        if i >= data.page * data.pageSize and i < (data.page + 1) * data.pageSize:
            items.append(item.properties)


    return JSONResponse(content={
        "properties": {
            property.name: property.data_type[:] for property in properties
        },
        "items": items,
        "error": ""
    }, status_code=200)
