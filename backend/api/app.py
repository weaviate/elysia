import os
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.websockets import WebSocketDisconnect

from backend.tree.tree import Tree, lm
from backend.util.logging import backend_print
from backend.api.types import ProcessData, GetCollectionData
from backend.util.collection_metadata import get_collection_data_types, get_collection_data
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

async def process(data: ProcessData, websocket: WebSocket):
    user_prompt = data.user_prompt
    async for yielded_result in tree.process(user_prompt):
        await websocket.send_json(yielded_result)

# Process endpoint
@app.post("/ws/query")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for processing pipelines.
    Handles real-time communication for pipeline execution and status updates.
    """
    global tree
    try:
        await websocket.accept()
        while True:
            try:
                
                # Wait for a message from the client
                data = await websocket.receive_json()

                # Check if the message is a disconnect request
                if data.get("type") == "disconnect":
                    logger.info("Received disconnect request")
                    break  # Exit the loop instead of closing the websocket here
                
                logger.info(f"Received message: {data}")

                await process(data, websocket)

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break  # Exit the loop on disconnect

            except RuntimeError as e:
                if (
                    "Cannot call 'receive' once a disconnect message has been received"
                    in str(e)
                ):
                    logger.info("WebSocket already disconnected")
                    break  # Exit the loop if the connection is already closed
                else:
                    raise  # Re-raise other RuntimeErrors

            except Exception as e:
                logger.error(f"Error in WebSocket: {str(e)}")
                try:
                    await websocket.send_json(
                        {
                            "status": "error",
                            "data": f"Error while processing query: {str(e)}",
                            "type": "ERROR",
                        },
                    )
                except RuntimeError:
                    logger.warning(
                        "Failed to send error message, WebSocket might be closed"
                    )
                break  # Exit the loop after sending the error message

    except Exception as e:
        logger.warning(f"Closing WebSocket: {str(e)}")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            logger.info("WebSocket already closed")

@app.get("/api/collections")
async def collections():

    # for now, only get collections that are in the tree
    collection_names = list(tree.decision_nodes["collection"].options.keys())

    # get collection metadata
    metadata = []
    for collection_name in collection_names:
        collection = client.collections.get(collection_name)
        metadata.append({
            "name": collection_name,
            "total": len(collection),
            "vectorizer": collection.config.get().vectorizer # None when using namedvectors, TODO: implement for named vectors
        })

    return JSONResponse(content={
        "collections": metadata,
        "error": ""
    }, status_code=200)

@app.post("/api/get_collection")
async def get_collection(data: GetCollectionData):
    
    # get collection properties
    data_types = get_collection_data_types(data.collection_name)

    # obtain paginated results from collection
    items = get_collection_data(
        collection_name = data.collection_name, 
        lower_bound = data.page * data.pageSize, 
        upper_bound = (data.page + 1) * data.pageSize
    )

    return JSONResponse(content={
        "properties": data_types,
        "items": items,
        "error": ""
    }, status_code=200)
