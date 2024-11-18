import os
import logging
import asyncio
import json
import spacy

from pathlib import Path

from fastapi import FastAPI, WebSocket, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.websockets import WebSocketDisconnect

from elysia.tree.tree import Tree, lm, RecursionLimitException
from elysia.util.logging import backend_print
from elysia.util.api import parse_error
from elysia.api.api_types import (
    QueryData, 
    GetCollectionData, 
    GetCollectionsData, 
    NERData, 
    TitleData, 
    SetCollectionsData, 
    GetObjectData,
    ObjectRelevanceData,
    InitialiseTreeData
)
from elysia.util.collection_metadata import (
    get_collection_data_types,
    get_collection_data,
)
from elysia.globals.weaviate_client import client
from elysia.api.prompt_executors import TitleCreatorExecutor, ObjectRelevanceExecutor

# Load the English language model
nlp = spacy.load("en_core_web_sm")

collection_names = ["example_verba_github_issues", "example_verba_email_chains", "example_verba_slack_conversations"]

class TreeManager:
    """
    Manages trees for different conversations.
    """

    def __init__(self):
        self.trees = {}

    def add_tree(self, user_id: str, conversation_id: str):
        if user_id not in self.trees:
            self.trees[user_id] = {}
        if conversation_id not in self.trees[user_id]:
            self.trees[user_id][conversation_id] = Tree(verbosity=2, conversation_id=conversation_id, collection_names=collection_names)

    def get_tree(self, user_id: str, conversation_id: str):
        if user_id not in self.trees :
            self.add_tree(user_id, conversation_id)
        elif conversation_id not in self.trees[user_id]:
            self.add_tree(user_id, conversation_id)
        return self.trees[user_id][conversation_id]


# App variable declaration
version = "0.1.0"
logger = logging.getLogger("uvicorn")


# app declaration
app = FastAPI(title="Elysia API", version=version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables, to be changed to user-based later
global tree_manager
tree_manager = TreeManager()


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


@app.post("/api/initialise_tree")
async def initialise_tree(data: InitialiseTreeData):
    global tree_manager
    tree_manager.add_tree(data.user_id, data.conversation_id)
    return JSONResponse(
        content={
            "conversation_id": data.conversation_id,
            "tree": tree_manager.get_tree(data.user_id, data.conversation_id).tree, 
            "error": ""
        }, 
        status_code=200
    )

async def process(data: QueryData, websocket: WebSocket):
    global tree_manager
    user_prompt = data["query"]
    tree = tree_manager.get_tree(data["user_id"], data["conversation_id"])
    tree.soft_reset()
    try:
        async for yielded_result in tree.process(user_prompt):
            await websocket.send_json(yielded_result)
            await asyncio.sleep(0)
    except RecursionLimitException:
        pass

# Process endpoint
@app.websocket("/ws/query")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for processing pipelines.
    Handles real-time communication for pipeline execution and status updates.
    """
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

                # == main code ==
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
                    if data and "conversation_id" in data:
                        await websocket.send_json(parse_error(str(e), data["conversation_id"]))
                    else:
                        await websocket.send_json(parse_error(str(e), ""))
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

    # hardcoded for now

    # conversation_id = list(tree_manager.trees[data.user_id].keys())[0]
    # tree = tree_manager.get_tree(data.user_id, conversation_id)
    # collection_names = list(tree.decision_nodes["collection"].options.keys())

    # get collection metadata
    metadata = []
    for collection_name in collection_names:
        collection = client.collections.get(collection_name)
        metadata.append(
            {
                "name": collection_name,
                "total": len(collection),
                "vectorizer": collection.config.get().vectorizer,  # None when using namedvectors, TODO: implement for named vectors
            }
        )

    logger.info(f"Returning collections: {metadata}")

    return JSONResponse(content={"collections": metadata, "error": ""}, status_code=200)


@app.post("/api/get_collection")
async def get_collection(data: GetCollectionData):

    # get collection properties
    data_types = get_collection_data_types(data.collection_name)
    print(data_types)

    # obtain paginated results from collection
    items = get_collection_data(
        collection_name=data.collection_name,
        lower_bound=data.page * data.pageSize,
        upper_bound=(data.page + 1) * data.pageSize,
    )

    logger.info(f"Returning collection info for {data.collection_name}")
    return JSONResponse(
        content={"properties": data_types, "items": items, "error": ""}, status_code=200
    )

@app.post("/api/ner")
async def named_entity_recognition(data: NERData):
    """
    Performs Named Entity Recognition using spaCy.
    Returns a list of entities with their labels, start and end positions.
    """
    doc = nlp(data.text)
    
    out = {
        "text": data.text,  
        "entity_spans": [],
        "noun_spans": [],
        "error": ""
    }
    
    try:
        # Get entity spans
        for ent in doc.ents:
            out["entity_spans"].append((ent.start_char, ent.end_char))
        
        # Get noun spans
        for token in doc:
            if token.pos_ == "NOUN":
                span = doc[token.i:token.i + 1]  
                out["noun_spans"].append((span.start_char, span.end_char))
        
        logger.info(f"Returning NER results: {out}")
    except Exception as e:
        logger.error(f"Error in NER: {str(e)}")
        out["error"] = str(e)
    
    return JSONResponse(content=out, status_code=200)

@app.post("/api/title")
async def title(data: TitleData):
    try:
        title_creator = TitleCreatorExecutor()
        title = title_creator(data.text)
        logger.info(f"Returning title: {title.title}")
    except Exception as e:
        logger.error(f"Error in title: {str(e)}")
        out = {
            "title": "",
            "error": str(e)
        }
        return JSONResponse(content=out, status_code=200)

    out = {
        "title": title.title,
        "error": ""
    }
    return JSONResponse(content=out, status_code=200)

@app.post("/api/set_collections")
async def set_collections(data: SetCollectionsData):
    global tree_manager
    tree_manager.get_tree(data.user_id, data.conversation_id).set_collection_names(data.collection_names, remove_data=data.remove_data)
    return JSONResponse(content={"error": ""}, status_code=200)

@app.post("/api/get_object")
async def get_object(data: GetObjectData):
    error = ""

    collection = client.collections.get(data.collection_name)
    
    try:
        object = collection.query.fetch_object_by_id(data.uuid).properties
    except Exception as e:
        error = "No object found with this UUID."
    
    data_types = get_collection_data_types(data.collection_name)
    
    return JSONResponse(content={
        "properties": data_types,
        "items": [object],
        "error": error
    }, status_code=200)

@app.post("/api/object_relevance")
async def object_relevance(data: ObjectRelevanceData):
    error = ""
    object_relevance = ObjectRelevanceExecutor()
    try:
        prediction = object_relevance(data.user_prompt, data.objects)
        any_relevant = eval(prediction.any_relevant)
        assert isinstance(any_relevant, bool)
    except Exception as e:
        any_relevant = False
        error = str(e)
    return JSONResponse(
        content={
            "conversation_id": data.conversation_id,
            "any_relevant": any_relevant,
            "error": error
        }, 
        status_code=200
    )