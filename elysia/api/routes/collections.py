from fastapi import APIRouter, Depends
from elysia.api.api_types import InitialiseTreeData
from elysia.api.services.tree import TreeManager
from elysia.api.dependencies.common import get_tree_manager
from fastapi.responses import JSONResponse
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

import asyncio

# Util
from elysia.util.collection_metadata import (
    get_collection_data_types,
    get_collection_data,
)

# Websocket
from elysia.api.utils.websocket import help_websocket

# Preprocessing
from elysia.preprocess.collection import CollectionPreprocessor

# API Types
from elysia.api.api_types import (
    GetCollectionData, 
    CollectionsData,
    ProcessCollectionData,
    SetCollectionsData,
    GetObjectData,
    CollectionMetadataData
)

# client
from elysia.globals.weaviate_client import client

# Logging
from elysia.api.core.logging import logger

router = APIRouter()

@router.get("/collections")
async def collections(
    tree_manager: TreeManager = Depends(get_tree_manager)
):

    # get collection metadata
    metadata = []
    for collection_name in tree_manager.collection_names:
        collection = client.collections.get(collection_name)
        metadata.append(
            {
                "name": collection_name,
                "total": len(collection),
                "vectorizer": collection.config.get().vectorizer,  # None when using namedvectors, TODO: implement for named vectors
                "processed": client.collections.exists(f"ELYSIA_METADATA_{collection_name}__")
            }
        )

    logger.info(f"Returning collections: {metadata}")

    return JSONResponse(content={"collections": metadata, "error": ""}, status_code=200)

@router.post("/get_collection")
async def get_collection(data: GetCollectionData):

    # get collection properties
    data_types = get_collection_data_types(data.collection_name)

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


@router.post("/set_collections")
async def set_collections(
    data: SetCollectionsData,
    tree_manager: TreeManager = Depends(get_tree_manager)
):
    tree_manager.get_tree(data.user_id, data.conversation_id).set_collection_names(data.collection_names, remove_data=data.remove_data)
    return JSONResponse(content={"error": ""}, status_code=200)

@router.post("/get_object")
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


@router.post("/collection_metadata")
async def collection_metadata(
    data: CollectionMetadataData,
    tree_manager: TreeManager = Depends(get_tree_manager)
):
    error_message = ""
    try:
        tree = tree_manager.get_tree(data.user_id, data.conversation_id)
    except Exception as e:
        error_message = str(e)
    return JSONResponse(content={"metadata": tree.collection_information, "error": error_message}, status_code=200)