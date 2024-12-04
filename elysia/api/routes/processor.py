from fastapi import APIRouter
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

import asyncio

# Websocket
from elysia.api.utils.websocket import help_websocket

# Preprocessing
from elysia.preprocess.collection import CollectionPreprocessor

# API Types
from elysia.api.api_types import ProcessCollectionData

# Logging
from elysia.api.core.logging import logger

router = APIRouter()

async def process_collection(data: ProcessCollectionData, websocket: WebSocket):
    if "lm" in data:
        lm = data["lm"]
    else:
        lm = "claude-3-5-sonnet-20241022"

    try:
        preprocessor = CollectionPreprocessor(lm=lm)
        async for result in preprocessor(data["collection_name"], force=data["force"]):
            try:
                await websocket.send_json(result)
            except WebSocketDisconnect:
                logger.info("Client disconnected during processing process_collection")
                break
            # Add a small delay between messages to prevent overwhelming
            await asyncio.sleep(0.001)
        
            
    except Exception as e:
        logger.error(f"Error in process_collection_websocket: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "collection_name": data["collection_name"],
            "progress": 0,
            "error": str(e)
        })


@router.websocket("/process_collection")
async def process_collection_websocket(websocket: WebSocket):
    await help_websocket(websocket, process_collection)

