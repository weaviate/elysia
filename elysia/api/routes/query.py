import asyncio

from fastapi import WebSocket, APIRouter

from starlette.websockets import WebSocketDisconnect

# API Types
from elysia.api.api_types import QueryData

# Websocket
from elysia.api.utils.websocket import help_websocket

# Logging
from elysia.api.core.logging import logger

# Types
from elysia.api.objects import Error

# Services
from elysia.api.services.tree import TreeManager
from elysia.api.dependencies.common import get_tree_manager

router = APIRouter()

async def process(data: dict, websocket: WebSocket, tree_manager):
    tree = tree_manager.get_tree(data["user_id"], data["conversation_id"])
    tree.soft_reset()
    
    try:

        if "route" in data:
            route = data["route"]
        else:
            route = None
        
        if "mimick" in data:
            mimick = data["mimick"]
        else:
            mimick = False
    
        async for yielded_result in tree.process(
            data["query"], 
            query_id=data["query_id"],
            training_route=route,
            training_mimick_model=mimick
        ):
            try:
                await websocket.send_json(yielded_result)
            except WebSocketDisconnect:
                logger.info("Client disconnected during processing")
                break
            # Add a small delay between messages to prevent overwhelming
            await asyncio.sleep(0.001)
    
    except Exception as e:
        logger.error(f"Error in process function: {str(e)}")

        if "conversation_id" in data:
            await websocket.send_json(
                Error(
                    text=str(e)
                ).to_frontend(data["conversation_id"])
            )
        else:
            await websocket.send_json(
                Error(
                    text=str(e)
                ).to_frontend("")
            )

# Process endpoint
@router.websocket("/query")
async def query_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for processing pipelines.
    Handles real-time communication for pipeline execution and status updates.
    """
    tree_manager = get_tree_manager()
    await help_websocket(
        websocket, 
        lambda data, ws: process(data, ws, tree_manager)
    )