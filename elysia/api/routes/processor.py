import asyncio

from fastapi import APIRouter, Depends
from fastapi.websockets import WebSocket, WebSocketDisconnect
from typing import Callable
import psutil
import time

# Logging
from elysia.api.core.log import logger

# User manager
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager

# Websocket
from elysia.api.utils.websocket import help_websocket

# Preprocessing
from elysia.preprocessing.collection import preprocess_async

router = APIRouter()


async def process_collection(
    data: dict, websocket: WebSocket, user_manager: UserManager
):
    logger.debug(f"/process_collection API request received")
    logger.debug(f"User ID: {data['user_id']}")
    logger.debug(f"Collection name: {data['collection_name']}")

    user = await user_manager.get_user_local(data["user_id"])
    settings = user["tree_manager"].settings

    async for result in preprocess_async(
        collection_name=data["collection_name"],
        client_manager=user["client_manager"],
        force=True,
        settings=settings,
    ):
        try:
            logger.debug(
                f"(process_collection) sending result with progress: {result['progress']*100}%"
            )
            await websocket.send_json(result)
        except WebSocketDisconnect:
            logger.info("Client disconnected during process_collection")
            break
        await asyncio.sleep(0.001)
    logger.debug(f"(process_collection) FINISHED!")


@router.websocket("/process_collection")
async def process_collection_websocket(
    websocket: WebSocket, user_manager: UserManager = Depends(get_user_manager)
):
    await help_websocket(
        websocket, lambda data, ws: process_collection(data, ws, user_manager)
    )
