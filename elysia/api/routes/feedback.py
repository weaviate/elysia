from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from elysia.api.api_types import (
    AddFeedbackData,
    RemoveFeedbackData,
)
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.api.utils.feedback import (
    create_feedback,
    feedback_metadata,
    remove_feedback,
)
from elysia.api.core.log import logger
from elysia.util.client import ClientManager

router = APIRouter()


@router.post("/add")
async def run_add_feedback(
    data: AddFeedbackData, user_manager: UserManager = Depends(get_user_manager)
):
    logger.debug(f"/add_feedback API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Conversation ID: {data.conversation_id}")
    logger.debug(f"Query ID: {data.query_id}")
    logger.debug(f"Feedback: {data.feedback}")

    tree = await user_manager.get_tree(data.user_id, data.conversation_id)
    user = user_manager.users[data.user_id]
    client_manager: ClientManager = user["client_manager"]

    async with client_manager.connect_to_async_client() as client:
        try:
            await create_feedback(
                data.user_id,
                data.conversation_id,
                data.query_id,
                data.feedback,
                tree,
                client,
            )
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    return JSONResponse(content={"error": ""}, status_code=200)


@router.post("/remove")
async def run_remove_feedback(
    data: RemoveFeedbackData, user_manager: UserManager = Depends(get_user_manager)
):
    logger.debug(f"/remove_feedback API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Conversation ID: {data.conversation_id}")
    logger.debug(f"Query ID: {data.query_id}")

    user = user_manager.users[data.user_id]
    client_manager: ClientManager = user["client_manager"]

    async with client_manager.connect_to_async_client() as client:
        try:
            await remove_feedback(
                data.user_id, data.conversation_id, data.query_id, client
            )
            return JSONResponse(content={"error": ""}, status_code=200)
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/metadata/{user_id}")
async def run_feedback_metadata(
    user_id: str, user_manager: UserManager = Depends(get_user_manager)
):
    logger.debug(f"/feedback_metadata API request received")
    logger.debug(f"User ID: {user_id}")

    user = user_manager.users[user_id]
    client_manager: ClientManager = user["client_manager"]

    async with client_manager.connect_to_async_client() as client:
        metadata = await feedback_metadata(client, user_id)

    return JSONResponse(
        content={
            "error": "",
            **metadata,
        },
        status_code=200,
    )
