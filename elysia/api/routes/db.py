from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager

# Logging
from elysia.api.core.log import logger

router = APIRouter()


@router.get("/{user_id}/saved_trees")
async def get_saved_trees(
    user_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):

    headers = {"Cache-Control": "no-cache"}

    user = await user_manager.get_user_local(user_id)
    save_location_client_manager = user["frontend_config"].save_location_client_manager
    if not save_location_client_manager.is_client:
        logger.warning(
            "In /get_saved_trees API, "
            "no valid destination for trees location found. "
            "Returning no error but an empty list of trees."
        )
        return JSONResponse(
            content={"trees": {}, "error": ""},
            status_code=200,
            headers=headers,
        )

    try:
        trees = await user_manager.get_saved_trees(
            user_id, save_location_client_manager
        )
        return JSONResponse(
            content={"trees": trees, "error": ""}, status_code=200, headers=headers
        )

    except Exception as e:
        logger.error(f"Error getting saved trees: {str(e)}")
        return JSONResponse(
            content={"trees": {}, "error": str(e)}, status_code=500, headers=headers
        )


@router.get("/{user_id}/load_tree/{conversation_id}")
async def load_tree(
    user_id: str,
    conversation_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):

    headers = {"Cache-Control": "no-cache"}

    try:
        frontend_rebuild = await user_manager.load_tree(user_id, conversation_id)
        return JSONResponse(
            content={"rebuild": frontend_rebuild, "error": ""},
            status_code=200,
            headers=headers,
        )
    except Exception as e:
        logger.error(f"Error loading tree: {str(e)}")
        return JSONResponse(
            content={"rebuild": [], "error": str(e)},
            status_code=500,
            headers=headers,
        )


@router.post("/{user_id}/save_tree/{conversation_id}")
async def save_tree(
    user_id: str,
    conversation_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    try:
        await user_manager.save_tree(user_id, conversation_id)
        return JSONResponse(content={"error": ""}, status_code=200)
    except Exception as e:
        logger.error(f"Error saving tree: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.delete("/{user_id}/delete_tree/{conversation_id}")
async def delete_tree(
    user_id: str,
    conversation_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    try:
        await user_manager.delete_tree(user_id, conversation_id)
        return JSONResponse(content={"error": ""}, status_code=200)
    except Exception as e:
        logger.error(f"Error deleting tree: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
