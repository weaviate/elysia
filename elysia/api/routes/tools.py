from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.api.core.log import logger
from elysia.api.utils.config import FrontendConfig
from elysia.api.api_types import TreeGraph
from elysia.api.utils.tools import (
    add_preset_weaviate,
    delete_preset_weaviate,
    find_tool_metadata,
)
from elysia.util.client import ClientManager

router = APIRouter()


@router.get("/available")
async def get_available_tools():
    headers = {"Cache-Control": "no-cache"}

    try:
        return JSONResponse(
            content={"tools": find_tool_metadata(), "error": ""},
            status_code=200,
            headers=headers,
        )
    except Exception as e:
        logger.error(f"Error getting available tools: {str(e)}")
        return JSONResponse(
            content={"tools": {}, "error": str(e)},
            status_code=500,
            headers=headers,
        )


@router.post("/{user_id}")
async def add_tool_preset(
    user_id: str,
    data: TreeGraph,
    user_manager: UserManager = Depends(get_user_manager),
):
    user = await user_manager.get_user_local(user_id)
    fe_config: FrontendConfig = user["frontend_config"]

    if (
        fe_config.config["save_configs_to_weaviate"]
        and fe_config.save_location_client_manager.is_client
    ):
        logger.debug(f"Adding tool preset {data.id} to Weaviate")
        client_manager: ClientManager = fe_config.save_location_client_manager
        await add_preset_weaviate(
            user_id,
            data.id,
            data.name,
            data.nodes,
            data.edges,
            data.default,
            client_manager,
        )
        await user["tree_graph_manager"].sync(user_id, client_manager)
    else:
        logger.debug(f"Adding tool preset {data.id} to local tree graph manager")
        user["tree_graph_manager"].add(
            id=data.id,
            name=data.name,
            nodes=data.nodes,
            edges=data.edges,
            default=data.default,
        )
    return JSONResponse(
        content={"presets": user["tree_graph_manager"].to_json(), "error": ""},
        status_code=200,
    )


@router.get("/{user_id}")
async def get_tool_presets(
    user_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    user = await user_manager.get_user_local(user_id)
    return JSONResponse(
        content={"presets": user["tree_graph_manager"].to_json(), "error": ""},
        status_code=200,
    )


@router.delete("/{user_id}/{preset_id}")
async def delete_tool_preset(
    user_id: str,
    preset_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    user = await user_manager.get_user_local(user_id)
    fe_config: FrontendConfig = user["frontend_config"]

    if (
        fe_config.config["save_configs_to_weaviate"]
        and fe_config.save_location_client_manager.is_client
    ):
        logger.debug(f"Deleting tool preset {preset_id} from Weaviate")
        client_manager: ClientManager = fe_config.save_location_client_manager
        await delete_preset_weaviate(user_id, preset_id, client_manager)
        await user["tree_graph_manager"].sync(user_id, client_manager)
    else:
        logger.debug(f"Deleting tool preset {preset_id} from local tree graph manager")
        user["tree_graph_manager"].remove(preset_id)

    return JSONResponse(
        content={"presets": user["tree_graph_manager"].to_json(), "error": ""},
        status_code=200,
    )
