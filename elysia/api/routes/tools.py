import elysia.api.custom_tools as custom_tools
from elysia import Tool
from typing import Dict, Type
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.api.core.log import logger
from elysia.api.utils.config import FrontendConfig
from elysia.api.api_types import ToolPreset
from elysia.api.utils.tools import (
    add_preset_weaviate,
    get_presets_weaviate,
    delete_preset_weaviate,
)
from elysia.util.client import ClientManager

router = APIRouter()


def find_tool_classes() -> Dict[str, Type[Tool]]:
    """
    Find all Tool subclasses in the custom_tools module.

    Returns:
        (dict): Dictionary mapping class names to Tool subclass types
    """
    tool_classes = {}

    # Get all objects from the custom_tools module
    module_objects = dict(
        [
            (name, cls)
            for name, cls in custom_tools.__dict__.items()
            if isinstance(cls, type)
        ]
    )

    # Filter for Tool subclasses (excluding the base Tool class)
    for name, cls in module_objects.items():
        if issubclass(cls, Tool) and cls.__name__ != "Tool":
            tool_classes[name] = cls

    return tool_classes


def find_tool_metadata() -> Dict[str, Dict[str, str]]:
    """
    Find all tool metadata in the custom_tools module.

    Returns:
        (dict): Dictionary mapping tool names to tool metadata
    """
    tools = find_tool_classes()
    metadata = {}
    for name, cls in tools.items():
        metadata[name] = cls.get_metadata()
    return metadata


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
    data: ToolPreset,
    user_manager: UserManager = Depends(get_user_manager),
):
    user = await user_manager.get_user_local(user_id)
    fe_config: FrontendConfig = user["frontend_config"]

    if fe_config.config["save_configs_to_weaviate"]:
        client_manager: ClientManager = fe_config.save_location_client_manager
        await add_preset_weaviate(
            user_id,
            data.preset_id,
            data.name,
            data.order,
            data.branches,
            data.default,
            client_manager,
        )
        await user["tool_preset_manager"].retrieve(user_id, client_manager)
    else:
        user["tool_preset_manager"].add(
            preset_id=data.preset_id,
            name=data.name,
            order=data.order,
            branches=data.branches,
            default=data.default,
        )
    return JSONResponse(content={"error": ""}, status_code=200)


@router.get("/{user_id}")
async def get_tool_presets(
    user_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    user = await user_manager.get_user_local(user_id)
    return JSONResponse(
        content={"presets": user["tool_preset_manager"].to_json(), "error": ""},
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

    if fe_config.config["save_configs_to_weaviate"]:
        client_manager: ClientManager = fe_config.save_location_client_manager
        await delete_preset_weaviate(user_id, preset_id, client_manager)
        await user["tool_preset_manager"].retrieve(user_id, client_manager)
    else:
        user["tool_preset_manager"].remove(preset_id)
    return JSONResponse(content={"error": ""}, status_code=200)
