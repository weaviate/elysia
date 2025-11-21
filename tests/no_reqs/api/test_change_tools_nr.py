import pytest
import os
from uuid import uuid4

from elysia.api.utils.config import default_presets
from weaviate.util import generate_uuid5
from elysia.api.services.user import UserManager
from elysia.api.api_types import (
    InitialiseTreeData,
    SaveConfigUserData,
    UpdateFrontendConfigData,
    ToolPreset,
    ToolItem,
    BranchInfo,
)
from elysia.api.routes.query import process
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.init import initialise_user, initialise_tree
from elysia.api.routes.user_config import (
    update_frontend_config,
    save_config_user,
    get_current_user_config,
)
from elysia.api.routes.tools import (
    get_available_tools,
    add_tool_preset,
    get_tool_presets,
    delete_tool_preset,
)
from elysia.api.custom_tools import TellAJoke
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.init import initialise_user, initialise_tree
from elysia.api.routes.user_config import (
    update_frontend_config,
    save_config_user,
    get_current_user_config,
)
from elysia.api.routes.query import process

from fastapi.responses import JSONResponse
import json

from elysia.api.core.log import logger, set_log_level

set_log_level("CRITICAL")


def read_response(response: JSONResponse):
    return json.loads(response.body)


async def initialise_user_and_tree(user_id: str, conversation_id: str):
    user_manager = get_user_manager()

    response = await initialise_user(
        user_id,
        user_manager,
    )

    response = await initialise_tree(
        user_id,
        conversation_id,
        InitialiseTreeData(
            low_memory=False,
        ),
        user_manager,
    )


@pytest.mark.asyncio
async def test_tools_exist():
    joke_tool = TellAJoke()
    response = await get_available_tools()
    response = read_response(response)
    assert response["error"] == ""
    assert "tools" in response
    assert response["tools"] is not None
    assert "tell_a_joke" in response["tools"]
    assert response["tools"]["tell_a_joke"]["description"] == joke_tool.description
    assert response["tools"]["tell_a_joke"]["inputs"].keys() == joke_tool.inputs.keys()
    assert response["tools"]["tell_a_joke"]["end"] == joke_tool.end
    assert response["tools"]["tell_a_joke"]["name"] == joke_tool.name


"""
Tests when not saving configs to Weaviate.
"""


@pytest.mark.asyncio
async def test_cycle():
    user_id = f"test_add_tool_preset_{uuid4()}"
    user_manager = get_user_manager()
    user = await initialise_user(user_id, user_manager)

    # save config so that save_configs_to_weaviate is False
    response = await save_config_user(
        user_id=user_id,
        config_id=f"test_add_tool_preset_{uuid4()}",
        data=SaveConfigUserData(
            name="test_add_tool_preset",
            default=True,
            config={},
            frontend_config={"save_configs_to_weaviate": False},
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # get available tools
    response = await get_available_tools()
    available_tools = read_response(response)["tools"]

    # add tool preset
    tool_preset_id = f"test_add_tool_preset_{uuid4()}"
    response = await add_tool_preset(
        user_id=user_id,
        data=ToolPreset(
            preset_id=tool_preset_id,
            name="Test Add Tool Preset",
            order=[
                ToolItem(
                    instance_id="base",
                    name="base",
                    from_branch="",
                    from_tools=[],
                    is_branch=True,
                ),
            ]
            + [
                ToolItem(
                    name=tool["name"],
                    from_branch="base",
                    from_tools=[],
                    is_branch=False,
                )
                for tool in list(available_tools.values())[:2]
            ]
            + [
                ToolItem(
                    instance_id="secondary_branch",
                    name="secondary_branch",
                    from_branch="base",
                    from_tools=[],
                    is_branch=True,
                )
            ]
            + [
                ToolItem(
                    name=tool["name"],
                    from_branch="secondary_branch",
                    from_tools=[],
                    is_branch=False,
                )
                for tool in list(available_tools.values())[2:4]
            ],
            branches=[
                BranchInfo(
                    reference_id="base",
                    description="Secondary branch",
                    instruction="Use the secondary branch to get the information",
                    is_root=False,
                ),
                BranchInfo(
                    reference_id="secondary_branch",
                    description="Secondary branch",
                    instruction="Use the secondary branch to get the information",
                    is_root=False,
                ),
            ],
            default=True,
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    user_local = await user_manager.get_user_local(user_id)
    tool_preset_manager = user_local["tool_preset_manager"]
    assert len(tool_preset_manager.tool_presets) == len(default_presets) + 1
    assert tool_preset_manager.tool_presets[-1].preset_id == tool_preset_id
    assert tool_preset_manager.tool_presets[-1].name == "Test Add Tool Preset"
    assert len(tool_preset_manager.tool_presets[-1].order) == 6

    # get tool presets
    response = await get_tool_presets(user_id=user_id, user_manager=user_manager)
    response = read_response(response)
    assert response["error"] == ""
    assert "presets" in response
    assert response["presets"] is not None
    assert len(response["presets"]) == len(default_presets) + 1
    assert response["presets"][-1]["preset_id"] == tool_preset_id
    assert response["presets"][-1]["name"] == "Test Add Tool Preset"
    assert len(response["presets"][-1]["order"]) == 6

    # remove tool preset
    response = await delete_tool_preset(
        user_id=user_id, preset_id=tool_preset_id, user_manager=user_manager
    )
    response = read_response(response)
    assert response["error"] == ""

    # check if its been deleted
    response = await get_tool_presets(user_id=user_id, user_manager=user_manager)
    response = read_response(response)
    assert response["error"] == ""
    assert "presets" in response
    assert response["presets"] is not None
    assert len(response["presets"]) == len(default_presets)
